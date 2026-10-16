/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import ai.rapids.cudf.{ColumnVector, OutOfBoundsPolicy, Scalar, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, Expression,
  NamedExpression, SortOrder}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuAddBase, GpuAnd, GpuDecimalDivide, GpuDecimalMultiply,
  GpuEqualTo, GpuGreaterThan,
  GpuGreaterThanOrEqual, GpuLessThan, GpuLessThanOrEqual, GpuMultiply, GpuNot, GpuOr,
  GpuSubtractBase}
import org.apache.spark.sql.types.{BooleanType, ByteType, DataType, DecimalType, IntegerType, LongType, ShortType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuFilterProjectExec extends Logging {
  private def supportedType(dt: DataType): Boolean = dt match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType => true
    case _: DecimalType => true
    case _ => false
  }

  private def decimalCast(cast: GpuCast): Boolean = {
    def numeric(dt: DataType): Boolean = supportedType(dt) && dt != BooleanType
    !cast.ansiMode && numeric(cast.child.dataType) && numeric(cast.dataType) &&
      (cast.child.dataType.isInstanceOf[DecimalType] || cast.dataType.isInstanceOf[DecimalType])
  }

  private def pure(expr: Expression): Boolean = expr.deterministic && (expr match {
    case g: GpuExpression => !g.hasSideEffects
    case _: AttributeReference => true
    case _ => false
  })

  private def direct(expr: Expression): Boolean = pure(expr) && (expr match {
    case a: GpuAlias => direct(a.child)
    case _: AttributeReference => true
    case _ => false
  })

  // Deliberately closed whitelist. Retryability alone also admits stateful expressions.
  private def safe(expr: Expression): Boolean = {
    pure(expr) && supportedType(expr.dataType) && (expr match {
      case _: AttributeReference | _: GpuLiteral => true
      // Decimal wrappers and casts can throw without being fully described by hasSideEffects
      // in every Spark shim. Keep explicit guards and preserve their existing evaluation.
      case a: GpuProjectAstExpression => safe(a.child)
      case c: GpuCast => decimalCast(c) && safe(c.child)
      case c: GpuCheckOverflow => c.nullOnOverflow && safe(c.child)
      case m: GpuDecimalMultiply => !m.failOnError && m.children.forall(safe)
      case d: GpuDecimalDivide =>
        !d.failOnError && !d.failOnDivideByZero && d.children.forall(safe)
      case p: GpuPromotePrecision => safe(p.child)
      case _: GpuAlias | _: GpuNot | _: GpuAnd | _: GpuOr |
           _: GpuEqualTo | _: GpuGreaterThan | _: GpuGreaterThanOrEqual |
           _: GpuLessThan | _: GpuLessThanOrEqual | _: GpuIsNull | _: GpuIsNotNull |
           _: GpuAddBase | _: GpuSubtractBase | _: GpuMultiply => expr.children.forall(safe)
      case _ => false
    })
  }

  private[rapids] def eligible(condition: Expression, projects: Seq[Expression]): Boolean =
    safe(condition) && projects.forall(p => direct(p) || safe(p))

  private def skipped(project: SparkPlan, explain: Boolean): SparkPlan = {
    if (explain) {
      logInfo(s"Filter/project fusion skipped for ${project.nodeName}: expressions must be " +
        "deterministic, side-effect free, and in the fusion whitelist; " +
        "the project must allow splitting and the filter must allow coalescing.")
    }
    project
  }

  private[rapids] def fuse(
      plan: SparkPlan,
      explain: Boolean = false): SparkPlan = plan.transformUp {
    // Some callers require a single batch and intentionally disable project splitting.
    case p @ GpuProjectExec(projects, f: GpuFilterExec, allowSplit) =>
      if (allowSplit && f.coalesceAfter && eligible(f.condition, projects)) {
        GpuFilterProjectExec(f.condition, projects, f.child, useAst = false)
      } else {
        skipped(p, explain)
      }
  }
}

/** Filter selection is internal; downstream operators receive ordinary compact batches. */
case class GpuFilterProjectExec(
    condition: Expression,
    projectList: List[Expression],
    child: SparkPlan,
    useAst: Boolean) extends GpuPartitioningPreservingUnaryExecNode with GpuExec {

  override def output: Seq[Attribute] = projectList.map {
    case ne: NamedExpression => ne.toAttribute
  }

  override val coalesceAfter: Boolean = true
  override def outputBatching: CoalesceGoal = null
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override protected def buildAttributeMap(): Map[Attribute, Attribute] = {
    // Preserve the original project's mapping, including Filter's nullability refinement.
    GpuFilterExec(condition, child)().output.zip(output).toMap
  }

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    NUM_INPUT_ROWS -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_ROWS),
    NUM_INPUT_BATCHES -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_BATCHES),
    "dependencyColumns" -> createMetric(DEBUG_LEVEL, "computed input columns per output batch"),
    "directColumns" -> createMetric(DEBUG_LEVEL, "direct output columns per output batch"))

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val inputRows = gpuLongMetric(NUM_INPUT_ROWS)
    val inputBatches = gpuLongMetric(NUM_INPUT_BATCHES)
    val outputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val outputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val dependencies = gpuLongMetric("dependencyColumns")
    val directColumns = gpuLongMetric("directColumns")
    val opTime = gpuLongMetric(OP_TIME_LEGACY)
    // Bind once on the driver, as the existing filter/project operators do. SQLConf is transient.
    val bound = GpuFilterProject.bind(condition, projectList, child.output, useAst, conf,
      allMetrics, Some(GpuFilterExec(condition, child)().output))
    child.executeColumnar().mapPartitions { input =>
      new GpuColumnarBatchIterator(true) {
        private val evaluator = new GpuFilterProject(bound)
        private var pending: Iterator[ColumnarBatch] = Iterator.empty

        override def hasNext: Boolean = {
          val ret = pending.hasNext || input.hasNext
          if (!ret) {
            close()
          }
          ret
        }

        override def next(): ColumnarBatch = {
          if (!hasNext) {
            throw new NoSuchElementException()
          }
          if (!pending.hasNext) {
            val batch = input.next()
            inputRows += batch.numRows()
            inputBatches += 1
            val sb = SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
            pending = GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
          }
          val ret = opTime.ns { pending.next() }
          outputRows += ret.numRows()
          outputBatches += 1
          dependencies += evaluator.dependencyOrdinals.length
          directColumns += evaluator.directOrdinals.length
          ret
        }

        override def doClose(): Unit = {
          if (evaluator != null) {
            evaluator.close()
          }
        }
      }
    }
  }
}

private[rapids] case class BoundFilterProject(
    predicate: GpuTieredProject,
    layout: Seq[Option[Int]],
    directOrdinals: Seq[Int],
    dependencyOrdinals: Seq[Int],
    computedTypes: Array[DataType],
    tiered: GpuTieredProject,
    astExpressions: Seq[GpuExpression],
    fullProjection: Option[GpuTieredProject],
    allInputsNeeded: Boolean,
    useAst: Boolean)

private[rapids] object GpuFilterProject {
  private def directOrdinal(expr: Expression): Option[Int] = expr match {
    case a: GpuAlias => directOrdinal(a.child)
    case r: GpuBoundReference => Some(r.ordinal)
    case _ => None
  }
  def bind(condition: Expression, projects: Seq[Expression], input: Seq[Attribute],
      useAst: Boolean, conf: SQLConf, metrics: Map[String, GpuMetric],
      projectionInput: Option[Seq[Attribute]] = None): BoundFilterProject = {
    val projectAttrs = projectionInput.getOrElse(input)
    val predicate = GpuBindReferences.bindGpuReferencesTiered(Seq(condition), input, conf, metrics)
    val layout = GpuBindReferences.bindGpuReferences(projects, projectAttrs, metrics)
      .map(directOrdinal)
    val directOrdinals = layout.flatten.distinct
    val computed = projects.zip(layout).collect { case (p, None) => p }
    val references = computed.flatMap(_.references).map(_.exprId).toSet
    val dependencies = input.indices.filter(i => references(input(i).exprId))
    val computedInput = dependencies.map(projectAttrs)
    val tiered = GpuBindReferences.bindGpuReferencesTiered(computed, computedInput, conf, metrics)
    val astExpressions = if (useAst) {
      GpuBindReferences.bindGpuReferences(computed, computedInput, metrics)
    } else {
      Nil
    }
    val fullProjection = if (!useAst && projects.nonEmpty) {
      Some(GpuBindReferences.bindGpuReferencesTiered(projects, projectAttrs, conf, metrics))
    } else {
      None
    }
    val allInputsNeeded = input.nonEmpty &&
      (dependencies ++ directOrdinals).distinct.length == input.length
    BoundFilterProject(predicate, layout, directOrdinals, dependencies,
      computed.map(_.dataType).toArray, tiered, astExpressions, fullProjection, allInputsNeeded,
      useAst)
  }
}

/** A partition-local evaluator. Does not take ownership of its input batch. */
private[rapids] class GpuFilterProject(bound: BoundFilterProject) extends AutoCloseable {
  def this(condition: Expression, projects: Seq[Expression], input: Seq[Attribute],
      useAst: Boolean, conf: SQLConf, metrics: Map[String, GpuMetric]) =
    this(GpuFilterProject.bind(condition, projects, input, useAst, conf, metrics))

  val directOrdinals: Seq[Int] = bound.directOrdinals
  val dependencyOrdinals: Seq[Int] = bound.dependencyOrdinals
  private val asts = if (bound.useAst) {
    bound.astExpressions.safeMap(_.convertToAst(Int.MaxValue).compile())
  } else {
    Nil
  }

  private def select(batch: ColumnarBatch, ordinals: Seq[Int]): ColumnarBatch = {
    val columns = ordinals.safeMap { i =>
      batch.column(i).asInstanceOf[GpuColumnVector].incRefCount()
    }
    closeOnExcept(columns) { _ =>
      new ColumnarBatch(columns.toArray, batch.numRows())
    }
  }

  private def gather(batch: ColumnarBatch, ordinals: Seq[Int],
      indices: Option[ColumnVector]): ColumnarBatch = {
    val rows = indices.map(_.getRowCount.toInt).getOrElse(batch.numRows())
    if (ordinals.isEmpty) {
      new ColumnarBatch(Array.empty, rows)
    } else {
      withResource(select(batch, ordinals)) { selected =>
        indices.map { map =>
          withResource(GpuColumnVector.from(selected)) { table =>
            // Indices come from filtering [0, numRows), so bounds checks are unnecessary.
            withResource(table.gather(map, OutOfBoundsPolicy.DONT_CHECK)) { gathered =>
              GpuColumnVector.from(gathered, ordinals.map(i => batch.column(i).dataType()).toArray)
            }
          }
        }.getOrElse(GpuColumnVector.incRefCounts(selected))
      }
    }
  }

  private def project(batch: ColumnarBatch): ColumnarBatch = {
    if (bound.computedTypes.isEmpty) {
      new ColumnarBatch(Array.empty, batch.numRows())
    } else if (!bound.useAst) {
      bound.tiered.project(batch)
    } else {
      // cudf cannot represent a zero-column table; literal-only ASTs need a row-count carrier.
      val table = if (batch.numCols() > 0) {
        GpuColumnVector.from(batch)
      } else {
        withResource(Scalar.fromBool(false)) { scalar =>
          withResource(ColumnVector.fromScalar(scalar, batch.numRows())) { column =>
            new Table(column)
          }
        }
      }
      withResource(table) { t =>
        withResource(asts.safeMap(_.computeColumn(t))) { columns =>
          withResource(new Table(columns: _*)) { result =>
            GpuColumnVector.from(result, bound.computedTypes)
          }
        }
      }
    }
  }

  private def evaluateSelected(batch: ColumnarBatch,
      selection: Option[ColumnVector]): ColumnarBatch = {
    withResource(gather(batch, dependencyOrdinals, selection)) { inputs =>
      withResource(project(inputs)) { computedBatch =>
        withResource(gather(batch, directOrdinals, selection)) { directBatch =>
          var computedIndex = 0
          val columns = bound.layout.safeMap {
            case Some(ordinal) =>
              directBatch.column(directOrdinals.indexOf(ordinal))
                .asInstanceOf[GpuColumnVector].incRefCount()
            case None =>
              val column = computedBatch.column(computedIndex)
                .asInstanceOf[GpuColumnVector].incRefCount()
              computedIndex += 1
              column
          }
          closeOnExcept(columns) { _ =>
            new ColumnarBatch(columns.toArray, inputs.numRows())
          }
        }
      }
    }
  }

  private def filterNeededInput(batch: ColumnarBatch, mask: ColumnVector): ColumnarBatch = {
    withResource(GpuColumnVector.from(batch)) { table =>
      withResource(table.filter(mask)) { filtered =>
        GpuColumnVector.from(filtered, GpuColumnVector.extractTypes(batch))
      }
    }
  }

  def apply(batch: ColumnarBatch): ColumnarBatch = {
    withResource(GpuFilter.computeCheckedFilterMask(bound.predicate, batch)) { mask =>
      if (mask.isEmpty && bound.fullProjection.isDefined) {
        bound.fullProjection.get.project(batch)
      } else if (mask.isDefined && bound.allInputsNeeded) {
        // All columns are necessary: filtering them directly avoids an extra index column.
        withResource(filterNeededInput(batch, mask.get)) { filtered =>
          bound.fullProjection.map(_.project(filtered)).getOrElse(evaluateSelected(filtered, None))
        }
      } else {
        val indices = mask.map { m =>
          withResource(Scalar.fromInt(0)) { start =>
            withResource(ColumnVector.sequence(start, batch.numRows())) { sequence =>
              withResource(new Table(sequence)) { table =>
                withResource(table.filter(m)) { selected =>
                  selected.getColumn(0).incRefCount()
                }
              }
            }
          }
        }
        withResource(indices)(selection => evaluateSelected(batch, selection))
      }
    }
  }

  override def close(): Unit = asts.safeClose()
}
