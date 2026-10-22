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

import ai.rapids.cudf.{ColumnVector, OutOfBoundsPolicy, Scalar}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.shims.ShimUnaryExecNode

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, Expression}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuGenerateFilterExec extends Logging {

  /** Name is internal; only its position in the probe schema matters. */
  private[rapids] def indexAttribute: AttributeReference =
    AttributeReference("__gpu_generate_row_index__", IntegerType, nullable = false)()

  /**
   * Bytes per row of the columns the explode would replicate. The fused path replaces that
   * replication with a four byte index, so this is what it stands to save per discarded row.
   */
  private[rapids] def replicatedWidth(requiredChildOutput: Seq[Attribute]): Long =
    requiredChildOutput.map(_.dataType.defaultSize.toLong).sum

  /**
   * A predicate that reads only the generator's own output can be evaluated before the payload
   * is ever replicated. One that also reads a replicated column would have to gather that column
   * first, which is a larger change and is left to the unfused path.
   */
  private[rapids] def readsOnlyGeneratorOutput(condition: Expression,
      generatorOutput: Seq[Attribute]): Boolean = {
    val produced = generatorOutput.map(_.exprId).toSet
    condition.references.forall(ref => produced(ref.exprId))
  }

  private[rapids] def eligible(gen: GpuGenerateExec, condition: Expression,
      minWidth: Long): Option[String] = gen.generator match {
    case explode: GpuExplodeBase if explode.child.dataType.isInstanceOf[ArrayType] =>
      if (explode.fixedLenLazyExpressions.nonEmpty) {
        Some("the generator uses the fixed length lazy array path")
      } else if (gen.requiredChildOutput.isEmpty) {
        Some("there are no replicated columns to defer")
      } else if (!readsOnlyGeneratorOutput(condition, gen.generatorOutput)) {
        Some("the predicate also reads a replicated column")
      } else if (replicatedWidth(gen.requiredChildOutput) < minWidth) {
        Some(s"replicated columns are only " +
          s"${replicatedWidth(gen.requiredChildOutput)} bytes per row")
      } else {
        None
      }
    case other => Some(s"${other.getClass.getSimpleName} over " +
      s"${gen.generator.children.head.dataType.simpleString} is not supported")
  }

  def fuse(plan: SparkPlan, minWidth: Long, explain: Boolean = false): SparkPlan =
    plan.transformUp {
      case filter @ GpuFilterExec(condition, gen: GpuGenerateExec) =>
        eligible(gen, condition, minWidth) match {
          case Some(reason) =>
            if (explain) {
              logInfo(s"Generate/filter fusion skipped for ${gen.nodeName}: $reason")
            }
            filter
          case None =>
            if (explain) {
              logInfo(s"Generate/filter fusion keeps " +
                s"${replicatedWidth(gen.requiredChildOutput)} bytes per row unreplicated " +
                s"for ${gen.nodeName}")
            }
            GpuGenerateFilterExec(gen.generator, gen.requiredChildOutput, gen.outer,
              gen.generatorOutput, condition, gen.child)
        }
    }
}

/**
 * An explode whose consuming filter runs before the replicated columns are materialized.
 *
 * The unfused pair replicates every retained child column once per generated row and then throws
 * away the rows the predicate rejects. This explodes a row index alongside the generator input,
 * applies the predicate to the narrow result, and gathers the payload once for the rows that
 * survive. Rows the predicate rejects never have their payload copied.
 */
case class GpuGenerateFilterExec(
    generator: GpuGenerator,
    requiredChildOutput: Seq[Attribute],
    outer: Boolean,
    generatorOutput: Seq[Attribute],
    condition: Expression,
    child: SparkPlan) extends ShimUnaryExecNode with GpuExec {

  // Built from the operators this replaces so the schema, including the nullability a filter
  // refines through its constraints, is identical to the unfused pair.
  @transient private lazy val unfused: SparkPlan = GpuFilterExec(condition,
    GpuGenerateExec(generator, requiredChildOutput, outer, generatorOutput, child))()

  override def output: Seq[Attribute] = unfused.output

  override val coalesceAfter: Boolean = true

  override def outputBatching: CoalesceGoal = null

  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    NUM_INPUT_ROWS -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_ROWS),
    NUM_INPUT_BATCHES -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_BATCHES),
    "generatedRows" -> createMetric(DEBUG_LEVEL, "rows produced before the predicate"))

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val numOutputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val inputRows = gpuLongMetric(NUM_INPUT_ROWS)
    val inputBatches = gpuLongMetric(NUM_INPUT_BATCHES)
    val generatedRows = gpuLongMetric("generatedRows")
    val opTime = gpuLongMetric(OP_TIME_LEGACY)
    val targetSize = new RapidsConf(conf).gpuTargetBatchSizeBytes
    val genProject = GpuBindReferences.bindGpuReferences(
      generator.children, child.output, allMetrics)
    val payloadProject = GpuBindReferences.bindGpuReferences(
      requiredChildOutput, child.output, allMetrics)
    // The probe batch is the row index followed by the generator's own output, so binding
    // against that schema shifts the predicate's ordinals past the index for free.
    val boundCondition = GpuBindReferences.bindGpuReference(condition,
      GpuGenerateFilterExec.indexAttribute +: generatorOutput, allMetrics)
    val payloadTypes = requiredChildOutput.map(_.dataType).toArray
    val localGenerator = generator
    val localOuter = outer

    child.executeColumnar().flatMap { input =>
      inputBatches += 1
      inputRows += input.numRows()
      // Splits are measured on the full width batch, so the memory bound is the one the
      // unfused generator would have used even though the exploded batch is now narrow.
      val projected = NvtxIdWithMetrics(NvtxRegistry.GPU_GENERATE_PROJECT_SPLIT, opTime) {
        GpuProjectExec.projectAndCloseWithRetrySingleBatch(
          SpillableColumnarBatch(input, SpillPriorities.ACTIVE_ON_DECK_PRIORITY),
          payloadProject ++ genProject)
      }
      val splits = GpuGenerateUtils.getSplitsWithRetryAndClose(projected, localGenerator,
        payloadProject.length, localOuter, targetSize, opTime)
      new GpuFusedGenerateIterator(splits, localGenerator, payloadProject.length, payloadTypes,
        localOuter, boundCondition, numOutputRows, numOutputBatches, generatedRows, opTime)
    }
  }
}

/**
 * Produces one split at a time: explode a row index with the generator input, apply the
 * predicate to that narrow result, then gather the split's payload for the surviving rows.
 */
class GpuFusedGenerateIterator(
    inputs: Iterator[Array[SpillableColumnarBatch]],
    generator: GpuGenerator,
    payloadWidth: Int,
    payloadTypes: Array[DataType],
    outer: Boolean,
    boundCondition: Expression,
    numOutputRows: GpuMetric,
    numOutputBatches: GpuMetric,
    generatedRows: GpuMetric,
    opTime: GpuMetric) extends Iterator[ColumnarBatch] with AutoCloseable {

  private val pending = scala.collection.mutable.Queue.empty[SpillableColumnarBatch]
  private var payload: Option[SpillableColumnarBatch] = None
  private var exploded: Iterator[ColumnarBatch] = Iterator.empty
  private var readyBatch: Option[ColumnarBatch] = None
  private var closed = false

  Option(org.apache.spark.TaskContext.get()).foreach { tc =>
    ScalableTaskCompletion.onTaskCompletion(tc)(close())
  }

  private def rowIndex(rows: Int): ColumnVector =
    withResource(Scalar.fromInt(0))(ColumnVector.sequence(_, rows))

  /** Separate a split into the payload that stays put and the narrow batch that explodes. */
  private def prepare(split: SpillableColumnarBatch): Unit = {
    val (payloadBatch, narrow) = withRetryNoSplit(split) { attempt =>
      withResource(attempt.getColumnarBatch()) { cb =>
        val payloadColumns = (0 until payloadWidth).safeMap { i =>
          cb.column(i).asInstanceOf[GpuColumnVector].incRefCount()
        }
        closeOnExcept(payloadColumns) { _ =>
          val payloadBatch = new ColumnarBatch(payloadColumns.toArray, cb.numRows())
          closeOnExcept(payloadBatch) { _ =>
            val genColumn = cb.column(payloadWidth).asInstanceOf[GpuColumnVector]
            val narrowColumns = closeOnExcept(rowIndex(cb.numRows())) { index =>
              Array[org.apache.spark.sql.vectorized.ColumnVector](
                GpuColumnVector.from(index, IntegerType), genColumn.incRefCount())
            }
            (payloadBatch, new ColumnarBatch(narrowColumns, cb.numRows()))
          }
        }
      }
    }
    payload = closeOnExcept(narrow) { _ =>
      Some(SpillableColumnarBatch(payloadBatch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
    }
    val narrowSpillable = SpillableColumnarBatch(narrow, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
    exploded = generator.generate(Iterator(narrowSpillable), 1, outer)
  }

  /** Filter the narrow result, then gather the payload rows that survived. */
  private def materialize(probe: ColumnarBatch): ColumnarBatch = {
    generatedRows += probe.numRows()
    withResource(GpuFilter(probe, boundCondition)) { survivors =>
      if (survivors.numRows() == 0) {
        new ColumnarBatch(Array.empty, 0)
      } else {
        val indices = survivors.column(0).asInstanceOf[GpuColumnVector].getBase
        val gathered = withRetryNoSplit[ColumnarBatch] {
          withResource(payload.get.getColumnarBatch()) { payloadBatch =>
            withResource(GpuColumnVector.from(payloadBatch)) { payloadTable =>
              withResource(payloadTable.gather(indices, OutOfBoundsPolicy.DONT_CHECK)) { table =>
                GpuColumnVector.from(table, payloadTypes)
              }
            }
          }
        }
        withResource(gathered) { gathered =>
          val columns = (0 until gathered.numCols()).safeMap { i =>
            gathered.column(i).asInstanceOf[GpuColumnVector].incRefCount()
          } ++ (1 until survivors.numCols()).safeMap { i =>
            survivors.column(i).asInstanceOf[GpuColumnVector].incRefCount()
          }
          closeOnExcept(columns) { _ =>
            new ColumnarBatch(columns.toArray, survivors.numRows())
          }
        }
      }
    }
  }

  private def advance(): Unit = {
    while (readyBatch.isEmpty) {
      if (exploded.hasNext) {
        val candidate = opTime.ns(withResource(exploded.next())(materialize))
        if (candidate.numRows() > 0) {
          readyBatch = Some(candidate)
        } else {
          candidate.close()
        }
      } else {
        payload.foreach(_.close())
        payload = None
        if (pending.isEmpty && !inputs.hasNext) {
          return
        }
        if (pending.isEmpty) {
          pending ++= inputs.next()
        }
        opTime.ns(prepare(pending.dequeue()))
      }
    }
  }

  override def hasNext: Boolean = {
    if (closed) {
      return false
    }
    advance()
    if (readyBatch.isEmpty) {
      close()
    }
    readyBatch.isDefined
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("no more batches")
    }
    val ret = readyBatch.get
    readyBatch = None
    numOutputBatches += 1
    numOutputRows += ret.numRows()
    ret
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      readyBatch.foreach(_.close())
      readyBatch = None
      payload.foreach(_.close())
      payload = None
      pending.foreach(_.close())
      pending.clear()
      exploded match {
        case closeable: AutoCloseable => closeable.close()
        case _ =>
      }
      exploded = Iterator.empty
    }
  }
}
