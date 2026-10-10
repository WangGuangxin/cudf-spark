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

import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows,
  AutoCloseableAttemptSpliterator, RmmRapidsRetryAutoCloseableIterator, Spliterator}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.shims.ShimUnaryExecNode

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, AttributeSet,
  Expression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, UnknownPartitioning}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum, GpuCount,
  GpuMax, GpuMin}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuExpandAggregateExec extends Logging {
  private def integral(dt: DataType): Boolean = dt match {
    case ByteType | ShortType | IntegerType | LongType => true
    case _ => false
  }

  private def keyType(dt: DataType): Boolean = dt match {
    case BooleanType | DateType | TimestampType | StringType => true
    case _ => integral(dt)
  }

  private def simple(expr: Expression): Boolean = expr match {
    case _: AttributeReference | _: GpuLiteral => true
    case _ => false
  }

  private def groupingReference(expr: Expression): Option[AttributeReference] = expr match {
    case a: AttributeReference => Some(a)
    case GpuAlias(a: AttributeReference, _) => Some(a)
    case _ => None
  }

  private def supportedAggregate(expr: GpuAggregateExpression): Boolean = {
    val fn = expr.aggregateFunction
    expr.mode == Partial && !expr.isDistinct && expr.filter.isEmpty &&
      fn.deterministic && fn.children.forall(simple) && (fn match {
        case c: GpuCount => c.children.length == 1 && !c.failOnError &&
          keyType(c.children.head.dataType)
        case s: GpuBasicSum => !s.failOnErrorOverride && integral(s.child.dataType) &&
          integral(s.resultType)
        case _: GpuMin | _: GpuMax => fn.children.length == 1 &&
          integral(fn.children.head.dataType)
        case _ => false
      })
  }

  private[rapids] def eligible(agg: GpuHashAggregateExec, expand: GpuExpandExec): Boolean = {
    val keys = agg.groupingExpressions.flatMap(groupingReference)
    val keyOrdinals = keys.map(k => expand.output.indexWhere(_.exprId == k.exprId))
    val projections = expand.projections
    val baseShape = !agg.allowSinglePassAgg && !agg.forceSinglePassAgg &&
      agg.requiredChildDistributionExpressions.isEmpty && agg.aggregateExpressions.nonEmpty &&
      agg.aggregateExpressions.forall(supportedAggregate) && keys.nonEmpty &&
      keys.length == agg.groupingExpressions.length && keys.forall(k => keyType(k.dataType)) &&
      keyOrdinals.forall(_ >= 0) && projections.length > 1 &&
      projections.forall(p => p.length == expand.output.length && p.forall(simple))
    baseShape && {
      val validInputs = projections.forall(_.zip(expand.output).forall { case (e, out) =>
        e.dataType == out.dataType && (e match {
          case a: AttributeReference => expand.child.output.exists(_.exprId == a.exprId)
          case _: GpuLiteral => true
          case _ => false
        })
      })
      // Grouping-set payloads do not change between branches; DISTINCT rewrites do.
      val stablePayload = expand.output.indices.filterNot(keyOrdinals.contains).forall { i =>
        projections.tail.forall(p => p(i).semanticEquals(projections.head(i)))
      }
      val stableKeys = keyOrdinals.forall { i =>
        val refs = projections.map(_(i)).collect { case a: AttributeReference => a }
        refs.isEmpty || refs.tail.forall(_.semanticEquals(refs.head))
      }
      val discriminators = keyOrdinals.filter(i => projections.forall(_(i)
        .isInstanceOf[GpuLiteral]))
      // Prove uniqueness structurally, including Spark's extra key for duplicate sets.
      // Names (including spark_grouping_id) are not a discriminator contract.
      val uniqueBranches = discriminators.nonEmpty && projections.indices.forall { i =>
        (0 until i).forall { j =>
          discriminators.exists(k => !projections(i)(k).semanticEquals(projections(j)(k)))
        }
      }
      val bufferOutput = agg.groupingExpressions.map(_.toAttribute) ++
        agg.aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)
      val plainPartialOutput = agg.resultExpressions.length == bufferOutput.length &&
        agg.resultExpressions.forall(_.isInstanceOf[AttributeReference]) &&
        agg.resultExpressions.zip(bufferOutput)
          .forall { case (e, a) => e.dataType == a.dataType } &&
        agg.resultExpressions.take(keys.length).zip(agg.groupingExpressions)
          .forall { case (e, key) => e.semanticEquals(key.toAttribute) }
      validInputs && stablePayload && stableKeys && uniqueBranches && plainPartialOutput
    }
  }

  private[rapids] def fuse(plan: SparkPlan, explain: Boolean = false): SparkPlan =
    plan.transformUp {
      case agg: GpuHashAggregateExec => agg.child match {
        case expand: GpuExpandExec if eligible(agg, expand) =>
          GpuExpandAggregateExec(agg, expand.projections, expand.output, expand.child)
        case _: GpuExpandExec =>
          if (explain) {
            logInfo("Expand/aggregate fusion skipped: requires simple grouping sets with " +
              "unique constant branch keys, supported partial aggregates, and full hash mode.")
          }
          agg
        case _ => agg
      }
    }

  private[rapids] def bind(agg: GpuHashAggregateExec, projections: Seq[Seq[Expression]],
      expandOutput: Seq[Attribute], input: Seq[Attribute], conf: SQLConf,
      metrics: Map[String, GpuMetric]): Seq[ExpandAggregateBranch] = {
    val keyOrdinals = agg.groupingExpressions.map { e =>
      val ref = groupingReference(e).get
      expandOutput.indexWhere(_.exprId == ref.exprId)
    }
    val functionsByBranch = projections.map { projection =>
      val replacements = expandOutput.map(_.exprId).zip(projection).toMap
      agg.aggregateExpressions.map { e =>
        val fn = e.origAggregateFunction.transform {
          case a: AttributeReference if replacements.contains(a.exprId) => replacements(a.exprId)
        }
        e.copy(origAggregateFunction = fn.asInstanceOf[
          org.apache.spark.sql.rapids.aggregate.GpuAggregateFunction])
      }
    }
    val sources = projections.indices.map { target =>
      (0 until target).filter { source =>
        keyOrdinals.forall { i =>
          projections(target)(i).isInstanceOf[GpuLiteral] ||
            projections(target)(i).semanticEquals(projections(source)(i))
        } &&
          functionsByBranch(target).zip(functionsByBranch(source)).forall { case (a, b) =>
            a.aggregateFunction.semanticEquals(b.aggregateFunction)
          }
      }.toSet
    }
    val fullBuffers = agg.groupingExpressions.map(_.toAttribute) ++
      agg.aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)
    projections.zipWithIndex.map { case (projection, index) =>
      val liveKeys = agg.groupingExpressions.zip(keyOrdinals).collect {
        case (key, i) if !projection(i).isInstanceOf[GpuLiteral] =>
          GpuAlias(projection(i), key.name)(exprId = key.exprId)
      }
      val functions = functionsByBranch(index)
      val helper = new AggHelper(input, liveKeys, functions,
        forceMerge = false, conf = conf, metrics = metrics)
      val buffers = functions.flatMap(_.aggregateFunction.aggBufferAttributes)
      val restoreExprs = agg.groupingExpressions.zip(keyOrdinals).map { case (key, i) =>
        projection(i) match {
          case lit: GpuLiteral => lit
          case _ => liveKeys.find(_.exprId == key.exprId).get.toAttribute
        }
      } ++ buffers
      val restore = GpuBindReferences.bindGpuReferencesTiered(restoreExprs,
        liveKeys.map(_.toAttribute) ++ buffers, conf, metrics)
      val merge = if (sources(index).isEmpty) None else {
        val mergeHelper = new AggHelper(input, liveKeys, functions,
          forceMerge = true, conf = conf, metrics = metrics)
        // All branch buffers have the same positional layout, even when their generated
        // attribute IDs differ. Select live keys and buffers before the merge helper.
        val mergeInput = GpuBindReferences.bindGpuReferencesTiered(
          liveKeys.map(_.toAttribute) ++ fullBuffers.drop(keyOrdinals.length),
          fullBuffers, conf, metrics)
        Some(ExpandAggregateMerge(mergeHelper, mergeInput, sources(index)))
      }
      ExpandAggregateBranch(helper, restore, merge,
        cacheResult = sources.drop(index + 1).exists(_.contains(index)))
    }
  }
}

/** Aggregate is an immutable description of the original partial stage, not an executed child. */
case class GpuExpandAggregateExec(
    aggregate: GpuHashAggregateExec,
    projections: Seq[Seq[Expression]],
    expandOutput: Seq[Attribute],
    child: SparkPlan) extends ShimUnaryExecNode with GpuExec {
  override def output: Seq[Attribute] = aggregate.output
  override def outputPartitioning: Partitioning = UnknownPartitioning(0)
  override def producedAttributes: AttributeSet = aggregate.producedAttributes
  override lazy val references: AttributeSet =
    AttributeSet(projections.flatten.flatMap(_.references))
  override def gpuExpressions: Seq[Expression] = aggregate.gpuExpressions ++ projections.flatten
  override protected val outputRowsLevel: MetricsLevel = ESSENTIAL_LEVEL
  override protected val outputBatchesLevel: MetricsLevel = MODERATE_LEVEL
  override lazy val additionalMetrics: Map[String, GpuMetric] = aggregate.additionalMetrics ++ Map(
    NUM_INPUT_ROWS -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_ROWS),
    NUM_INPUT_BATCHES -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_BATCHES),
    "branchBatches" -> createMetric(DEBUG_LEVEL, "successful branch partial batches"),
    "reusedBranchBatches" -> createMetric(DEBUG_LEVEL, "branch batches using partial buffers"),
    "branchInputRows" -> createMetric(DEBUG_LEVEL, "branch input rows before split retry"))

  override def simpleString(maxFields: Int): String =
    s"$nodeName (branches=${projections.length}, " +
      s"keys=${aggregate.groupingExpressions.mkString(",")})"

  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("Row-based execution is not supported")

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val metrics = allMetrics
    val aggMetrics = GpuHashAggregateMetrics(
      metrics(NUM_OUTPUT_ROWS), metrics(NUM_OUTPUT_BATCHES), metrics(NUM_TASKS_REPARTITIONED),
      metrics(NUM_TASKS_SKIPPED_AGG), metrics(OP_TIME_LEGACY), metrics(AGG_TIME),
      metrics(CONCAT_TIME), metrics(SORT_TIME), metrics(REPARTITION_TIME), metrics("NUM_AGGS"),
      metrics("NUM_PRE_SPLITS"), metrics("NUM_TASKS_SINGLE_PASS"), metrics("HEURISTIC_TIME"))
    val branches = GpuExpandAggregateExec.bind(aggregate, projections, expandOutput,
      child.output, conf, metrics)
    val grouping = aggregate.groupingExpressions
    val functions = aggregate.aggregateExpressions
    val aggAttrs = aggregate.aggregateAttributes
    val results = aggregate.resultExpressions
    val inputAttrs = expandOutput
    val targetSize = aggregate.configuredTargetBatchSize
    val allowUnmerged = aggregate.allowNonFullyAggregatedOutput
    val skipRatio = aggregate.skipAggPassReductionRatio
    val modeInfo = AggregateModeInfo(Seq(Partial))
    val finalRefs = GpuAggFinalPassIterator.setupReferences(grouping, functions, aggAttrs,
      results, modeInfo, metrics)
    val settings = conf.getAllConfs
    child.executeColumnar().mapPartitions { input =>
      val localConf = new SQLConf()
      settings.foreach { case (key, value) => localConf.setConfString(key, value) }
      val expandedRows = new LocalGpuMetric
      val firstPass = new GpuExpandAggregateIterator(input, branches, aggMetrics, expandedRows,
        metrics(NUM_INPUT_ROWS), metrics(NUM_INPUT_BATCHES), metrics("branchBatches"),
        metrics("reusedBranchBatches"), metrics("branchInputRows"))
      val firstPassBuffered = new CloseableBufferedIterator(firstPass)
      val merge = new GpuMergeAggregateIterator(firstPassBuffered, inputAttrs, grouping, functions,
        aggAttrs, results, modeInfo, aggMetrics, targetSize, localConf, allowUnmerged,
        skipRatio, expandedRows, metrics, () => firstPass.hasUniqueKeys)
      val outputIter = GpuAggFinalPassIterator.makeIter(merge, finalRefs, aggMetrics)
      new GpuColumnarBatchIterator(true) {
        override def hasNext: Boolean = closeOnExcept(this) { _ =>
          val more = outputIter.hasNext
          if (!more) close()
          more
        }
        override def next(): ColumnarBatch = closeOnExcept(this)(_ => outputIter.next())
        override def doClose(): Unit = {
          withResource(firstPass) { _ =>
            withResource(firstPassBuffered)(_ => withResource(merge)(_ => ()))
          }
        }
      }
    }
  }
}

private[rapids] case class ExpandAggregateMerge(
    helper: AggHelper, input: GpuTieredProject, sources: Set[Int])

private[rapids] case class ExpandAggregateBranch(
    helper: AggHelper,
    restore: GpuTieredProject,
    merge: Option[ExpandAggregateMerge] = None,
    cacheResult: Boolean = false) {
  // Does not own input. The outer retry restarts every allocation from the original rows.
  def apply(input: ColumnarBatch, mergeBuffers: Boolean = false): ColumnarBatch = {
    require(input.numRows() > 0, "Grouped Expand cannot reduce empty input into a row")
    val (aggregation, projection) = if (mergeBuffers) {
      val reuse = merge.get
      (reuse.helper, reuse.input)
    } else {
      (helper, helper.preStepBound)
    }
    withResource(projection.project(input)) { pre =>
      withResource(aggregation.aggregate(pre, NoopMetric)) { aggregated =>
        withResource(aggregation.postStepBound.project(aggregated)) { post =>
          restore.project(post)
        }
      }
    }
  }
}

private[rapids] class GpuExpandAggregateIterator(
    input: Iterator[ColumnarBatch],
    branches: Seq[ExpandAggregateBranch],
    metrics: GpuHashAggregateMetrics,
    expandedRows: LocalGpuMetric,
    inputRows: GpuMetric = NoopMetric,
    inputBatches: GpuMetric = NoopMetric,
    branchBatches: GpuMetric = NoopMetric,
    reusedBranchBatches: GpuMetric = NoopMetric,
    branchInputRows: GpuMetric = NoopMetric)
    extends Iterator[SpillableColumnarBatch] with AutoCloseable {
  private var original: Option[SpillableColumnarBatch] = None
  private var branchIndex = 0
  private var attempts: Option[Spliterator[SpillableColumnarBatch]] = None
  private var pending: Iterator[SpillableColumnarBatch] = Iterator.empty
  private var cached: Option[(Int, SpillableColumnarBatch)] = None
  private var mergingBuffers = false
  private var branchPieces = 0
  private var nonemptyInputs = 0
  private var outputBatches = 0
  private var exhausted = false
  private var closed = false
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      // Retry attempts have their own callback. Closing them here would remove another
      // callback during task completion, which ScalableTaskCompletion forbids.
      closed = true
      withResource(cached.map(_._2))(_ => withResource(original)(_ => ()))
      cached = None
      original = None
      attempts = None
    }
  }

  // Each branch has unique keys; branch discriminators make unsplit results disjoint.
  // This proof is valid only after the complete input has been consumed normally.
  def hasUniqueKeys: Boolean = exhausted && nonemptyInputs == 1 &&
    outputBatches == branches.length

  override def hasNext: Boolean = closeOnExcept(this) { _ =>
    while (!closed && !pending.hasNext && (original.isDefined || input.hasNext)) {
      withResource(attempts)(_ => ())
      attempts = None
      if (original.isDefined && branchIndex == branches.length) {
        withResource(cached.map(_._2))(_ => ())
        cached = None
        withResource(original)(_ => ())
        original = None
      }
      if (original.isEmpty && input.hasNext) {
        val batch = input.next()
        inputBatches += 1
        inputRows += batch.numRows()
        if (batch.numRows() == 0) {
          withResource(batch)(_ => ())
        } else {
          original = Some(SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
          nonemptyInputs += 1
          branchIndex = 0
        }
      }
      original.foreach { sb =>
        val branch = branches(branchIndex)
        expandedRows += sb.numRows()
        val reusable = cached.filter { case (index, _) =>
          branch.merge.exists(_.sources.contains(index))
        }
        mergingBuffers = reusable.isDefined
        val source = reusable.map(_._2).getOrElse(sb)
        val mergeBuffers = mergingBuffers
        branchPieces = 0
        branchInputRows += source.numRows()
        val attempt = new AutoCloseableAttemptSpliterator[SpillableColumnarBatch,
            SpillableColumnarBatch](Iterator.empty, current => {
          GpuMetric.ns(metrics.opTime, metrics.computeAggTime) {
            withResource(current.getColumnarBatch()) { cb =>
              SpillableColumnarBatch(branch(cb, mergeBuffers),
                SpillPriorities.ACTIVE_BATCHING_PRIORITY)
            }
          }
        }, splitSpillableInHalfByRows) {
          // Own the reference even before next(); close() also releases abandoned splits.
          attemptStack.push(source.incRefCount())
        }
        attempts = Some(attempt)
        pending = new RmmRapidsRetryAutoCloseableIterator[SpillableColumnarBatch,
          SpillableColumnarBatch](attempt)
        branchIndex += 1
      }
    }
    val more = !closed && pending.hasNext
    if (!more) {
      if (!closed) exhausted = true
      close()
    }
    more
  }

  override def next(): SpillableColumnarBatch = closeOnExcept(this) { _ =>
    if (!hasNext) throw new NoSuchElementException()
    closeOnExcept(pending.next()) { result =>
      branchPieces += 1
      outputBatches += 1
      metrics.numAggOps += 1
      branchBatches += 1
      if (mergingBuffers) reusedBranchBatches += 1
      // Retain only a complete, compressed branch. Split pieces can contain duplicate
      // keys, so they cannot be reused as a complete branch. A twofold reduction bounds
      // the extra retained buffers and avoids rescanning high-cardinality partials.
      val index = branchIndex - 1
      val reuseReductionFactor = 2
      if (branches(index).cacheResult && branchPieces == 1 && !pending.hasNext &&
          result.numRows().toLong * reuseReductionFactor <= original.get.numRows() &&
          cached.forall(_._2.numRows() > result.numRows())) {
        withResource(cached.map(_._2))(_ => ())
        cached = Some((index, result.incRefCount()))
      }
      result
    }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      withResource(attempts) { _ =>
        withResource(cached.map(_._2))(_ => withResource(original)(_ => ()))
      }
      cached = None
      attempts = None
      original = None
    }
  }
}
