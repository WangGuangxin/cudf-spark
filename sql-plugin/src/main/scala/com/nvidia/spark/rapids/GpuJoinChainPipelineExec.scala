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

import com.nvidia.spark.rapids.GpuMetric._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference,
  AttributeSet, Expression, ExprId, NamedExpression, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.rapids.GpuAnd
import org.apache.spark.sql.rapids.execution.{GpuJoinChain, JoinChainFilter, JoinChainProjection}
import org.apache.spark.sql.types.BooleanType
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuJoinChainPipelineExec {
  def supportsExpressions(condition: Option[Expression], projects: Seq[Expression]): Boolean =
    GpuFilterProjectExec.eligible(condition.getOrElse(GpuLiteral(true, BooleanType)), projects)

  private def value(expr: Expression): Expression = expr match {
    case alias: GpuAlias => value(alias.child)
    case other => other
  }

  private def substitute(expr: Expression, projects: Seq[NamedExpression]): Expression = {
    val replacements = projects.map(p => p.exprId -> value(p)).toMap
    expr.transformUp {
      case a: AttributeReference if replacements.contains(a.exprId) => replacements(a.exprId)
    }
  }

  private def fromFilter(filter: GpuJoinChainFilterExec): GpuJoinChainPipelineExec =
    GpuJoinChainPipelineExec(Some(filter.condition), filter.output.toList, filter.child,
      if (filter.residualJoin) Some(filter.condition) else None, filter.output)

  /** Keep batching boundaries while pruning columns, including COUNT(*)'s empty input schema. */
  private def aggregateInput(plan: SparkPlan, needed: Set[ExprId]): Option[SparkPlan] = plan match {
    case c: GpuCoalesceBatches if c.goal.isInstanceOf[TargetSize] =>
      aggregateInput(c.child, needed).map(child => c.copy(child = child))
    case p: GpuJoinChainPipelineExec =>
      Some(p.copy(projectList = p.projectList.filter(e => needed(e.exprId))))
    case f: GpuJoinChainFilterExec => aggregateInput(fromFilter(f), needed)
    case join if GpuJoinChain.canFilter(join) =>
      Some(GpuJoinChainPipelineExec(None, join.output.filter(a => needed(a.exprId)).toList, join))
    case _ => None
  }

  def fuse(plan: SparkPlan): SparkPlan = if (!RapidsConf.ENABLE_JOIN_CHAIN.get(plan.conf)) {
    plan
  } else plan.transformUp {
    case p @ GpuProjectExec(projects, f: GpuJoinChainFilterExec, true)
        if supportsExpressions(Some(f.condition), projects) =>
      fromFilter(f).copy(projectList = projects, outputAttributes = p.output)
    case p @ GpuProjectExec(projects, pipeline: GpuJoinChainPipelineExec, true) =>
      val composed = projects.map(p =>
        substitute(p, pipeline.projectList).asInstanceOf[NamedExpression])
      if (supportsExpressions(pipeline.condition, composed)) {
        pipeline.copy(projectList = composed, outputAttributes = p.output)
      } else p
    case p @ GpuProjectExec(projects, child, true) if
        RapidsConf.ENABLE_JOIN_CHAIN.get(p.conf) && supportsExpressions(None, projects) &&
        GpuJoinChain.canFilter(child) =>
      GpuJoinChainPipelineExec(None, projects, child, outputAttributes = p.output)
    case f @ GpuFilterExec(condition, pipeline: GpuJoinChainPipelineExec)
        if f.coalesceAfter =>
      val translated = substitute(condition, pipeline.projectList)
      val combined = pipeline.condition.map(GpuAnd(_, translated)).getOrElse(translated)
      if (supportsExpressions(Some(combined), pipeline.projectList)) {
        pipeline.copy(condition = Some(combined), outputAttributes = f.output)
      } else f
    case agg: GpuHashAggregateExec if RapidsConf.ENABLE_JOIN_CHAIN.get(agg.conf) &&
        agg.requiredChildDistributionExpressions.isEmpty && agg.aggregateExpressions.nonEmpty &&
        agg.aggregateExpressions.forall(e => e.mode == Partial && !e.isDistinct) =>
      val needed = (agg.groupingExpressions ++ agg.aggregateExpressions)
        .flatMap(_.references).map(_.exprId).toSet
      aggregateInput(agg.child, needed).map(GpuJoinChainAggregateExec(agg, _)).getOrElse(agg)
  }
}

/** Select source columns by row maps before computing the final project. */
case class GpuJoinChainPipelineExec(
    condition: Option[Expression],
    projectList: List[NamedExpression],
    child: SparkPlan,
    residualCondition: Option[Expression] = None,
    outputAttributes: Seq[Attribute] = Nil)
    extends GpuPartitioningPreservingUnaryExecNode with GpuExec {
  override def output: Seq[Attribute] = {
    // Folding a filter through a computed alias must preserve the original non-null guarantee.
    val refined = outputAttributes.map(a => a.exprId -> a).toMap
    projectList.map(p => refined.getOrElse(p.exprId, p.toAttribute))
  }
  // Combining row-only batches into one Int row count can overflow and saves no payload work.
  override def coalesceAfter: Boolean = projectList.nonEmpty
  override def gpuExpressions: Seq[Expression] = condition.toSeq ++ projectList
  @transient override lazy val references: AttributeSet =
    AttributeSet(gpuExpressions.flatMap(_.references))
  override protected def buildAttributeMap(): Map[Attribute, Attribute] = {
    def source(expr: Expression): Option[Attribute] = expr match {
      case alias: GpuAlias => source(alias.child)
      case a: AttributeReference => child.output.find(_.exprId == a.exprId)
      case _ => None
    }
    projectList.zip(output).flatMap { case (expr, out) => source(expr).map(_ -> out) }.toMap
  }
  override def outputOrdering: Seq[SortOrder] = {
    val aliases = buildAttributeMap()
    child.outputOrdering.map { order =>
      order.transform {
        case a: Attribute if aliases.contains(a) => aliases(a)
      }.asInstanceOf[SortOrder]
    }.takeWhile(_.references.subsetOf(outputSet))
  }
  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("Row-based execution is not supported")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    "candidateRows" -> createMetric(MODERATE_LEVEL, "rows before deferred filter"),
    "candidateBatches" -> createMetric(DEBUG_LEVEL, "map batches before deferred filter"),
    "avoidedPayloadRows" -> createMetric(MODERATE_LEVEL, "full payload rows not materialized"),
    "fusedTasks" -> createMetric(MODERATE_LEVEL, "tasks with deferred join pipeline"),
    "payloadColumnsAvoided" -> createMetric(MODERATE_LEVEL,
      "source columns not gathered across output batches"),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY))

  def bind(): JoinChainFilter = {
    val (attributes, lineage) = GpuJoinChain.pipelineInput(child)
    val replacements = child.output.map(_.exprId).zip(lineage).toMap
    def rewrite(expr: Expression): Expression = expr.transformUp {
      case a: AttributeReference if replacements.contains(a.exprId) => replacements(a.exprId)
    }
    def project(exprs: Seq[Expression], input: Seq[Attribute]): JoinChainProjection = {
      val refs = exprs.flatMap(_.references).map(_.exprId).toSet
      val ordinals = input.indices.filter(i => refs(input(i).exprId))
      JoinChainProjection(ordinals, GpuBindReferences.bindGpuReferencesTiered(exprs,
        ordinals.map(input), conf, allMetrics), gpuLongMetric("payloadColumnsAvoided"))
    }
    val predicateExpr = condition.getOrElse(GpuLiteral(true, BooleanType))
    val predicate = project(Seq(rewrite(predicateExpr)), attributes)
    val fullPredicate = GpuBindReferences.bindGpuReferencesTiered(Seq(predicateExpr),
      child.output, conf, allMetrics)
    val filteredAttrs = condition.map(GpuFilterExec(_, child)().output).getOrElse(child.output)
    val fallback = project(projectList, filteredAttrs)
    val projected = project(projectList.map(rewrite), attributes)
    JoinChainFilter(predicate.ordinals, predicate.expressions, fullPredicate,
      gpuLongMetric("candidateRows"), gpuLongMetric("candidateBatches"),
      gpuLongMetric("avoidedPayloadRows"), gpuLongMetric("fusedTasks"),
      gpuLongMetric(OP_TIME_LEGACY), residualCondition, Some(projected),
      filterEnabled = condition.isDefined,
      filterOnResidualFallback = residualCondition.isDefined && condition != residualCondition,
      fallbackProjection = Some(fallback))
  }

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val terminal = bind()
    val rows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val batches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val input = GpuJoinChain.executeFiltered(child, terminal).getOrElse {
      child.executeColumnar().mapPartitions(terminal.eager)
    }
    input.mapPartitions { iter =>
      iter.map { batch =>
        rows += batch.numRows()
        batches += 1
        batch
      }
    }
  }
}

/** Keep the native partial/merge algorithms and buffer schema at the pipeline's dense boundary. */
case class GpuJoinChainAggregateExec(aggregate: GpuHashAggregateExec, child: SparkPlan)
    extends GpuPartitioningPreservingUnaryExecNode with GpuExec {
  override def output: Seq[Attribute] = aggregate.output
  override def gpuExpressions: Seq[Expression] = aggregate.gpuExpressions
  override def producedAttributes: AttributeSet = aggregate.producedAttributes
  @transient override lazy val references: AttributeSet = AttributeSet(
    (aggregate.groupingExpressions ++ aggregate.aggregateExpressions).flatMap(_.references))
  override protected def buildAttributeMap(): Map[Attribute, Attribute] =
    aggregate.resultExpressions.collect {
      case a @ GpuAlias(child: Attribute, _) => child -> a.toAttribute
      case a @ Alias(child: Attribute, _) => child -> a.toAttribute
    }.toMap
  // The descriptor is not executed as a child. Share its metrics with the visible pipeline node.
  override lazy val allMetrics: Map[String, GpuMetric] = aggregate.allMetrics
  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("Row-based execution is not supported")
  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] =
    aggregate.executeWithInput(child.executeColumnar(), child.output, child.outputOrdering)
}
