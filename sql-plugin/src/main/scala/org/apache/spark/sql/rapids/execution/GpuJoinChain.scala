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

package org.apache.spark.sql.rapids.execution

import ai.rapids.cudf.{ColumnVector, ColumnView, DType, GroupByAggregation, GroupByOptions, NullEquality, NullPolicy, OutOfBoundsPolicy, Scalar, Table}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{withRestoreOnRetry, withRetryNoSplit}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, Expression,
  NamedExpression}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.{ArrayType, BinaryType, IntegerType, MapType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Why a plan did not fuse into a join chain, or why an otherwise fusable chain stopped growing.
 * `reason` is a stable, greppable token; `detail` narrows it down to the specific plan node or
 * predicate that ended the chain.
 */
private[execution] case class ChainBreak(reason: String, detail: String = "") {
  override def toString: String = if (detail.isEmpty) reason else s"$reason($detail)"
}

private[execution] object ChainBreak {
  /** The chain is turned off by configuration. */
  val disabled: ChainBreak = ChainBreak("DISABLED")
  /** This plan node is not a join kind that can take part in a chain (reports the node class). */
  def notChainableJoin(plan: SparkPlan): ChainBreak =
    ChainBreak("NOT_CHAINABLE_JOIN", plan.getClass.getSimpleName)
  /** A chainable join kind, but one of the chain preconditions does not hold. */
  def ineligible(why: String): ChainBreak = ChainBreak("INELIGIBLE", why)
  /** A projection carrying computed expressions, which cannot be deferred past a probe. */
  def computedProjection(plan: SparkPlan): ChainBreak =
    ChainBreak("COMPUTED_PROJECTION", plan.getClass.getSimpleName)
  /** A coalesce whose goal is not a plain target size, so batches cannot be streamed through. */
  def coalesceGoal(goal: CoalesceGoal): ChainBreak =
    ChainBreak("COALESCE_GOAL", goal.getClass.getSimpleName)
  /** The chain formed but failed the per-row byte model in `worthwhile`. */
  val notWorthwhile: ChainBreak = ChainBreak("NOT_WORTHWHILE")
}

/** Only immutable CPU metadata is shared between tasks using the same broadcast. */
private[execution] case class JoinChainBuildKeyStats(unique: Boolean, maxMatches: Long)

private[execution] class JoinChainBuildKeyStatsCache(maxEntries: Int = 64) {
  require(maxEntries > 0)
  private var entries = Map.empty[(Vector[Int], Boolean), JoinChainBuildKeyStats]

  def clear(): Unit = synchronized { entries = Map.empty }

  def getOrCompute(ordinals: Seq[Int], nullsEqual: Boolean)
      (compute: => JoinChainBuildKeyStats): JoinChainBuildKeyStats = {
    val key = (ordinals.toVector, nullsEqual)
    val found = synchronized { entries.get(key) }
    found.getOrElse {
      // Never hold the cache monitor during GPU work or a retry. Concurrent misses may
      // compute twice, but neither wait for another task while holding its GPU semaphore.
      val result = compute
      synchronized {
        entries.get(key).getOrElse {
          if (entries.size < maxEntries) entries += key -> result
          result
        }
      }
    }
  }
}

/** A stage consumes the preceding logical output, but its payload stays in the original sources. */
private[execution] case class JoinChainStage(
    buildKeys: Seq[GpuExpression],
    streamKeyOrdinals: Seq[Int],
    streamProjection: Seq[Int],
    buildSide: GpuBuildSide,
    nullsEqual: Boolean,
    outputRows: GpuMetric = NoopMetric,
    outputBatches: GpuMetric = NoopMetric,
    joinTime: GpuMetric = NoopMetric,
    isLeftSemi: Boolean = false,
    buildKeysUnique: Boolean = true,
    maxBuildMatches: Long = 1L,
    inputFilters: Seq[JoinChainFilter] = Nil,
    streamKeyProject: Option[JoinChainProjection] = None,
    virtualInput: Boolean = false)

/** Projection bound to the distinct source columns needed after filtering, including no columns. */
case class JoinChainProjection(
    ordinals: Seq[Int],
    expressions: GpuTieredProject,
    inputColumnsPruned: GpuMetric = NoopMetric) {
  // Include validity bits when budgeting computed output alongside gathered dependencies.
  val extraBitsPerRow: Int = expressions.outputExprs.map { expr =>
    expr.dataType.defaultSize * 8 + (if (expr.nullable) 1 else 0)
  }.sum

  def apply(input: ColumnarBatch): ColumnarBatch = expressions.project(input)

  def eager(input: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = input.flatMap { batch =>
    val selected = withResource(batch)(GpuJoinChain.select(_, ordinals))
    val owned = SpillableColumnarBatch(selected, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
    expressions.projectAndCloseStreamingWithSplitRetry(owned)
  }
}

/** A deferred terminal predicate/projection and the equivalent ordinary fallback. */
case class JoinChainFilter(
    ordinals: Seq[Int],
    predicate: GpuTieredProject,
    fullPredicate: GpuTieredProject,
    inputRows: GpuMetric = NoopMetric,
    inputBatches: GpuMetric = NoopMetric,
    avoidedPayloadRows: GpuMetric = NoopMetric,
    fusedTasks: GpuMetric = NoopMetric,
    filterTime: GpuMetric = NoopMetric,
    joinCondition: Option[Expression] = None,
    projection: Option[JoinChainProjection] = None,
    filterEnabled: Boolean = true,
    filterOnResidualFallback: Boolean = false,
    fallbackProjection: Option[JoinChainProjection] = None,
    selectedRows: GpuMetric = NoopMetric,
    selectedBatches: GpuMetric = NoopMetric) {
  def finish(input: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] =
    fallbackProjection.orElse(projection).map(_.eager(input)).getOrElse(input)

  def eager(input: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = {
    val selected = if (filterEnabled) input.flatMap { batch =>
      inputRows += batch.numRows()
      inputBatches += 1
      GpuFilter.filterAndClose(batch, fullPredicate, NoopMetric, NoopMetric, filterTime)
    } else input
    finish(selected)
  }

  def residualFallback(input: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] =
    if (filterOnResidualFallback) eager(input) else finish(input)
}

/** A local execution pipeline; exchanges and unsupported operators remain stage boundaries. */
object GpuJoinChain extends Logging {
  /** Prefix for the per-join diagnostic line, so a whole benchmark run can be grepped at once. */
  private val LOG_TAG = "RAPIDS_JOIN_CHAIN"

  private type Batches = Iterator[ColumnarBatch]
  private type Built = Either[ColumnarBatch, Batches]
  private case class Node(plan: GpuJoinExec, side: GpuBuildSide) {
    def stream: SparkPlan = if (side == GpuBuildRight) plan.left else plan.right
    def build: SparkPlan = if (side == GpuBuildRight) plan.right else plan.left
    lazy val buildKeys: Seq[GpuExpression] = GpuBindReferences.bindGpuReferences(
      if (side == GpuBuildRight) plan.rightKeys else plan.leftKeys, build.output, plan.allMetrics)
    lazy val streamKeys: Seq[GpuExpression] = GpuBindReferences.bindGpuReferences(
      if (side == GpuBuildRight) plan.leftKeys else plan.rightKeys, stream.output, plan.allMetrics)
    def target: Long = plan match {
      case join: GpuShuffledHashJoinExec => join.joinChainTarget
      case join: GpuShuffledSymmetricHashJoinExec => join.gpuBatchSizeBytes
      case _ => RapidsConf.GPU_BATCH_SIZE_BYTES.get(plan.conf)
    }
    def broadcast: Boolean = plan.isInstanceOf[GpuBroadcastHashJoinExecBase]
  }
  private case class VirtualInput(
      attributes: Seq[Attribute],
      keys: Seq[Expression],
      filters: Seq[(GpuFilterExec, Expression)],
      eagerOperations: Seq[SparkPlan])
  private case class Link(node: Node, projection: Seq[Int],
      virtualInput: Option[VirtualInput] = None, eagerCondition: Option[Expression] = None)
  /** `break` records why the chain stopped growing past `input`, for diagnostics. */
  private case class Chain(input: SparkPlan, links: Seq[Link], break: ChainBreak,
      rawOutput: Seq[Attribute] = Nil, outputExpressions: Seq[Expression] = Nil,
      tailFilters: Seq[(GpuFilterExec, Expression)] = Nil,
      tailOperations: Seq[SparkPlan] = Nil) {
    def virtual: Boolean = links.exists(_.virtualInput.isDefined)
  }
  private case class Inputs(stream: Batches, builds: Seq[Batches])
  private case class Prepared(stage: JoinChainStage,
      load: (Batches, Batches) => (Built, Batches), eager: (Built, Batches) => Batches,
      statsCache: () => Option[JoinChainBuildKeyStatsCache],
      eagerInput: Batches => Batches)

  /** The result of analyzing one plan node as a potential chain anchor. */
  private sealed trait Outcome
  /** A chain of more than one link that passed every check. */
  private case class Fused(chain: Chain) extends Outcome
  /**
   * The node could anchor a chain but none was formed. `candidateLength` is how long the chain
   * would have been, which distinguishes "nothing to chain with" from "rejected by the model".
   */
  private case class Unfused(reason: ChainBreak, candidateLength: Int) extends Outcome
  /** The node itself cannot anchor a chain at all. */
  private case class NotAnchored(reason: ChainBreak) extends Outcome

  private def ordinal(expr: Expression): Option[Int] = expr match {
    case ref: GpuBoundReference => Some(ref.ordinal)
    case alias: GpuAlias => ordinal(alias.child)
    case _ => None
  }

  private def node(plan: SparkPlan): Option[Node] = plan match {
    case join: GpuBroadcastHashJoinExecBase if join.supportsJoinChain =>
      Some(Node(join, join.buildSide))
    case join: GpuShuffledHashJoinExec =>
      val original = Node(join, join.buildSide)
      val side = if (join.joinType == Inner && join.condition.isEmpty &&
          childJoin(original.stream).isLeft && childJoin(original.build).isRight) {
        if (join.buildSide == GpuBuildRight) GpuBuildLeft else GpuBuildRight
      } else join.buildSide
      Some(Node(join, side))
    // Symmetric joins choose their build side using actual input sizes. Treat them as chain
    // boundaries so a selective prefix can finish before sizing, instead of forcing the other
    // (potentially much larger) input to be the build merely to extend the chain.
    case _: GpuShuffledSymmetricHashJoinExec => None
    case _ => None
  }

  /**
   * The first chain precondition this node violates, or None when it may anchor/extend a chain.
   * Conditions are checked in a fixed order so the reported reason is stable across runs.
   */
  private def ineligible(node: Node): Option[String] = {
    def simpleKey(expr: GpuExpression): Boolean = expr match {
      case ref: GpuBoundReference => ref.dataType match {
        case _: ArrayType | _: MapType | _: StructType | BinaryType => false
        case _ => true
      }
      case _ => false
    }
    val options = RapidsConf.getJoinOptions(node.plan.conf, node.target)
    val hashStrategy = options.strategy == JoinStrategy.AUTO ||
      options.strategy == JoinStrategy.HASH_ONLY
    if (!hashStrategy) Some(s"strategy=${options.strategy}")
    else if (node.plan.joinType != Inner && node.plan.joinType != LeftSemi) {
      Some(s"joinType=${node.plan.joinType}")
    } else if (node.plan.joinType == LeftSemi && node.side != GpuBuildRight) {
      Some("leftSemiBuildSide")
    } else if (node.plan.condition.isDefined) Some("joinCondition")
    else if (node.plan.isSkewJoin) Some("skewJoin")
    else if (node.buildKeys.isEmpty) Some("noBuildKeys")
    else if (!node.buildKeys.forall(simpleKey)) Some("buildKeyType")
    else if (!node.streamKeys.forall(simpleKey)) Some("streamKeyType")
    else None
  }

  /** Walk one step down the stream side, reporting what ended the chain when it cannot continue. */
  private def childJoin(plan: SparkPlan): Either[ChainBreak, (Node, Seq[Int])] = plan match {
    case coalesce: GpuCoalesceBatches if coalesce.goal.isInstanceOf[TargetSize] =>
      childJoin(coalesce.child)
    case coalesce: GpuCoalesceBatches => Left(ChainBreak.coalesceGoal(coalesce.goal))
    case project: GpuProjectExec =>
      val refs = GpuBindReferences.bindGpuReferences(
        project.projectList, project.child.output, project.allMetrics).map(ordinal)
      if (refs.forall(_.isDefined)) {
        childJoin(project.child).map { case (join, projection) =>
          (join, refs.map(ref => projection(ref.get)))
        }
      } else Left(ChainBreak.computedProjection(project))
    case _ => node(plan) match {
      case None => Left(ChainBreak.notChainableJoin(plan))
      case Some(n) => ineligible(n) match {
        case Some(why) => Left(ChainBreak.ineligible(why))
        case None => Right((n, plan.output.indices))
      }
    }
  }

  private def collect(join: Node): Chain = childJoin(join.stream) match {
    case Right((child, projection)) =>
      val prefix = collect(child)
      prefix.copy(links = prefix.links :+ Link(join, projection))
    case Left(why) => Chain(join.stream, Seq(Link(join, join.stream.output.indices)), why)
  }

  /** Substitute values, never aliases, while retaining each physical source's ExprId. */
  private def substitute(expr: Expression, attributes: Seq[Attribute],
      values: Seq[Expression]): Expression = {
    val replacements = attributes.map(_.exprId).zip(values).toMap
    expr.transformUp {
      case a: AttributeReference if replacements.contains(a.exprId) => replacements(a.exprId)
    }
  }

  private def value(expr: Expression): Expression = expr match {
    case alias: GpuAlias => value(alias.child)
    case other => other
  }

  /**
   * Keep projections as expression lineage over the original fact/build columns. Predicates
   * retain their position before the next probe; they are never moved past a selective join.
   * The ordinary operators are also retained as descriptors for out-of-core/fanout fallback.
   */
  private case class VirtualPlan(chain: Chain, attributes: Seq[Attribute],
      values: Seq[Expression], filters: Seq[(GpuFilterExec, Expression)] = Nil,
      operations: Seq[SparkPlan] = Nil, changed: Boolean = false)

  private def virtualPlan(plan: SparkPlan): VirtualPlan = {
    def boundary: VirtualPlan = VirtualPlan(
      Chain(plan, Nil, ChainBreak.notChainableJoin(plan), plan.output, plan.output),
      plan.output, plan.output)
    def project(projects: Seq[NamedExpression], child: SparkPlan,
        output: Seq[Attribute]): VirtualPlan = {
      val previous = virtualPlan(child)
      val values = projects.map(p => substitute(value(p), previous.attributes, previous.values))
      previous.copy(attributes = output, values = values,
        operations = previous.operations :+ GpuProjectExec(projects.toList, child),
        changed = previous.changed || values.exists(!_.isInstanceOf[AttributeReference]))
    }
    def preserveResidual(previous: VirtualPlan, condition: Option[Expression]): VirtualPlan = {
      if (condition.isEmpty || previous.chain.links.isEmpty) previous else {
        val links = previous.chain.links
        // A nested terminal node may have extracted a native INNER join residual. Restore it
        // on that exact join during fallback, before its payload could multiply. Replaying
        // the safe pending filter afterwards also preserves any combined post-filter.
        previous.copy(chain = previous.chain.copy(links = links.updated(links.size - 1,
          links.last.copy(eagerCondition = condition))))
      }
    }
    def filtered(filter: GpuFilterExec): VirtualPlan = {
      val previous = virtualPlan(filter.child)
      val condition = substitute(filter.condition, previous.attributes, previous.values)
      previous.copy(attributes = filter.output,
        filters = previous.filters :+ (filter -> condition),
        operations = previous.operations :+ filter, changed = true)
    }
    plan match {
      case c: GpuCoalesceBatches if c.goal.isInstanceOf[TargetSize] => virtualPlan(c.child)
      case p: GpuProjectExec if p.enablePreSplit &&
          GpuJoinChainPipelineExec.supportsExpressions(None, p.projectList) =>
        project(p.projectList, p.child, p.output)
      case f: GpuFilterExec if f.coalesceAfter &&
          GpuJoinChainPipelineExec.supportsExpressions(Some(f.condition), Nil) => filtered(f)
      case p: GpuJoinChainPipelineExec =>
        val child = p.condition.map(GpuFilterExec(_, p.child)()).getOrElse(p.child)
        preserveResidual(project(p.projectList, child, p.output), p.residualCondition)
      case f: GpuJoinChainFilterExec =>
        preserveResidual(filtered(GpuFilterExec(f.condition, f.child)()),
          if (f.residualJoin) Some(f.condition) else None)
      case p: GpuFilterProjectExec if !p.useAst =>
        val child = GpuFilterExec(p.condition, p.child)()
        project(p.projectList.map(_.asInstanceOf[NamedExpression]), child, p.output)
      case _ => node(plan) match {
        case Some(n) if ineligible(n).isEmpty =>
          val previous = virtualPlan(n.stream)
          val streamKeys = if (n.side == GpuBuildRight) n.plan.leftKeys else n.plan.rightKeys
          val keys = streamKeys.map(substitute(_, previous.attributes, previous.values))
          if (!GpuJoinChainPipelineExec.supportsExpressions(None, keys)) boundary else {
            val raw = previous.chain.rawOutput
            val input = VirtualInput(raw, keys, previous.filters, previous.operations)
            val link = Link(n, raw.indices, Some(input))
            val build = n.build.output
            val values = if (n.plan.joinType == LeftSemi) previous.values
              else if (n.side == GpuBuildRight) previous.values ++ build
              else build ++ previous.values
            val chain = previous.chain.copy(links = previous.chain.links :+ link,
              rawOutput = raw ++ (if (n.plan.joinType == LeftSemi) Nil else build),
              outputExpressions = values)
            VirtualPlan(chain, plan.output, values, changed = previous.changed)
          }
        case _ => boundary
      }
    }
  }

  /** Driver-only metadata used to bind a terminal project/filter to the physical source columns. */
  def pipelineInput(plan: SparkPlan): (Seq[Attribute], Seq[Expression]) =
    analyze(filterAnchor(plan)) match {
      case Fused(chain) if chain.virtual => (chain.rawOutput, chain.outputExpressions)
      case _ => (plan.output, plan.output)
    }

  /** Bind an expression to just its physical source dependencies. */
  private def dependencyProject(exprs: Seq[Expression], input: Seq[Attribute],
      plan: GpuExec): JoinChainProjection = {
    val refs = exprs.flatMap(_.references).map(_.exprId).toSet
    val ordinals = input.indices.filter(i => refs(input(i).exprId))
    JoinChainProjection(ordinals, GpuBindReferences.bindGpuReferencesTiered(exprs,
      ordinals.map(input), plan.conf, plan.allMetrics))
  }

  private def inputFilter(filter: GpuFilterExec, condition: Expression,
      attributes: Seq[Attribute]): JoinChainFilter = {
    val bound = dependencyProject(Seq(condition), attributes, filter)
    JoinChainFilter(bound.ordinals, bound.expressions, bound.expressions,
      inputRows = optionalMetric(filter, GpuMetric.NUM_INPUT_ROWS),
      inputBatches = optionalMetric(filter, GpuMetric.NUM_INPUT_BATCHES),
      filterTime = filter.gpuLongMetric(GpuMetric.OP_TIME_LEGACY),
      selectedRows = filter.gpuLongMetric(GpuMetric.NUM_OUTPUT_ROWS),
      selectedBatches = filter.gpuLongMetric(GpuMetric.NUM_OUTPUT_BATCHES))
  }

  /** Bind ordinary fallback operations on the driver, before their transient SQLConf is lost. */
  private def eagerOperations(operations: Seq[SparkPlan]): Batches => Batches = {
    val bound: Seq[Batches => Batches] = operations.map {
      case f: GpuFilterExec =>
        val predicate = GpuBindReferences.bindGpuReferencesTiered(Seq(f.condition),
          f.child.output, f.conf, f.allMetrics)
        val rows = f.gpuLongMetric(GpuMetric.NUM_OUTPUT_ROWS)
        val batches = f.gpuLongMetric(GpuMetric.NUM_OUTPUT_BATCHES)
        val time = f.gpuLongMetric(GpuMetric.OP_TIME_LEGACY)
        (input: Batches) => input.flatMap { batch =>
          GpuFilter.filterAndClose(batch, predicate, rows, batches, time)
        }
      case p: GpuProjectExec =>
        val projection = GpuBindReferences.bindGpuReferencesTiered(p.projectList,
          p.child.output, p.conf, p.allMetrics)
        (input: Batches) => input.flatMap { batch =>
          projection.projectAndCloseStreamingWithSplitRetry(
            SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
        }
    }
    (input: Batches) => bound.foldLeft(input)((stream, operation) => operation(stream))
  }

  private def worthwhile(chain: Chain): Boolean = {
    // Per-row bytes screen out narrow chains; this does not predict selectivity or elapsed time.
    val savedPayload = chain.links.dropRight(1).flatMap(_.node.plan.output)
      .map(_.dataType.defaultSize.toLong).sum
    // Semi stages contribute no build payload map to subsequent probes.
    val composedMaps = chain.links.indices.drop(1).map { i =>
      1L + chain.links.take(i).count(_.node.plan.joinType != LeftSemi)
    }.sum
    val mapTraffic = 2L * Integer.BYTES * composedMaps
    val keyTraffic = chain.links.drop(1).flatMap(_.node.streamKeys)
      .map(key => 2L * key.dataType.defaultSize + Integer.BYTES).sum
    // Broadcast chains can filter a stream through small dimension tables before gathering
    // its payload. The schema-only traffic estimate cannot account for that reduction or the
    // avoided intermediate gathers, so do not reject narrow broadcast chains with this model.
    // Runtime build-size and fanout checks still select the eager path when necessary.
    chain.links.forall(_.node.broadcast) || savedPayload > mapTraffic + keyTraffic
  }

  /**
   * Classify one plan node as a chain anchor. This is a pure function of plan shape and does not
   * consider `spark.rapids.sql.join.chain.enabled`, so an A/B run with the chain disabled still
   * reports which plans would have fused.
   */
  private def analyze(plan: SparkPlan): Outcome = {
    val virtual = virtualPlan(plan)
    if (virtual.changed && virtual.chain.links.nonEmpty && worthwhile(virtual.chain)) {
      // Pending operations after the last probe have no next link to consume them. Keep
      // their predicates over raw sources, their final lineage, and their ordinary fallback.
      Fused(virtual.chain.copy(outputExpressions = virtual.values,
        tailFilters = virtual.filters, tailOperations = virtual.operations))
    } else analyzeSimple(plan)
  }

  private def analyzeSimple(plan: SparkPlan): Outcome = node(plan) match {
    case None => NotAnchored(ChainBreak.notChainableJoin(plan))
    case Some(root) => ineligible(root) match {
      case Some(why) => NotAnchored(ChainBreak.ineligible(why))
      case None =>
        val chain = collect(root)
        if (chain.links.size <= 1) Unfused(chain.break, chain.links.size)
        else if (!worthwhile(chain)) Unfused(ChainBreak.notWorthwhile, chain.links.size)
        else Fused(chain)
    }
  }

  /**
   * One greppable line per join operator, emitted on the driver as the RDD is built. Aggregate the
   * trigger rate and the break-reason histogram across a benchmark run with something like:
   * {{{
   *   grep -hoE 'fused=true|reason=[A-Z_]+' driver.log | sort | uniq -c | sort -rn
   * }}}
   */
  private def logOutcome(plan: SparkPlan, outcome: Outcome, enabled: Boolean,
      verbose: Boolean): Unit = {
    if (verbose || log.isDebugEnabled) {
      val root = s"root=${plan.getClass.getSimpleName}"
      val message = outcome match {
        case Fused(chain) if enabled =>
          val inputs = chain.links.flatMap(_.virtualInput)
          val filters = inputs.map(_.filters.size).sum
          val projects = inputs.flatMap(_.eagerOperations).count(_.isInstanceOf[GpuProjectExec])
          s"$LOG_TAG fused=true length=${chain.links.size} virtual=${chain.virtual} " +
            s"inputFilters=$filters inputProjects=$projects stoppedBy=${chain.break} $root"
        case Fused(chain) =>
          s"$LOG_TAG fused=false reason=${ChainBreak.disabled} wouldFuse=true " +
            s"length=${chain.links.size} $root"
        case Unfused(reason, candidateLength) =>
          s"$LOG_TAG fused=false reason=$reason candidateLength=$candidateLength $root"
        case NotAnchored(reason) =>
          s"$LOG_TAG fused=false reason=$reason $root"
      }
      if (verbose) logWarning(message) else logDebug(message)
    }
  }

  private[execution] def chainLength(plan: SparkPlan): Int = analyze(plan) match {
    case Fused(chain) => chain.links.size
    case _: Unfused => 1
    case _: NotAnchored => 0
  }

  private def prepare(link: Link, eagerCondition: Option[Expression]): Prepared = {
    val n = link.node
    val plan = n.plan
    val options = RapidsConf.getJoinOptions(plan.conf, n.target)
    val rows = plan.gpuLongMetric(GpuMetric.NUM_OUTPUT_ROWS)
    val batches = plan.gpuLongMetric(GpuMetric.NUM_OUTPUT_BATCHES)
    val opTime = plan.gpuLongMetric(GpuMetric.OP_TIME_LEGACY)
    val joinTime = plan.gpuLongMetric(GpuMetric.JOIN_TIME)
    val originalStage = JoinChainStage(n.buildKeys,
      n.streamKeys.map(_.asInstanceOf[GpuBoundReference].ordinal), link.projection,
      n.side, GpuHashJoin.compareNullsEqual(plan.joinType, n.buildKeys), rows, batches, joinTime,
      isLeftSemi = plan.joinType == LeftSemi)
    val (load, eager, statsCache): ((Batches, Batches) => (Built, Batches),
        (Built, Batches) => Batches, () => Option[JoinChainBuildKeyStatsCache]) =
      plan match {
        case join: GpuBroadcastHashJoinExecBase =>
          val broadcast = join.broadcastExchange.executeColumnarBroadcast[Any]()
          val schema = join.chainBuildSchema
          val projection = join.chainBuildProjection
          val loader = (_: Batches, stream: Batches) => {
            val (raw, buffered) = GpuBroadcastHelper.getBroadcastBuiltBatchAndStreamIter(
              broadcast, schema, stream)
            (Left(projection.map(_(raw)).getOrElse(raw)): Built, buffered)
          }
          // Restore an extracted residual condition on runtime fallback. Native mixed joins
          // can reject matches before gathering, unlike an unfiltered join followed by Filter.
          val eagerJoin = (join, eagerCondition) match {
            case (concrete: GpuBroadcastHashJoinExec, Some(condition)) =>
              val restored = concrete.copy(condition = Some(condition))
              // Binding inside doJoin needs this map on the executor. Its factory uses a
              // transient SQLConf/SparkContext, so initialize the copied plan on the driver.
              val _ = restored.allMetrics
              restored
            case _ => join
          }
          val fallback = (built: Built, stream: Batches) => built match {
            case Left(batch) =>
              eagerJoin.doJoin(batch, stream, options, rows, batches, opTime, joinTime)
            case Right(_) => throw new IllegalStateException("broadcast build must be one batch")
          }
          val stats = () => {
            // Post-broadcast projections can change values and key positions. Cache only
            // keys bound directly to this immutable broadcast's original columns.
            if (options.buildKeyCacheEnabled && projection.isEmpty) {
              broadcast.value match {
                case batch: SerializeConcatHostBuffersDeserializeBatch =>
                  Some(batch.joinChainBuildKeyStats)
                case _ => None
              }
            } else None
          }
          (loader, fallback, stats)
        case join: GpuShuffledHashJoinExec =>
          (join.joinChainBuildLoader(n.side), join.joinChainEager(n.side), () => None)
        case join: GpuShuffledSymmetricHashJoinExec =>
          (join.joinChainBuildLoader(n.side), join.joinChainEager(n.side), () => None)
      }
    val stage = link.virtualInput.map { input =>
      val keyProject = dependencyProject(input.keys, input.attributes, plan)
      originalStage.copy(streamKeyOrdinals = keyProject.ordinals,
        // Virtual layouts always append physical sources; Spark's logical left/right order
        // is represented by the final expression lineage instead of intermediate payloads.
        buildSide = GpuBuildRight,
        inputFilters = input.filters.map { case (filter, expr) =>
          inputFilter(filter, expr, input.attributes)
        }, streamKeyProject = Some(keyProject), virtualInput = true)
    }.getOrElse(originalStage)
    Prepared(stage, load, eager, statsCache,
      eagerOperations(link.virtualInput.map(_.eagerOperations).getOrElse(Nil)))
  }

  /**
   * A metric the chain reports but a given join exec may not declare. Missing metrics degrade to
   * no-ops rather than failing the query, since the chain spans several join implementations.
   */
  private def optionalMetric(plan: GpuExec, name: String): GpuMetric =
    plan.allMetrics.getOrElse(name, NoopMetric)

  /** Narrow RDD dependencies align local build partitions without collecting across an exchange. */
  def execute(plan: SparkPlan): Option[RDD[ColumnarBatch]] = {
    val enabled = RapidsConf.ENABLE_JOIN_CHAIN.get(plan.conf)
    val verbose = RapidsConf.JOIN_CHAIN_LOG_DIAGNOSTICS.get(plan.conf)
    // Analyze even when the chain is disabled so both sides of an A/B run report the same shape
    // information. The analysis is plan-time only and runs once per join operator.
    val outcome = if (enabled || verbose || log.isDebugEnabled) {
      analyze(plan)
    } else {
      NotAnchored(ChainBreak.disabled)
    }
    logOutcome(plan, outcome, enabled, verbose)
    outcome match {
      case Fused(chain) if enabled => Some(buildRdd(plan, chain))
      case _ => None
    }
  }

  // Only size-based coalesces are transparent at this boundary. A SingleBatch consumer must
  // retain its batching contract; exchanges and computed projections remain boundaries.
  @scala.annotation.tailrec
  private def filterAnchor(plan: SparkPlan): SparkPlan = plan match {
    case c: GpuCoalesceBatches if c.goal.isInstanceOf[TargetSize] => filterAnchor(c.child)
    case _ => plan
  }

  def canFilter(plan: SparkPlan): Boolean = analyze(filterAnchor(plan)).isInstanceOf[Fused]

  def executeFiltered(plan: SparkPlan, filter: JoinChainFilter): Option[RDD[ColumnarBatch]] = {
    val anchor = filterAnchor(plan)
    def eagerResidual(): Option[RDD[ColumnarBatch]] = filter.joinCondition.flatMap { condition =>
      anchor match {
        case join: GpuBroadcastHashJoinExec =>
          Some(join.copy(condition = Some(condition)).executeColumnar()
            .mapPartitions(filter.residualFallback))
        case _ => None
      }
    }
    if (RapidsConf.ENABLE_JOIN_CHAIN.get(plan.conf)) {
      analyze(anchor) match {
        case outcome @ Fused(chain) =>
          val verbose = RapidsConf.JOIN_CHAIN_LOG_DIAGNOSTICS.get(plan.conf)
          logOutcome(anchor, outcome, enabled = true, verbose)
          if (verbose) {
            val terminal = if (filter.projection.isDefined) {
              if (filter.filterEnabled) "filterProject" else "project"
            } else "filter"
            val columns = filter.projection.map(_.ordinals.size)
            logWarning(s"$LOG_TAG terminal=$terminal predicateColumns=${filter.ordinals.size} " +
              s"projectInputColumns=$columns")
          }
          Some(buildRdd(anchor, chain, Some(filter)))
        case _ => eagerResidual()
      }
    } else eagerResidual()
  }

  /** Keep the first live batch buffered; empty inner/semi streams need no build work. */
  private[execution] def bufferNonemptyStream(input: Iterator[ColumnarBatch]):
      CloseableBufferedIterator[ColumnarBatch] = {
    val buffered = new CloseableBufferedIterator(input)
    closeOnExcept(buffered) { stream =>
      while (stream.hasNext && stream.head.numRows() == 0) {
        withResource(stream.next())(_ => ())
      }
      stream
    }
  }

  private def buildRdd(plan: SparkPlan, chain: Chain,
      filter: Option[JoinChainFilter] = None): RDD[ColumnarBatch] = {
    val terminal = filter.orElse {
      if (chain.virtual) Some(GpuJoinChainPipelineExec(None, plan.output.toList, plan).bind())
      else None
    }
    val prepared = chain.links.zipWithIndex.map { case (link, i) =>
      val residual = link.eagerCondition.orElse {
        if (i == chain.links.size - 1) terminal.flatMap(_.joinCondition) else None
      }
      prepare(link, residual)
    }
    val tailFilters = chain.tailFilters.map { case (filter, condition) =>
      inputFilter(filter, condition, chain.rawOutput)
    }
    val eagerTail = eagerOperations(chain.tailOperations)
    val broadcastOnly = chain.links.forall(_.node.broadcast)
    val root = chain.links.last.node
    val attrs = chain.input.output
    val options = RapidsConf.getJoinOptions(plan.conf, root.target)
    val opTime = root.plan.gpuLongMetric(GpuMetric.OP_TIME_LEGACY)
    val streamTime = root.plan.gpuLongMetric(GpuMetric.STREAM_TIME)
    // Plan-time analysis only says a chain was built. These report whether it survived at
    // runtime, per task, which the driver-side diagnostic line cannot observe.
    val fusedTasks = optionalMetric(root.plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS)
    val oversizedFallbacks = optionalMetric(root.plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED)
    val nonUniqueFallbacks = optionalMetric(root.plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE)
    val classifications = optionalMetric(root.plan, GpuMetric.JOIN_CHAIN_BUILD_KEY_STATS)
    val statsCacheHits = optionalMetric(root.plan, GpuMetric.JOIN_CHAIN_BUILD_KEY_STATS_CACHE_HITS)
    val streamNvtx = if (root.broadcast) NvtxRegistry.BROADCAST_JOIN_STREAM
      else NvtxRegistry.SHUFFLED_JOIN_STREAM
    var inputs = chain.input.executeColumnar().mapPartitions { stream =>
      Iterator.single(Inputs(stream, Seq.empty))
    }
    chain.links.foreach { link =>
      if (!link.node.broadcast) {
        inputs = inputs.zipPartitions(link.node.build.executeColumnar()) { (bundles, build) =>
          val bundle = bundles.next()
          Iterator.single(bundle.copy(builds = bundle.builds :+ build))
        }
      } else {
        inputs = inputs.mapPartitions { bundles =>
          val bundle = bundles.next()
          Iterator.single(bundle.copy(builds = bundle.builds :+ Iterator.empty))
        }
      }
    }
    inputs.mapPartitions { bundles =>
      val bundle = bundles.next()
      val timedStream = new CollectTimeIterator(streamNvtx, bundle.stream, streamTime)
      // Sized/shuffled loaders may need to read the build before pulling a GPU stream batch.
      val buffered = if (broadcastOnly) bufferNonemptyStream(timedStream)
        else new CloseableBufferedIterator(timedStream)
      if (broadcastOnly && !buffered.hasNext) {
        withResource(buffered)(_ => Iterator.empty)
      } else closeOnExcept(buffered) { _ =>
        var stream: Batches = buffered
        val owners = scala.collection.mutable.ArrayBuffer[LazySpillableColumnarBatch]()
        val built = scala.collection.mutable.ArrayBuffer[
          Either[LazySpillableColumnarBatch, Batches]]()
        closeOnExcept(owners) { _ =>
          prepared.zip(bundle.builds).foreach { case (stage, buildInput) =>
            val (data, buffered) = stage.load(buildInput, stream)
            stream = buffered
            data match {
              case Left(batch) =>
                withResource(batch) { batch =>
                  val owned = LazySpillableColumnarBatch(batch, "join_chain_build")
                  closeOnExcept(owned) { _ =>
                    owned.allowSpilling()
                    owners += owned
                    built += Left(owned)
                  }
                }
              case Right(iterator) => built += Right(iterator)
            }
          }
          def eager(): Batches = {
            prepared.indices.foreach { i =>
              stream = prepared(i).eagerInput(stream)
              if (!prepared(i).stage.virtualInput) {
                val refs = prepared(i).stage.streamProjection
                stream = stream.map(batch => withResource(batch)(select(_, refs)))
              }
              val data = built(i).map(identity).left.map(_.releaseBatch())
              stream = prepared(i).eager(data, stream)
            }
            stream = eagerTail(stream)
            // A residual was already applied by the restored native root join. Candidate and
            // avoided-row counters describe only the deferred path in this case.
            terminal.map { terminal =>
              if (terminal.joinCondition.isDefined) terminal.residualFallback(stream)
              else terminal.eager(stream)
            }.getOrElse(stream)
          }
          opTime.ns {
            if (built.exists(_.isRight)) {
              // Preserve out-of-core joins. A chain must never concatenate an oversized build.
              oversizedFallbacks += 1
              withResource(owners)(_ => eager())
            } else {
              val builds = owners.toVector
              val stages = prepared.map(_.stage)
              val keys = buildKeys(builds, stages)
              closeOnExcept(keys) { _ =>
                val classified = keys.zip(prepared).map { case (key, stage) =>
                  classifyBuildKey(key, stage.stage, stage.statsCache(),
                    classifications, statsCacheHits)
                }
                // Splitting the stream cannot bound a single row with excessive fanout. Keep
                // the ordinary join's out-of-core path for that case, before probing any stage.
                val fits = classified.forall(_.buildKeysUnique) ||
                  mapWorkPerRow(classified) <= maxMapRows(options.targetSize)
                if (fits) {
                  chain.links.zip(builds).foreach { case (link, data) =>
                    if (!link.node.broadcast) {
                      link.node.plan.gpuLongMetric(GpuMetric.BUILD_DATA_SIZE) +=
                        data.deviceMemorySize
                    }
                  }
                  fusedTasks += 1
                  terminal.foreach(_.fusedTasks += 1)
                  new JoinChainIterator(stream, attrs, builds, keys, classified, options, opTime,
                    terminal, tailFilters)
                } else {
                  nonUniqueFallbacks += 1
                  withResource(keys)(_ => withResource(owners)(_ => eager()))
                }
              }
            }
          }
        }
      }
    }
  }

  /** Select the distinct fast path or bound the complete-key fanout, including skew. */
  private[execution] def classifyBuildKeys(keys: Seq[LazySpillableColumnarBatch],
      stages: Seq[JoinChainStage]): Seq[JoinChainStage] = {
    stages.indices.map(i => classifyBuildKey(keys(i), stages(i)))
  }

  private[execution] def classifyBuildKey(key: LazySpillableColumnarBatch,
      stage: JoinChainStage, cache: Option[JoinChainBuildKeyStatsCache] = None,
      classifications: GpuMetric = NoopMetric, cacheHits: GpuMetric = NoopMetric):
      JoinChainStage = {
    if (stage.isLeftSemi) {
      // Semi keys were deduplicated and never multiply the stream rows.
      stage.copy(buildKeysUnique = true, maxBuildMatches = 1)
    } else {
      def compute(): JoinChainBuildKeyStats = {
        key.checkpoint()
        val result = withRetryNoSplit {
          withRestoreOnRetry(key) {
            withResource(GpuColumnVector.from(key.getBatch)) { table =>
              val unique = table.distinctCount(NullEquality.EQUAL) == key.numRows
              val maxMatches = if (unique) {
                1L
              } else {
                val groupOptions = GroupByOptions.builder()
                  .withIgnoreNullKeys(!stage.nullsEqual).build()
                withResource(table.groupBy(groupOptions, stage.buildKeys.indices: _*)
                    .aggregate(GroupByAggregation.count(NullPolicy.INCLUDE).onColumn(0))) {
                  counts =>
                    withResource(counts.getColumn(counts.getNumberOfColumns - 1).max()) { max =>
                      if (max.isValid) math.max(1L, max.getInt.toLong) else 1L
                    }
                }
              }
              key.allowSpilling()
              JoinChainBuildKeyStats(unique, maxMatches)
            }
          }
        }
        classifications += 1
        result
      }
      val stats = cache match {
        case Some(shared) =>
          val ordinals = stage.buildKeys.map(_.asInstanceOf[GpuBoundReference].ordinal)
          var computed = false
          val result = shared.getOrCompute(ordinals, stage.nullsEqual) {
            computed = true
            compute()
          }
          if (!computed) cacheHits += 1
          result
        case None => compute()
      }
      // Never cache a stage: its metrics and stream projections belong to the current task.
      stage.copy(buildKeysUnique = stats.unique, maxBuildMatches = stats.maxMatches)
    }
  }

  private[execution] def saturatingMultiply(left: Long, right: Long): Long =
    if (left > Long.MaxValue / right) Long.MaxValue else left * right

  private[execution] def mapTargetSize(target: Long): Long =
    math.min(target, Int.MaxValue.toLong * 2 * Integer.BYTES)

  private[execution] def maxMapRows(target: Long): Long =
    math.max(1L, mapTargetSize(target) / (2 * Integer.BYTES))

  /** Upper bound in pairs of INT32 indices: previous/composed source maps plus join maps.
   * Products are conservative across stages since key correlations are not known here.
   */
  private[execution] def mapWorkPerRow(stages: Seq[JoinChainStage]): Long = {
    var fanout = 1L
    var sources = 1L
    var work = 1L
    stages.foreach { stage =>
      fanout = saturatingMultiply(fanout, stage.maxBuildMatches)
      if (!stage.isLeftSemi) sources += 1
      work = math.max(work, saturatingMultiply(fanout, sources + 1))
    }
    work
  }

  /** Sources needed after each probe, including the terminal predicate and project inputs. */
  private[execution] def liveSourcesAfterStages(factWidth: Int, buildWidths: Seq[Int],
      stages: Seq[JoinChainStage], outputOrdinals: Option[Seq[Int]] = None): Seq[Set[Int]] = {
    var layout: Seq[Int] = Seq.fill(factWidth)(0)
    val probeSources = stages.zipWithIndex.map { case (stage, i) =>
      layout = stage.streamProjection.map(layout)
      val needed = (stage.streamKeyOrdinals ++ stage.inputFilters.flatMap(_.ordinals))
        .map(layout).toSet
      if (!stage.isLeftSemi) {
        val build = Seq.fill(buildWidths(i))(i + 1)
        layout = if (stage.buildSide == GpuBuildRight) layout ++ build else build ++ layout
      }
      needed
    }
    var needed = outputOrdinals.map(_.map(layout)).getOrElse(layout).toSet
    stages.indices.reverse.map { i =>
      // Keep source slots aligned, but do not mark a future build as already live.
      val live = needed.filter(_ <= i + 1)
      needed = needed ++ probeSources(i)
      live
    }.reverse
  }

  /** Reference-only projection, stripping contiguous-batch subclasses when rearranging columns. */
  private[execution] def select(batch: ColumnarBatch, ordinals: Seq[Int]): ColumnarBatch = {
    val columns = ordinals.safeMap { i =>
      val col = batch.column(i).asInstanceOf[GpuColumnVector]
      GpuColumnVector.from(col.getBase.incRefCount(), col.dataType())
    }
    closeOnExcept(columns)(_ => new ColumnarBatch(columns.toArray, batch.numRows()))
  }

  private[execution] def buildKeys(builds: Seq[LazySpillableColumnarBatch],
      stages: Seq[JoinChainStage]): Seq[LazySpillableColumnarBatch] = {
    builds.foreach(_.checkpoint())
    builds.indices.safeMap { i =>
      val build = builds(i)
      val stage = stages(i)
      val identity = !stage.isLeftSemi && stage.buildKeys.length == build.numCols &&
        stage.buildKeys.zipWithIndex.forall {
          case (ref: GpuBoundReference, ordinal) =>
            ref.ordinal == ordinal && ref.dataType == build.dataTypes(ordinal)
          case _ => false
        }
      if (identity) {
        // The entire payload is the key table, so detaching would only duplicate its buffers.
        // Borrow the owner: closing the key wrapper must not close the chain's build payload.
        LazySpillableColumnarBatch.spillOnly(build)
      } else withRetryNoSplit {
        withRestoreOnRetry(builds(i)) {
          withResource(GpuProjectExec.project(builds(i).getBatch, stages(i).buildKeys)) { keys =>
            // Partial/reordered inner keys need an independent copy. Semi keys are deduplicated,
            // retaining one row even for repeated NULL keys; probe null equality is unchanged.
            val semi = stages(i).isLeftSemi
            withResource(GpuColumnVector.extractBases(keys).toSeq.safeMap { column =>
              if (semi) column.incRefCount() else column.subVector(0, keys.numRows())
            }) { columns =>
              withResource(new Table(columns: _*)) { table =>
                val types = GpuColumnVector.extractTypes(keys)
                val detached = if (semi) {
                  withResource(table.dropDuplicates(columns.indices.toArray,
                      Table.DuplicateKeepOption.KEEP_ANY, true)) { unique =>
                    GpuColumnVector.from(unique, types)
                  }
                } else {
                  GpuColumnVector.from(table, types)
                }
                withResource(detached) { batch =>
                  val result = LazySpillableColumnarBatch(batch, "join_chain_build_keys")
                  closeOnExcept(result) { owned =>
                    builds(i).allowSpilling()
                    owned.allowSpilling()
                    owned
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/** Original payloads + composed INT32 indices. Only keys are gathered between probes. */
private[execution] class JoinChainIterator(
    input: Iterator[ColumnarBatch],
    attrs: Seq[org.apache.spark.sql.catalyst.expressions.Attribute],
    builds: Seq[LazySpillableColumnarBatch],
    keys: Seq[LazySpillableColumnarBatch],
    stages: Seq[JoinChainStage],
    options: JoinOptions,
    opTime: GpuMetric = NoopMetric,
    filter: Option[JoinChainFilter] = None,
    tailFilters: Seq[JoinChainFilter] = Nil)
    extends SplittableJoinIterator(NvtxRegistry.JOIN_GATHER,
      input.map { batch =>
        withResource(batch)(LazySpillableColumnarBatch(_, "join_chain_fact"))
      }, attrs, builds.head, GpuJoinChain.mapTargetSize(options.targetSize),
      options.sizeEstimateThreshold,
      opTime, stages.last.joinTime) {
  private case class Origin(source: Int, column: Int)
  private val hashes = stages.map(stage =>
    new GpuJoinHashCache(stage.buildKeysUnique, stage.nullsEqual, options.hashReuseEnabled))
  private val terminalOrdinals = filter.flatMap { terminal =>
    terminal.projection.map { projection =>
      (projection.ordinals ++ (if (terminal.filterEnabled) terminal.ordinals else Nil) ++
        tailFilters.flatMap(_.ordinals)).distinct
    }
  }
  private val liveSources = GpuJoinChain.liveSourcesAfterStages(
    attrs.length, builds.map(_.numCols), stages, terminalOrdinals)
  private val hasFanout = stages.exists(!_.buildKeysUnique)
  private val mapWork = GpuJoinChain.mapWorkPerRow(stages)
  override protected def splitBeforePending: Boolean = hasFanout
  private[execution] def hashBuildCount: Long = hashes.map(_.buildCount).sum
  override def close(): Unit = {
    if (!closed) {
      withResource(hashes ++ builds.drop(1) ++ keys) { _ =>
        val closeableInput = input match {
          case closeable: AutoCloseable => Some(closeable)
          case _ => None
        }
        withResource(closeableInput)(_ => super.close())
      }
    }
  }

  override protected def computeNumJoinRows(batch: LazySpillableColumnarBatch): Long = {
    if (hasFanout) {
      // Limit each splitting pass to eight pieces rather than queuing potentially millions
      // of batches at once. Each piece is checked again before any probe allocation.
      val maxSplits = 8L
      math.min(GpuJoinChain.saturatingMultiply(batch.numRows.toLong, mapWork),
        GpuJoinChain.saturatingMultiply(GpuJoinChain.maxMapRows(options.targetSize), maxSplits))
    } else {
      batch.numRows.toLong * stages.size
    }
  }

  private class IndexMap(val data: LazySpillableColumnarBatch) extends LazySpillableGatherMap {
    override val getRowCount: Long = data.numRows
    override def toColumnView(start: Long, rows: Int): ColumnView = {
      val column = GpuColumnVector.extractBases(data.getBatch)(0)
      withResource(column.getData) { buffer =>
        ColumnView.fromDeviceBuffer(buffer, start * Integer.BYTES, DType.INT32, rows)
      }
    }
    override def close(): Unit = withResource(data)(_ => ())
    override def allowSpilling(): Unit = data.allowSpilling()
    override def checkpoint(): Unit = data.checkpoint()
    override def restore(): Unit = data.restore()
  }

  private def index(column: ColumnVector): IndexMap = {
    withResource(new Table(column)) { table =>
      withResource(GpuColumnVector.from(table, Array[org.apache.spark.sql.types.DataType](
          IntegerType))) { batch =>
        new IndexMap(LazySpillableColumnarBatch(batch, "join_chain_indices"))
      }
    }
  }

  protected def beforeProbe(stage: Int): Unit = ()
  protected def beforeFilter(): Unit = ()
  protected def beforeProject(): Unit = ()

  override protected def createGatherer(fact: LazySpillableColumnarBatch,
      numJoinRows: Option[Long]): Option[JoinGatherer] = {
    if (fact.numRows == 0 || builds.exists(_.numRows == 0)) {
      None
    } else {
      val inputs = Seq(fact) ++ builds ++ keys
      inputs.foreach(_.checkpoint())
      try {
        val (gatherer, counts, selectedRows, inputCounts) = withRetryNoSplit {
          withRestoreOnRetry(hashes) {
            withRestoreOnRetry(inputs) { buildGatherer(fact) }
          }
        }
        // Commit metrics only after the complete attempt succeeds, avoiding retry double counts.
        // With a terminal filter, the root join reports logical candidate map batches; it
        // must not report the filtered payload batches emitted by ProjectedGatherer.
        val countedStages = if (filter.isDefined) stages else stages.dropRight(1)
        countedStages.zip(counts).foreach { case (stage, rows) =>
          stage.outputRows += rows
          if (rows > 0) stage.outputBatches += 1
        }
        inputCounts.foreach { case (predicate, before, after) =>
          predicate.inputRows += before
          if (before > 0) predicate.inputBatches += 1
          predicate.selectedRows += after
          if (after > 0) predicate.selectedBatches += 1
        }
        filter.foreach { terminal =>
          val candidates = if (tailFilters.nonEmpty) inputCounts.last._3.toLong else counts.last
          terminal.inputRows += candidates
          if (candidates > 0) terminal.inputBatches += 1
          terminal.avoidedPayloadRows += candidates - selectedRows
        }
        gatherer
      } catch {
        case oom @ (_: OutOfMemoryError | _: com.nvidia.spark.rapids.jni.GpuOOM) =>
          if (hasFanout && fact.numRows > 1) {
            // A small input can still produce large maps. Allow reducing it down to one row.
            splitAndSave(fact.getBatch, 2)
          } else {
            splitAndSave(fact.getBatch, 2, Some(oom))
          }
          None
      }
    }
  }

  private def buildGatherer(fact: LazySpillableColumnarBatch):
      (Option[JoinGatherer], Seq[Long], Int, Seq[(JoinChainFilter, Int, Int)]) = {
    val sources = Seq(fact) ++ builds
    var layout: Seq[Origin] = fact.dataTypes.indices.map(Origin(0, _)).toSeq
    var maps = Seq.empty[IndexMap]
    var rows = fact.numRows
    val counts = scala.collection.mutable.ArrayBuffer[Long]()
    val inputCounts = scala.collection.mutable.ArrayBuffer[(JoinChainFilter, Int, Int)]()
    try {
      if (stages.head.virtualInput) {
        // An explicit identity map lets source-side predicates select rows without gathering
        // any payload. It participates in the same spill/retry lifetime as composed maps.
        maps = withResource(Scalar.fromInt(0)) { zero =>
          withResource(ColumnVector.sequence(zero, rows))(column => Seq(index(column)))
        }
      }
      stages.zipWithIndex.foreach { case (stage, i) =>
        layout = stage.streamProjection.map(layout)
        stage.inputFilters.foreach { predicate =>
          val before = rows
          if (rows > 0) {
            predicate.filterTime.ns {
              beforeFilter()
              filterMaps(predicate, layout, sources, maps, rows)
            }.foreach { case (selected, selectedRows) =>
              withResource(maps) { _ => maps = selected }
              rows = selectedRows
              maps.filter(_ != null).foreach(_.allowSpilling())
            }
          }
          inputCounts += ((predicate, before, rows))
        }
        if (rows > 0) {
          beforeProbe(i)
          val next = stage.joinTime.ns {
            def streamKeyColumns(): Seq[ColumnVector] = stage.streamKeyProject match {
              case Some(project) =>
                withResource(predicateInput(project.ordinals.map(layout), sources, maps, rows)) {
                  input => withResource(project.expressions.project(input)) { evaluated =>
                    GpuColumnVector.extractBases(evaluated).toSeq.safeMap(_.incRefCount())
                  }
                }
              case None =>
                val origins = stage.streamKeyOrdinals.map(layout)
                origins.safeMap { origin =>
                  val source = sources(origin.source)
                  withResource(GpuColumnVector.from(source.getBatch)) { table =>
                    withResource(new Table(table.getColumn(origin.column))) { key =>
                      val column = if (maps.isEmpty) {
                        key.getColumn(0).copyToColumnVector()
                      } else {
                        withResource(maps(origin.source).toColumnView(0, rows)) { map =>
                          withResource(key.gather(map, OutOfBoundsPolicy.DONT_CHECK)) {
                            _.getColumn(0).incRefCount()
                          }
                        }
                      }
                      closeOnExcept(column) { owned =>
                        source.allowSpilling()
                        owned
                      }
                    }
                  }
                }
            }
            withResource(streamKeyColumns()) { streamColumns =>
              withResource(new Table(streamColumns: _*)) { streamKeys =>
                withResource(GpuColumnVector.from(keys(i).getBatch)) { buildKeys =>
                  withResource(hashes(i).innerJoin(buildKeys, streamKeys)) { joined =>
                    keys(i).allowSpilling()
                    val joinedRows = joined(0).getRowCount.toInt
                    if (joinedRows == 0) {
                      (Seq.empty[IndexMap], 0)
                    } else {
                      withResource(joined(0).toColumnView(0, joinedRows)) { selection =>
                        val live = liveSources(i)
                        val prefix = if (maps.isEmpty) {
                          Seq(if (live.contains(0)) {
                            withResource(selection.copyToColumnVector())(index)
                          } else null)
                        } else {
                          maps.zipWithIndex.safeMap { case (old, source) =>
                            if (old == null || !live.contains(source)) {
                              null
                            } else {
                              withResource(GpuColumnVector.from(old.data.getBatch)) { table =>
                                withResource(table.gather(selection,
                                    OutOfBoundsPolicy.DONT_CHECK)) {
                                  gathered => index(gathered.getColumn(0))
                                }
                              }
                            }
                          }
                        }
                        closeOnExcept(prefix) { _ =>
                          val result = if (stage.isLeftSemi || !live.contains(i + 1)) {
                            // Preserve source slots for semi builds and payloads that no later
                            // key or output uses, without allocating or composing their maps.
                            prefix :+ null
                          } else {
                            withResource(joined(1).toColumnView(0, joinedRows)) { right =>
                              prefix :+ withResource(right.copyToColumnVector())(index)
                            }
                          }
                          (result, joinedRows)
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          withResource(maps) { _ => maps = next._1 }
          // The fact map can be dead even when dimension-derived rows are still live.
          rows = next._2
          maps.filter(_ != null).foreach(_.allowSpilling())
        }
        counts += rows
        if (!stage.isLeftSemi) {
          val buildLayout = builds(i).dataTypes.indices.map(Origin(i + 1, _))
          layout = if (stage.buildSide == GpuBuildRight) layout ++ buildLayout
            else buildLayout ++ layout
        }
      }
      // Preserve each predicate's order and retain its source maps even when the final
      // projection (including COUNT(*)) does not reference those columns.
      tailFilters.foreach { predicate =>
        val before = rows
        if (rows > 0) {
          predicate.filterTime.ns {
            beforeFilter()
            filterMaps(predicate, layout, sources, maps, rows)
          }.foreach { case (selected, selectedRows) =>
            withResource(maps) { _ => maps = selected }
            rows = selectedRows
            maps.filter(_ != null).foreach(_.allowSpilling())
          }
        }
        inputCounts += ((predicate, before, rows))
      }
      filter.filter(_.filterEnabled).foreach { terminal =>
        if (rows > 0) {
          terminal.filterTime.ns {
            beforeFilter()
            filterMaps(terminal, layout, sources, maps, rows)
          }.foreach { case (selected, selectedRows) =>
            withResource(maps) { _ => maps = selected }
            rows = selectedRows
            maps.filter(_ != null).foreach(_.allowSpilling())
          }
        }
      }
      if (rows == 0) {
        (None, counts.toSeq, 0, inputCounts.toSeq)
      } else {
        // Gather each surviving source column once, then restore the logical output order by refs.
        val outputLayout = filter.flatMap(_.projection).map(_.ordinals.map(layout))
          .getOrElse(layout)
        val origins = outputLayout.distinct.sortBy(o => (o.source, o.column))
        val groups = origins.groupBy(_.source).toSeq.sortBy(_._1)
        val gatherers = groups.safeMap { case (source, columns) =>
          withResource(GpuJoinChain.select(sources(source).getBatch,
              columns.map(_.column))) { selected =>
            val data = LazySpillableColumnarBatch(selected, "join_chain_final_payload")
            closeOnExcept(data) { _ =>
              sources(source).allowSpilling()
              val map = maps(source)
              val result = JoinGatherer(map, data, OutOfBoundsPolicy.DONT_CHECK)
              maps = maps.updated(source, null)
              result
            }
          }
        }
        closeOnExcept(gatherers) { _ =>
          val combined = if (gatherers.isEmpty) new RowCountGatherer(rows)
            else gatherers.reduceLeft[JoinGatherer](MultiJoinGather)
          val projection = outputLayout.map(origins.indexOf)
          val pruned = layout.distinct.length - origins.length
          (Some(new ProjectedGatherer(combined, projection, pruned)), counts.toSeq, rows,
            inputCounts.toSeq)
        }
      }
    } finally {
      maps.filter(_ != null).safeClose()
    }
  }

  /** Read only predicate dependencies; payloads remain in their original spillable sources. */
  private def predicateInput(origins: Seq[Origin], sources: Seq[LazySpillableColumnarBatch],
      maps: Seq[IndexMap], rows: Int): ColumnarBatch = {
    val groups = origins.distinct.groupBy(_.source).toSeq.sortBy(_._1)
    withResource(groups.safeMap { case (source, columns) =>
      withResource(GpuJoinChain.select(sources(source).getBatch, columns.map(_.column))) {
        selected =>
          withResource(GpuColumnVector.from(selected)) { table =>
            withResource(maps(source).toColumnView(0, rows)) { map =>
              withResource(table.gather(map, OutOfBoundsPolicy.DONT_CHECK)) { gathered =>
                val batch = GpuColumnVector.from(gathered, GpuColumnVector.extractTypes(selected))
                closeOnExcept(batch) { _ =>
                  sources(source).allowSpilling()
                  batch
                }
              }
            }
          }
      }
    }) { batches =>
      val locations = groups.zipWithIndex.flatMap { case ((_, columns), batch) =>
        columns.zipWithIndex.map { case (origin, column) => origin -> ((batch, column)) }
      }.toMap
      val columns = origins.safeMap { origin =>
        val (batch, column) = locations(origin)
        batches(batch).column(column).asInstanceOf[GpuColumnVector].incRefCount()
      }
      closeOnExcept(columns)(_ => new ColumnarBatch(columns.toArray, rows))
    }
  }

  /** Compact INT32 maps together. Null predicates reject rows just as Table.filter does. */
  private def filterMaps(terminal: JoinChainFilter, layout: Seq[Origin],
      sources: Seq[LazySpillableColumnarBatch], maps: Seq[IndexMap], rows: Int):
      Option[(Seq[IndexMap], Int)] = {
    withResource(predicateInput(terminal.ordinals.map(layout), sources, maps, rows)) { input =>
      withResource(terminal.predicate.project(input)) { evaluated =>
        val mask = evaluated.column(0).asInstanceOf[GpuColumnVector].getBase
        // Match the ordinary filter's all-true fast path, retaining the existing maps.
        val allTrue = !mask.hasNulls && withResource(mask.all())(_.getBoolean)
        val live = maps.indices.filter(maps(_) != null)
        if (allTrue) None else if (live.isEmpty) {
          // Constant projections such as COUNT(*) may make every payload map dead.
          val selectedRows = withResource(mask.sum(DType.INT32)) { count =>
            if (count.isValid) count.getInt else 0
          }
          Some((Seq.fill(maps.length)(null), selectedRows))
        } else {
          // Borrowed views remain owned by their maps throughout this attempt.
          val columns = live.map(i => GpuColumnVector.extractBases(maps(i).data.getBatch)(0))
          withResource(new Table(columns: _*)) { table =>
            withResource(table.filter(mask)) { selected =>
              val selectedRows = selected.getRowCount.toInt
              val result = if (selectedRows == 0) Seq.fill(maps.length)(null) else {
                maps.indices.safeMap { i =>
                  if (maps(i) == null) null else index(selected.getColumn(live.indexOf(i)))
                }
              }
              Some((result, selectedRows))
            }
          }
        }
      }
    }
  }

  /** A count-only pipeline carries row cardinality without allocating a payload column. */
  private class RowCountGatherer(total: Int) extends JoinGatherer {
    private var offset = 0
    private var saved = 0
    override def gatherNext(rows: Int): ColumnarBatch = {
      require(rows > 0 && rows <= numRowsLeft)
      val result = new ColumnarBatch(Array.empty, rows)
      offset += rows
      result
    }
    override def isDone: Boolean = offset == total
    override def numRowsLeft: Long = total.toLong - offset
    override def realCheapPerRowSizeEstimate: Double = 1.0
    override def getFixedWidthBitSize: Option[Int] = Some(8)
    override def getBitSizeMap(rows: Int): ColumnView =
      withResource(Scalar.fromLong(8))(ColumnVector.fromScalar(_, rows))
    override def checkpoint(): Unit = { saved = offset }
    override def restore(): Unit = { offset = saved }
    override def allowSpilling(): Unit = ()
    override def close(): Unit = { offset = total }
  }

  private class ProjectedGatherer(delegate: JoinGatherer, projection: Seq[Int], pruned: Int)
      extends JoinGatherer {
    override def gatherNext(rows: Int): ColumnarBatch = {
      withResource(delegate.gatherNext(rows)) { gathered =>
        val result = withResource(GpuJoinChain.select(gathered, projection)) { inputs =>
          filter.flatMap(_.projection).map { terminal =>
            beforeProject()
            val output = terminal(inputs)
            terminal.inputColumnsPruned += pruned
            output
          }.getOrElse(GpuColumnVector.incRefCounts(inputs))
        }
        if (filter.isEmpty) {
          stages.last.outputRows += rows
          stages.last.outputBatches += 1
        }
        result
      }
    }
    override def isDone: Boolean = delegate.isDone
    override def numRowsLeft: Long = delegate.numRowsLeft
    private val extraBits = filter.flatMap(_.projection).map(_.extraBitsPerRow).getOrElse(0)
    override def realCheapPerRowSizeEstimate: Double =
      delegate.realCheapPerRowSizeEstimate + extraBits / 8.0
    override def getFixedWidthBitSize: Option[Int] =
      delegate.getFixedWidthBitSize.map(_ + extraBits)
    override def getBitSizeMap(rows: Int): ColumnView = {
      withResource(delegate.getBitSizeMap(rows)) { sizes =>
        withResource(Scalar.fromLong(extraBits.toLong))(sizes.add(_))
      }
    }
    override def checkpoint(): Unit = delegate.checkpoint()
    override def restore(): Unit = delegate.restore()
    override def allowSpilling(): Unit = delegate.allowSpilling()
    override def close(): Unit = withResource(delegate)(_ => ())
  }
}
