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

import ai.rapids.cudf.{ColumnVector, ColumnView, DType, NullEquality, OutOfBoundsPolicy, Table}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{withRestoreOnRetry, withRetryNoSplit}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.plans.Inner
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

/** A stage consumes the preceding logical output, but its payload stays in the original sources. */
private[execution] case class JoinChainStage(
    buildKeys: Seq[GpuExpression],
    streamKeyOrdinals: Seq[Int],
    streamProjection: Seq[Int],
    buildSide: GpuBuildSide,
    nullsEqual: Boolean,
    outputRows: GpuMetric = NoopMetric,
    outputBatches: GpuMetric = NoopMetric,
    joinTime: GpuMetric = NoopMetric)

/** Fuse only a local stream-side chain. Exchanges, filters and computed projections end a chain. */
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
  private case class Link(node: Node, projection: Seq[Int])
  /** `break` records why the chain stopped growing past `input`, for diagnostics. */
  private case class Chain(input: SparkPlan, links: Seq[Link], break: ChainBreak)
  private case class Inputs(stream: Batches, builds: Seq[Batches])
  private case class Prepared(stage: JoinChainStage,
      load: (Batches, Batches) => (Built, Batches), eager: (Built, Batches) => Batches)

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
    case join: GpuShuffledSymmetricHashJoinExec =>
      // Keep the chain on the streaming side. The alternate input must fit the normal build limit;
      // otherwise the existing symmetric sizing/sub-partition path is resumed at runtime.
      val side = if (childJoin(join.left).isRight) GpuBuildRight
        else if (childJoin(join.right).isRight) GpuBuildLeft
        else {
          // At the anchor, prefer retaining the wider payload as the streaming source. Runtime
          // size and uniqueness checks still decide whether the selected build can be deferred.
          val leftWidth = join.left.output.map(_.dataType.defaultSize.toLong).sum
          val rightWidth = join.right.output.map(_.dataType.defaultSize.toLong).sum
          if (leftWidth < rightWidth) GpuBuildLeft else GpuBuildRight
        }
      Some(Node(join, side))
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
    else if (node.plan.joinType != Inner) Some(s"joinType=${node.plan.joinType}")
    else if (node.plan.condition.isDefined) Some("joinCondition")
    else if (node.plan.isSkewJoin) Some("skewJoin")
    else if (node.buildKeys.isEmpty) Some("noBuildKeys")
    else if (!node.buildKeys.forall(simpleKey)) Some("buildKeyType")
    else if (!node.streamKeys.forall(simpleKey)) Some("streamKeyType")
    else None
  }

  private def eligible(node: Node): Boolean = ineligible(node).isEmpty

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

  private def worthwhile(chain: Chain): Boolean = {
    // Per-row bytes screen out narrow chains; this does not predict selectivity or elapsed time.
    val stages = chain.links.size.toLong
    val savedPayload = chain.links.dropRight(1).flatMap(_.node.plan.output)
      .map(_.dataType.defaultSize.toLong).sum
    val composedMaps = (stages + 2) * (stages - 1) / 2
    val mapTraffic = 2L * Integer.BYTES * composedMaps
    val keyTraffic = chain.links.drop(1).flatMap(_.node.streamKeys)
      .map(key => 2L * key.dataType.defaultSize + Integer.BYTES).sum
    savedPayload > mapTraffic + keyTraffic
  }

  /**
   * Classify one plan node as a chain anchor. This is a pure function of plan shape and does not
   * consider `spark.rapids.sql.join.chain.enabled`, so an A/B run with the chain disabled still
   * reports which plans would have fused.
   */
  private def analyze(plan: SparkPlan): Outcome = node(plan) match {
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
          s"$LOG_TAG fused=true length=${chain.links.size} stoppedBy=${chain.break} $root"
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

  private def prepare(link: Link): Prepared = {
    val n = link.node
    val plan = n.plan
    val options = RapidsConf.getJoinOptions(plan.conf, n.target)
    val rows = plan.gpuLongMetric(GpuMetric.NUM_OUTPUT_ROWS)
    val batches = plan.gpuLongMetric(GpuMetric.NUM_OUTPUT_BATCHES)
    val opTime = plan.gpuLongMetric(GpuMetric.OP_TIME_LEGACY)
    val joinTime = plan.gpuLongMetric(GpuMetric.JOIN_TIME)
    val stage = JoinChainStage(n.buildKeys,
      n.streamKeys.map(_.asInstanceOf[GpuBoundReference].ordinal), link.projection,
      n.side, GpuHashJoin.compareNullsEqual(plan.joinType, n.buildKeys), rows, batches, joinTime)
    val (load, eager): ((Batches, Batches) => (Built, Batches), (Built, Batches) => Batches) =
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
          val fallback = (built: Built, stream: Batches) => built match {
            case Left(batch) =>
              join.doJoin(batch, stream, options, rows, batches, opTime, joinTime)
            case Right(_) => throw new IllegalStateException("broadcast build must be one batch")
          }
          (loader, fallback)
        case join: GpuShuffledHashJoinExec =>
          (join.joinChainBuildLoader(n.side), join.joinChainEager(n.side))
        case join: GpuShuffledSymmetricHashJoinExec =>
          (join.joinChainBuildLoader(n.side), join.joinChainEager(n.side))
      }
    Prepared(stage, load, eager)
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

  private def buildRdd(plan: SparkPlan, chain: Chain): RDD[ColumnarBatch] = {
    val prepared = chain.links.map(prepare)
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
      var stream: Batches = new CollectTimeIterator(streamNvtx, bundle.stream, streamTime)
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
            val refs = prepared(i).stage.streamProjection
            stream = stream.map(batch => withResource(batch)(select(_, refs)))
            val data = built(i).map(identity).left.map(_.releaseBatch())
            stream = prepared(i).eager(data, stream)
          }
          stream
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
              val unique = builds.indices.forall { i =>
                keys(i).checkpoint()
                withRetryNoSplit {
                  withRestoreOnRetry(keys(i)) {
                    withResource(GpuColumnVector.from(keys(i).getBatch)) { table =>
                      val result = table.distinctCount(NullEquality.EQUAL) == builds(i).numRows
                      keys(i).allowSpilling()
                      result
                    }
                  }
                }
              }
              if (unique) {
                chain.links.zip(builds).foreach { case (link, data) =>
                  if (!link.node.broadcast) {
                    link.node.plan.gpuLongMetric(GpuMetric.BUILD_DATA_SIZE) +=
                      data.deviceMemorySize
                  }
                }
                fusedTasks += 1
                new JoinChainIterator(stream, attrs, builds, keys, stages, options, opTime)
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

  /** Reference-only projection, stripping contiguous-batch subclasses when rearranging columns. */
  private[execution] def select(batch: ColumnarBatch, ordinals: Seq[Int]): ColumnarBatch = {
    val columns = ordinals.safeMap { i =>
      val col = batch.column(i).asInstanceOf[GpuColumnVector]
      GpuColumnVector.from(col.getBase.incRefCount(), col.dataType())
    }
    new ColumnarBatch(columns.toArray, batch.numRows())
  }

  private[execution] def buildKeys(builds: Seq[LazySpillableColumnarBatch],
      stages: Seq[JoinChainStage]): Seq[LazySpillableColumnarBatch] = {
    builds.foreach(_.checkpoint())
    builds.indices.safeMap { i =>
      withRetryNoSplit {
        withRestoreOnRetry(builds(i)) {
          withResource(GpuProjectExec.project(builds(i).getBatch, stages(i).buildKeys)) { keys =>
            // Copy build keys once, so they cannot pin an entire contiguous payload.
            withResource(GpuColumnVector.extractBases(keys).toSeq.safeMap { column =>
              column.subVector(0, keys.numRows())
            }) { columns =>
                withResource(new Table(columns: _*)) { table =>
                  withResource(GpuColumnVector.from(table, GpuColumnVector.extractTypes(keys))) {
                    batch =>
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
    opTime: GpuMetric = NoopMetric)
    extends SplittableJoinIterator(NvtxRegistry.JOIN_GATHER,
      input.map { batch =>
        withResource(batch)(LazySpillableColumnarBatch(_, "join_chain_fact"))
      }, attrs, builds.head, options.targetSize, options.sizeEstimateThreshold,
      opTime, stages.last.joinTime) {
  private case class Origin(source: Int, column: Int)
  private val hashes = stages.map(stage => new GpuJoinHashCache(true, stage.nullsEqual))
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

  override protected def computeNumJoinRows(batch: LazySpillableColumnarBatch): Long =
    batch.numRows.toLong * stages.size

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

  override protected def createGatherer(fact: LazySpillableColumnarBatch,
      numJoinRows: Option[Long]): Option[JoinGatherer] = {
    if (fact.numRows == 0 || builds.exists(_.numRows == 0)) {
      None
    } else {
      val inputs = Seq(fact) ++ builds ++ keys
      inputs.foreach(_.checkpoint())
      try {
        val (gatherer, counts) = withRetryNoSplit {
          withRestoreOnRetry(hashes) {
            withRestoreOnRetry(inputs) { buildGatherer(fact) }
          }
        }
        // Commit metrics only after the complete attempt succeeds, avoiding retry double counts.
        stages.dropRight(1).zip(counts).foreach { case (stage, rows) =>
          stage.outputRows += rows
          if (rows > 0) stage.outputBatches += 1
        }
        gatherer
      } catch {
        case oom @ (_: OutOfMemoryError | _: com.nvidia.spark.rapids.jni.GpuOOM) =>
          splitAndSave(fact.getBatch, 2, Some(oom))
          None
      }
    }
  }

  private def buildGatherer(fact: LazySpillableColumnarBatch):
      (Option[JoinGatherer], Seq[Long]) = {
    val sources = Seq(fact) ++ builds
    var layout: Seq[Origin] = fact.dataTypes.indices.map(Origin(0, _)).toSeq
    var maps = Seq.empty[IndexMap]
    var rows = fact.numRows
    val counts = scala.collection.mutable.ArrayBuffer[Long]()
    try {
      stages.zipWithIndex.foreach { case (stage, i) =>
        layout = stage.streamProjection.map(layout)
        if (rows > 0) {
          beforeProbe(i)
          val next = stage.joinTime.ns {
            val origins = stage.streamKeyOrdinals.map(layout)
            withResource(origins.safeMap { origin =>
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
            }) { streamColumns =>
              withResource(new Table(streamColumns: _*)) { streamKeys =>
                withResource(GpuColumnVector.from(keys(i).getBatch)) { buildKeys =>
                  withResource(hashes(i).innerJoin(buildKeys, streamKeys)) { joined =>
                    keys(i).allowSpilling()
                    val joinedRows = joined(0).getRowCount.toInt
                    if (joinedRows == 0) {
                      Seq.empty[IndexMap]
                    } else {
                      withResource(joined(0).toColumnView(0, joinedRows)) { selection =>
                        val prefix = if (maps.isEmpty) {
                          Seq(withResource(selection.copyToColumnVector())(index))
                        } else {
                          maps.safeMap { old =>
                            withResource(GpuColumnVector.from(old.data.getBatch)) { table =>
                              withResource(table.gather(selection, OutOfBoundsPolicy.DONT_CHECK)) {
                                gathered => index(gathered.getColumn(0))
                              }
                            }
                          }
                        }
                        closeOnExcept(prefix) { _ =>
                          withResource(joined(1).toColumnView(0, joinedRows)) { right =>
                            prefix :+ withResource(right.copyToColumnVector())(index)
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          withResource(maps) { _ => maps = next }
          rows = maps.headOption.map(_.getRowCount.toInt).getOrElse(0)
          maps.foreach(_.allowSpilling())
        }
        counts += rows
        val buildLayout = builds(i).dataTypes.indices.map(Origin(i + 1, _))
        layout = if (stage.buildSide == GpuBuildRight) layout ++ buildLayout
          else buildLayout ++ layout
      }
      if (rows == 0) {
        (None, counts.toSeq)
      } else {
        // Gather each surviving source column once, then restore the logical output order by refs.
        val origins = layout.distinct.sortBy(o => (o.source, o.column))
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
          val combined = gatherers.reduceLeft[JoinGatherer](MultiJoinGather)
          val projection = layout.map(origins.indexOf)
          (Some(new ProjectedGatherer(combined, projection)), counts.toSeq)
        }
      }
    } finally {
      maps.filter(_ != null).safeClose()
    }
  }

  private class ProjectedGatherer(delegate: JoinGatherer, projection: Seq[Int])
      extends JoinGatherer {
    override def gatherNext(rows: Int): ColumnarBatch = {
      withResource(delegate.gatherNext(rows)) { gathered =>
        val result = GpuJoinChain.select(gathered, projection)
        stages.last.outputRows += rows
        stages.last.outputBatches += 1
        result
      }
    }
    override def isDone: Boolean = delegate.isDone
    override def numRowsLeft: Long = delegate.numRowsLeft
    override def realCheapPerRowSizeEstimate: Double = delegate.realCheapPerRowSizeEstimate
    override def getFixedWidthBitSize: Option[Int] = delegate.getFixedWidthBitSize
    override def getBitSizeMap(rows: Int): ColumnView = delegate.getBitSizeMap(rows)
    override def checkpoint(): Unit = delegate.checkpoint()
    override def restore(): Unit = delegate.restore()
    override def allowSpilling(): Unit = delegate.allowSpilling()
    override def close(): Unit = withResource(delegate)(_ => ())
  }
}
