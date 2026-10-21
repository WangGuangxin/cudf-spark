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

import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.JavaSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeReference,
  NamedExpression, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Complete, Final, Partial}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning}
import org.apache.spark.sql.execution.{LeafExecNode, LocalTableScanExec, SparkPlan}
import org.apache.spark.sql.execution.exchange.{ENSURE_REQUIREMENTS, ShuffleExchangeExec}
import org.apache.spark.sql.rapids.{GpuGreaterThan, GpuLessThan, GpuMultiply}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum, GpuCount}
import org.apache.spark.sql.rapids.execution.{GpuBroadcastHashJoinExec, GpuJoinChain}
import org.apache.spark.sql.types.{IntegerType, LongType}

class JoinChainPipelinePlanningSuite extends AnyFunSuite {
  private def session(body: SparkSession => Unit): Unit =
    withResource(SparkSession.builder().master("local[1]")
      .appName("join pipeline planning").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.sql.warehouse.dir", "/tmp/join-pipeline-planning").getOrCreate()) { spark =>
      spark.conf.set(RapidsConf.ENABLE_JOIN_CHAIN.key, "true")
      spark.conf.set("spark.sql.ansi.enabled", "false")
      body(spark)
    }

  private def leaf(names: String*) = LocalTableScanExec(
    names.map(name => AttributeReference(name, IntegerType, nullable = true)()), Nil)

  private def root(): SparkPlan = {
    val fact = leaf("k", "v", "unused")
    val d1 = leaf("d1")
    val first = GpuBroadcastHashJoinExec(Seq(fact.output.head), Seq(d1.output.head),
      Inner, GpuBuildRight, None, fact, d1, isNullAwareAntiJoin = false)
    val d2 = leaf("d2")
    GpuBroadcastHashJoinExec(Seq(fact.output.head), Seq(d2.output.head),
      Inner, GpuBuildRight, None, first, d2, isNullAwareAntiJoin = false)
  }

  private def fuse(plan: SparkPlan): SparkPlan =
    GpuJoinChainPipelineExec.fuse(GpuJoinChainFilterExec.fuse(plan))

  private def aggregate(child: SparkPlan, countOnly: Boolean = false): GpuHashAggregateExec = {
    val groups = if (countOnly) Nil else Seq(child.output.head)
    val function = if (countOnly) GpuCount(Seq(GpuLiteral(1, IntegerType)), failOnError = false)
      else GpuBasicSum(child.output(1), LongType, failOnErrorOverride = false)
    val expression = GpuAggregateExpression(function, Partial, isDistinct = false,
      None, NamedExpression.newExprId)
    val buffers = function.aggBufferAttributes
    GpuHashAggregateExec(None, groups, Seq(expression), buffers, groups ++ buffers, child,
      1024 * 1024, 1.0, forceSinglePassAgg = false, allowSinglePassAgg = false,
      allowNonFullyAggregatedOutput = false, skipAggPassReductionRatio = 0.85)
  }

  test("join, filter, computed project and partial aggregate form one input pipeline") {
    session { _ =>
      val join = root()
      val filtered = GpuFilterExec(GpuLessThan(join.output(1), join.output(4)), join)()
      val twice = GpuAlias(GpuMultiply(join.output(1), GpuLiteral(2, IntegerType),
        failOnError = false)(), "twice")()
      val unused = GpuAlias(join.output(2), "unused_alias")()
      val projected = GpuProjectExec(List(join.output(3), twice, unused), filtered)
      val original = aggregate(projected)
      val result = fuse(original).asInstanceOf[GpuJoinChainAggregateExec]
      val pipeline = result.child.asInstanceOf[GpuJoinChainPipelineExec]
      assert(result.output == original.output)
      assert(pipeline.output.map(_.exprId) == projected.output.take(2).map(_.exprId))
      assert(pipeline.condition.contains(filtered.condition))
      val bound = pipeline.bind()
      assert(bound.ordinals == Seq(1, 4))
      assert(bound.projection.get.ordinals == Seq(1, 3))
      assert(bound.projection.get.expressions.outputExprs.length == 2)
    }
  }

  test("join and computed project feed a partial aggregate without a filter") {
    session { _ =>
      val join = root()
      val twice = GpuAlias(GpuMultiply(join.output(1), GpuLiteral(2, IntegerType),
        failOnError = false)(), "twice")()
      val projected = GpuProjectExec(List(join.output(3), twice, join.output(2)), join)
      val original = aggregate(projected)
      val result = fuse(original).asInstanceOf[GpuJoinChainAggregateExec]
      val pipeline = result.child.asInstanceOf[GpuJoinChainPipelineExec]
      assert(result.output == original.output)
      assert(pipeline.condition.isEmpty)
      assert(!pipeline.bind().filterEnabled)
      assert(pipeline.output.map(_.exprId) == projected.output.take(2).map(_.exprId))
      assert(pipeline.bind().projection.get.ordinals == Seq(1, 3))
    }
  }

  private def sourcePipeline(semi: Boolean = false): SparkPlan = {
    val fact = leaf("k", "v", "unused")
    val selected = GpuFilterExec(GpuGreaterThan(fact.output(1),
      GpuLiteral(0, IntegerType)), fact)()
    val twice = GpuAlias(GpuMultiply(selected.output(1), GpuLiteral(2, IntegerType),
      failOnError = false)(), "twice")()
    val projected = GpuProjectExec(List(selected.output.head, twice, selected.output(2)), selected)
    val d1 = leaf("d1")
    val first = GpuBroadcastHashJoinExec(Seq(projected.output.head), Seq(d1.output.head),
      Inner, GpuBuildRight, None, projected, d1, isNullAwareAntiJoin = false)
    val filtered = GpuFilterExec(GpuLessThan(first.output(1), first.output(3)), first)()
    val key = GpuAlias(GpuMultiply(first.output.head, GpuLiteral(2, IntegerType),
      failOnError = false)(), "next_key")()
    val measure = GpuAlias(GpuMultiply(first.output(1), first.output(3),
      failOnError = false)(), "measure")()
    val middle = GpuProjectExec(List(key, measure, first.output(2)), filtered)
    val d2 = leaf("d2")
    GpuBroadcastHashJoinExec(Seq(middle.output.head), Seq(d2.output.head),
      if (semi) LeftSemi else Inner, GpuBuildRight, None, middle, d2,
      isNullAwareAntiJoin = false)
  }

  for (semi <- Seq(false, true)) {
    test(s"source and inter-join filter/projects keep lineage through aggregate semi=$semi") {
      session { _ =>
        val join = sourcePipeline(semi)
        assert(GpuJoinChain.canFilter(join))
        val (raw, expressions) = GpuJoinChain.pipelineInput(join)
        assert(raw.map(_.name) == Seq("k", "v", "unused", "d1") ++
          (if (semi) Nil else Seq("d2")))
        assert(expressions.length == join.output.length)
        assert(expressions.head.isInstanceOf[GpuMultiply])
        assert(expressions(1).references.map(_.name).toSet == Set("v", "d1"))
        val partial = aggregate(join)
        val result = fuse(partial).asInstanceOf[GpuJoinChainAggregateExec]
        val pipeline = result.child.asInstanceOf[GpuJoinChainPipelineExec]
        assert(result.output == partial.output)
        assert(pipeline.bind().projection.get.ordinals == Seq(0, 1, 3))
        assert(!pipeline.bind().filterEnabled)
      }
    }
  }

  test("the full source pipeline emits the original partial buffers into Exchange") {
    session { _ =>
      val partial = aggregate(sourcePipeline())
      val exchange = ShuffleExchangeExec(HashPartitioning(partial.output.take(1), 3),
        partial, ENSURE_REQUIREMENTS)
      val result = fuse(exchange).asInstanceOf[ShuffleExchangeExec]
      assert(result.output == exchange.output)
      assert(result.outputPartitioning == exchange.outputPartitioning)
      assert(result.child.isInstanceOf[GpuJoinChainAggregateExec])
      assert(result.child.output == partial.output)
      assert(result.child.children.head.isInstanceOf[GpuJoinChainPipelineExec])
    }
  }

  test("a source-side predicate can extend even a single join into a pipeline") {
    session { _ =>
      val fact = leaf("k", "v")
      val selected = GpuFilterExec(GpuGreaterThan(fact.output(1),
        GpuLiteral(0, IntegerType)), fact)()
      val dim = leaf("d")
      val join = GpuBroadcastHashJoinExec(Seq(selected.output.head), Seq(dim.output.head),
        Inner, GpuBuildRight, None, selected, dim, isNullAwareAntiJoin = false)
      assert(GpuJoinChain.canFilter(join))
      val result = fuse(aggregate(join, countOnly = true))
        .asInstanceOf[GpuJoinChainAggregateExec]
      assert(result.child.asInstanceOf[GpuJoinChainPipelineExec].bind()
        .projection.get.ordinals.isEmpty)
    }
  }

  test("unsupported source operators and batching contracts remain execution boundaries") {
    session { _ =>
      val base = sourcePipeline().asInstanceOf[GpuBroadcastHashJoinExec]
      val blocked = GpuCoalesceBatches(base.left, RequireSingleBatch)
      val join = base.copy(left = blocked)
      assert(!GpuJoinChain.canFilter(join))
      val project = GpuProjectExec(base.left.output.toList, base.left, enablePreSplit = false)
      assert(!GpuJoinChain.canFilter(base.copy(left = project)))
      val filtered = GpuFilterExec(GpuGreaterThan(base.left.output(1),
        GpuLiteral(0, IntegerType)), base.left)(coalesceAfter = false)
      assert(!GpuJoinChain.canFilter(base.copy(left = filtered)))
    }
  }

  test("global COUNT(*) needs row cardinality and no payload columns") {
    session { _ =>
      val result = fuse(aggregate(root(), countOnly = true))
        .asInstanceOf[GpuJoinChainAggregateExec]
      val pipeline = result.child.asInstanceOf[GpuJoinChainPipelineExec]
      val bound = pipeline.bind()
      assert(pipeline.output.isEmpty)
      assert(!pipeline.coalesceAfter)
      assert(bound.projection.get.ordinals.isEmpty)
      assert(bound.projection.get.expressions.outputExprs.isEmpty)
      assert(!bound.filterEnabled)
    }
  }

  test("composed projects and a post-project filter bind to original source attributes") {
    session { _ =>
      val join = root()
      val twice = GpuAlias(GpuMultiply(join.output(1), GpuLiteral(2, IntegerType),
        failOnError = false)(), "twice")()
      val first = GpuProjectExec(List(twice, join.output.head), join)
      val second = GpuProjectExec(List(GpuAlias(twice.toAttribute, "renamed")()), first)
      val filtered = GpuFilterExec(GpuGreaterThan(second.output.head,
        GpuLiteral(20, IntegerType)), second)()
      val pipeline = fuse(filtered).asInstanceOf[GpuJoinChainPipelineExec]
      assert(pipeline.output == filtered.output)
      assert(pipeline.references.forall(a => join.output.exists(_.exprId == a.exprId)))
      assert(pipeline.bind().projection.get.ordinals == Seq(1))
    }
  }

  test("folding IS NOT NULL through a computed project preserves output nullability") {
    session { _ =>
      val join = root()
      val alias = GpuAlias(GpuMultiply(join.output(1), GpuLiteral(2, IntegerType),
        failOnError = false)(), "computed")()
      val projected = GpuProjectExec(List(alias), join)
      val filtered = GpuFilterExec(GpuIsNotNull(projected.output.head), projected)()
      assert(!filtered.output.head.nullable)
      val pipeline = fuse(filtered).asInstanceOf[GpuJoinChainPipelineExec]
      assert(pipeline.output == filtered.output)
      val renamed = GpuProjectExec(List(GpuAlias(filtered.output.head, "renamed")()), filtered)
      assert(fuse(renamed).output == renamed.output)
    }
  }

  test("a folded post-filter preserves the original native residual for fallback") {
    session { _ =>
      val base = root().asInstanceOf[GpuBroadcastHashJoinExec]
      val predicate = GpuLessThan(base.output(1), base.output(4))
      val join = base.copy(condition = Some(predicate))
      val projected = GpuProjectExec(List(join.output(1)), join)
      val filtered = GpuFilterExec(GpuGreaterThan(projected.output.head,
        GpuLiteral(10, IntegerType)), projected)()
      val pipeline = fuse(filtered).asInstanceOf[GpuJoinChainPipelineExec]
      assert(pipeline.residualCondition.contains(predicate))
      assert(pipeline.bind().filterOnResidualFallback)
    }
  }

  test("final aggregate, complete aggregate, SingleBatch and non-splitting project stay bounded") {
    session { _ =>
      val join = root()
      val partial = aggregate(join, countOnly = true)
      for (mode <- Seq(Final, Complete)) {
        val other = partial.copy(aggregateExpressions =
          partial.aggregateExpressions.map(_.copy(mode = mode)))
        assert(fuse(other).isInstanceOf[GpuHashAggregateExec])
      }
      val single = partial.copy(child = GpuCoalesceBatches(join, RequireSingleBatch))
      assert(fuse(single).isInstanceOf[GpuHashAggregateExec])
      val project = GpuProjectExec(List(join.output.head), join, enablePreSplit = false)
      assert(fuse(project).isInstanceOf[GpuProjectExec])
    }
  }

  test("pending tail projections preserve lineage through coalesces and nested pipelines") {
    session { _ =>
      val join = root()
      val filtered = GpuFilterExec(GpuGreaterThan(join.output(1),
        GpuLiteral(10, IntegerType)), join)()
      val twice = GpuAlias(GpuMultiply(join.output(1), GpuLiteral(2, IntegerType),
        failOnError = false)(), "twice")()
      val projected = GpuProjectExec(List(twice, join.output.head), filtered)
      val coalesced = GpuCoalesceBatches(projected, TargetSize(1024))
      val (raw, lineage) = GpuJoinChain.pipelineInput(coalesced)
      assert(raw == join.output)
      assert(lineage == Seq(twice.child, join.output.head))
      val nested = GpuJoinChainPipelineExec(None,
        List(GpuAlias(projected.output.head, "renamed")()), coalesced)
      val (_, nestedLineage) = GpuJoinChain.pipelineInput(nested)
      assert(nestedLineage == Seq(twice.child))
      val terminal = GpuJoinChainPipelineExec(None, List(projected.output.head), coalesced)
      assert(terminal.bind().projection.get.ordinals == Seq(1))
    }
  }

  test("pipeline reference caches survive task serialization after being populated") {
    session { spark =>
      val aggregate = fuse(this.aggregate(root())).asInstanceOf[GpuJoinChainAggregateExec]
      val pipeline = aggregate.child.asInstanceOf[GpuJoinChainPipelineExec]
      val serializer = new JavaSerializer(spark.sparkContext.getConf).newInstance()
      for (plan <- Seq(pipeline, aggregate)) {
        val references = plan.references.map(_.exprId).toSet
        assert(references.nonEmpty)
        val restored = serializer.deserialize[SparkPlan](serializer.serialize(plan))
        assert(restored.references.map(_.exprId).toSet == references)
        assert(restored.output == plan.output)
      }
    }
  }

  test("the existing join-chain switch disables pipeline planning") {
    session { spark =>
      val join = root()
      spark.conf.set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false")
      assert(fuse(GpuProjectExec(List(join.output.head), join)).isInstanceOf[GpuProjectExec])
      assert(fuse(aggregate(join, countOnly = true)).isInstanceOf[GpuHashAggregateExec])
    }
  }

  private case class OrderedLeaf(override val output: Seq[Attribute],
      override val outputOrdering: Seq[SortOrder], override val outputPartitioning: Partitioning)
      extends LeafExecNode {
    override def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException()
  }

  test("projection aliases preserve valid partitioning and only a surviving ordering prefix") {
    session { _ =>
      val k = AttributeReference("k", IntegerType)()
      val v = AttributeReference("v", IntegerType)()
      val leaf = OrderedLeaf(Seq(k, v), Seq(SortOrder(k, Ascending), SortOrder(v, Ascending)),
        HashPartitioning(Seq(k), 3))
      val alias = GpuAlias(k, "renamed")()
      val pipeline = GpuJoinChainPipelineExec(None, List(alias), leaf)
      assert(pipeline.outputPartitioning == HashPartitioning(Seq(alias.toAttribute), 3))
      assert(pipeline.outputOrdering.map(_.child) == Seq(alias.toAttribute))
      assert(pipeline.copy(projectList = List(v)).outputOrdering.isEmpty)
    }
  }
}
