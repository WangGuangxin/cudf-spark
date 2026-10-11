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

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Final, Partial}
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum, GpuCount,
  GpuMax, GpuMin}
import org.apache.spark.sql.types._

object ExpandAggregateFusionSuite {
  case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Planning-only leaf")
  }

  def fixture(): (GpuHashAggregateExec, GpuExpandExec) = {
    val key = AttributeReference("key", IntegerType)()
    val value = AttributeReference("value", LongType)()
    val unused = AttributeReference("unused", IntegerType)()
    val expanded = AttributeReference("expanded", IntegerType)()
    val gid = AttributeReference("discriminator", LongType, nullable = false)()
    val input = Seq(key, value, unused)
    val output = input ++ Seq(expanded, gid)
    val projections = Seq(
      input ++ Seq(key, GpuLiteral(0L, LongType)),
      input ++ Seq(GpuLiteral(null, IntegerType), GpuLiteral(1L, LongType)))
    val expand = GpuExpandExec(projections, output, TestLeaf(input))()
    val functions = Seq(GpuCount(Seq(value), failOnError = false),
      GpuBasicSum(value, LongType, failOnErrorOverride = false), GpuMin(value), GpuMax(value),
      GpuCount(Seq(GpuLiteral(1, IntegerType)), failOnError = false))
    val aggregates = functions.map(fn => GpuAggregateExpression(fn, Partial, isDistinct = false,
      None, NamedExpression.newExprId))
    val groups = Seq(expanded, gid)
    val buffers = functions.flatMap(_.aggBufferAttributes)
    val agg = GpuHashAggregateExec(None, groups, aggregates, buffers,
      groups ++ buffers, expand, 1024 * 1024, 1.0, forceSinglePassAgg = false,
      allowSinglePassAgg = false, allowNonFullyAggregatedOutput = false,
      skipAggPassReductionRatio = 0.85)
    (agg, expand)
  }
}

class ExpandAggregateFusionSuite extends SparkQueryCompareTestSuite {
  private def configuration(enabled: Boolean, filterFusion: Boolean = false,
      aqe: Boolean = false): SparkConf = new SparkConf()
    .set(RapidsConf.ENABLE_EXPAND_AGGREGATE_FUSION.key, enabled.toString)
    .set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, filterFusion.toString)
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.ansi.enabled", "false")
    .set("spark.sql.shuffle.partitions", "3")

  private def data(spark: SparkSession, empty: Boolean = false): DataFrame = {
    val schema = StructType(Seq(StructField("a", IntegerType), StructField("b", StringType),
      StructField("v", LongType), StructField("unused", StringType)))
    val rows = (0 until (if (empty) 0 else 80)).map { i =>
      Row(if (i % 5 == 0) null else Int.box(i % 4),
        if (i % 7 == 0) null else s"key-${i % 3}",
        if (i % 6 == 0) null else Long.box(i.toLong - 30), s"payload-$i")
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 3), schema)
  }

  for (sets <- Seq("ROLLUP(a,b)", "CUBE(a,b)",
      "GROUPING SETS ((a,b),(a),(b),())", "GROUPING SETS ((a,b),(a,b),())",
      "GROUPING SETS ((a),(b),())")) {
    test(s"CPU/off/on preserve nullable groups and all aggregate buffers: $sets") {
      def query(spark: SparkSession): DataFrame = {
        data(spark).createOrReplaceTempView("fusion_input")
        spark.sql(s"SELECT a,b,grouping_id(),count(*),count(v),sum(v),min(v),max(v) " +
          s"FROM fusion_input GROUP BY $sets")
      }
      val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
      for (enabled <- Seq(false, true)) {
        withGpuSparkSession({ spark =>
          val df = query(spark)
          compareResults(true, 0.0, cpu, df.collect())
          val nodes = PlanUtils.findOperators(df.queryExecution.executedPlan,
            _.isInstanceOf[GpuExpandAggregateExec])
          assert(nodes.nonEmpty == enabled, df.queryExecution.executedPlan.toString)
        }, configuration(enabled))
      }
    }
  }

  test("aggregate arguments follow branch-specific NULL substitution and integer widening") {
    def query(spark: SparkSession): DataFrame = {
      data(spark).createOrReplaceTempView("fusion_keys")
      spark.sql("SELECT a,grouping_id(),count(a),sum(a),min(a),max(a) " +
        "FROM fusion_keys GROUP BY ROLLUP(a)")
    }
    val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuExpandAggregateExec]).nonEmpty == enabled)
      }, configuration(enabled))
    }
  }

  test("boolean/date/timestamp keys retain grouping-set types and nullability") {
    def query(spark: SparkSession): DataFrame = {
      data(spark).selectExpr("CAST(a % 2 AS BOOLEAN) AS flag",
        "date_add(DATE '2025-01-01', a) AS d", "CAST(a AS TIMESTAMP) AS t", "v")
        .createOrReplaceTempView("fusion_types")
      spark.sql("SELECT flag,d,t,grouping_id(),count(v),sum(v) " +
        "FROM fusion_types GROUP BY ROLLUP(flag,d,t)")
    }
    val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        // The shared unordered comparator does not support Boolean values.
        def counts(rows: Array[Row]): Map[Row, Int] =
          rows.toSeq.groupBy(identity).map { case (row, copies) => (row, copies.length) }
        assert(counts(cpu) == counts(df.collect()))
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuExpandAggregateExec]).nonEmpty == enabled)
      }, configuration(enabled))
    }
  }

  test("non-ANSI long sum preserves wrapping across batches and partitions") {
    def query(spark: SparkSession): DataFrame = {
      val schema = StructType(Seq(StructField("k", IntegerType), StructField("v", LongType)))
      val rows = Seq(Row(1, Long.MaxValue), Row(1, 1L), Row(1, Long.MinValue),
        Row(2, Long.MaxValue), Row(2, Long.MaxValue), Row(2, null))
      spark.createDataFrame(spark.sparkContext.parallelize(rows, 3), schema)
        .createOrReplaceTempView("fusion_overflow")
      spark.sql("SELECT k,count(v),sum(v) FROM fusion_overflow GROUP BY ROLLUP(k)")
    }
    val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuExpandAggregateExec]).nonEmpty == enabled)
      }, configuration(enabled).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024"))
    }
  }

  test("empty grouping-set input does not manufacture a grand total partial row") {
    def query(spark: SparkSession): DataFrame = {
      data(spark, empty = true).createOrReplaceTempView("fusion_empty")
      spark.sql("SELECT a,grouping_id(),count(*),sum(v) FROM fusion_empty GROUP BY ROLLUP(a)")
    }
    val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
    assert(cpu.isEmpty)
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
      }, configuration(enabled))
    }
  }

  for (filterFusion <- Seq(false, true); aqe <- Seq(false, true)) {
    test(s"fusion flags are independent across adaptive stages: filter=$filterFusion AQE=$aqe") {
      def query(spark: SparkSession): DataFrame = {
        data(spark).filter("a > 0").selectExpr("a", "b", "v", "a + 1 AS next")
          .createOrReplaceTempView("fusion_combined")
        spark.sql("SELECT next,b,count(v),sum(v) FROM fusion_combined GROUP BY CUBE(next,b)")
      }
      val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false, false, aqe))
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuExpandAggregateExec]).nonEmpty, df.queryExecution.executedPlan.toString)
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuFilterProjectExec]).nonEmpty == filterFusion)
      }, configuration(true, filterFusion, aqe))
    }
  }

  for (aggregate <- Seq("avg(v)", "count(distinct v)", "sum(v + 1)",
      "sum(v) FILTER (WHERE v > 0)")) {
    test(s"unsupported aggregation retains the original plan: $aggregate") {
      withGpuSparkSession({ spark =>
        data(spark).createOrReplaceTempView("fusion_fallback")
        val df = spark.sql(s"SELECT a,$aggregate FROM fusion_fallback GROUP BY ROLLUP(a)")
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuExpandAggregateExec]).isEmpty, df.queryExecution.executedPlan.toString)
      }, configuration(true))
    }
  }

  test("rewrite preserves output and excludes unsafe strategies and discriminator shapes") {
    val (agg, expand) = ExpandAggregateFusionSuite.fixture()
    val fused = GpuExpandAggregateExec.fuse(agg)
    assert(fused.isInstanceOf[GpuExpandAggregateExec])
    assert(fused.output == agg.output)
    assert(fused.outputPartitioning == agg.outputPartitioning)
    assert(fused.outputOrdering == agg.outputOrdering)
    assert(fused.children == Seq(expand.child))
    assert(GpuExpandAggregateExec.fuse(agg.copy(allowSinglePassAgg = true))
      .isInstanceOf[GpuHashAggregateExec])
    assert(!GpuExpandAggregateExec.eligible(agg.copy(forceSinglePassAgg = true), expand))
    assert(!GpuExpandAggregateExec.eligible(agg.copy(aggregateExpressions =
      agg.aggregateExpressions.map(_.copy(mode = Final))), expand))
    val duplicate = expand.copy(projections = Seq(expand.projections.head,
      expand.projections.head))(preprojectEnabled = false, coalesceAfter = true)
    assert(!GpuExpandAggregateExec.eligible(agg, duplicate))
    val noId = agg.copy(groupingExpressions = agg.groupingExpressions.take(1))
    assert(!GpuExpandAggregateExec.eligible(noId, expand))
  }

  test("fused input is coalesced to the target batch size") {
    // The unfused plan coalesces Expand's output before the partial aggregate. Fusion must
    // keep equivalent batching on its own input, otherwise the first pass reduces poorly and
    // pushes the work into the merge pass. See GpuExpandAggregateExec.childrenCoalesceGoal.
    val (agg, _) = ExpandAggregateFusionSuite.fixture()
    val fused = GpuExpandAggregateExec.fuse(agg).asInstanceOf[GpuExpandAggregateExec]
    assert(fused.childrenCoalesceGoal == Seq(TargetSize(agg.configuredTargetBatchSize)))
    withGpuSparkSession({ spark =>
      data(spark).createOrReplaceTempView("fusion_batching")
      val df = spark.sql("SELECT a,b,count(v) FROM fusion_batching GROUP BY ROLLUP(a,b)")
      df.collect()
      val nodes = PlanUtils.findOperators(df.queryExecution.executedPlan,
        _.isInstanceOf[GpuExpandAggregateExec])
      assert(nodes.nonEmpty, df.queryExecution.executedPlan.toString)
      val goal = TargetSize(new RapidsConf(spark.sessionState.conf).gpuTargetBatchSizeBytes)
      nodes.foreach { node =>
        // insertCoalesce honours the goal, then optimizeCoalesce drops the node again when
        // the child already provides that batching. Either outcome satisfies the contract.
        val input = node.children.head
        assert(input.isInstanceOf[GpuCoalesceBatches] ||
          CoalesceGoal.satisfies(GpuExec.outputBatching(input), goal),
          df.queryExecution.executedPlan.toString)
      }
    }, configuration(true))
  }

  test("fusion defaults off independently of filter/project fusion") {
    val conf = new RapidsConf(new SparkConf()
      .set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, "true"))
    assert(!conf.isExpandAggregateFusionEnabled)
    assert(conf.isFilterProjectFusionEnabled)
  }

  test("reuse requires a live source key in the same expanded output position") {
    val (agg, expand) = ExpandAggregateFusionSuite.fixture()
    val input = expand.child.output
    val first = expand.output(3)
    val second = AttributeReference("second", IntegerType)()
    val gid = expand.output.last
    val output = input ++ Seq(first, second, gid)
    val projections = Seq(
      input ++ Seq(input.head, GpuLiteral(null, IntegerType), GpuLiteral(0L, LongType)),
      input ++ Seq(GpuLiteral(null, IntegerType), input.head, GpuLiteral(1L, LongType)))
    val modified = expand.copy(projections = projections, output = output)(
      preprojectEnabled = false, coalesceAfter = true)
    val groups = Seq(first, second, gid)
    val aggregate = agg.copy(groupingExpressions = groups,
      resultExpressions = groups ++ agg.aggregateAttributes, child = modified)
    assert(GpuExpandAggregateExec.eligible(aggregate, modified))
    val bound = GpuExpandAggregateExec.bind(aggregate, projections, output, input,
      new org.apache.spark.sql.internal.SQLConf(), Map.empty)
    assert(!bound.head.cacheResult)
    assert(bound(1).merge.isEmpty)
  }
}
