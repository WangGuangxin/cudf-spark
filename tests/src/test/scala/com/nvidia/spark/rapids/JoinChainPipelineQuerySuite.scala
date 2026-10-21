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
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{avg, broadcast, col, count, lit, max, min, sum}

class JoinChainPipelineQuerySuite extends SparkQueryCompareTestSuite {
  private def config(aqe: Boolean, enabled: Boolean = true): SparkConf = new SparkConf()
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.ansi.enabled", "false")
    .set("spark.rapids.sql.projectAstEnabled", "false")
    .set("spark.rapids.sql.projectPartialAstEnabled", "false")
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.shuffle.partitions", "3")
    .set(RapidsConf.ENABLE_JOIN_CHAIN.key, enabled.toString)
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  private def joined(spark: SparkSession, empty: Boolean,
      duplicates: Boolean = true): DataFrame = {
    val fact = spark.range(0, if (empty) 0 else 192, 1, 3).selectExpr(
      "CAST(id % 48 AS INT) AS k1", "CAST(id % 24 AS INT) AS k2",
      "CASE WHEN id % 11 = 0 THEN NULL ELSE CAST(id % 90 AS INT) END AS amount",
      "concat(CAST(id AS STRING), repeat('x',256)) AS payload")
    val d1 = spark.range(0, if (duplicates) 96 else 48).selectExpr(
      "CAST(id % 48 AS INT) AS d1", "CAST(id % 7 AS INT) AS delta")
    val d2 = spark.range(0, 24).selectExpr("CAST(id AS INT) AS d2",
      "CAST(5 + id % 5 AS INT) AS lim")
    fact.join(broadcast(d1), col("k1") === col("d1"))
      .join(broadcast(d2), col("k2") === col("d2"))
      .where(col("amount") + col("delta") > col("lim"))
  }

  private def query(spark: SparkSession, mode: String, empty: Boolean = false): DataFrame = {
    val input = joined(spark, empty)
    val projected = input.select(col("d1").as("g"),
      (col("amount") * col("delta")).as("measure"), col("payload"))
    mode match {
      case "project" => projected.select("measure", "g", "measure")
      case "literal" => input.select(lit(7).as("constant"))
      case "count" => input.agg(count(lit(1)).as("rows"))
      case "decimal" => projected.groupBy("g")
        .agg(sum(col("measure").cast("decimal(14,2)")),
          avg(col("measure").cast("decimal(14,2)")))
      case _ => projected.groupBy("g").agg(sum("measure"), count("measure"),
        min("measure"), max("measure"), avg("measure"))
    }
  }

  for (aqe <- Seq(false, true); empty <- Seq(false, true);
      mode <- Seq("project", "literal", "count", "aggregate", "decimal")) {
    testSparkResultsAreEqual(s"deferred pipeline $mode AQE=$aqe empty=$empty",
      spark => query(spark, mode, empty), config(aqe), sort = true,
      maxFloatDiff = 1e-9)(identity)
  }

  testSparkResultsAreEqual("deferred pipeline native residual fallback with projected aggregate",
    spark => query(spark, "aggregate"),
    config(aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "32"), sort = true,
    maxFloatDiff = 1e-9)(identity)

  for (enabled <- Seq(false, true); aqe <- Seq(false, true)) {
    test(s"partial aggregate pipeline executes and prunes payloads enabled=$enabled AQE=$aqe") {
      withGpuSparkSession(spark => {
        val df = query(spark, "aggregate")
        assert(df.collect().nonEmpty)
        val plan = df.queryExecution.executedPlan
        val aggregates = PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinChainAggregateExec])
        val pipelines = PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinChainPipelineExec])
          .map(_.asInstanceOf[GpuJoinChainPipelineExec])
        if (enabled) {
          assert(aggregates.nonEmpty && pipelines.nonEmpty, plan.toString)
          assert(pipelines.map(_.gpuLongMetric("fusedTasks").value).sum > 0)
          assert(pipelines.map(_.gpuLongMetric("payloadColumnsAvoided").value).sum > 0)
          assert(pipelines.forall(!_.output.exists(_.name == "payload")))
        } else {
          assert(aggregates.isEmpty && pipelines.isEmpty, plan.toString)
        }
      }, config(aqe, enabled))
    }
  }

  private def fullStage(spark: SparkSession, semi: Boolean): DataFrame = {
    val fact = spark.range(0, 192, 1, 3).selectExpr("CAST(id % 48 AS INT) AS k",
      "CASE WHEN id % 11 = 0 THEN NULL ELSE CAST(id % 90 AS INT) END AS amount",
      "concat(CAST(id AS STRING), repeat('x', 256)) AS payload")
      .where(col("amount") > 5)
      .select(col("k"), (col("amount") * 2).as("twice"), col("payload"))
    val d1 = spark.range(0, 96).selectExpr("CAST(id % 48 AS INT) AS d1",
      "CAST(id % 7 AS INT) AS delta")
    val first = fact.join(broadcast(d1), col("k") === col("d1"))
      .where(col("twice") > col("delta"))
      .select((col("k") * 2).as("next_key"),
        (col("twice") * col("delta")).as("measure"), col("payload"))
    val d2 = spark.range(0, 24).selectExpr("CAST(id * 2 AS INT) AS d2")
    first.join(broadcast(d2), col("next_key") === col("d2"),
      if (semi) "left_semi" else "inner")
      .where(col("measure") > 10)
      .select(col("next_key").as("g"), (col("measure") * 2).as("m"))
      .groupBy("g").agg(sum("m"), count("m"), min("m"), max("m"))
  }

  for (aqe <- Seq(false, true); semi <- Seq(false, true); enabled <- Seq(false, true)) {
    testSparkResultsAreEqual(s"source-to-shuffle pipeline AQE=$aqe semi=$semi enabled=$enabled",
      spark => fullStage(spark, semi), config(aqe, enabled), sort = true)(identity)
  }

  testSparkResultsAreEqual("source-to-shuffle pipeline preserves native fanout fallback",
    spark => fullStage(spark, semi = false),
    config(aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "32"), sort = true)(identity)

  // Decimal128 forces this predicate above the native join. The coalesce between Filter
  // and Project must not hide the pending predicate from the terminal pipeline.
  private def pendingTail(spark: SparkSession, countOnly: Boolean,
      duplicates: Boolean = false): DataFrame = {
    val thresholds = if (duplicates) "(1,100L), (2,150L), (2,150L)"
      else "(1,100L), (2,150L)"
    val selected = spark.sql(s"""
      WITH totals AS (
        SELECT /*+ REPARTITION(10, state) */ customer, state, SUM(amount) AS total
        FROM VALUES (1,1,10L), (1,1,20L), (2,2,200L) AS f(customer,state,amount)
        GROUP BY customer,state
      ), thresholds AS (
        SELECT state,threshold FROM VALUES $thresholds AS t(state,threshold)
      )
      SELECT /*+ BROADCAST(thresholds) */ totals.customer, totals.total
      FROM totals JOIN thresholds ON totals.state=thresholds.state
      WHERE CAST(totals.total AS DECIMAL(24,7)) >
            CAST(thresholds.threshold AS DECIMAL(24,7))
    """)
    if (countOnly) selected.agg(count(lit(1)).as("rows"))
    else selected.orderBy("customer").limit(100)
  }

  for (aqe <- Seq(false, true); countOnly <- Seq(false, true); tiny <- Seq(false, true)) {
    testSparkResultsAreEqual(
      s"pending tail filter survives terminal projection AQE=$aqe count=$countOnly tiny=$tiny",
      spark => pendingTail(spark, countOnly),
      config(aqe).set("spark.sql.adaptive.autoBroadcastJoinThreshold", "67108864")
        .set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, if (tiny) "32" else "268435456"),
      sort = true)(identity)
  }

  testSparkResultsAreEqualWithCapture("pending tail filter survives native fanout fallback",
    spark => pendingTail(spark, countOnly = false, duplicates = true),
    config(aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1"),
    sort = true)(identity) { (_, gpuPlan) =>
    val fallbackMetric = GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE
    val joins = PlanUtils.findOperators(gpuPlan, _.metrics.contains(fallbackMetric))
    assert(joins.map(_.metrics(fallbackMetric).value).sum > 0, gpuPlan.toString)
  }

  test("early downstream limit closes a deferred computed projection") {
    withGpuSparkSession(spark => {
      assert(query(spark, "project").limit(1).collect().length == 1)
    }, config(aqe = false))
  }
}
