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

import com.nvidia.spark.rapids._

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.{broadcast, col}

class NonUniqueJoinChainQuerySuite extends SparkQueryCompareTestSuite {
  private def config(aqe: Boolean, enabled: Boolean, reuse: Boolean): SparkConf = new SparkConf()
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set(RapidsConf.ENABLE_JOIN_CHAIN.key, enabled.toString)
    .set(RapidsConf.JOIN_HASH_REUSE_ENABLED.key, reuse.toString)
    .set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024")
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  // Every stage multiplies rows, carries distinct payloads for duplicate keys, and the next
  // stage probes a key from the preceding dimension across a reordered reference projection.
  private def query(spark: SparkSession, buildLeft: Boolean,
      keyType: String = "BIGINT"): DataFrame = {
    val fact = spark.range(0, 40).selectExpr(s"CAST(id % 5 AS $keyType) AS k", "id AS payload")
    val d1 = spark.range(0, 10).selectExpr(s"CAST(id % 5 AS $keyType) AS k1",
      "id AS v1", s"CAST(id % 3 AS $keyType) AS next1")
    val d2 = spark.range(0, 9).selectExpr(s"CAST(id % 3 AS $keyType) AS k2",
      s"CAST(id % 2 AS $keyType) AS next2",
      "concat('dim', CAST(id AS STRING)) AS v2")
    val d3 = spark.range(0, 4).selectExpr(s"CAST(id % 2 AS $keyType) AS k3",
      "array(id, id + 1) AS v3")
    val first = (if (buildLeft) broadcast(d1).join(fact, col("k") === col("k1"))
      else fact.join(broadcast(d1), col("k") === col("k1")))
      .select("next1", "v1", "payload")
    val second = (if (buildLeft) broadcast(d2).join(first, col("next1") === col("k2"))
      else first.join(broadcast(d2), col("next1") === col("k2")))
      .select("v2", "payload", "next2", "v1")
    if (buildLeft) broadcast(d3).join(second, col("next2") === col("k3"))
    else second.join(broadcast(d3), col("next2") === col("k3"))
  }

  private def metric(plan: SparkPlan, name: String): Long =
    PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
      .map(_.asInstanceOf[GpuJoinExec].gpuLongMetric(name).value).sum

  private def counts(rows: Array[Row]): Map[Seq[Any], Int] =
    rows.toSeq.groupBy(_.toSeq).map { case (row, matches) => row -> matches.size }

  for (keyType <- Seq("STRING", "DECIMAL(18,0)", "DECIMAL(30,0)")) {
    test(s"non-unique $keyType keys retain complete payload matches") {
      val conf = config(aqe = false, enabled = true, reuse = true)
      val cpu = withCpuSparkSession(spark => query(spark, false, keyType).collect(), conf)
      val gpu = withGpuSparkSession(spark => {
        val df = query(spark, false, keyType)
        val rows = df.collect()
        assert(metric(df.queryExecution.executedPlan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) > 0)
        rows
      }, conf)
      assert(counts(cpu) == counts(gpu))
    }
  }

  for (aqe <- Seq(false, true); buildLeft <- Seq(false, true);
      enabled <- Seq(false, true)) {
    test(s"duplicate chain AQE=$aqe buildLeft=$buildLeft enabled=$enabled") {
      val conf = config(aqe, enabled, reuse = true)
      val cpu = withCpuSparkSession(spark => query(spark, buildLeft).collect(), conf)
      val gpu = withGpuSparkSession(spark => query(spark, buildLeft).collect(), conf)
      // Compare multisets directly: the shared sorting helper does not support array payloads.
      assert(counts(cpu) == counts(gpu))
    }
  }

  for (reuse <- Seq(false, true); buildLeft <- Seq(false, true)) {
    test(s"duplicate chains actually fuse buildLeft=$buildLeft hashReuse=$reuse") {
      withGpuSparkSession(spark => {
        val df = query(spark, buildLeft)
        assert(df.collect().length == 480)
        val plan = df.queryExecution.executedPlan
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) > 0, plan.toString)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) == 0)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED) == 0)
      }, config(aqe = false, enabled = true, reuse))
    }
  }

  private def excessiveFanout(spark: SparkSession): DataFrame = {
    val fact = spark.range(0, 10).selectExpr("id % 2 AS k", "id AS payload")
    (0 until 2).foldLeft(fact) { (stream, i) =>
      val build = spark.range(0, 16).selectExpr(s"id % 2 AS d$i", s"id AS v$i")
      stream.join(broadcast(build), stream("k") === build(s"d$i"))
    }
  }

  testSparkResultsAreEqual("excessive single-row fanout retains eager correctness",
    excessiveFanout, config(aqe = false, enabled = true, reuse = true),
    sort = true)(identity)

  test("excessive single-row fanout falls back before composing maps") {
    withGpuSparkSession(spark => {
      val df = excessiveFanout(spark)
      assert(df.collect().length == 640)
      val plan = df.queryExecution.executedPlan
      assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) == 0)
      assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) > 0, plan.toString)
      assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED) == 0)
    }, config(aqe = false, enabled = true, reuse = true))
  }
}
