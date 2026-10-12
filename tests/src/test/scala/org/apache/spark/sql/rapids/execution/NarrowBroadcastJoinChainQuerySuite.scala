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
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.{broadcast, col}

class NarrowBroadcastJoinChainQuerySuite extends SparkQueryCompareTestSuite {
  private def config(aqe: Boolean): SparkConf = new SparkConf()
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  // Only INT keys survive each reference projection. These two/three-stage star joins were
  // rejected by the schema-width model even though the dimension probes reduce the fact rows.
  private def star(spark: SparkSession, length: Int, duplicates: Boolean): DataFrame = {
    val fact = spark.range(0, 240).selectExpr(
      (0 until length).map(i => s"CAST(id % ${60 - 20 * i} AS INT) AS k$i"): _*)
    (0 until length).foldLeft(fact) { (stream, i) =>
      val key = s"d$i"
      val rows = 50 - 20 * i
      val build = spark.range(0, rows * (if (duplicates && i == 0) 2 else 1))
        .selectExpr(s"CAST(id % $rows AS INT) AS $key")
      stream.join(broadcast(build), stream(s"k$i") === build(key))
        .select(stream.columns.map(stream(_)): _*)
    }
  }

  private def lookup(spark: SparkSession): DataFrame = {
    val fact = spark.range(0, 240).selectExpr("CAST(id % 60 AS INT) AS k")
    val dim1 = spark.range(0, 40).selectExpr("CAST(id AS INT) AS d1",
      "CAST(id % 12 AS INT) AS next_key")
    val dim2 = spark.range(0, 8).selectExpr("CAST(id AS INT) AS d2")
    fact.join(broadcast(dim1), col("k") === col("d1"))
      .select("k", "next_key")
      .join(broadcast(dim2), col("next_key") === col("d2"))
  }

  private def joins(plan: SparkPlan): Seq[GpuJoinExec] =
    PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
      .map(_.asInstanceOf[GpuJoinExec])

  private def metric(plan: SparkPlan, name: String): Long =
    joins(plan).map(_.gpuLongMetric(name).value).sum

  for (aqe <- Seq(false, true); length <- Seq(2, 3); duplicates <- Seq(false, true)) {
    testSparkResultsAreEqual(
      s"narrow broadcast star length=$length AQE=$aqe duplicate keys=$duplicates",
      spark => star(spark, length, duplicates), config(aqe), sort = true)(identity)
  }

  for (aqe <- Seq(false, true)) {
    testSparkResultsAreEqual(s"narrow broadcast dimension-derived lookup AQE=$aqe",
      lookup, config(aqe), sort = true)(identity)
  }

  for (length <- Seq(2, 3)) {
    testSparkResultsAreEqual(s"narrow broadcast star disabled length=$length",
      spark => star(spark, length, duplicates = false),
      config(aqe = false).set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false"),
      sort = true)(identity)

    test(s"narrow broadcast star actually fuses and can be disabled length=$length") {
      for (enabled <- Seq(true, false)) {
        withGpuSparkSession(spark => {
          val df = star(spark, length, duplicates = false)
          df.collect()
          val plan = df.queryExecution.executedPlan
          assert(joins(plan).exists(GpuJoinChain.chainLength(_) == length), plan.toString)
          val fused = metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS)
          if (enabled) {
            assert(fused > 0, plan.toString)
          } else {
            assert(fused == 0)
            assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) == 0)
            assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED) == 0)
          }
        }, config(aqe = false).set(RapidsConf.ENABLE_JOIN_CHAIN.key, enabled.toString))
      }
    }

    test(s"narrow broadcast duplicate keys retain eager fallback length=$length") {
      withGpuSparkSession(spark => {
        val df = star(spark, length, duplicates = true)
        df.collect()
        val plan = df.queryExecution.executedPlan
        assert(joins(plan).exists(GpuJoinChain.chainLength(_) == length), plan.toString)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) == 0)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) > 0)
      }, config(aqe = false))
    }
  }

  test("narrow shuffled chains retain the traffic guard") {
    withGpuSparkSession(spark => {
      val fact = spark.range(0, 120).selectExpr("CAST(id AS INT) AS k")
        .repartition(2, col("k"))
      val df = (0 until 2).foldLeft(fact) { (stream, i) =>
        val key = s"d$i"
        val build = spark.range(0, 80 - 20 * i).selectExpr(s"CAST(id AS INT) AS $key")
          .repartition(2, col(key)).hint("SHUFFLE_HASH")
        stream.join(build, stream("k") === build(key)).select(stream("k"))
      }
      df.collect()
      val plan = df.queryExecution.executedPlan
      assert(joins(plan).nonEmpty, plan.toString)
      assert(joins(plan).forall(GpuJoinChain.chainLength(_) <= 1), plan.toString)
      assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) == 0)
    }, config(aqe = false).set(RapidsConf.USE_SHUFFLED_SYMMETRIC_HASH_JOIN.key, "false"))
  }
}
