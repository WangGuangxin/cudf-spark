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
import org.apache.spark.sql.functions.{broadcast, col}

class JoinChainFilterQuerySuite extends SparkQueryCompareTestSuite {
  private def config(aqe: Boolean, enabled: Boolean = true): SparkConf = new SparkConf()
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set(RapidsConf.ENABLE_JOIN_CHAIN.key, enabled.toString)
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  private def query(spark: SparkSession, duplicates: Boolean,
      rows: Long = 2048): DataFrame = {
    val fact = spark.range(0, rows, 1, 3).selectExpr("CAST(id % 64 AS INT) AS k1",
      "CAST(id % 32 AS INT) AS k2", "CAST(id % 100 AS INT) AS amount",
      "concat(CAST(id AS STRING), repeat('x', 256)) AS payload")
    val d1 = spark.range(0, if (duplicates) 128 else 64)
      .selectExpr("CAST(id % 64 AS INT) AS d1", "CAST(id % 7 AS INT) AS delta")
    val d2 = spark.range(0, 32)
      .selectExpr("CAST(id AS INT) AS d2", "CAST(50 + id % 5 AS INT) AS lim")
    fact.join(broadcast(d1), col("k1") === col("d1"))
      .join(broadcast(d2), col("k2") === col("d2"))
      .where(col("amount") + col("delta") > col("lim"))
  }

  for (aqe <- Seq(false, true); enabled <- Seq(false, true); duplicates <- Seq(false, true)) {
    testSparkResultsAreEqual(
      s"deferred join filter AQE=$aqe enabled=$enabled duplicates=$duplicates",
      spark => query(spark, duplicates), config(aqe, enabled), sort = true)(identity)
  }

  for (aqe <- Seq(false, true); duplicates <- Seq(false, true)) {
    test(s"residual inner filter avoids wide payload gathers AQE=$aqe duplicates=$duplicates") {
      withGpuSparkSession(spark => {
        val df = query(spark, duplicates)
        val rows = df.collect()
        val plan = df.queryExecution.executedPlan
        val filters = PlanUtils.findOperators(plan, p =>
          p.isInstanceOf[GpuJoinChainFilterExec] || p.isInstanceOf[GpuJoinChainPipelineExec])
          .map(_.asInstanceOf[GpuExec])
        assert(filters.nonEmpty, plan.toString)
        assert(filters.map(_.gpuLongMetric("fusedTasks").value).sum > 0, plan.toString)
        val candidates = filters.map(_.gpuLongMetric("candidateRows").value).sum
        val avoided = filters.map(_.gpuLongMetric("avoidedPayloadRows").value).sum
        assert(avoided > 0 && candidates - avoided == rows.length)
      }, config(aqe))
    }
  }

  testSparkResultsAreEqual("residual deferred filter runtime fanout fallback",
    spark => query(spark, duplicates = true, rows = 72),
    config(aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "32"), sort = true)(identity)

  test("runtime fallback retains the native residual join") {
    withGpuSparkSession(spark => {
      val df = query(spark, duplicates = true, rows = 72)
      assert(df.collect().nonEmpty)
      val plan = df.queryExecution.executedPlan
      val filters = PlanUtils.findOperators(plan, p =>
        p.isInstanceOf[GpuJoinChainFilterExec] || p.isInstanceOf[GpuJoinChainPipelineExec])
        .map(_.asInstanceOf[GpuExec])
      assert(filters.nonEmpty && filters.forall {
        case f: GpuJoinChainFilterExec => f.residualJoin
        case p: GpuJoinChainPipelineExec => p.residualCondition.isDefined
        case _ => false
      }, plan.toString)
      assert(filters.map(_.gpuLongMetric("fusedTasks").value).sum == 0)
      val joins = PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
        .map(_.asInstanceOf[GpuJoinExec])
      assert(joins.map(_.gpuLongMetric(GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE).value).sum > 0)
    }, config(aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "32"))
  }

}
