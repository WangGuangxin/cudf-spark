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
import org.apache.spark.sql.functions.col

class JoinHashReuseQuerySuite extends SparkQueryCompareTestSuite {
  private val joinTypes = Seq("inner", "left_outer", "right_outer", "full_outer",
    "left_semi", "left_anti")

  private def config(strategy: String): SparkConf = new SparkConf()
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.adaptive.enabled", "false")
    .set("spark.sql.shuffle.partitions", "1")
    .set(RapidsConf.USE_SHUFFLED_SYMMETRIC_HASH_JOIN.key, "false")
    .set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024")
    .set(RapidsConf.JOIN_STRATEGY.key, strategy)

  private def query(spark: SparkSession, joinType: String, conditional: Boolean): DataFrame = {
    val left = spark.range(0, 1000).selectExpr(
      "CASE WHEN id % 13 = 0 THEN NULL ELSE id % 80 END AS lk", "id AS lv")
      .repartition(1, col("lk"))
    val right = spark.range(0, 160).selectExpr(
      "CASE WHEN id % 17 = 0 THEN NULL ELSE id % 60 END AS rk", "id AS rv")
      .repartition(1, col("rk"))
    val equality = left("lk") === right("rk")
    val condition = if (conditional) equality && left("lv") < right("rv") else equality
    left.join(right, condition, joinType)
  }

  for (strategy <- Seq("HASH_ONLY", "INNER_HASH_WITH_POST");
      joinType <- joinTypes; conditional <- Seq(false, true)) {
    testSparkResultsAreEqual(
      s"reusable hash join $joinType strategy=$strategy conditional=$conditional",
      spark => query(spark, joinType, conditional), config(strategy), sort = true)(identity)
  }

  test("reusable hash query exercises GPU joins") {
    withGpuSparkSession(spark => {
      joinTypes.foreach { joinType =>
        val df = query(spark, joinType, conditional = false)
        df.collect()
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuJoinExec]).nonEmpty)
      }
    }, config("HASH_ONLY"))
  }
}
