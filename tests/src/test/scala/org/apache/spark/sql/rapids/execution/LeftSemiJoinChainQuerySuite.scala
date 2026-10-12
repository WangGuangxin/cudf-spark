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
import org.apache.spark.sql.catalyst.plans.LeftSemi
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.col

class LeftSemiJoinChainQuerySuite extends SparkQueryCompareTestSuite {
  private def config(aqe: Boolean, sized: Boolean): SparkConf = new SparkConf()
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.shuffle.partitions", "2")
    .set(RapidsConf.USE_SHUFFLED_SYMMETRIC_HASH_JOIN.key, sized.toString)
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  private val recipes = Seq(Seq("inner", "left_semi", "inner"),
    Seq("left_semi", "inner", "left_semi"), Seq("left_semi", "left_semi", "left_semi"))

  // Semi builds deliberately repeat keys with different payloads. Wide fact rows make mixed
  // shuffled chains worthwhile as well, while projections retain only the left-side columns.
  private def query(spark: SparkSession, recipe: Seq[String], hint: String,
      nullable: Boolean = false, semiBuildRows: Int = 180): DataFrame = {
    val key = if (nullable) "CASE WHEN id % 13 = 0 THEN NULL ELSE CAST(id % 60 AS INT) END"
      else "CAST(id % 60 AS INT)"
    val input = spark.range(0, 240).selectExpr(
      (Seq(s"$key AS k") ++ (0 until 8).map(i => s"id * ${i + 2} AS p$i")): _*)
    // Spark can push a semi join below repartition, placing an exchange inside the chain.
    // Broadcast fixtures need no repartition; keep all three probes in the same pipeline.
    val fact = if (hint == "BROADCAST") input else input.repartition(2, col("k"))
    recipe.zipWithIndex.foldLeft(fact) { case (stream, (kind, i)) =>
      val name = s"d$i"
      val buildKey = if (nullable && kind == "left_semi") key else "CAST(id % 60 AS INT)"
      val raw = spark.range(0, if (kind == "left_semi") semiBuildRows else 60)
        .selectExpr(s"$buildKey AS $name", s"id AS unused_payload$i")
      val build = (if (hint == "BROADCAST") raw else raw.repartition(2, col(name))).hint(hint)
      stream.join(build, stream("k") === build(name), kind)
        .select(stream.columns.map(stream(_)): _*)
    }
  }

  private def joins(plan: SparkPlan): Seq[GpuJoinExec] =
    PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
      .map(_.asInstanceOf[GpuJoinExec])

  private def metric(plan: SparkPlan, name: String): Long =
    joins(plan).map(_.gpuLongMetric(name).value).sum

  for (aqe <- Seq(false, true); recipe <- recipes; hint <- Seq("BROADCAST", "SHUFFLE_HASH")) {
    testSparkResultsAreEqual(s"semi chain ${recipe.mkString(" -> ")} hint=$hint AQE=$aqe",
      spark => query(spark, recipe, hint), config(aqe, sized = false), sort = true)(identity)
  }

  for (recipe <- recipes) {
    testSparkResultsAreEqual(s"semi chain NULLs ${recipe.mkString(" -> ")}",
      spark => query(spark, recipe, "BROADCAST", nullable = true),
      config(aqe = false, sized = false), sort = true)(identity)

    testSparkResultsAreEqual(s"semi chain empty build ${recipe.mkString(" -> ")}",
      spark => query(spark, recipe, "BROADCAST", semiBuildRows = 0),
      config(aqe = false, sized = false), sort = true)(identity)
  }

  for (recipe <- recipes; hint <- Seq("BROADCAST", "SHUFFLE_HASH", "MERGE")) {
    test(s"semi chain fuses repeated build keys ${recipe.mkString(" -> ")} hint=$hint") {
      withGpuSparkSession(spark => {
        val df = query(spark, recipe, hint)
        df.collect()
        val plan = df.queryExecution.executedPlan
        assert(joins(plan).exists(GpuJoinChain.chainLength(_) == recipe.length), plan.toString)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) > 0, plan.toString)
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) == 0)
        if (hint == "MERGE") {
          assert(joins(plan).exists {
            case j: GpuShuffledHashJoinExec => j.joinType == LeftSemi
            case _ => false
          }, plan.toString)
        }
      }, config(aqe = false, sized = hint == "MERGE"))
    }
  }

  testSparkResultsAreEqual("semi chain oversized shuffled build resumes eager path",
    spark => query(spark, recipes(1), "SHUFFLE_HASH", semiBuildRows = 10000),
    config(aqe = false, sized = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024"),
    sort = true)(identity)

  test("semi chain oversized builds and disabled execution preserve fallback counters") {
    for (enabled <- Seq(true, false)) {
      withGpuSparkSession(spark => {
        val df = query(spark, recipes(1), "SHUFFLE_HASH", semiBuildRows = 10000)
        df.collect()
        val plan = df.queryExecution.executedPlan
        assert(metric(plan, GpuMetric.JOIN_CHAIN_FUSED_TASKS) == 0)
        if (enabled) {
          assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED) > 0)
        } else {
          assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED) == 0)
          assert(metric(plan, GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE) == 0)
        }
      }, config(aqe = false, sized = false)
        .set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024")
        .set(RapidsConf.ENABLE_JOIN_CHAIN.key, enabled.toString))
    }
  }

  testSparkResultsAreEqual("semi chain can be disabled with duplicate build keys",
    spark => query(spark, recipes(1), "BROADCAST"),
    config(aqe = false, sized = false).set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false"),
    sort = true)(identity)
}
