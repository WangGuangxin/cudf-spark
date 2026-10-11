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
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.functions.col

class MixedJoinChainQuerySuite extends SparkQueryCompareTestSuite {
  private def config(sized: Boolean, aqe: Boolean): SparkConf = new SparkConf()
    .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.adaptive.autoBroadcastJoinThreshold", "-1")
    .set("spark.sql.shuffle.partitions", "2")
    .set(RapidsConf.USE_SHUFFLED_SYMMETRIC_HASH_JOIN.key, sized.toString)
    .set(RapidsConf.METRICS_LEVEL.key, "DEBUG")

  private val recipes = Seq(Seq("BROADCAST", "SHUFFLE_HASH", "BROADCAST"),
    Seq("SHUFFLE_HASH", "BROADCAST", "SHUFFLE_HASH"),
    Seq("MERGE", "SHUFFLE_HASH", "BROADCAST"),
    Seq("SHUFFLE_HASH", "MERGE", "SHUFFLE_HASH"),
    Seq("MERGE", "MERGE", "MERGE"),
    Seq("SHUFFLE_HASH", "SHUFFLE_HASH", "SHUFFLE_HASH"))

  private def query(spark: SparkSession, recipe: Seq[String], duplicates: Boolean = false,
      wideBuild: Boolean = false, exchangeAfter: Int = -1, buildLeft: Boolean = false,
      repeatedFact: Boolean = false): DataFrame = {
    val payload = (0 until 8).map(i => s"id * ${i + 2} AS p$i")
    val fact = spark.range(0, 200).repartition(2, col("id"))
      .selectExpr((Seq(if (repeatedFact) "id % 160 AS k" else "id AS k") ++ payload): _*)
    recipe.zipWithIndex.foldLeft(fact) { case (stream, (hint, i)) =>
      val key = s"d$i"
      val value = s"value$i"
      val raw = spark.range(0, if (duplicates && i == 0) 320 else 160)
        .selectExpr((if (duplicates && i == 0) s"id % 160 AS $key" else s"id AS $key"),
          (if (wideBuild) s"concat(cast(id AS STRING), repeat('x', 200)) AS $value"
           else s"id * ${i + 3} AS $value"))
      val build = if (hint == "BROADCAST") raw.hint(hint)
        else raw.repartition(2, col(key)).hint(hint)
      val joined = (if (buildLeft) build.join(stream, build(key) === stream("k"))
        else stream.join(build, stream("k") === build(key)))
        .select((stream.columns.map(stream(_)) :+ build(value)): _*)
      if (i == exchangeAfter) joined.repartition(2, col(value)) else joined
    }
  }

  test("CPU plans exercise broadcast, shuffled hash and sort merge joins") {
    withCpuSparkSession(spark => {
      recipes.foreach { recipe =>
        val df = query(spark, recipe)
        df.collect()
        val joins = PlanUtils.findOperators(df.queryExecution.executedPlan, p =>
          p.isInstanceOf[BroadcastHashJoinExec] || p.isInstanceOf[ShuffledHashJoinExec] ||
            p.isInstanceOf[SortMergeJoinExec])
        val names = joins.map {
          case _: BroadcastHashJoinExec => "BROADCAST"
          case _: ShuffledHashJoinExec => "SHUFFLE_HASH"
          case _: SortMergeJoinExec => "MERGE"
        }
        assert(names.sorted == recipe.sorted)
      }
    }, config(sized = false, aqe = false))
  }

  for (sized <- Seq(false, true); aqe <- Seq(false, true); recipe <- recipes) {
    val label = recipe.mkString(" -> ")
    testSparkResultsAreEqual(s"mixed join chain $label sized=$sized AQE=$aqe",
      spark => query(spark, recipe), config(sized, aqe), sort = true)(identity)
  }

  for (sized <- Seq(false, true)) {
    testSparkResultsAreEqual(s"mixed join build-left chain sized=$sized",
      spark => query(spark, recipes(3), buildLeft = true, repeatedFact = true),
      config(sized, aqe = false), sort = true)(identity)

    testSparkResultsAreEqual(s"mixed join duplicate-key fallback sized=$sized",
      spark => query(spark, recipes(3), duplicates = true),
      config(sized, aqe = false), sort = true)(identity)

    testSparkResultsAreEqual(s"mixed join oversized-build fallback sized=$sized",
      spark => query(spark, recipes(1), wideBuild = true),
      config(sized, aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024"),
      sort = true)(identity)

    testSparkResultsAreEqual(s"mixed join build-left duplicate fallback sized=$sized",
      spark => query(spark, recipes(3), duplicates = true, buildLeft = true,
        repeatedFact = true), config(sized, aqe = false), sort = true)(identity)

    testSparkResultsAreEqual(s"mixed join build-left oversized fallback sized=$sized",
      spark => query(spark, recipes(3), wideBuild = true, buildLeft = true,
        repeatedFact = true),
      config(sized, aqe = false).set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024"),
      sort = true)(identity)

    testSparkResultsAreEqual(s"mixed join chain disabled sized=$sized",
      spark => query(spark, recipes(0)),
      config(sized, aqe = false).set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false"),
      sort = true)(identity)

    test(s"join chain can be turned off for A/B measurement sized=$sized") {
      // Sum the three runtime counters rather than asserting on `fused` alone: the point of the
      // switch is that none of the chain path runs when it is off, independently of whether a
      // given build would have passed the runtime uniqueness and sizing checks.
      def chainTaskCount(plan: org.apache.spark.sql.execution.SparkPlan): Long =
        PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
          .collect { case j: GpuJoinExec => j }
          .map { j =>
            j.gpuLongMetric(GpuMetric.JOIN_CHAIN_FUSED_TASKS).value +
              j.gpuLongMetric(GpuMetric.JOIN_CHAIN_FALLBACK_OVERSIZED).value +
              j.gpuLongMetric(GpuMetric.JOIN_CHAIN_FALLBACK_NON_UNIQUE).value
          }.sum

      withGpuSparkSession(spark => {
        val df = query(spark, recipes(0))
        df.collect()
        assert(chainTaskCount(df.queryExecution.executedPlan) > 0)
      }, config(sized, aqe = false))

      withGpuSparkSession(spark => {
        val df = query(spark, recipes(0))
        df.collect()
        // The plan shape is unchanged and would still fuse; only execution is turned off, so an
        // A/B run can still report which plans the baseline gave up.
        val plan = df.queryExecution.executedPlan
        assert(PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
          .exists(GpuJoinChain.chainLength(_) == 3), plan.toString)
        assert(chainTaskCount(plan) == 0)
      }, config(sized, aqe = false).set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false"))
    }

    test(s"mixed plans fuse same-partition joins and stop at exchanges sized=$sized") {
      withGpuSparkSession(spark => {
        recipes.foreach { recipe =>
          val df = query(spark, recipe)
          df.collect()
          val joins = PlanUtils.findOperators(df.queryExecution.executedPlan,
            _.isInstanceOf[GpuJoinExec])
          assert(joins.exists(GpuJoinChain.chainLength(_) == 3), df.queryExecution.toString)
        }
        val left = query(spark, recipes(3), buildLeft = true, repeatedFact = true)
        left.collect()
        val leftJoins = PlanUtils.findOperators(left.queryExecution.executedPlan,
          _.isInstanceOf[GpuJoinExec])
        assert(leftJoins.exists(GpuJoinChain.chainLength(_) == 3))
        if (sized) {
          val symmetric = leftJoins.collect { case j: GpuShuffledSymmetricHashJoinExec => j }
          assert(symmetric.nonEmpty)
          assert(symmetric.forall(_.gpuLongMetric(GpuMetric.NUM_OUTPUT_ROWS).value > 0))
          assert(symmetric.forall(_.gpuLongMetric(GpuMetric.SMALL_JOIN_COUNT).value == 0))
        }

        val shuffled = query(spark, recipes(3), exchangeAfter = 1)
        shuffled.collect()
        val root = PlanUtils.findOperators(shuffled.queryExecution.executedPlan,
          _.isInstanceOf[GpuJoinExec]).head
        assert(GpuJoinChain.chainLength(root) == 1)
      }, config(sized, aqe = false))
    }
  }
}
