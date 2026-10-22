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
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class GenerateFilterFusionSuite extends SparkQueryCompareTestSuite {

  private def configuration(enabled: Boolean, batchBytes: Long, minBytes: Int): SparkConf = {
    val conf = new SparkConf()
      .set(RapidsConf.ENABLE_GENERATE_FILTER_FUSION.key, enabled.toString)
      .set(RapidsConf.GENERATE_FILTER_FUSION_MIN_BYTES.key, minBytes.toString)
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.ansi.enabled", "false")
    if (batchBytes == Long.MaxValue) conf
    else conf.set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, batchBytes.toString)
  }

  /** Arrays that are long, short, empty, null, and carry null elements. */
  private def data(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("id", IntegerType),
      StructField("payload", StringType),
      StructField("weight", LongType),
      StructField("arr", ArrayType(IntegerType))))
    val rows = (0 until 200).map { i =>
      val arr = i % 7 match {
        case 0 => null
        case 1 => Seq.empty[Integer]
        case 2 => Seq[Integer](null, Int.box(i % 13))
        case _ => (0 until (i % 11)).map(j => Int.box((i + j) % 13))
      }
      Row(Int.box(i), if (i % 17 == 0) null else s"payload-value-$i",
        Long.box(i * 7L), arr)
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 3), schema)
  }

  private def fusedNodes(plan: SparkPlan) =
    PlanUtils.findOperators(plan, _.isInstanceOf[GpuGenerateFilterExec])

  private def compareOffOn(name: String, expectFused: Boolean,
      batchBytes: Long = Long.MaxValue, minBytes: Int = 16)(
      query: SparkSession => DataFrame): Unit = {
    val cpu = withCpuSparkSession(s => query(s).collect(),
      configuration(enabled = false, batchBytes, minBytes))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
        val plan = df.queryExecution.executedPlan
        val fused = fusedNodes(plan)
        assert(fused.nonEmpty == (enabled && expectFused),
          s"$name: fused=${fused.size} enabled=$enabled\n$plan")
      }, configuration(enabled, batchBytes, minBytes))
    }
  }

  test("explode with a predicate on the exploded value") {
    compareOffOn("basic", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x") > 5)
    }
  }

  test("explode across many output batches") {
    compareOffOn("batched", expectFused = true, batchBytes = 2048) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x") > 5)
    }
  }

  test("a predicate that keeps nothing") {
    compareOffOn("empty", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x") > 10000)
    }
  }

  test("a predicate that keeps everything") {
    compareOffOn("all", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x") > -1 || col("x").isNull)
    }
  }

  test("a predicate on null exploded elements") {
    compareOffOn("nulls", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x").isNull)
    }
  }

  test("posexplode filtered on the value") {
    compareOffOn("pos-value", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"),
        posexplode(col("arr")).as(Seq("p", "x"))).filter(col("x") > 5)
    }
  }

  test("posexplode filtered on the position") {
    // Position is the index inside the array before filtering, so it has to be produced by the
    // explode and then filtered, never renumbered afterwards.
    compareOffOn("pos-index", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"),
        posexplode(col("arr")).as(Seq("p", "x"))).filter(col("p") === 2)
    }
  }

  for (predicate <- Seq("value", "null")) {
    test(s"explode_outer keeps empty and null arrays consistent, predicate=$predicate") {
      // outer produces one NULL row for an empty or null array. A value predicate must drop it
      // and an IS NULL predicate must keep it, exactly as the unfused pair does.
      compareOffOn(s"outer-$predicate", expectFused = true) { spark =>
        val exploded = data(spark).select(col("id"), col("payload"), col("weight"),
          explode_outer(col("arr")).as("x"))
        if (predicate == "value") exploded.filter(col("x") > 5)
        else exploded.filter(col("x").isNull)
      }
    }
  }

  test("posexplode_outer with a position predicate") {
    compareOffOn("pos-outer", expectFused = true) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"),
        posexplode_outer(col("arr")).as(Seq("p", "x"))).filter(col("p") === 0)
    }
  }

  test("a predicate reading a replicated column is not fused") {
    compareOffOn("mixed", expectFused = false) { spark =>
      data(spark).select(col("id"), col("payload"), col("weight"), explode(col("arr")).as("x"))
        .filter(col("x") > col("id"))
    }
  }

  test("narrow replicated columns are not fused") {
    // A single int is below the default byte threshold, where the index and gather cost more
    // than the replication they avoid.
    compareOffOn("narrow", expectFused = false, minBytes = 16) { spark =>
      data(spark).select(col("id"), explode(col("arr")).as("x")).filter(col("x") > 5)
    }
  }

  test("the same narrow case fuses once the threshold allows it") {
    compareOffOn("narrow-allowed", expectFused = true, minBytes = 1) { spark =>
      data(spark).select(col("id"), explode(col("arr")).as("x")).filter(col("x") > 5)
    }
  }

  test("explode of a map is not fused") {
    compareOffOn("map", expectFused = false) { spark =>
      // Map keys cannot be null, so build the map from the non-null id instead of arr.
      data(spark).select(col("id"), col("payload"), col("weight"),
        map_from_arrays(array(col("id"), col("id") + 1),
          array(col("weight"), col("weight") + 1)).as("m"))
        .select(col("id"), col("payload"), col("weight"), explode(col("m")).as(Seq("k", "v")))
        .filter(col("v") > 5)
    }
  }

  test("fusion is disabled by default") {
    assert(!new RapidsConf(new SparkConf()).isGenerateFilterFusionEnabled)
    assert(new RapidsConf(new SparkConf()).generateFilterFusionMinBytes == 16L)
  }
}
