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

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.rapids.execution.{GpuShuffleExchangeExecBase, TrampolineUtil}
import org.apache.spark.sql.types._

class PartialAggregateShuffleSuite extends SparkQueryCompareTestSuite {
  // Shuffle-manager selection is a SparkContext setting, so each variant needs a fresh context.
  private def runSession[T](gpu: Boolean, settings: SparkConf)(f: SparkSession => T): T = {
    TrampolineUtil.cleanupAnyExistingSession()
    val spark = SparkSession.builder().master("local[2]").appName(getClass.getSimpleName)
      .config(settings).config("spark.ui.enabled", "false")
      .config("spark.sql.session.timeZone", "UTC")
      .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
      .config("spark.rapids.sql.enabled", gpu.toString).getOrCreate()
    withResource(spark)(f)
  }

  private def cpuSession[T](f: SparkSession => T): T = runSession(false, new SparkConf())(f)

  private def gpuSession[T](f: SparkSession => T, settings: SparkConf): T =
    runSession(true, settings)(f)

  private def configuration(enabled: Boolean, aqe: Boolean = false,
      codec: String = "none", kudo: Boolean = false): SparkConf = new SparkConf()
    .set("spark.shuffle.manager", GpuShuffleEnv.RAPIDS_SHUFFLE_CLASS)
    .set("spark.rapids.shuffle.mode", "CACHE_ONLY")
    .set(RapidsConf.SHUFFLE_KUDO_SERIALIZER_ENABLED.key, kudo.toString)
    .set(RapidsConf.SHUFFLE_PARTIAL_AGG_PACK_ENABLED.key, enabled.toString)
    .set(RapidsConf.SHUFFLE_PARTITION_PACK_ENABLED.key, "false")
    .set(RapidsConf.ENABLE_SINGLE_PASS_PARTIAL_SORT_AGG.key, "false")
    .set(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, codec)
    .set("spark.sql.adaptive.enabled", aqe.toString)
    .set("spark.sql.ansi.enabled", "false")
    .set("spark.sql.shuffle.partitions", "7")
    .set("spark.rapids.sql.batchSizeBytes", "4096")

  private def query(spark: SparkSession, expression: String,
      empty: Boolean = false): DataFrame = {
    val schema = StructType(Seq(StructField("k", LongType), StructField("v", LongType)))
    val rows = (0 until (if (empty) 0 else 2003)).map { i =>
      Row(if (i % 11 == 0) null else Long.box(i % 103 - 51),
        if (i % 7 == 0 || i % 103 == 0) null
        else Long.box(if (i % 13 == 0) Long.MaxValue else i - 1000L))
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 3), schema)
      .createOrReplaceTempView("partial_pack_input")
    spark.sql(s"SELECT $expression FROM partial_pack_input GROUP BY k")
  }

  private def fused(df: DataFrame): Boolean = PlanUtils.findOperators(
    df.queryExecution.executedPlan, {
      case exchange: GpuShuffleExchangeExecBase => exchange.isPartialAggregatePackFused
      case _ => false
    }).nonEmpty

  for (aqe <- Seq(false, true); codec <- Seq("none", "copy", "lz4", "zstd")) {
    test(s"CPU/off/on merge across batches with AQE=$aqe codec=$codec") {
      val expressions = "k,count(*),count(v),sum(v),min(v),max(v)"
      val cpu = cpuSession(s => query(s, expressions).collect())
      for (enabled <- Seq(false, true)) {
        gpuSession({ spark =>
          val df = query(spark, expressions)
          compareResults(true, 0.0, cpu, df.collect())
          assert(fused(df) == enabled, df.queryExecution.executedPlan.toString())
        }, configuration(enabled, aqe, codec))
      }
    }
  }

  test("empty grouping input has no partial shuffle rows") {
    gpuSession({ spark =>
      val df = query(spark, "k,count(*),sum(v)", empty = true)
      assert(df.collect().isEmpty)
      assert(fused(df))
    }, configuration(true))
  }

  for (expressions <- Seq("k,avg(v)", "k,count(DISTINCT v)",
      "k,sum(CAST(v AS DOUBLE))", "k,sum(v) FILTER (WHERE v > 0)")) {
    test(s"unsupported aggregate falls back: $expressions") {
      val cpu = cpuSession(s => query(s, expressions).collect())
      gpuSession({ spark =>
        val df = query(spark, expressions)
        // Floating aggregates retain the existing GPU path and its rounding differences.
        val epsilon = if (expressions.contains("avg") || expressions.contains("DOUBLE")) {
          1e-12
        } else 0.0
        compareResults(true, epsilon, cpu, df.collect())
        assert(!fused(df))
      }, configuration(true))
    }
  }

  test("GPU Kudo retains the existing aggregate and serialization path") {
    val settings = configuration(true, kudo = true)
      .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.SortShuffleManager")
      .set(RapidsConf.SHUFFLE_KUDO_WRITE_MODE.key, "GPU")
    val cpu = cpuSession(s => query(s, "k,count(*)").collect())
    gpuSession({ spark =>
      val df = query(spark, "k,count(*)")
      compareResults(true, 0.0, cpu, df.collect())
      assert(!fused(df))
    }, settings)
  }

  test("CPU shuffle retains the existing partial aggregate output") {
    val settings = configuration(true)
      .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.SortShuffleManager")
    val cpu = cpuSession(s => query(s, "k,count(*),sum(v)").collect())
    gpuSession({ spark =>
      val df = query(spark, "k,count(*),sum(v)")
      compareResults(true, 0.0, cpu, df.collect())
      assert(!fused(df))
    }, settings)
  }

  test("ANSI count retains overflow checking in the existing aggregate") {
    gpuSession({ spark =>
      val df = query(spark, "k,count(v)")
      df.collect()
      assert(!fused(df))
    }, configuration(true).set("spark.sql.ansi.enabled", "true"))
  }
}
