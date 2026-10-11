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

import java.io.File

import ai.rapids.cudf.{ColumnVector, Cuda, Rmm, RmmAllocationMode, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.shims.GpuHashPartitioning

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.ExprId
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Resident input, same packed-table endpoint. Includes hashing, partitioning, packing and close.
 */
object PartitionPackBenchmark {
  def main(args: Array[String]): Unit = {
    require(args.length >= 5 && args.length <= 7,
      "Arguments: rows partitions stringBytes payloadLongColumns iterations [rmmCsv] [OFF|ON|AB]")
    val rows = args(0).toInt
    val partitions = args(1).toInt
    val stringBytes = args(2).toInt
    val payloadColumns = args(3).toInt
    val iterations = args(4).toInt
    val mode = if (args.length == 7) args(6) else "AB"
    require(rows > 0 && partitions > 0 && stringBytes >= 0 && payloadColumns > 0 &&
      iterations > 0 && Set("OFF", "ON", "AB").contains(mode))
    val spark = SparkSession.builder().master("local[1]").appName(getClass.getSimpleName)
      .config("spark.ui.enabled", "false").getOrCreate()
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC,
      if (args.length >= 6 && args(5) != "-") Rmm.logTo(new File(args(5))) else null,
      1024L * 1024 * 1024)
    val conf = new SQLConf()
    conf.setConfString(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, "none")
    GpuShuffleEnv.init(new RapidsConf(conf))
    try {
      val keyType = if (stringBytes == 0) IntegerType else StringType
      val types = Array[DataType](keyType) ++ Array.fill[DataType](payloadColumns)(LongType)
      val columns = (0 to payloadColumns).toArray.safeMap { c =>
        if (c == 0 && stringBytes > 0) {
          ColumnVector.fromStrings(Array.tabulate(rows) { i =>
            val prefix = i.toString
            prefix + "x" * math.max(0, stringBytes - prefix.length)
          }: _*)
        } else if (c == 0) {
          ColumnVector.fromInts(Array.tabulate(rows)(i => i): _*)
        } else {
          ColumnVector.fromLongs(Array.tabulate(rows)(i => i.toLong + c): _*)
        }
      }
      withResource(columns) { cols =>
        withResource(new Table(cols: _*)) { table =>
          withResource(GpuColumnVector.from(table, types)) { input =>
            val variants = Seq(false, true).filter(on => mode == "AB" || on == (mode == "ON"))
              .map { enabled =>
                conf.setConfString(RapidsConf.SHUFFLE_PARTITION_PACK_ENABLED.key, enabled.toString)
                val hp = SQLConf.withExistingConf(conf) {
                  new GpuHashPartitioning(
                    Seq(GpuBoundReference(0, keyType, false)(ExprId(0), "key")), partitions) {
                    override def usesGPUShuffle: Boolean = true
                    override def usesKudoGPUSlicing: Boolean = false
                  }
                }
                (enabled, hp)
              }
            val times = scala.collection.mutable.Map[Boolean, Vector[Double]]()
            variants.foreach { case (enabled, _) => times(enabled) = Vector.empty }
            for (iteration <- -5 until iterations) {
              if (iteration == 0) Cuda.profilerStart()
              // Alternate ordering to limit drift. Warmup and source construction are unmeasured.
              val order = if (iteration % 2 == 0) variants else variants.reverse
              order.foreach { case (enabled, hp) =>
                Cuda.DEFAULT_STREAM.sync()
                Rmm.resetScopedMaximumBytesAllocated(Rmm.getTotalBytesAllocated)
                val before = Rmm.getTotalBytesAllocated
                val start = System.nanoTime()
                val output = hp.columnarEvalAny(GpuColumnVector.incRefCounts(input))
                  .asInstanceOf[Array[(ColumnarBatch, Int)]]
                withResource(output.map(_._1)) { _ =>
                  require(output.map(_._1.numRows().toLong).sum == rows)
                }
                Cuda.DEFAULT_STREAM.sync()
                val millis = (System.nanoTime() - start) / 1000000.0
                if (iteration >= 0) {
                  times(enabled) = times(enabled) :+ millis
                  println(s"SAMPLE on=$enabled iteration=$iteration ms=$millis " +
                    s"peakDelta=${Rmm.getScopedMaximumBytesAllocated - before}")
                }
              }
            }
            Cuda.profilerStop()
            times.toSeq.sortBy(_._1).foreach { case (enabled, samples) =>
              val sorted = samples.sorted
              println(s"RESULT rows=$rows partitions=$partitions stringBytes=$stringBytes " +
                s"payloadLongColumns=$payloadColumns on=$enabled medianMs=" +
                s"${sorted(sorted.length / 2)}")
            }
          }
        }
      }
    } finally {
      GpuShuffleEnv.shutdown()
      Rmm.shutdown()
      spark.stop()
    }
  }
}
