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

import ai.rapids.cudf.{ColumnVector, Cuda, GroupByAggregation, Rmm, RmmAllocationMode, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.shims.GpuHashPartitioning

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.ExprId
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.types.{DataType, IntegerType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/** Terminal merge + shuffle preparation; resident canonical buffers, no transport or first pass. */
object PartialAggregatePackBenchmark {
  def main(args: Array[String]): Unit = {
    require(args.length == 4, "Arguments: rows distinctKeys partitions iterations")
    val rows = args(0).toInt
    val distinct = args(1).toInt
    val partitions = args(2).toInt
    val iterations = args(3).toInt
    require(rows > 0 && distinct > 0 && distinct <= rows && partitions > 0 && iterations > 0)
    val spark = SparkSession.builder().master("local[1]").appName(getClass.getSimpleName)
      .config("spark.ui.enabled", "false").getOrCreate()
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, 1024L * 1024 * 1024)
    val conf = new SQLConf()
    conf.setConfString(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, "none")
    // The reference already uses partition+pack, so this measures the additional merge fusion.
    conf.setConfString(RapidsConf.SHUFFLE_PARTITION_PACK_ENABLED.key, "true")
    GpuShuffleEnv.init(new RapidsConf(conf))
    try {
      val hp = SQLConf.withExistingConf(conf) {
        new GpuHashPartitioning(Seq(GpuBoundReference(0, IntegerType, false)
          (ExprId(0), "key")), partitions) {
          override def usesGPUShuffle: Boolean = true
          override def usesKudoGPUSlicing: Boolean = false
        }
      }
      withResource(ColumnVector.fromInts(Array.tabulate(rows)(i => i % distinct): _*)) { keys =>
        withResource(ColumnVector.fromLongs(Array.tabulate(rows)(i => i.toLong): _*)) { values =>
          withResource(new Table(keys, values)) { table =>
            val types = Array[DataType](IntegerType, LongType)
            withResource(GpuColumnVector.from(table, types)) { input =>
              val times = scala.collection.mutable.Map(
                false -> Vector.empty[Double], true -> Vector.empty[Double])
              for (iteration <- -5 until iterations) {
                val order = if (iteration % 2 == 0) Seq(false, true) else Seq(true, false)
                order.foreach { on =>
                  Cuda.DEFAULT_STREAM.sync()
                  val before = Rmm.getTotalBytesAllocated
                  Rmm.resetScopedMaximumBytesAllocated(before)
                  val start = System.nanoTime()
                  val output = if (on) {
                    hp.partialAggregateAndPack(GpuColumnVector.incRefCounts(input), Array(0))
                  } else {
                    val merged = withResource(table.groupBy(0)
                        .aggregate(GroupByAggregation.sum().onColumn(1))) { result =>
                      GpuColumnVector.from(result, types)
                    }
                    hp.columnarEvalAny(merged).asInstanceOf[Array[(ColumnarBatch, Int)]]
                  }
                  withResource(output.map(_._1)) { _ =>
                    require(output.map(_._1.numRows().toLong).sum == distinct)
                  }
                  Cuda.DEFAULT_STREAM.sync()
                  val ms = (System.nanoTime() - start) / 1000000.0
                  if (iteration >= 0) {
                    times(on) = times(on) :+ ms
                    println(s"SAMPLE on=$on iteration=$iteration ms=$ms " +
                      s"peakDelta=${Rmm.getScopedMaximumBytesAllocated - before}")
                  }
                }
              }
              times.toSeq.sortBy(_._1).foreach { case (on, samples) =>
                println(s"RESULT rows=$rows distinct=$distinct partitions=$partitions " +
                  s"on=$on medianMs=${samples.sorted.apply(samples.size / 2)}")
              }
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
