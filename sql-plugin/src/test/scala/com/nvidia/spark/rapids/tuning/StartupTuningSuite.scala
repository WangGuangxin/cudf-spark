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

package com.nvidia.spark.rapids.tuning

import com.nvidia.spark.rapids.RapidsConf
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf

class StartupTuningSuite extends AnyFunSuite {
  private val MiB = 1024L * 1024
  private val GiB = 1024L * MiB

  test("scan split is the batch size within bounds") {
    assert(StartupTuning.scanSplitBytes(1 * GiB) == 1 * GiB)
    assert(StartupTuning.scanSplitBytes(64 * MiB) == 128 * MiB)
    assert(StartupTuning.scanSplitBytes(8 * GiB) == 2 * GiB)
  }

  test("min partition size is a fraction of the advisory size") {
    assert(StartupTuning.minPartitionSize(64 * MiB) == 4 * MiB)
    assert(StartupTuning.minPartitionSize(256 * MiB) == 16 * MiB)
    assert(StartupTuning.minPartitionSize(0L) == 1L)
  }

  test("pinned pool is a capped fraction of the off-heap budget") {
    assert(StartupTuning.pinnedPoolBytes(32 * GiB) == 4 * GiB)
    assert(StartupTuning.pinnedPoolBytes(512 * GiB) == 16 * GiB)
    assert(StartupTuning.pinnedPoolBytes(-1L) == 0L)
  }

  test("executor pinned pool keeps explicit and disabled settings") {
    def pinned(settings: (String, String)*): Long = {
      val sparkConf = new SparkConf(false).setAll(settings)
      StartupTuning.executorPinnedPoolBytes(new RapidsConf(sparkConf), sparkConf, 64 * GiB)
    }
    val enabled = RapidsConf.ADAPTIVE_TUNING_ENABLED.key -> "true"
    assert(pinned() == 0L)
    assert(pinned(enabled) == 8 * GiB)
    assert(pinned(enabled, RapidsConf.PINNED_POOL_SIZE.key -> "0") == 0L)
    assert(pinned(enabled, RapidsConf.PINNED_POOL_SIZE.key -> "2g") == 2 * GiB)
  }

  test("skip merge is derived only when its prerequisites hold") {
    def derives(settings: (String, String)*): Boolean = {
      val sparkConf = new SparkConf(false).setAll(settings)
      StartupTuning.shouldDeriveSkipMerge(sparkConf, new RapidsConf(sparkConf))
    }
    val mt = RapidsConf.SHUFFLE_MANAGER_MODE.key -> "MULTITHREADED"
    val offHeap = RapidsConf.OFF_HEAP_LIMIT_ENABLED.key -> "true"
    assert(derives(mt, offHeap))
    assert(!derives(mt))
    assert(!derives(mt, offHeap, "spark.shuffle.service.enabled" -> "true"))
    assert(!derives(mt, offHeap, RapidsConf.MULTITHREADED_SHUFFLE_SKIP_MERGE.key -> "false"))
    assert(!derives(RapidsConf.SHUFFLE_MANAGER_MODE.key -> "UCX", offHeap))
  }

  test("connections per peer are derived only when skip merge serves the blocks") {
    def derives(settings: (String, String)*): Boolean =
      StartupTuning.shouldDeriveConnectionsPerPeer(new SparkConf(false).setAll(settings),
        rapidsShuffleConfigured = true)
    val mt = RapidsConf.SHUFFLE_MANAGER_MODE.key -> "MULTITHREADED"
    val skipMerge = RapidsConf.MULTITHREADED_SHUFFLE_SKIP_MERGE.key -> "true"
    val offHeap = RapidsConf.OFF_HEAP_LIMIT_ENABLED.key -> "true"
    assert(derives(mt, skipMerge, offHeap))
    assert(!derives(mt, offHeap))
    assert(!derives(mt, skipMerge))
    assert(!derives(mt, skipMerge, offHeap, "spark.shuffle.service.enabled" -> "true"))
    assert(!derives(mt, skipMerge, offHeap, StartupTuning.NumConnectionsPerPeerKey -> "1"))
    assert(!derives(RapidsConf.SHUFFLE_MANAGER_MODE.key -> "UCX", skipMerge, offHeap))
    // Spark's own shuffle manager never uses the catalog
    assert(!StartupTuning.shouldDeriveConnectionsPerPeer(
      new SparkConf(false).setAll(Seq(mt, skipMerge, offHeap)), rapidsShuffleConfigured = false))
  }

  test("initial partitions are raised to the GPU task slots") {
    def floor(settings: (String, String)*): Option[Int] = {
      val sparkConf = new SparkConf(false).setAll(settings)
      StartupTuning.initialPartitionFloor(sparkConf, new RapidsConf(sparkConf))
    }
    val standalone = Seq("spark.master" -> "spark://h:7077", "spark.executor.instances" -> "8",
      RapidsConf.CONCURRENT_GPU_TASKS.key -> "4")
    val initial = CpuMigration.InitialPartitionNumKey
    assert(floor(standalone :+ (initial -> "10"): _*).contains(32))
    assert(floor(standalone :+ (initial -> "64"): _*).isEmpty)
    // Without an initial number the number of shuffle partitions is the initial number.
    assert(floor(standalone :+ (CpuMigration.ShufflePartitionsKey -> "16"): _*).contains(32))
    assert(floor(standalone: _*).isEmpty)
    // Not used without AQE coalescing.
    assert(floor(standalone ++ Seq(initial -> "10",
      CpuMigration.CoalescePartitionsEnabledKey -> "false"): _*).isEmpty)
    // Standalone without an executor count, two tasks per GPU by default.
    assert(floor("spark.master" -> "spark://h:7077", "spark.cores.max" -> "128",
      "spark.executor.cores" -> "16", initial -> "10").contains(16))
    // Two GPUs per executor; dynamic allocation uses the most executors.
    assert(floor("spark.master" -> "yarn", "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.maxExecutors" -> "5", "spark.executor.resource.gpu.amount" -> "2",
      RapidsConf.CONCURRENT_GPU_TASKS.key -> "3", initial -> "10").contains(30))
    assert(floor("spark.master" -> "local[8]", RapidsConf.CONCURRENT_GPU_TASKS.key -> "4",
      initial -> "1").contains(4))
    // Unknown number of executors.
    assert(floor("spark.master" -> "yarn", initial -> "10").isEmpty)
  }
}
