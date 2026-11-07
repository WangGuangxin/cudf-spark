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

  test("scan split is half the batch size within bounds") {
    assert(StartupTuning.scanSplitBytes(1 * GiB) == 512 * MiB)
    assert(StartupTuning.scanSplitBytes(64 * MiB) == 128 * MiB)
    assert(StartupTuning.scanSplitBytes(8 * GiB) == 1 * GiB)
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
}
