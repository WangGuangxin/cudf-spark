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

class CpuMigrationSuite extends AnyFunSuite {
  private val MiB = 1024L * 1024
  private val GiB = 1024L * MiB

  private def migrated(settings: (String, String)*): Map[String, String] = {
    val sparkConf = new SparkConf(false).setAll(settings)
    val decisions = CpuMigration.decisions(sparkConf, new RapidsConf(sparkConf))
    assert(decisions.forall(_.source == TuningDecisionLog.MigrationSource))
    decisions.map(d => d.knob -> d.newValue).toMap
  }

  private val split = StartupTuning.MaxPartitionBytesKey
  private val partitions = CpuMigration.ShufflePartitionsKey
  private val initial = CpuMigration.InitialPartitionNumKey

  test("nothing is migrated when nothing is set") {
    assert(migrated().isEmpty)
  }

  test("scan split is at least the GPU split and at most the largest split") {
    assert(migrated(split -> "128m") == Map(split -> (1 * GiB).toString))
    assert(migrated(split -> "1g").isEmpty)
    assert(migrated(split -> "1536m").isEmpty)
    assert(migrated(split -> "4g") == Map(split -> (2 * GiB).toString))
    assert(migrated(split -> "128m", RapidsConf.GPU_BATCH_SIZE_BYTES.key -> "256m") ==
      Map(split -> (256 * MiB).toString))
  }

  test("initial shuffle partitions are capped with AQE coalescing") {
    assert(migrated(partitions -> "1000") == Map(initial -> "200"))
    assert(migrated(partitions -> "4000", initial -> "300") == Map(initial -> "200"))
    assert(migrated(partitions -> "100", initial -> "1000") == Map(initial -> "200"))
    assert(migrated(partitions -> "1000", initial -> "100").isEmpty)
    assert(migrated(partitions -> "200").isEmpty)
    assert(migrated(partitions -> "1000", CpuMigration.AdaptiveEnabledKey -> "false").isEmpty)
    assert(migrated(partitions -> "1000",
      CpuMigration.CoalescePartitionsEnabledKey -> "false").isEmpty)
  }

  test("broadcast threshold is capped and a disabled threshold is kept") {
    val threshold = CpuMigration.BroadcastThresholdKey
    assert(migrated(threshold -> "64m") == Map(threshold -> (10 * MiB).toString))
    assert(migrated(threshold -> "10m").isEmpty)
    assert(migrated(threshold -> "1m").isEmpty)
    assert(migrated(threshold -> "-1").isEmpty)
  }
}
