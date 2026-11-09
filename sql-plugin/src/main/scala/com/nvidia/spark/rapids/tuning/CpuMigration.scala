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

import com.nvidia.spark.rapids.{RapidsConf, SparkRapidsTuningDecisionEvent}

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Translates explicitly set Spark settings, assumed to be tuned for running on the CPU, into
 * settings that suit the GPU. Settings that suit both are kept as they are.
 */
object CpuMigration {
  val ShufflePartitionsKey = "spark.sql.shuffle.partitions"
  val AdaptiveEnabledKey = "spark.sql.adaptive.enabled"
  val CoalescePartitionsEnabledKey = "spark.sql.adaptive.coalescePartitions.enabled"
  val InitialPartitionNumKey = "spark.sql.adaptive.coalescePartitions.initialPartitionNum"

  // CPU configurations size the number of shuffle partitions for the number of cores, which
  // is far larger than the number of tasks a GPU runs at once. AQE only coalesces the
  // partitions on the read side, so the map side still pays for every partition. On NDS and
  // NDS-H at 1 and 3 TB, 200 partitions, Spark's default, were faster than the 500 to 4000
  // used by tuned CPU configurations, which took up to twice as long.
  private val MaxInitialShufflePartitions = 200

  val BroadcastThresholdKey = "spark.sql.autoBroadcastJoinThreshold"
  // Larger thresholds, common in CPU configurations, make broadcast joins of tables that the
  // GPU joins faster after a shuffle. On NDS at 1 and 3 TB, 64 MiB made several queries 2 to
  // 3 times slower than Spark's default of 10 MiB. A negative threshold disables broadcast
  // joins and is kept.
  private val MaxBroadcastThresholdBytes = 10L * 1024 * 1024

  /** Settings to change and why. Pure, so that it can be tested without a SparkContext. */
  def decisions(sparkConf: SparkConf, conf: RapidsConf): Seq[SparkRapidsTuningDecisionEvent] =
    scanSplit(sparkConf, conf).toSeq ++ initialShufflePartitions(sparkConf).toSeq ++
      broadcastThreshold(sparkConf).toSeq

  def apply(sc: SparkContext, sparkConf: SparkConf, conf: RapidsConf): Unit = {
    decisions(sparkConf, conf).foreach { d =>
      sparkConf.set(d.knob, d.newValue)
      TuningDecisionLog.record(sc, d)
    }
  }

  private def decision(knob: String, oldValue: String, newValue: String,
      reason: String): SparkRapidsTuningDecisionEvent =
    SparkRapidsTuningDecisionEvent(AdaptiveTuningListener.NoExecutionId, -1, knob, oldValue,
      newValue, reason, TuningDecisionLog.MigrationSource)

  /**
   * CPU scan splits are sized for the throughput of a core, which makes them too small to
   * amortize the fixed cost of a GPU task. The split derived for the GPU is a lower bound,
   * and a larger CPU split is kept up to the largest split the GPU is known to handle.
   */
  private def scanSplit(sparkConf: SparkConf,
      conf: RapidsConf): Option[SparkRapidsTuningDecisionEvent] = {
    val key = StartupTuning.MaxPartitionBytesKey
    sparkConf.getOption(key).flatMap { old =>
      val cpuSplit = sparkConf.getSizeAsBytes(key)
      val gpuSplit = StartupTuning.scanSplitBytes(conf.gpuTargetBatchSizeBytes)
      val split = math.min(math.max(cpuSplit, gpuSplit), StartupTuning.MaxScanSplitBytes)
      if (split != cpuSplit) {
        Some(decision(key, old, split.toString,
          s"CPU split $cpuSplit raised to at least the GPU split $gpuSplit, at most " +
            s"${StartupTuning.MaxScanSplitBytes}"))
      } else {
        None
      }
    }
  }

  /**
   * With AQE coalescing, the initial number of shuffle partitions, which is the explicit
   * initial partition number or else the number of shuffle partitions, is capped. The number
   * of shuffle partitions itself is kept for the plans that AQE does not handle.
   */
  private def initialShufflePartitions(
      sparkConf: SparkConf): Option[SparkRapidsTuningDecisionEvent] = {
    val coalescing = sparkConf.getBoolean(AdaptiveEnabledKey, true) &&
      sparkConf.getBoolean(CoalescePartitionsEnabledKey, true)
    val cpuKey = Seq(InitialPartitionNumKey, ShufflePartitionsKey).find(sparkConf.contains)
    cpuKey.filter(_ => coalescing).flatMap { key =>
      val cpuPartitions = sparkConf.getInt(key, MaxInitialShufflePartitions)
      if (cpuPartitions > MaxInitialShufflePartitions) {
        val old = sparkConf.getOption(InitialPartitionNumKey).getOrElse("unset")
        Some(decision(InitialPartitionNumKey, old, MaxInitialShufflePartitions.toString,
          s"CPU $key=$cpuPartitions capped at $MaxInitialShufflePartitions"))
      } else {
        None
      }
    }
  }

  private def broadcastThreshold(sparkConf: SparkConf): Option[SparkRapidsTuningDecisionEvent] =
    sparkConf.getOption(BroadcastThresholdKey).filterNot(_.trim.startsWith("-")).flatMap { old =>
      val cpuThreshold = sparkConf.getSizeAsBytes(BroadcastThresholdKey)
      if (cpuThreshold > MaxBroadcastThresholdBytes) {
        Some(decision(BroadcastThresholdKey, old, MaxBroadcastThresholdBytes.toString,
          s"CPU threshold $cpuThreshold capped at $MaxBroadcastThresholdBytes"))
      } else {
        None
      }
    }
}
