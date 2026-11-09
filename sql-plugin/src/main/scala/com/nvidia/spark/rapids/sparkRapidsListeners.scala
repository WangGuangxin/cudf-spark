/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.tuning.StageRuntimeStats

import org.apache.spark.scheduler.SparkListenerEvent

case class SparkRapidsBuildInfoEvent(
  sparkRapidsBuildInfo: Map[String, String],
  sparkRapidsJniBuildInfo: Map[String, String],
  cudfBuildInfo: Map[String, String],
  sparkRapidsPrivateBuildInfo: Map[String, String]
) extends SparkListenerEvent

/**
 * Event posted when a shuffle is unregistered, containing disk I/O savings statistics.
 * This tracks how much data stayed in memory throughout the shuffle lifecycle,
 * avoiding disk writes compared to the baseline implementation.
 *
 * @param shuffleId The shuffle ID being unregistered
 * @param bytesFromMemory Bytes that were read from memory (never spilled to disk)
 * @param bytesFromDisk Bytes that were read from disk (spilled at some point)
 * @param numExpansions Number of buffer expansions that occurred
 * @param numSpills Number of buffers that were spilled to disk
 * @param numForcedFileOnly Number of buffers that used forced file-only mode
 */
case class SparkRapidsShuffleDiskSavingsEvent(
  shuffleId: Int,
  bytesFromMemory: Long,
  bytesFromDisk: Long,
  numExpansions: Int = 0,
  numSpills: Int = 0,
  numForcedFileOnly: Int = 0
) extends SparkListenerEvent

/**
 * Event posted by adaptive tuning when a stage attempt completes, carrying the aggregated
 * runtime signals for that stage. Written to the event log so that tuning decisions can be
 * analyzed and the tuning controller can be replayed offline.
 */
case class SparkRapidsStageRuntimeStatsEvent(
  stats: StageRuntimeStats
) extends SparkListenerEvent

/**
 * Event posted whenever adaptive tuning changes a setting.
 *
 * @param executionId SQL execution the decision applies to, or -1 for application scope
 * @param stageId stage whose signals triggered the decision, or -1 when not stage driven
 * @param knob name of the setting that was changed
 * @param oldValue value before the change
 * @param newValue value after the change
 * @param reason human readable explanation, including the signals that drove the change
 * @param source strategy that made the decision, "rule" for settings derived from the
 *               resources or "migration" for settings translated from a CPU configuration
 */
case class SparkRapidsTuningDecisionEvent(
  executionId: Long,
  stageId: Int,
  knob: String,
  oldValue: String,
  newValue: String,
  reason: String,
  source: String
) extends SparkListenerEvent
