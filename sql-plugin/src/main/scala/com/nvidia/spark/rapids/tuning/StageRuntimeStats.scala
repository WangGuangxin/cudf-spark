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

import scala.collection.mutable.ArrayBuffer

/**
 * Per-task measurements needed by adaptive tuning. This is a plain value with no Spark
 * dependencies so that the aggregation and, later, the tuning controller can be unit tested
 * and driven by values replayed from an event log.
 *
 * @param inputBytes bytes read from data sources plus bytes read from shuffle by this task
 */
case class TaskRuntimeSample(
    successful: Boolean,
    runTimeMs: Long,
    inputBytes: Long,
    shuffleWriteBytes: Long,
    retryCount: Long,
    splitAndRetryCount: Long,
    spillToHostBytes: Long,
    spillToDiskBytes: Long,
    semaphoreWaitNs: Long,
    gpuTimeNs: Long,
    maxDeviceMemoryBytes: Long,
    maxConcurrentGpuTasks: Long)

/**
 * Aggregated runtime statistics for one stage attempt. These are the signals the adaptive
 * tuning controller consumes; collecting them does not change any execution behavior.
 *
 * Per-task input size percentiles describe how well the stage was partitioned: a large
 * max relative to the median indicates skew, a small median indicates over-partitioning.
 */
case class StageRuntimeStats(
    executionId: Long,
    stageId: Int,
    stageAttemptId: Int,
    numTasks: Int,
    numFailedTasks: Int,
    totalRunTimeMs: Long,
    totalInputBytes: Long,
    totalShuffleWriteBytes: Long,
    medianTaskInputBytes: Long,
    p90TaskInputBytes: Long,
    maxTaskInputBytes: Long,
    totalRetryCount: Long,
    totalSplitAndRetryCount: Long,
    numTasksWithRetry: Int,
    totalSpillToHostBytes: Long,
    totalSpillToDiskBytes: Long,
    numTasksWithSpill: Int,
    totalSemaphoreWaitNs: Long,
    totalGpuTimeNs: Long,
    maxDeviceMemoryBytes: Long,
    maxConcurrentGpuTasks: Long) {

  /** Fraction of the stage input that was spilled from device memory to host. */
  def spillToHostRatio: Double =
    if (totalInputBytes > 0) totalSpillToHostBytes.toDouble / totalInputBytes else 0.0

  /** Fraction of successful tasks that hit at least one retry or split-and-retry. */
  def retryTaskRatio: Double = {
    val succeeded = numTasks - numFailedTasks
    if (succeeded > 0) numTasksWithRetry.toDouble / succeeded else 0.0
  }
}

/** Accumulates task samples for a single stage attempt. Not thread safe. */
class StageRuntimeStatsBuilder(
    val executionId: Long,
    val stageId: Int,
    val stageAttemptId: Int) {

  private val samples = ArrayBuffer[TaskRuntimeSample]()

  def add(sample: TaskRuntimeSample): Unit = samples += sample

  def numSamples: Int = samples.size

  def build(): StageRuntimeStats = {
    val succeeded = samples.filter(_.successful)
    // Partition size statistics only make sense over tasks that processed their whole input.
    val sortedInputs = succeeded.map(_.inputBytes).sorted
    StageRuntimeStats(
      executionId = executionId,
      stageId = stageId,
      stageAttemptId = stageAttemptId,
      numTasks = samples.size,
      numFailedTasks = samples.size - succeeded.size,
      totalRunTimeMs = samples.map(_.runTimeMs).sum,
      totalInputBytes = succeeded.map(_.inputBytes).sum,
      totalShuffleWriteBytes = succeeded.map(_.shuffleWriteBytes).sum,
      medianTaskInputBytes = StageRuntimeStatsBuilder.percentile(sortedInputs, 0.5),
      p90TaskInputBytes = StageRuntimeStatsBuilder.percentile(sortedInputs, 0.9),
      maxTaskInputBytes = sortedInputs.lastOption.getOrElse(0L),
      totalRetryCount = samples.map(_.retryCount).sum,
      totalSplitAndRetryCount = samples.map(_.splitAndRetryCount).sum,
      numTasksWithRetry =
        succeeded.count(s => s.retryCount > 0 || s.splitAndRetryCount > 0),
      totalSpillToHostBytes = samples.map(_.spillToHostBytes).sum,
      totalSpillToDiskBytes = samples.map(_.spillToDiskBytes).sum,
      numTasksWithSpill = succeeded.count(_.spillToHostBytes > 0),
      totalSemaphoreWaitNs = samples.map(_.semaphoreWaitNs).sum,
      totalGpuTimeNs = samples.map(_.gpuTimeNs).sum,
      maxDeviceMemoryBytes =
        if (samples.isEmpty) 0L else samples.map(_.maxDeviceMemoryBytes).max,
      maxConcurrentGpuTasks =
        if (samples.isEmpty) 0L else samples.map(_.maxConcurrentGpuTasks).max)
  }
}

object StageRuntimeStatsBuilder {
  /** Nearest-rank percentile over an ascending sequence; 0 when empty. */
  def percentile(sorted: Seq[Long], p: Double): Long = {
    require(p >= 0.0 && p <= 1.0, s"percentile $p must be in [0, 1]")
    if (sorted.isEmpty) {
      0L
    } else {
      val rank = math.ceil(p * sorted.size).toInt.max(1)
      sorted(rank - 1)
    }
  }
}
