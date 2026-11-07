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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.{NanoTime, SizeInBytes}

class StageRuntimeStatsSuite extends AnyFunSuite {
  private def sample(
      inputBytes: Long,
      successful: Boolean = true,
      retry: Long = 0,
      splitRetry: Long = 0,
      spillToHost: Long = 0,
      maxDeviceMem: Long = 0): TaskRuntimeSample =
    TaskRuntimeSample(
      successful = successful,
      runTimeMs = 10,
      inputBytes = inputBytes,
      shuffleWriteBytes = inputBytes / 2,
      retryCount = retry,
      splitAndRetryCount = splitRetry,
      spillToHostBytes = spillToHost,
      spillToDiskBytes = 0,
      semaphoreWaitNs = 5,
      gpuTimeNs = 7,
      maxDeviceMemoryBytes = maxDeviceMem,
      maxConcurrentGpuTasks = 2)

  test("percentile uses nearest rank") {
    val sorted = (1L to 10L).toSeq
    assert(StageRuntimeStatsBuilder.percentile(sorted, 0.5) == 5L)
    assert(StageRuntimeStatsBuilder.percentile(sorted, 0.9) == 9L)
    assert(StageRuntimeStatsBuilder.percentile(sorted, 1.0) == 10L)
    assert(StageRuntimeStatsBuilder.percentile(sorted, 0.0) == 1L)
    assert(StageRuntimeStatsBuilder.percentile(Seq.empty, 0.5) == 0L)
  }

  test("empty stage builds zeroed stats") {
    val stats = new StageRuntimeStatsBuilder(1L, 2, 0).build()
    assert(stats.numTasks == 0)
    assert(stats.maxTaskInputBytes == 0L)
    assert(stats.maxDeviceMemoryBytes == 0L)
    assert(stats.spillToHostRatio == 0.0)
    assert(stats.retryTaskRatio == 0.0)
  }

  test("aggregates task samples and excludes failed tasks from partition sizes") {
    val builder = new StageRuntimeStatsBuilder(3L, 4, 1)
    builder.add(sample(100, retry = 1, maxDeviceMem = 50))
    builder.add(sample(200, spillToHost = 60))
    builder.add(sample(300, splitRetry = 2, spillToHost = 40, maxDeviceMem = 80))
    builder.add(sample(1000))
    // A failed attempt only read part of its input, so it must not skew the size stats,
    // but its retries and spills still reflect memory pressure.
    builder.add(sample(5000, successful = false, retry = 3, maxDeviceMem = 90))
    val stats = builder.build()

    assert(stats.executionId == 3L)
    assert(stats.stageId == 4)
    assert(stats.stageAttemptId == 1)
    assert(stats.numTasks == 5)
    assert(stats.numFailedTasks == 1)
    assert(stats.totalInputBytes == 1600L)
    assert(stats.totalShuffleWriteBytes == 800L)
    assert(stats.medianTaskInputBytes == 200L)
    assert(stats.p90TaskInputBytes == 1000L)
    assert(stats.maxTaskInputBytes == 1000L)
    assert(stats.totalRetryCount == 4L)
    assert(stats.totalSplitAndRetryCount == 2L)
    assert(stats.numTasksWithRetry == 2)
    assert(stats.totalSpillToHostBytes == 100L)
    assert(stats.numTasksWithSpill == 2)
    assert(stats.totalSemaphoreWaitNs == 25L)
    assert(stats.totalGpuTimeNs == 35L)
    assert(stats.maxDeviceMemoryBytes == 90L)
    assert(stats.maxConcurrentGpuTasks == 2L)
    assert(stats.spillToHostRatio == 100.0 / 1600)
    assert(stats.retryTaskRatio == 0.5)
  }

  test("accumulator updates of GPU metric types convert to longs") {
    assert(AdaptiveTuningListener.accumulatorValueAsLong(NanoTime(12L)) == Some(12L))
    assert(AdaptiveTuningListener.accumulatorValueAsLong(new SizeInBytes(34L)) == Some(34L))
    assert(AdaptiveTuningListener.accumulatorValueAsLong(java.lang.Long.valueOf(56L)) ==
      Some(56L))
    assert(AdaptiveTuningListener.accumulatorValueAsLong(java.lang.Double.valueOf(1.5)).isEmpty)
    assert(AdaptiveTuningListener.accumulatorValueAsLong("x").isEmpty)
  }
}
