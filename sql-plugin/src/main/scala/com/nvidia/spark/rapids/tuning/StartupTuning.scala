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
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.execution.TrampolineUtil

/**
 * Settings that adaptive tuning derives once at startup from the resources an application
 * has, for the settings the user did not set explicitly.
 */
object StartupTuning extends Logging {
  private val MiB = 1024L * 1024
  private val GiB = 1024L * MiB

  val MaxPartitionBytesKey = "spark.sql.files.maxPartitionBytes"

  // A scan split is compressed on disk and typically expands several times when decoded, so
  // half of the target batch size keeps decoded scan batches near that target. The bounds
  // keep tasks from becoming too small to amortize their fixed cost on the GPU, or too large
  // to balance across executors.
  private val MinScanSplitBytes = 128 * MiB
  private val MaxScanSplitBytes = 1 * GiB

  // Pinned memory makes host to device copies of scan and shuffle data much faster than
  // pageable memory. One eighth of the off-heap budget leaves most of it for the
  // non-pinned host allocations and spill, and the cap keeps the up-front pinned
  // allocation, which is not returned to the OS, bounded on large hosts.
  private val PinnedPoolBudgetFraction = 8
  private val MaxPinnedPoolBytes = 16 * GiB

  def scanSplitBytes(batchSizeBytes: Long): Long =
    math.min(math.max(batchSizeBytes / 2, MinScanSplitBytes), MaxScanSplitBytes)

  def pinnedPoolBytes(offHeapBudgetBytes: Long): Long =
    math.min(math.max(offHeapBudgetBytes, 0L) / PinnedPoolBudgetFraction, MaxPinnedPoolBytes)

  /**
   * Sets the application level settings that are derived on the driver, when adaptive
   * tuning is enabled. This must run before the driver plugin reads its configuration, so
   * that driver side components see the derived settings, and before any SparkSession is
   * created, so that SQL settings are picked up as session defaults. Executors receive the
   * derived settings with the rest of the application configuration.
   */
  def applyOnDriver(sc: SparkContext, sparkConf: SparkConf): Unit = {
    val conf = new RapidsConf(sparkConf)
    if (conf.adaptiveTuningEnabled) {
      deriveScanSplit(sc, sparkConf, conf)
      deriveShuffleSkipMerge(sc, sparkConf, conf)
    }
  }

  private def deriveScanSplit(sc: SparkContext, sparkConf: SparkConf, conf: RapidsConf): Unit = {
    if (!sparkConf.contains(MaxPartitionBytesKey)) {
      val split = scanSplitBytes(conf.gpuTargetBatchSizeBytes)
      sparkConf.set(MaxPartitionBytesKey, split.toString)
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, MaxPartitionBytesKey, "unset",
        split.toString,
        s"half of ${RapidsConf.GPU_BATCH_SIZE_BYTES.key}=${conf.gpuTargetBatchSizeBytes}, " +
          s"bounded to [$MinScanSplitBytes, $MaxScanSplitBytes]"))
    }
  }

  /**
   * Serving MULTITHREADED shuffle blocks from host memory instead of merging them into one
   * file per map task avoids rewriting all shuffle data through local disk, which on shuffle
   * heavy queries is the dominant non GPU cost. It is only safe when the external shuffle
   * service, which cannot see the in-memory blocks, is disabled and the off-heap limit
   * bounds the buffered data so that it spills instead of growing without limit.
   */
  def shouldDeriveSkipMerge(sparkConf: SparkConf, conf: RapidsConf): Boolean =
    !sparkConf.contains(RapidsConf.MULTITHREADED_SHUFFLE_SKIP_MERGE.key) &&
      conf.isMultiThreadedShuffleManagerMode &&
      !sparkConf.getBoolean(TrampolineUtil.shuffleServiceEnabledKey, false) &&
      conf.offHeapLimitEnabled

  private def deriveShuffleSkipMerge(sc: SparkContext, sparkConf: SparkConf,
      conf: RapidsConf): Unit = {
    val key = RapidsConf.MULTITHREADED_SHUFFLE_SKIP_MERGE.key
    if (shouldDeriveSkipMerge(sparkConf, conf)) {
      sparkConf.set(key, "true")
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, key, "unset", "true",
        "MULTITHREADED shuffle with the external shuffle service disabled and the off-heap " +
          "limit enabled"))
    }
  }

  /**
   * The pinned pool size an executor should use: the configured size if the user set one,
   * otherwise, when adaptive tuning is enabled, a size derived from the off-heap budget.
   */
  def executorPinnedPoolBytes(conf: RapidsConf, sparkConf: SparkConf,
      offHeapBudgetBytes: => Long): Long = {
    if (conf.adaptiveTuningEnabled && !sparkConf.contains(RapidsConf.PINNED_POOL_SIZE.key)) {
      val budget = offHeapBudgetBytes
      val pinned = pinnedPoolBytes(budget)
      logInfo(s"Adaptive tuning decision: ${RapidsConf.PINNED_POOL_SIZE.key}: unset -> " +
        s"$pinned (1/$PinnedPoolBudgetFraction of the off-heap budget $budget, " +
        s"capped at $MaxPinnedPoolBytes)")
      pinned
    } else {
      conf.pinnedPoolSize
    }
  }
}
