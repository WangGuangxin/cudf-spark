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

import scala.util.Try

import com.nvidia.spark.rapids.{RapidsConf, SparkRapidsTuningDecisionEvent}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.rapids.execution.TrampolineUtil

/**
 * Settings that adaptive tuning derives once at startup from the resources an application
 * has, for the settings the user did not set explicitly.
 */
object StartupTuning extends Logging {
  private val MiB = 1024L * 1024
  private val GiB = 1024L * MiB

  val MaxPartitionBytesKey = "spark.sql.files.maxPartitionBytes"
  val ParallelismFirstKey = "spark.sql.adaptive.coalescePartitions.parallelismFirst"
  val MinPartitionSizeKey = "spark.sql.adaptive.coalescePartitions.minPartitionSize"
  val AdvisoryPartitionSizeKey = "spark.sql.adaptive.advisoryPartitionSizeInBytes"
  val NumConnectionsPerPeerKey = "spark.shuffle.io.numConnectionsPerPeer"
  // Spark's default for spark.sql.adaptive.advisoryPartitionSizeInBytes.
  private val DefaultAdvisoryPartitionBytes = 64 * MiB
  // Partitions smaller than this fraction of the advisory size are merged with neighbors.
  private val MinPartitionSizeFraction = 16

  // A scan split as large as the target batch size keeps the number of scan tasks, and so
  // their fixed cost on the GPU, low. Splits that decode to more than fits are handled by
  // the chunked reader and the retry framework; on NDS and NDS-H, splits up to 2 GiB ran
  // without failures with only 8 GiB of GPU memory. The bounds keep tasks from becoming
  // too small to amortize their fixed cost, or too large to balance across executors.
  private val MinScanSplitBytes = 128 * MiB
  private[tuning] val MaxScanSplitBytes = 2 * GiB

  // Pinned memory makes host to device copies of scan and shuffle data much faster than
  // pageable memory. One eighth of the off-heap budget leaves most of it for the
  // non-pinned host allocations and spill, and the cap keeps the up-front pinned
  // allocation, which is not returned to the OS, bounded on large hosts.
  private val PinnedPoolBudgetFraction = 8
  private val MaxPinnedPoolBytes = 16 * GiB

  // Connections per peer for MULTITHREADED shuffle served from the in-memory catalog. With
  // two executors on one host, same-host shuffle reads of SSB at 3 TB stopped improving
  // beyond 4 connections.
  private[tuning] val SkipMergeConnectionsPerPeer = 4

  // Spark's default for spark.sql.shuffle.partitions.
  private val DefaultShufflePartitions = 200
  // Tasks per GPU assumed when spark.rapids.sql.concurrentGpuTasks is not set, in which case
  // executors pick their concurrency from the GPU memory at runtime.
  private val DefaultTasksPerGpu = 2

  def scanSplitBytes(batchSizeBytes: Long): Long =
    math.min(math.max(batchSizeBytes, MinScanSplitBytes), MaxScanSplitBytes)

  def pinnedPoolBytes(offHeapBudgetBytes: Long): Long =
    math.min(math.max(offHeapBudgetBytes, 0L) / PinnedPoolBudgetFraction, MaxPinnedPoolBytes)

  /**
   * Sets the application level settings that are translated from a CPU configuration, when
   * CPU migration is enabled, and then those that are derived on the driver, when adaptive
   * tuning is enabled. This must run before the driver plugin reads its configuration, so
   * that driver side components see the derived settings, and before any SparkSession is
   * created, so that SQL settings are picked up as session defaults. Executors receive the
   * derived settings with the rest of the application configuration.
   */
  def applyOnDriver(sc: SparkContext, sparkConf: SparkConf): Unit = {
    val conf = new RapidsConf(sparkConf)
    // Translated settings count as explicitly set, so the rules below do not override them.
    if (conf.cpuMigrationEnabled) {
      CpuMigration.apply(sc, sparkConf, conf)
    }
    if (conf.adaptiveTuningEnabled) {
      deriveScanSplit(sc, sparkConf, conf)
      deriveShuffleSkipMerge(sc, sparkConf, conf)
      // after skip merge, which it depends on
      deriveShuffleConnectionsPerPeer(sc, sparkConf)
      deriveCoalescing(sc, sparkConf)
      deriveInitialPartitionFloor(sc, sparkConf, conf)
    }
  }

  private def deriveScanSplit(sc: SparkContext, sparkConf: SparkConf, conf: RapidsConf): Unit = {
    if (!sparkConf.contains(MaxPartitionBytesKey)) {
      val split = scanSplitBytes(conf.gpuTargetBatchSizeBytes)
      sparkConf.set(MaxPartitionBytesKey, split.toString)
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, MaxPartitionBytesKey, "unset",
        split.toString,
        s"${RapidsConf.GPU_BATCH_SIZE_BYTES.key}=${conf.gpuTargetBatchSizeBytes}, " +
          s"bounded to [$MinScanSplitBytes, $MaxScanSplitBytes]",
        TuningDecisionLog.RuleSource))
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
          "limit enabled", TuningDecisionLog.RuleSource))
    }
  }

  /**
   * When MULTITHREADED shuffle serves its blocks from the in-memory catalog (skip merge),
   * there are no shuffle files to read from local disk, so the blocks of other executors on
   * the same host are fetched over the network like remote ones. Spark opens one connection
   * per peer by default, which caps that traffic at what one channel can move: on NDS-H at
   * 1 TB, with two executors on one host, fetch wait was about 80% of the task time of the
   * largest shuffled joins. The conditions are those under which the catalog is used: the
   * RAPIDS shuffle manager in MULTITHREADED mode with skip merge, no external shuffle service
   * and the off-heap limit enabled (GpuShuffleEnv.init). Executors read the value when they
   * create their shuffle client, from the application configuration they receive from the
   * driver, so deriving it at startup reaches them.
   */
  def shouldDeriveConnectionsPerPeer(sparkConf: SparkConf,
      rapidsShuffleConfigured: Boolean): Boolean = {
    // read the configuration again: skip merge may just have been derived
    val conf = new RapidsConf(sparkConf)
    rapidsShuffleConfigured &&
      !sparkConf.contains(NumConnectionsPerPeerKey) &&
      conf.isMultiThreadedShuffleManagerMode &&
      conf.isMultithreadedShuffleSkipMergeEnabled &&
      !sparkConf.getBoolean(TrampolineUtil.shuffleServiceEnabledKey, false) &&
      conf.offHeapLimitEnabled
  }

  private def deriveShuffleConnectionsPerPeer(sc: SparkContext, sparkConf: SparkConf): Unit = {
    if (shouldDeriveConnectionsPerPeer(sparkConf,
        GpuShuffleEnv.isRapidsShuffleConfigured(sparkConf))) {
      sparkConf.set(NumConnectionsPerPeerKey, SkipMergeConnectionsPerPeer.toString)
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, NumConnectionsPerPeerKey, "unset",
        SkipMergeConnectionsPerPeer.toString,
        "MULTITHREADED shuffle with skip merge fetches the blocks of same-host executors " +
          "over the network", TuningDecisionLog.RuleSource))
    }
  }

  /**
   * By default AQE coalesces shuffle partitions so that there are at least as many as there
   * are CPU cores, with partitions as small as 1 MiB. On the GPU the number of tasks running
   * at once is bounded by the GPU concurrency rather than by the cores, so that only makes
   * tasks too small to amortize their fixed cost. Coalescing toward the advisory partition
   * size instead, and merging partitions much smaller than it, keeps tasks large enough.
   */
  private def deriveCoalescing(sc: SparkContext, sparkConf: SparkConf): Unit = {
    if (!sparkConf.contains(ParallelismFirstKey)) {
      sparkConf.set(ParallelismFirstKey, "false")
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, ParallelismFirstKey, "unset", "false",
        "GPU task concurrency, not the number of cores, bounds the useful parallelism",
        TuningDecisionLog.RuleSource))
    }
    if (!sparkConf.contains(MinPartitionSizeKey)) {
      val advisory = sparkConf.getSizeAsBytes(AdvisoryPartitionSizeKey,
        DefaultAdvisoryPartitionBytes.toString)
      val minSize = minPartitionSize(advisory)
      sparkConf.set(MinPartitionSizeKey, minSize.toString)
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, MinPartitionSizeKey, "unset",
        minSize.toString, s"1/$MinPartitionSizeFraction of $AdvisoryPartitionSizeKey=$advisory",
        TuningDecisionLog.RuleSource))
    }
  }

  /**
   * Number of executors the application runs with, or None if the driver cannot know it
   * before the executors register. With dynamic allocation this is the most it may use.
   */
  private[tuning] def numExecutors(sparkConf: SparkConf): Option[Int] = {
    def positive(key: String): Option[Int] =
      sparkConf.getOption(key).flatMap(v => Try(v.trim.toInt).toOption).filter(_ > 0)
    if (sparkConf.get("spark.master", "").startsWith("local")) {
      Some(1)
    } else if (sparkConf.getBoolean("spark.dynamicAllocation.enabled", false)) {
      positive("spark.dynamicAllocation.maxExecutors").filter(_ < Int.MaxValue)
    } else {
      positive("spark.executor.instances").orElse {
        // Standalone mode without an executor count.
        for {
          coresMax <- positive("spark.cores.max")
          execCores <- positive("spark.executor.cores")
        } yield math.max(1, coresMax / execCores)
      }
    }
  }

  /**
   * Number of tasks the application's GPUs run at once: GPUs times concurrent GPU tasks,
   * or None if the number of executors is unknown on the driver.
   */
  private[tuning] def gpuTaskSlots(sparkConf: SparkConf, conf: RapidsConf): Option[Int] = {
    val gpusPerExecutor = sparkConf.getOption("spark.executor.resource.gpu.amount")
      .flatMap(v => Try(v.trim.toDouble).toOption).filter(_ > 0).getOrElse(1.0)
    val tasksPerGpu = conf.concurrentGpuTasks.map(_.intValue()).filter(_ > 0)
      .getOrElse(DefaultTasksPerGpu)
    numExecutors(sparkConf).map { e =>
      math.max(1, math.ceil(e * gpusPerExecutor).toInt) * tasksPerGpu
    }
  }

  /**
   * The initial number of AQE shuffle partitions to use instead of the current one, when the
   * current one is smaller than the number of GPU task slots, or None to keep it.
   */
  private[tuning] def initialPartitionFloor(sparkConf: SparkConf,
      conf: RapidsConf): Option[Int] = {
    val coalescing = sparkConf.getBoolean(CpuMigration.AdaptiveEnabledKey, true) &&
      sparkConf.getBoolean(CpuMigration.CoalescePartitionsEnabledKey, true)
    if (!coalescing) {
      None
    } else {
      val current = sparkConf.getOption(CpuMigration.InitialPartitionNumKey)
        .orElse(sparkConf.getOption(CpuMigration.ShufflePartitionsKey))
        .flatMap(v => Try(v.trim.toInt).toOption)
        .getOrElse(DefaultShufflePartitions)
      gpuTaskSlots(sparkConf, conf).filter(_ > current)
    }
  }

  /**
   * With AQE coalescing the initial number of shuffle partitions bounds the number of tasks
   * of every stage that reads a shuffle. A value tuned for fewer GPUs leaves GPUs idle: on
   * NDS at 3 TB on 8 GPUs running 4 tasks each, 10 initial partitions took 1.38 times as
   * long as 32, because the stages after the large shuffles ran 10 tasks of up to 12 GiB.
   * The value is raised, even when it was set explicitly, to at least one partition per GPU
   * task slot. Larger values are kept: AQE coalesces them toward the advisory size.
   */
  private def deriveInitialPartitionFloor(sc: SparkContext, sparkConf: SparkConf,
      conf: RapidsConf): Unit = {
    initialPartitionFloor(sparkConf, conf).foreach { slots =>
      val key = CpuMigration.InitialPartitionNumKey
      val old = sparkConf.getOption(key).getOrElse("unset")
      val current = sparkConf.getOption(key).map(k => s"$key=$k").getOrElse(
        s"${CpuMigration.ShufflePartitionsKey}=" +
          sparkConf.get(CpuMigration.ShufflePartitionsKey, DefaultShufflePartitions.toString))
      sparkConf.set(key, slots.toString)
      TuningDecisionLog.record(sc, SparkRapidsTuningDecisionEvent(
        AdaptiveTuningListener.NoExecutionId, -1, key, old, slots.toString,
        s"$current is below the $slots GPU task slots (GPUs x " +
          s"${RapidsConf.CONCURRENT_GPU_TASKS.key})", TuningDecisionLog.RuleSource))
    }
  }

  def minPartitionSize(advisoryPartitionBytes: Long): Long =
    math.max(advisoryPartitionBytes / MinPartitionSizeFraction, 1L)

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
