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

import java.util.concurrent.ConcurrentHashMap

import com.nvidia.spark.rapids.{SparkRapidsStageRuntimeStatsEvent, SparkRapidsTuningDecisionEvent}

import org.apache.spark.{SparkContext, Success}
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.rapids.{NanoTime, SizeInBytes}
import org.apache.spark.sql.rapids.execution.TrampolineUtil

/**
 * Driver side collector of per-stage runtime signals for adaptive tuning.
 *
 * It reads the GPU task metrics that executors already report as named accumulators,
 * together with Spark's own task metrics, and aggregates them per stage attempt. When a
 * stage completes the aggregate is logged, posted as a [[SparkRapidsStageRuntimeStatsEvent]]
 * and kept, per SQL execution, for tuning rules planning the remaining stages of that query.
 * Collecting these statistics does not change execution behavior.
 *
 * Spark delivers events to a listener from a single thread, but `completedStages` is read by
 * query planning threads, so it uses a concurrent map.
 */
class AdaptiveTuningListener(postEvent: SparkListenerEvent => Unit)
    extends SparkListener with Logging {
  import AdaptiveTuningListener._

  private val stageToExecution = new ConcurrentHashMap[Int, Long]()
  private val activeStages = new ConcurrentHashMap[(Int, Int), StageRuntimeStatsBuilder]()
  private val executionStats = new ConcurrentHashMap[Long, Vector[StageRuntimeStats]]()

  /** Stats of the stages of a SQL execution that have completed so far, in completion order. */
  def completedStages(executionId: Long): Seq[StageRuntimeStats] =
    Option(executionStats.get(executionId)).getOrElse(Vector.empty)

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionId = Option(jobStart.properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(s => scala.util.Try(s.toLong).toOption)
      .getOrElse(NoExecutionId)
    jobStart.stageInfos.foreach(si => stageToExecution.putIfAbsent(si.stageId, executionId))
  }

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = {
    val si = stageSubmitted.stageInfo
    val executionId = stageToExecution.getOrDefault(si.stageId, NoExecutionId)
    activeStages.put((si.stageId, si.attemptNumber()),
      new StageRuntimeStatsBuilder(executionId, si.stageId, si.attemptNumber()))
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    // Tasks that end after their stage attempt completed (e.g. speculative or zombie tasks)
    // have no builder and are ignored.
    Option(activeStages.get((taskEnd.stageId, taskEnd.stageAttemptId))).foreach { builder =>
      builder.add(toSample(taskEnd))
    }
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    val si = stageCompleted.stageInfo
    stageToExecution.remove(si.stageId)
    Option(activeStages.remove((si.stageId, si.attemptNumber()))).foreach { builder =>
      val stats = builder.build()
      logInfo(s"Adaptive tuning stage stats: $stats")
      if (stats.executionId != NoExecutionId) {
        executionStats.merge(stats.executionId, Vector(stats),
          (a: Vector[StageRuntimeStats], b: Vector[StageRuntimeStats]) => a ++ b)
      }
      postEvent(SparkRapidsStageRuntimeStatsEvent(stats))
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
    case e: SparkListenerSQLExecutionEnd => executionStats.remove(e.executionId)
    case _ =>
  }
}

object AdaptiveTuningListener extends Logging {
  /** Execution id used for stages that do not belong to a SQL execution. */
  val NoExecutionId: Long = -1L

  // Names under which GpuTaskMetrics registers its accumulators.
  private val RetryCount = "gpuRetryCount"
  private val SplitAndRetryCount = "gpuSplitAndRetryCount"
  private val SpillToHostBytes = "gpuSpillToHostBytes"
  private val SpillToDiskBytes = "gpuSpillToDiskBytes"
  private val SemaphoreWait = "gpuSemaphoreWait"
  private val GpuTime = "gpuTime"
  private val MaxDeviceMemoryBytes = "gpuMaxDeviceMemoryBytes"
  private val MaxConcurrentGpuTasks = "gpuMaxConcurrentGpuTasks"

  @volatile private var listener: Option[AdaptiveTuningListener] = None

  /** Registers the listener on the driver. Called once from the driver plugin. */
  def install(sc: SparkContext): Unit = synchronized {
    if (listener.isEmpty) {
      val l = new AdaptiveTuningListener(e => TrampolineUtil.postEvent(sc, e))
      sc.addSparkListener(l)
      listener = Some(l)
      logInfo("Adaptive tuning runtime statistics collection enabled")
    }
  }

  /** The installed listener, if adaptive tuning is enabled. */
  def get: Option[AdaptiveTuningListener] = listener

  private[tuning] def toSample(taskEnd: SparkListenerTaskEnd): TaskRuntimeSample = {
    val accums: Map[String, Long] = taskEnd.taskInfo.accumulables.flatMap { info =>
      for {
        name <- info.name
        update <- info.update
        value <- accumulatorValueAsLong(update)
      } yield name -> value
    }.toMap
    def acc(name: String): Long = accums.getOrElse(name, 0L)
    val tm = Option(taskEnd.taskMetrics)
    TaskRuntimeSample(
      successful = taskEnd.reason == Success,
      runTimeMs = tm.map(_.executorRunTime).getOrElse(0L),
      inputBytes = tm.map(m =>
        m.inputMetrics.bytesRead + m.shuffleReadMetrics.totalBytesRead).getOrElse(0L),
      shuffleWriteBytes = tm.map(_.shuffleWriteMetrics.bytesWritten).getOrElse(0L),
      retryCount = acc(RetryCount),
      splitAndRetryCount = acc(SplitAndRetryCount),
      spillToHostBytes = acc(SpillToHostBytes),
      spillToDiskBytes = acc(SpillToDiskBytes),
      semaphoreWaitNs = acc(SemaphoreWait),
      gpuTimeNs = acc(GpuTime),
      maxDeviceMemoryBytes = acc(MaxDeviceMemoryBytes),
      maxConcurrentGpuTasks = acc(MaxConcurrentGpuTasks))
  }

  private[tuning] def accumulatorValueAsLong(v: Any): Option[Long] = v match {
    case n: NanoTime => Some(n.value.longValue())
    case s: SizeInBytes => Some(s.value.longValue())
    case l: java.lang.Long => Some(l.longValue())
    case i: java.lang.Integer => Some(i.longValue())
    case _ => None
  }
}

/** Records adaptive tuning decisions in the driver log and the event log. */
object TuningDecisionLog extends Logging {
  val RuleSource = "rule"
  val MigrationSource = "migration"

  def record(sc: SparkContext, decision: SparkRapidsTuningDecisionEvent): Unit = {
    logInfo(s"Adaptive tuning ${decision.source} decision: execution ${decision.executionId} " +
      s"stage ${decision.stageId} ${decision.knob}: ${decision.oldValue} -> " +
      s"${decision.newValue} (${decision.reason})")
    TrampolineUtil.postEvent(sc, decision)
  }
}
