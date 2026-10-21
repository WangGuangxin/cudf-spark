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

package com.nvidia.spark.rapids

import com.nvidia.spark.rapids.GpuMetric._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, SortOrder}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.rapids.execution.{GpuBroadcastHashJoinExec, GpuJoinChain}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuJoinChainFilterExec {
  private def eligible(condition: Expression, child: SparkPlan): Boolean =
    RapidsConf.ENABLE_JOIN_CHAIN.get(child.conf) && condition.references.nonEmpty &&
      condition.references.size < child.output.size &&
      GpuFilterProjectExec.eligible(condition, Nil) && GpuJoinChain.canFilter(child)

  def fuse(plan: SparkPlan): SparkPlan = plan.transformUp {
    // Use the same closed safety whitelist as Filter/Project fusion. In particular, never
    // replay random/stateful expressions, Python UDFs or ANSI expressions across join retries.
    case f: GpuFilterExec if f.coalesceAfter && eligible(f.condition, f.child) =>
      GpuJoinChainFilterExec(f.condition, f.child)
    // Catalyst normally incorporates a post-join filter into the join's residual condition.
    // Extract only INNER conditions: doing this for semi/outer joins would change matching.
    // Named copy arguments preserve the extra executorBroadcast/isSkewJoin shim parameters.
    case join: GpuBroadcastHashJoinExec if join.joinType == Inner && join.condition.isDefined &&
        eligible(join.condition.get, join.copy(condition = None)) =>
      GpuJoinChainFilterExec(join.condition.get, join.copy(condition = None), residualJoin = true)
  }
}

/**
 * A terminal filter in the deferred join pipeline. Only predicate columns are gathered before
 * selection; downstream operators still receive ordinary compact batches. An explicit physical
 * node makes this boundary visible in explain plans, including when Filter/Project fusion is on.
 */
case class GpuJoinChainFilterExec(condition: Expression, child: SparkPlan,
    residualJoin: Boolean = false)
    extends GpuPartitioningPreservingUnaryExecNode with GpuExec {
  override val coalesceAfter: Boolean = true
  override def output: Seq[Attribute] = GpuFilterExec(condition, child)().output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    "candidateRows" -> createMetric(MODERATE_LEVEL, "rows before deferred filter"),
    "candidateBatches" -> createMetric(DEBUG_LEVEL, "map batches before deferred filter"),
    "avoidedPayloadRows" -> createMetric(MODERATE_LEVEL, "full payload rows not materialized"),
    "fusedTasks" -> createMetric(MODERATE_LEVEL, "tasks with deferred join filter"),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY))

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val descriptor = GpuJoinChainPipelineExec(Some(condition), output.toList, child,
      if (residualJoin) Some(condition) else None, output)
    val terminal = descriptor.bind().copy(
      inputRows = gpuLongMetric("candidateRows"),
      inputBatches = gpuLongMetric("candidateBatches"),
      avoidedPayloadRows = gpuLongMetric("avoidedPayloadRows"),
      fusedTasks = gpuLongMetric("fusedTasks"),
      filterTime = gpuLongMetric(OP_TIME_LEGACY))
    val outputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val outputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val rdd = GpuJoinChain.executeFiltered(child, terminal).getOrElse {
      child.executeColumnar().mapPartitions(terminal.eager)
    }
    rdd.mapPartitions { input =>
      input.map { batch =>
        outputRows += batch.numRows()
        outputBatches += 1
        batch
      }
    }
  }
}
