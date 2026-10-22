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

import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, SortOrder}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.rapids.execution.{GpuJoinChain, GpuJoinExec}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Output columns a join still has to gather, as strictly increasing ordinals into each side's
 * own schema. An empty side contributes nothing to the output, which the gatherer already
 * handles the same way it handles a semi join.
 *
 * This is a carrier for one join's execution, never compared or used as a plan node, so the
 * arrays' reference equality does not matter.
 */
case class JoinOutputPruning(left: Array[Int], right: Array[Int]) {
  require(left.nonEmpty, "the left side must keep at least one column")

  override def toString: String =
    s"JoinOutputPruning(left=[${left.mkString(",")}], right=[${right.mkString(",")}])"
}

/**
 * Narrows the output of a hash join to the columns its parent projection actually reads.
 *
 * The join still produces the same rows; it just stops gathering columns that the projection
 * above it drops right away. Equi-join keys are the systematic case: the build side's key is
 * materialized by the gather and then discarded, once per join in a chain of joins.
 *
 * This is a separate node rather than a field on each join exec so that the pruned `output`
 * lives in a first parameter list and survives tree rewrites.
 */
case class GpuPrunedJoinExec(
    child: SparkPlan,
    required: Seq[Int]) extends GpuPartitioningPreservingUnaryExecNode with GpuExec {

  require(required.nonEmpty, "a pruned join must keep at least one column")
  require(required.sliding(2).forall(p => p.length < 2 || p(0) < p(1)),
    "required ordinals must be strictly increasing")

  override def output: Seq[Attribute] = required.map(child.output)

  override def outputBatching: CoalesceGoal = GpuExec.outputBatching(child)

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override protected def buildAttributeMap(): Map[Attribute, Attribute] =
    output.map(attr => attr -> attr).toMap

  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val pruning = GpuJoinProjectExec.pruningFor(child, required)
    val pruned = child match {
      case join: GpuJoinExec => pruning.flatMap(join.executeColumnarPruned)
      case _ => None
    }
    // The join declined to narrow its gather, so produce the advertised schema anyway. This is
    // only a correctness net; it saves nothing. The rule already keeps chained joins out of
    // here, so reaching the eager path never costs a chain.
    pruned.getOrElse {
      val ordinals = required
      child.executeColumnar().map { batch =>
        withResource(batch) { batch =>
          val columns = ordinals.safeMap { i =>
            batch.column(i).asInstanceOf[GpuColumnVector].incRefCount()
          }
          closeOnExcept(columns)(_ => new ColumnarBatch(columns.toArray, batch.numRows()))
        }
      }
    }
  }
}

object GpuJoinProjectExec extends Logging {

  /** How many of a join's output columns come from its left child. */
  private[rapids] def leftWidth(join: GpuJoinExec): Int =
    math.min(join.left.output.length, join.output.length)

  /** Split required join-output ordinals into per-side ordinals. */
  private[rapids] def pruningFor(child: SparkPlan,
      required: Seq[Int]): Option[JoinOutputPruning] = child match {
    case join: GpuJoinExec =>
      val width = leftWidth(join)
      Some(JoinOutputPruning(required.filter(_ < width).toArray,
        required.filter(_ >= width).map(_ - width).toArray))
    case _ => None
  }

  /** Joins that said they can narrow their gather. Everything else is left alone. */
  private[rapids] def supported(plan: SparkPlan): Boolean = plan match {
    case join: GpuJoinExec => join.canPruneOutput
    case _ => false
  }

  /**
   * Ordinals of `join.output` that `projects` still reads, padded so the left side keeps at
   * least one column. The padded column is never referenced; it only keeps the left gatherer
   * from being handed an empty table.
   */
  private[rapids] def requiredOrdinals(join: GpuJoinExec,
      projects: Seq[Expression]): Seq[Int] = {
    val referenced = projects.flatMap(_.references).map(_.exprId).toSet
    val used = join.output.indices.filter(i => referenced(join.output(i).exprId))
    if (used.exists(_ < leftWidth(join))) used else (0 +: used).distinct.sorted
  }

  private def skipped(project: SparkPlan, reason: String, explain: Boolean): SparkPlan = {
    if (explain) {
      logInfo(s"Join output pruning skipped for ${project.nodeName}: $reason")
    }
    project
  }

  /**
   * Joins that a fused chain already covers. The chain gathers each payload once instead of once
   * per join, which is the larger saving where both apply, and it absorbs the projections between
   * its joins itself. Inserting a pruning node anywhere along it would stop the chain forming.
   */
  private[rapids] def chained(plan: SparkPlan): Set[SparkPlan] =
    plan.collect { case node => GpuJoinChain.fusedMembers(node) }.flatten.toSet

  def fuse(plan: SparkPlan, explain: Boolean = false): SparkPlan = {
    val inChain = chained(plan)
    plan.transformUp {
    case p @ GpuProjectExec(projects, join: GpuJoinExec, _) =>
      if (inChain(join)) {
        skipped(p, s"${join.nodeName} is part of a fused join chain", explain)
      } else if (!supported(join)) {
        skipped(p, s"${join.nodeName} cannot narrow its gather for ${join.joinType}", explain)
      } else {
        val required = requiredOrdinals(join, projects)
        if (required.length >= join.output.length) {
          skipped(p, "the projection reads every join output column", explain)
        } else {
          if (explain) {
            logInfo(s"Join output pruning keeps ${required.length} of ${join.output.length} " +
              s"columns for ${join.nodeName}")
          }
          GpuProjectExec(projects, GpuPrunedJoinExec(join, required), p.enablePreSplit)
        }
      }
    }
  }
}
