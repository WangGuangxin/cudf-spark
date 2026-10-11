/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.shims.GpuHashPartitioning

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.rapids.aggregate.{GpuBasicSum, GpuCount, GpuMax, GpuMin}
import org.apache.spark.sql.types.{IntegerType, LongType}

/** Eligibility for fusing the terminal partial merge with packed-table hash shuffle. */
object GpuPartialAggregateShuffle {
  def operations(agg: GpuHashAggregateExec,
      partitioning: GpuHashPartitioning): Option[Array[Int]] = {
    val keys = agg.groupingExpressions
    val functions = agg.aggregateExpressions.map(_.aggregateFunction)
    val buffers = keys.map(_.toAttribute) ++ functions.flatMap(_.aggBufferAttributes)
    val shape = keys.length == 1 &&
      (keys.head.dataType == IntegerType || keys.head.dataType == LongType) &&
      !agg.allowSinglePassAgg && !agg.forceSinglePassAgg &&
      agg.requiredChildDistributionExpressions.isEmpty && functions.nonEmpty &&
      agg.aggregateExpressions.forall(e => e.mode == Partial && !e.isDistinct &&
        e.filter.isEmpty && e.deterministic) &&
      partitioning.hashMode == Murmur3Mode && partitioning.expressions.length == 1 &&
      partitioning.expressions.head.semanticEquals(keys.head.toAttribute) &&
      agg.resultExpressions.length == buffers.length &&
      agg.resultExpressions.forall(_.isInstanceOf[AttributeReference]) &&
      // Spark's partial result attributes and GPU buffer attributes have different exprIds.
      // The existing final pass binds partial results to themselves, preserving buffer order.
      // Unique result attributes prove that this binding is an identity projection.
      agg.resultExpressions.map(_.exprId).distinct.length == buffers.length &&
      agg.resultExpressions.head.semanticEquals(keys.head.toAttribute) &&
      agg.resultExpressions.zip(buffers).forall { case (out, buffer) =>
        out.dataType == buffer.dataType
      }
    if (!shape) None
    else {
      val ops = functions.map {
        case c: GpuCount if !c.failOnError => Some(3)
        case s: GpuBasicSum if !s.failOnErrorOverride && s.resultType == LongType &&
            (s.child.dataType == IntegerType || s.child.dataType == LongType) => Some(0)
        case m: GpuMin if m.dataType == LongType => Some(1)
        case m: GpuMax if m.dataType == LongType => Some(2)
        case _ => None
      }
      if (ops.forall(_.isDefined)) Some(ops.flatten.toArray) else None
    }
  }
}
