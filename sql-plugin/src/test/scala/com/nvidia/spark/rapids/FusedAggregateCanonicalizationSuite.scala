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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum}
import org.apache.spark.sql.types.{IntegerType, LongType}

object FusedAggregateCanonicalizationSuite {
  case class PlanningLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Planning-only leaf")
  }
}

class FusedAggregateCanonicalizationSuite extends AnyFunSuite {
  import FusedAggregateCanonicalizationSuite.PlanningLeaf

  private def aggregate(child: SparkPlan, valueOrdinal: Int = 1): GpuHashAggregateExec = {
    val groups = Seq(child.output.head)
    val sum = GpuBasicSum(child.output(valueOrdinal), LongType, failOnErrorOverride = true)
    val expression = GpuAggregateExpression(sum, Partial, isDistinct = false,
      None, NamedExpression.newExprId)
    val buffers = sum.aggBufferAttributes
    GpuHashAggregateExec(None, groups, Seq(expression), buffers, groups ++ buffers, child,
      1024 * 1024, 1.0, forceSinglePassAgg = false, allowSinglePassAgg = false,
      allowNonFullyAggregatedOutput = false, skipAggPassReductionRatio = 0.85)
  }

  private def scan(): SparkPlan = PlanningLeaf(Seq(
    AttributeReference("key", IntegerType)(), AttributeReference("value", LongType)(),
    AttributeReference("other", LongType)()))

  test("join-chain aggregate retains semantic equality across fresh expression identifiers") {
    val left = aggregate(scan())
    val right = aggregate(scan())
    assert(left.sameResult(right), "unfused equivalent aggregates must match")
    val fusedLeft = GpuJoinChainAggregateExec(left, left.child)
    val fusedRight = GpuJoinChainAggregateExec(right, right.child)
    assert(fusedLeft.sameResult(fusedRight),
      "fusion must preserve equality used by Spark exchange and query-stage reuse")
  }

  test("canonicalization keeps aggregates over different input columns distinct") {
    val child = scan()
    val left = aggregate(child, 1)
    val right = aggregate(child, 2)
    assert(!left.sameResult(right), "different original aggregates must remain distinct")
    assert(!GpuJoinChainAggregateExec(left, child).sameResult(
      GpuJoinChainAggregateExec(right, child)))
  }

  private def expanded(discriminator: Long): GpuExpandAggregateExec = {
    val child = scan()
    val gid = AttributeReference("gid", LongType, nullable = false)()
    val output = child.output ++ Seq(gid)
    val projections = Seq(child.output ++ Seq(GpuLiteral(discriminator, LongType)))
    val expand = GpuExpandExec(projections, output, child)()
    GpuExpandAggregateExec(aggregate(expand), projections, output, child)
  }

  test("expand aggregate retains semantic equality across fresh expression identifiers") {
    val left = expanded(0)
    val right = expanded(0)
    assert(left.aggregate.sameResult(right.aggregate), "original descriptors must match")
    assert(left.sameResult(right), "the nested aggregate descriptor must be canonicalized")
  }

  test("canonicalization keeps different expand projections distinct") {
    assert(!expanded(0).sameResult(expanded(1)))
  }
}
