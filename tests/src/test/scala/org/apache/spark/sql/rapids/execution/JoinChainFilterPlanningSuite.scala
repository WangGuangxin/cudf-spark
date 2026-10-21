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

package org.apache.spark.sql.rapids.execution

import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.execution.{LocalTableScanExec, SparkPlan}
import org.apache.spark.sql.rapids.{GpuGreaterThan, GpuLessThan}
import org.apache.spark.sql.types.{DoubleType, IntegerType}

class JoinChainFilterPlanningSuite extends AnyFunSuite {
  private def attr(name: String) = AttributeReference(name, IntegerType, nullable = true)()
  private def leaf(names: String*) = LocalTableScanExec(names.map(attr), Seq.empty)

  // Plan-only cases also run without CUDA: they protect execution/batching boundaries.
  test("deferred filter planning preserves safety and batching boundaries") {
    withResource(SparkSession.builder().master("local[1]")
        .appName("deferred join filter planning")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.warehouse.dir", "/tmp/join-chain-filter-planning")
        .getOrCreate()) { spark =>
      spark.conf.set(RapidsConf.ENABLE_JOIN_CHAIN.key, "true")
      val fact = leaf("k", "payload")
      val d1 = leaf("d1")
      val first = GpuBroadcastHashJoinExec(Seq(fact.output.head), Seq(d1.output.head),
        Inner, GpuBuildRight, None, fact, d1, isNullAwareAntiJoin = false)
      val d2 = leaf("d2")
      val root = GpuBroadcastHashJoinExec(Seq(fact.output.head), Seq(d2.output.head),
        Inner, GpuBuildRight, None, first, d2, isNullAwareAntiJoin = false)
      val predicate = GpuLessThan(fact.output(1), d2.output.head)
      def fused(child: SparkPlan, coalesce: Boolean = true): SparkPlan =
        GpuJoinChainFilterExec.fuse(GpuFilterExec(predicate, child)(coalesce))
      assert(fused(root).isInstanceOf[GpuJoinChainFilterExec])
      assert(fused(GpuCoalesceBatches(root, TargetSize(1024)))
        .isInstanceOf[GpuJoinChainFilterExec])
      assert(fused(GpuCoalesceBatches(root, RequireSingleBatch)).isInstanceOf[GpuFilterExec])
      assert(fused(root, coalesce = false).isInstanceOf[GpuFilterExec])
      assert(fused(first).isInstanceOf[GpuFilterExec])
      val withResidual = root.copy(condition = Some(predicate))
      assert(GpuJoinChainFilterExec.fuse(withResidual).isInstanceOf[GpuJoinChainFilterExec])
      assert(GpuJoinChainFilterExec.fuse(withResidual.copy(joinType = LeftSemi))
        .isInstanceOf[GpuBroadcastHashJoinExec])
      val unsupported = GpuGreaterThan(GpuLiteral(1.0, DoubleType),
        GpuCast(fact.output(1), DoubleType)())
      assert(GpuJoinChainFilterExec.fuse(GpuFilterExec(unsupported, root)())
        .isInstanceOf[GpuFilterExec])
      // A reference-only projection remains an ordinary downstream node, preserving aliases.
      val projected = GpuProjectExec(List(root.output(1)),
        GpuFilterExec(predicate, root)())
      val result = GpuJoinChainFilterExec.fuse(projected)
      assert(result.output == projected.output)
      assert(result.children.head.isInstanceOf[GpuJoinChainFilterExec])
      spark.conf.set(RapidsConf.ENABLE_JOIN_CHAIN.key, "false")
      assert(fused(root).isInstanceOf[GpuFilterExec])
      assert(GpuJoinChainFilterExec.fuse(withResidual).isInstanceOf[GpuBroadcastHashJoinExec])
    }
  }
}
