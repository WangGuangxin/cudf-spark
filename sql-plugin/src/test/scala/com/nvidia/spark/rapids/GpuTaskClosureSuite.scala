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

import java.io.{ByteArrayOutputStream, ObjectOutputStream}

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeReference, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.plans.{FullOuter, Inner, JoinType, LeftAnti, LeftOuter, LeftSemi, RightOuter}
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.rapids.GpuGreaterThan
import org.apache.spark.sql.rapids.execution.GpuJoinChain
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuTaskClosureSuite {
  case class EmptyInput(override val output: Seq[Attribute]) extends LeafExecNode with GpuExec {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Columnar input only")

    override def internalDoExecuteColumnar(): RDD[ColumnarBatch] =
      sparkContext.emptyRDD[ColumnarBatch]
  }
}

class GpuTaskClosureSuite extends AnyFunSuite with BeforeAndAfterAll {
  import Arm.withResource
  import GpuTaskClosureSuite.EmptyInput

  private var session: SparkSession = _
  private var ownsSession = false

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    ownsSession = SparkSession.getDefaultSession.isEmpty
    session = SparkSession.builder.master("local[1]").appName("GpuTaskClosureSuite")
      .config("spark.ui.enabled", "false").getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try {
      if (ownsSession && session != null) session.stop()
    } finally {
      super.afterAll()
    }
  }

  private def assertNoCapturedPlan(plan: SparkPlan): Unit = {
    // Construct and serialize the real RDD lineage, without running a GPU task.
    assertNoCapturedPlanValue(plan.executeColumnar())
  }

  private def assertNoCapturedPlanValue(value: AnyRef): Unit = {
    withResource(new ObjectOutputStream(new ByteArrayOutputStream()) {
      enableReplaceObject(true)
      override protected def replaceObject(value: AnyRef): AnyRef = {
        assert(!value.isInstanceOf[SparkPlan],
          s"Task closure captured ${value.getClass.getName}")
        value
      }
    }) { stream =>
      stream.writeObject(value)
    }
  }

  for (offset <- Seq(0, 3)) {
    test(s"TopN task closure does not capture a plan with offset $offset") {
      val key = AttributeReference("key", IntegerType)()
      val child = EmptyInput(Seq(key))
      val order = SortOrder(key, Ascending, NullsFirst, Seq.empty)
      assertNoCapturedPlan(GpuTopN(10, Seq(order), child.output, child, offset)(Seq(order)))
    }
  }

  for ((kind, name) <- Seq((OutOfCoreSort, "OutOfCoreSort"),
      (FullSortSingleBatch, "FullSortSingleBatch"), (SortEachBatch, "SortEachBatch"))) {
    test(s"$name task closure does not capture a sort plan") {
      val key = AttributeReference("key", IntegerType)()
      val order = SortOrder(key, Ascending, NullsFirst, Seq.empty)
      assertNoCapturedPlan(GpuSortExec(Seq(order), false, EmptyInput(Seq(key)), kind)(
        Seq(order)))
    }
  }

  test("LocalLimit task closure does not capture its output plan") {
    assertNoCapturedPlan(GpuLocalLimitExec(10,
      EmptyInput(Seq(AttributeReference("key", IntegerType)()))))
  }

  test("GlobalLimit task closure does not capture its output plan") {
    assertNoCapturedPlan(GpuGlobalLimitExec(10,
      EmptyInput(Seq(AttributeReference("key", IntegerType)())), offset = 3))
  }

  private def shuffledJoin(
      joinType: JoinType = Inner,
      side: GpuBuildSide = GpuBuildRight,
      withCondition: Boolean = false): GpuShuffledHashJoinExec = {
    val leftKey = AttributeReference("leftKey", IntegerType)()
    val rightKey = AttributeReference("rightKey", IntegerType)()
    val condition = if (withCondition) Some(GpuGreaterThan(leftKey, rightKey)) else None
    GpuShuffledHashJoinExec(Seq(leftKey), Seq(rightKey), joinType, side, condition,
      EmptyInput(Seq(leftKey)), EmptyInput(Seq(rightKey)),
      CoalesceReadOption(session.sessionState.conf), isSkewJoin = false)(
      Seq(leftKey), Seq(rightKey))
  }

  private def sizedJoin(): GpuShuffledSymmetricHashJoinExec = {
    val leftKey = AttributeReference("leftKey", IntegerType)()
    val rightKey = AttributeReference("rightKey", IntegerType)()
    GpuShuffledSymmetricHashJoinExec(Inner, Seq(leftKey), Seq(rightKey), None,
      EmptyInput(Seq(leftKey)), EmptyInput(Seq(rightKey)), isGpuShuffle = true,
      gpuBatchSizeBytes = 1024 * 1024, partitionNumAmplification = 8.0,
      readOption = CoalesceReadOption(session.sessionState.conf), isSkewJoin = false)(
      Seq(leftKey), Seq(rightKey))
  }

  for (side <- Seq(GpuBuildLeft, GpuBuildRight)) {
    test(s"Shuffled hash join chain adapters do not capture a plan with $side") {
      val join = shuffledJoin()
      assertNoCapturedPlanValue(join.joinChainBuildLoader(side))
      assertNoCapturedPlanValue(join.joinChainEager(side))
    }

    test(s"Sized hash join chain adapters do not capture a plan with $side") {
      val join = sizedJoin()
      assertNoCapturedPlanValue(join.joinChainBuildLoader(side))
      assertNoCapturedPlanValue(join.joinChainEager(side))
    }
  }

  test("Shuffled hash join task RDD does not capture a plan") {
    assertNoCapturedPlan(shuffledJoin())
  }

  test("Sized hash join task RDD does not capture a plan") {
    assertNoCapturedPlan(sizedJoin())
  }

  for (joinType <- Seq(Inner, LeftOuter, RightOuter, LeftSemi, LeftAnti);
      side <- Seq(GpuBuildLeft, GpuBuildRight)) {
    test(s"Pruned hash join $joinType $side task does not capture a plan") {
      val join = shuffledJoin(joinType, side)
      assert(join.canPruneOutput)
      assertNoCapturedPlanValue(join.executeColumnarPruned(
        JoinOutputPruning(Array(0), Array.empty[Int])).get)
    }
  }

  test("Pruned sized hash join task does not capture a plan") {
    val join = sizedJoin()
    assert(join.canPruneOutput)
    assertNoCapturedPlanValue(join.executeColumnarPruned(
      JoinOutputPruning(Array(0), Array.empty[Int])).get)
  }

  for (joinType <- Seq(Inner, LeftOuter, RightOuter, FullOuter, LeftSemi, LeftAnti);
      side <- Seq(GpuBuildLeft, GpuBuildRight);
      withCondition <- Seq(false, true)) {
    test(s"Hash join $joinType $side condition=$withCondition has no plan in its task") {
      assertNoCapturedPlan(shuffledJoin(joinType, side, withCondition))
    }
  }

  test("Symmetric host and GPU join sizers do not retain an enclosing plan") {
    val read = CoalesceReadOption(session.sessionState.conf)
    assertNoCapturedPlanValue(
      new GpuShuffledSymmetricHashJoinExec.HostHostSymmetricJoinSizer(read))
    for (startWithLeft <- Seq(false, true)) {
      assertNoCapturedPlanValue(
        new GpuShuffledSymmetricHashJoinExec.SpillableColumnarBatchSymmetricJoinSizer(
          startWithLeft))
    }
  }

  test("Asymmetric host and GPU join sizers do not retain an enclosing plan") {
    val read = CoalesceReadOption(session.sessionState.conf)
    assertNoCapturedPlanValue(
      new GpuShuffledAsymmetricHashJoinExec.HostHostAsymmetricJoinSizer(10, read))
    assertNoCapturedPlanValue(
      new GpuShuffledAsymmetricHashJoinExec.SpillableColumnarBatchAsymmetricJoinSizer(10))
  }

  for (sized <- Seq(false, true)) {
    test(s"Shuffled chain sized=$sized retains eligibility without capturing plans") {
      val key = AttributeReference("key", IntegerType)()
      val rightKey = AttributeReference("rightKey", IntegerType)()
      val lastKey = AttributeReference("lastKey", IntegerType)()
      // Wide payload makes this a worthwhile chain under the unchanged traffic heuristic.
      val payload = (0 until 8).map(i => AttributeReference(s"payload$i", StringType)())
      val input = EmptyInput(Seq(key) ++ payload)
      val read = CoalesceReadOption(session.sessionState.conf)
      def join(left: SparkPlan, rightKey: Attribute): SparkPlan = {
        val right = EmptyInput(Seq(rightKey))
        if (sized) {
          GpuShuffledSymmetricHashJoinExec(Inner, Seq(key), Seq(rightKey), None,
            left, right, isGpuShuffle = true, gpuBatchSizeBytes = 1024 * 1024,
            partitionNumAmplification = 8.0, readOption = read, isSkewJoin = false)(
            Seq(key), Seq(rightKey))
        } else {
          GpuShuffledHashJoinExec(Seq(key), Seq(rightKey), Inner, GpuBuildRight, None,
            left, right, read, isSkewJoin = false)(Seq(key), Seq(rightKey))
        }
      }
      val root = join(join(input, rightKey), lastKey)
      val rdd = GpuJoinChain.execute(root)
      if (sized) {
        // Existing sized-join boundaries must not change merely to optimize serialization.
        assert(rdd.isEmpty)
      } else {
        assert(rdd.isDefined, "fixture must use the fused runtime chain")
        assertNoCapturedPlanValue(rdd.get)
      }
    }
  }
}
