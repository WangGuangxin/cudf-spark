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
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.types.IntegerType
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
    withResource(new ObjectOutputStream(new ByteArrayOutputStream()) {
      enableReplaceObject(true)
      override protected def replaceObject(value: AnyRef): AnyRef = {
        assert(!value.isInstanceOf[SparkPlan],
          s"Task closure captured ${value.getClass.getName}")
        value
      }
    }) { stream =>
      stream.writeObject(plan.executeColumnar())
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
}
