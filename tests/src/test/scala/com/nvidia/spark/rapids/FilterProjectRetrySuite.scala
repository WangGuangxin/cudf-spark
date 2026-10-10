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

import ai.rapids.cudf.{ColumnVector, Rmm}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.{GpuRetryOOM, GpuSplitAndRetryOOM, RmmSpark}
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuAdd, GpuGreaterThan}
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.rapids.metrics.source.MockTaskContext
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

object FilterProjectRetrySuite {
  private[rapids] case class FailOnce(child: Expression, failure: RuntimeException)
      extends GpuUnaryExpression {
    private var failed = false
    override def dataType = child.dataType
    override def nullable: Boolean = child.nullable
    override protected def doColumnar(input: GpuColumnVector): ColumnVector = {
      if (!failed) {
        failed = true
        throw failure
      }
      input.getBase.incRefCount()
    }
  }
}

class FilterProjectRetrySuite extends RmmSparkRetrySuiteBase {
  import FilterProjectRetrySuite.FailOnce

  private val a = AttributeReference("a", IntegerType)()
  private val b = AttributeReference("b", IntegerType)()
  private val unused = AttributeReference("unused", IntegerType)()
  private val inputAttrs = Seq(a, b, unused)
  private val rows = 100
  private val projects = Seq(a, GpuAlias(GpuAdd(b, GpuLiteral(1, IntegerType),
    failOnError = false)(), "sum")(), GpuAlias(a, "duplicate")())

  private def batch(rowCount: Int = rows, columnCount: Int = 3): ColumnarBatch = {
    val values = (0 until rowCount).toArray
    val column = GpuColumnVector.from(ColumnVector.fromInts(values: _*), IntegerType)
    val columns = (0 until columnCount).map { i =>
      if (i == 0) column else column.incRefCount()
    }
    new ColumnarBatch(columns.toArray, rowCount)
  }

  override def afterEach(): Unit = {
    RmmSpark.getAndResetNumRetryThrow(1)
    RmmSpark.getAndResetNumSplitRetryThrow(1)
    super.afterEach()
  }

  for (ast <- Seq(false, true); threshold <- Seq(-1, 40, 101);
       retry <- Seq("none", "retry", "split")) {
    test(s"selection/project recomputes on $retry, AST=$ast, threshold=$threshold") {
      val condition = GpuGreaterThan(a, GpuLiteral(threshold, IntegerType))
      withResource(new GpuFilterProject(condition, projects, inputAttrs, ast,
          new SQLConf(), Map.empty)) { evaluator =>
        assert(evaluator.dependencyOrdinals == Seq(1))
        assert(evaluator.directOrdinals == Seq(0))
        val sb = SpillableColumnarBatch(batch(), SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        retry match {
          case "retry" => RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case "split" => RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case _ =>
        }
        val result = GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
        val actual = scala.collection.mutable.ArrayBuffer[Int]()
        result.foreach { out =>
          withResource(out) { cb =>
            withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { first =>
              withResource(cb.column(1).asInstanceOf[GpuColumnVector].copyToHost()) { sum =>
                withResource(cb.column(2).asInstanceOf[GpuColumnVector].copyToHost()) { duplicate =>
                  (0 until cb.numRows()).foreach { i =>
                    assert(sum.getInt(i) == first.getInt(i) + 1)
                    assert(duplicate.getInt(i) == first.getInt(i))
                    actual += first.getInt(i)
                  }
                }
              }
            }
          }
        }
        assert(actual.toSeq == (0 until rows).filter(_ > threshold))
        assert(SpillFramework.stores.deviceStore.numHandles == 0)
        if (retry == "retry") {
          assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
        } else if (retry == "split") {
          assert(RmmSpark.getAndResetNumSplitRetryThrow(1) > 0)
        }
      }
    }
  }

  test("spilled original input is restored before selection") {
    withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(40, IntegerType)),
        projects, inputAttrs, false, new SQLConf(), Map.empty)) { evaluator =>
      val sb = SpillableColumnarBatch(batch(), SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
      val result = GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
      var count = 0
      result.foreach(out => withResource(out)(cb => count += cb.numRows()))
      assert(count == rows - 41)
      assert(SpillFramework.stores.deviceStore.numHandles == 0)
      assert(SpillFramework.stores.hostStore.numHandles == 0)
    }
  }

  for (ast <- Seq(false, true); literalOnly <- Seq(false, true)) {
    test(s"empty dependencies and count-only output, AST=$ast, literal=$literalOnly") {
      val output = if (literalOnly) Seq(GpuAlias(GpuLiteral(17, IntegerType), "literal")())
        else Seq.empty
      withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(40, IntegerType)),
          output, inputAttrs, ast, new SQLConf(), Map.empty)) { evaluator =>
        withResource(batch()) { input =>
          withResource(evaluator(input)) { cb =>
            assert(cb.numRows() == rows - 41)
            assert(cb.numCols() == output.length)
            if (literalOnly) {
              withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { host =>
                assert((0 until cb.numRows()).forall(i => host.getInt(i) == 17))
              }
            }
          }
        }
      }
    }
  }

  for (consumeFirst <- Seq(false, true)) {
    test(s"task completion closes abandoned split input, consumeFirst=$consumeFirst") {
      val context = new MockTaskContext(taskAttemptId = 1, partitionId = 0)
      TrampolineUtil.setTaskContext(context)
      try {
        withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(40, IntegerType)),
            projects, inputAttrs, false, new SQLConf(), Map.empty)) { evaluator =>
          val sb = SpillableColumnarBatch(batch(), SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
          val result = GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
          assert(result.hasNext)
          if (consumeFirst) {
            RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
            // This standalone RMM suite has no SparkEnv for GPU semaphore initialization.
            // The retry iterator already registered its cleanup with the mock task.
            TrampolineUtil.unsetTaskContext()
            try {
              withResource(result.next()) { _ => }
            } finally {
              TrampolineUtil.setTaskContext(context)
            }
            assert(result.hasNext)
          }
          context.markTaskComplete()
          assert(SpillFramework.stores.deviceStore.numHandles == 0)
          assert(SpillFramework.stores.hostStore.numHandles == 0)
        }
      } finally {
        TrampolineUtil.unsetTaskContext()
        ScalableTaskCompletion.reset()
      }
    }
  }

  for (ast <- Seq(false, true)) {
    test(s"empty input, AST=$ast") {
      withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(0, IntegerType)),
          projects, inputAttrs, ast, new SQLConf(), Map.empty)) { evaluator =>
        withResource(batch(0)) { input =>
          withResource(evaluator(input)) { cb =>
            assert(cb.numRows() == 0)
            assert(cb.numCols() == projects.length)
          }
        }
      }
    }
  }

  for (split <- Seq(false, true)) {
    test(s"OOM after selection and dependency gather restarts the full pipeline, split=$split") {
      val failure = if (split) new GpuSplitAndRetryOOM("after selection")
        else new GpuRetryOOM("after selection")
      val expression = GpuAlias(FailOnce(
        GpuAdd(b, GpuLiteral(1, IntegerType), failOnError = false)(), failure), "sum")()
      withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(40, IntegerType)),
          Seq(expression), inputAttrs, false, new SQLConf(), Map.empty)) { evaluator =>
        val sb = SpillableColumnarBatch(batch(), SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        val result = GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
        val actual = scala.collection.mutable.ArrayBuffer[Int]()
        var batches = 0
        result.foreach { out =>
          withResource(out) { cb =>
            batches += 1
            withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { host =>
              (0 until cb.numRows()).foreach(i => actual += host.getInt(i))
            }
          }
        }
        assert(actual.toSeq == (41 until rows).map(_ + 1))
        if (split) {
          assert(batches >= 2)
        } else {
          assert(batches == 1)
        }
        assert(SpillFramework.stores.deviceStore.numHandles == 0)
      }
    }
  }

  test("unexpected projection failures propagate and release selection temporaries") {
    val expected = new IllegalStateException("unexpected projection failure")
    withResource(new GpuFilterProject(GpuGreaterThan(a, GpuLiteral(40, IntegerType)),
        Seq(GpuAlias(FailOnce(b, expected), "sum")()), inputAttrs, false,
        new SQLConf(), Map.empty)) { evaluator =>
      withResource(batch()) { input =>
        val bytes = Rmm.getTotalBytesAllocated
        assert(intercept[IllegalStateException] {
          withResource(evaluator(input)) { _ => }
        } eq expected)
        assert(Rmm.getTotalBytesAllocated == bytes)
        assert(input.numRows() == rows)
      }
    }
  }

  for (ast <- Seq(false, true); threshold <- Seq(-1, 40, 101);
       split <- Seq(false, true)) {
    test(s"all-needed input fast path, AST=$ast, threshold=$threshold, split=$split") {
      val condition = GpuGreaterThan(a, GpuLiteral(threshold, IntegerType))
      val bound = GpuFilterProject.bind(condition, projects, inputAttrs.take(2), ast,
        new SQLConf(), Map.empty)
      assert(bound.allInputsNeeded)
      withResource(new GpuFilterProject(bound)) { evaluator =>
        val sb = SpillableColumnarBatch(batch(columnCount = 2),
          SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        if (split) {
          RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
        }
        val actual = scala.collection.mutable.ArrayBuffer[Int]()
        GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply).foreach { out =>
          withResource(out) { cb =>
            withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { first =>
              withResource(cb.column(1).asInstanceOf[GpuColumnVector].copyToHost()) { sum =>
                (0 until cb.numRows()).foreach { i =>
                  actual += first.getInt(i)
                  assert(sum.getInt(i) == first.getInt(i) + 1)
                }
              }
            }
          }
        }
        assert(actual.toSeq == (0 until rows).filter(_ > threshold))
        assert(SpillFramework.stores.deviceStore.numHandles == 0)
      }
    }
  }
}
