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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.jni.RmmSpark

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuAdd
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.rapids.metrics.source.MockTaskContext
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

class PartialProjectAstRetrySuite extends RmmSparkRetrySuiteBase {
  private val rowCount = 64
  private val input = AttributeReference("input", IntegerType)()

  override def afterEach(): Unit = {
    RmmSpark.getAndResetNumRetryThrow(1)
    RmmSpark.getAndResetNumSplitRetryThrow(1)
    super.afterEach()
  }

  private def expression: GpuExpression = GpuAdd(
    GpuAdd(input, GpuLiteral(2, IntegerType), failOnError = false)(),
    GpuLiteral(3, IntegerType), failOnError = false)()

  private def boundProject(tiered: Boolean): GpuTieredProject = {
    val arithmetic = expression
    GpuPartialProjectAst.markEligible(arithmetic)
    val conf = new SQLConf()
    conf.setConfString(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "true")
    conf.setConfString(RapidsConf.ENABLE_TIERED_PROJECT.key, tiered.toString)
    GpuBindReferences.bindGpuReferencesTiered(
      Seq(GpuAlias(arithmetic, "result")(), GpuAlias(input, "direct")()),
      Seq(input), conf, Map.empty)
  }

  private def batch(): ColumnarBatch = closeOnExcept(
      ColumnVector.fromInts((0 until rowCount): _*)) { column =>
    new ColumnarBatch(Array(GpuColumnVector.from(column, IntegerType)), rowCount)
  }

  private def assertValues(batches: Seq[ColumnarBatch]): Unit = {
    val actual = batches.flatMap { b =>
      withResource(b.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { result =>
        withResource(b.column(1).asInstanceOf[GpuColumnVector].copyToHost()) { direct =>
          (0 until b.numRows()).map { i =>
            assert(result.getInt(i) == direct.getInt(i) + 5)
            direct.getInt(i)
          }
        }
      }
    }
    assert(actual == (0 until rowCount))
  }

  for (tiered <- Seq(false, true); split <- Seq(false, true)) {
    test(s"mixed AST and direct output survive injected OOM: tiered=$tiered, split=$split") {
      val conf = new SQLConf()
      conf.setConfString(RapidsConf.PROJECT_SPLIT_RETRY_ENABLED.key, "true")
      SQLConf.withExistingConf(conf) {
        val project = boundProject(tiered)
        val hasAst = project.exprTiers.flatten.exists(
          _.exists(_.isInstanceOf[GpuPartialProjectAst]))
        assert(hasAst)
        withResource(SpillableColumnarBatch(batch(),
            SpillPriorities.ACTIVE_ON_DECK_PRIORITY)) { sb =>
          if (split) {
            RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
          } else {
            RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
          }
          val outputs = project.projectAndCloseStreamingWithSplitRetry(sb.incRefCount())
          val batches = closeOnExcept(ArrayBuffer[ColumnarBatch]()) { buffer =>
            while (outputs.hasNext) {
              buffer += outputs.next()
            }
            buffer
          }
          withResource(batches) { batches =>
            assertValues(batches)
            if (split) assert(batches.size > 1)
          }
        }
        if (split) {
          assert(RmmSpark.getAndResetNumSplitRetryThrow(1) > 0)
        } else {
          assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
        }
      }
    }
  }

  test("task-owned AST can be serialized and evaluated after task completion") {
    val context = new MockTaskContext(taskAttemptId = 1, partitionId = 0)
    TrampolineUtil.setTaskContext(context)
    try {
      val project = boundProject(tiered = true)
      withResource(batch()) { inputBatch =>
        withResource(project.project(inputBatch)) { result => assertValues(Seq(result)) }
        // Reuse the compiled expression across batches, then serialize while it is live.
        withResource(project.project(inputBatch)) { result => assertValues(Seq(result)) }
        val bytes = new ByteArrayOutputStream()
        withResource(new ObjectOutputStream(bytes))(_.writeObject(project))
        val copy = withResource(new ObjectInputStream(
            new ByteArrayInputStream(bytes.toByteArray))) { stream =>
          stream.readObject().asInstanceOf[GpuTieredProject]
        }
        context.markTaskComplete()
        TrampolineUtil.unsetTaskContext()
        withResource(copy.project(inputBatch)) { result => assertValues(Seq(result)) }
        withResource(project.project(inputBatch)) { result => assertValues(Seq(result)) }
      }
    } finally {
      context.markTaskComplete()
      TrampolineUtil.unsetTaskContext()
      ScalableTaskCompletion.reset()
    }
  }

  test("shared subtree remains a single tier output across AST and ordinary expressions") {
    val shared = expression
    GpuPartialProjectAst.markEligible(shared)
    // The last output is deliberately unmarked, modelling an ordinary output sharing arithmetic.
    val exprs = Seq(GpuAlias(GpuAdd(shared, GpuLiteral(1, IntegerType), false)(), "ast")(),
      GpuAlias(shared, "shared")(),
      GpuAlias(GpuAdd(shared, GpuLiteral(2, IntegerType), false)(), "ordinary")())
    GpuPartialProjectAst.markEligible(exprs.head)
    val config = new SQLConf()
    config.setConfString(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "true")
    val project = GpuBindReferences.bindGpuReferencesTiered(exprs, Seq(input), config, Map.empty)
    assert(project.exprTiers.size == 2)
    assert(project.exprTiers.head.count(_.exists(_.isInstanceOf[GpuPartialProjectAst])) == 1)
    val last = project.exprTiers.last
    assert(!last.last.exists(_.isInstanceOf[GpuPartialProjectAst]))
    assert(GpuProjectExec.extractSingleBoundIndex(Seq(last(1))).head.nonEmpty)
  }

  test("direct columns remain references and unapproved arithmetic remains ordinary") {
    val config = new SQLConf()
    config.setConfString(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "true")
    val project = GpuBindReferences.bindGpuReferencesTiered(
      Seq(GpuAlias(expression, "ordinary")(), GpuAlias(input, "direct")()),
      Seq(input), config, Map.empty)
    val hasAst = project.exprTiers.flatten.exists(_.exists(_.isInstanceOf[GpuPartialProjectAst]))
    assert(!hasAst)
    assert(project.getPassThroughIndex(1).contains(0))
  }

  test("eligible constant expression supports count-only and empty batches") {
    val expression = GpuPartialProjectAst(GpuAdd(
      GpuLiteral(2, IntegerType), GpuLiteral(3, IntegerType), failOnError = false)())
    for (rows <- Seq(0, rowCount)) {
      withResource(new ColumnarBatch(Array.empty, rows)) { inputBatch =>
        withResource(expression.columnarEval(inputBatch)) { result =>
          withResource(result.copyToHost()) { host =>
            assert(host.getRowCount == rows)
            (0 until rows).foreach(i => assert(host.getInt(i) == 5))
          }
        }
      }
    }
  }
}
