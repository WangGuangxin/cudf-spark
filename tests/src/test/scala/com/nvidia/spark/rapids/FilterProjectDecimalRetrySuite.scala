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

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, DType}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.FilterProjectRetrySuite.FailOnce
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{GpuRetryOOM, GpuSplitAndRetryOOM, RmmSpark}
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{DecimalMultiplyChecks, GpuAdd, GpuDecimalDivide, GpuDecimalMultiply, GpuGreaterThan}
import org.apache.spark.sql.types.{BooleanType, Decimal, DecimalType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class FilterProjectDecimalRetrySuite extends RmmSparkRetrySuiteBase {
  private val rowCount = 64
  private val decimalTypes = Seq(DecimalType(9, 2), DecimalType(18, 6), DecimalType(38, 10))

  override def afterEach(): Unit = {
    RmmSpark.getAndResetNumRetryThrow(1)
    RmmSpark.getAndResetNumSplitRetryThrow(1)
    super.afterEach()
  }

  private def batch(dt: DecimalType): ColumnarBatch = {
    def decimalColumn(factor: Boolean): GpuColumnVector = {
      val values = (0 until rowCount).map { i =>
        if (i % (if (factor) 7 else 11) == 0) {
          null
        } else {
          val value = if (factor) BigDecimal(i % 3) else BigDecimal(i - rowCount / 2)
          value.setScale(dt.scale).bigDecimal
        }
      }
      withResource(ColumnVector.fromDecimals(values: _*)) { input =>
        GpuColumnVector.from(input.castTo(GpuColumnVector.getNonNestedRapidsType(dt)), dt)
      }
    }
    val columns = Seq(false, true).safeMap(decimalColumn)
    closeOnExcept(columns) { _ =>
      withResource(GpuColumnVector.from(
          ColumnVector.fromInts((0 until rowCount): _*), IntegerType)) { unused =>
        new ColumnarBatch((columns :+ unused.incRefCount()).toArray, rowCount)
      }
    }
  }

  private def project(dt: DecimalType, scenario: String,
      failure: Option[RuntimeException] = None): GpuFilterProject = {
    val a = AttributeReference("a", dt)()
    val b = AttributeReference("b", dt)()
    val unused = AttributeReference("unused", IntegerType)()
    val condition = scenario match {
      case "all" => GpuLiteral(true, BooleanType)
      case "none" => GpuLiteral(false, BooleanType)
      case _ => GpuGreaterThan(a, GpuLiteral(Decimal(0), dt))
    }
    val add = GpuAdd(a, b, failOnError = false)()
    // Match Spark's result type: precision 38 addition reduces scale to make room for carry.
    val sum = GpuCheckOverflow(add, add.dataType.asInstanceOf[DecimalType], nullOnOverflow = true)
    val quotient = GpuDecimalDivide(a, b, dt, failOnError = false, failOnDivideByZero = false)
    val last: Expression = failure.map(FailOnce(quotient, _)).getOrElse(quotient)
    val useLongMultiply = DecimalMultiplyChecks.nonRoundedIntermediatePrecision(dt, dt, dt) >
      DType.DECIMAL128_MAX_PRECISION
    val projects = Seq(a, GpuAlias(a, "duplicate")(), GpuAlias(sum, "sum")(),
      GpuAlias(GpuDecimalMultiply(a, b, dt, useLongMultiply, failOnError = false), "product")(),
      GpuAlias(GpuCast(a, DecimalType(dt.precision - 1, dt.scale - 1))(), "rounded")(),
      GpuAlias(last, "quotient")())
    if (failure.isEmpty) {
      assert(GpuFilterProjectExec.eligible(condition, projects))
    }
    new GpuFilterProject(condition, projects, Seq(a, b, unused), useAst = false,
      new SQLConf(), Map.empty)
  }

  private def values(batch: ColumnarBatch): Seq[Seq[Option[BigDecimal]]] = {
    val columns = (0 until batch.numCols()).safeMap { i =>
      batch.column(i).asInstanceOf[GpuColumnVector].copyToHost()
    }
    withResource(columns) { hosts =>
      (0 until batch.numRows()).map { row =>
        hosts.map { host =>
          val column = host.getBase
          if (column.isNull(row)) None else Some(BigDecimal(column.getBigDecimal(row)))
        }
      }
    }
  }

  private def checkRetry(dt: DecimalType, scenario: String, split: Boolean,
      spill: Boolean, failAfterProjection: Boolean): Unit = {
    val expected = withResource(project(dt, scenario)) { evaluator =>
      withResource(batch(dt)) { input =>
        withResource(evaluator(input))(values)
      }
    }
    val failure = if (!failAfterProjection) None else {
      Some(if (split) new GpuSplitAndRetryOOM("after decimal projection")
        else new GpuRetryOOM("after decimal projection"))
    }
    withResource(project(dt, scenario, failure)) { evaluator =>
      val sb = SpillableColumnarBatch(batch(dt), SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
      val result = closeOnExcept(sb) { _ =>
        if (spill) {
          assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
        } else if (!failAfterProjection) {
          if (split) {
            RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
          } else {
            RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
          }
        }
        GpuProjectExec.runStreamingWithSplitRetry(sb, Nil, evaluator.apply)
      }
      val actual = ArrayBuffer[Seq[Option[BigDecimal]]]()
      result.foreach(out => withResource(out)(b => actual ++= values(b)))
      assert(actual.toSeq == expected)
      if (!spill && !failAfterProjection) {
        if (split) assert(RmmSpark.getAndResetNumSplitRetryThrow(1) > 0)
        else assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
  }

  for (dt <- decimalTypes; scenario <- Seq("all", "some", "none"); split <- Seq(false, true)) {
    test(s"decimal projection retry: $dt, selection=$scenario, split=$split") {
      checkRetry(dt, scenario, split, spill = false, failAfterProjection = false)
    }
  }

  for (dt <- decimalTypes) {
    test(s"decimal spilled input is restored: $dt") {
      checkRetry(dt, "some", split = false, spill = true, failAfterProjection = false)
    }
  }

  for (split <- Seq(false, true)) {
    test(s"retry after decimal outputs have been allocated: split=$split") {
      checkRetry(DecimalType(38, 10), "some", split, spill = false, failAfterProjection = true)
    }
  }
}
