/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types._

class ExpandAnsiEligibilitySuite extends AnyFunSuite {
  test("accepted decimal types bound every branch sum within result precision") {
    for (precision <- 1 to 28; scale <- Seq(0, precision / 2, precision).distinct) {
      val input = DecimalType(precision, scale)
      val output = DecimalType(precision + 10, scale)
      assert(GpuExpandAggregateExec.sumCannotOverflowInBatch(input, output))
      val maxSum = (BigInt(10).pow(precision) - 1) * BigInt(Int.MaxValue)
      assert(maxSum <= BigInt(10).pow(output.precision) - 1)
    }
  }

  test("high precision decimal and scale changes retain the unfused path") {
    for (precision <- 29 to 38) {
      assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(
        DecimalType(precision, 2), DecimalType(38, 2)))
    }
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(
      DecimalType(7, 2), DecimalType(17, 3)))
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(
      DecimalType(7, 2), DecimalType(16, 2)))
  }

  test("narrow integral sums fit Long while Long and narrower results fall back") {
    Seq(ByteType, ShortType, IntegerType).foreach { input =>
      assert(GpuExpandAggregateExec.sumCannotOverflowInBatch(input, LongType))
    }
    assert((BigInt(1) << 31) * BigInt(Int.MaxValue) <= BigInt(Long.MaxValue))
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(LongType, LongType))
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(ByteType, IntegerType))
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(DoubleType, DoubleType))
  }

  test("Spark unscaled decimal sums use the original precision bound") {
    for (precision <- 1 to 18) {
      val child = GpuUnscaledValue(
        AttributeReference("v", DecimalType(precision, math.min(precision, 2)))())
      assert(GpuExpandAggregateExec.sumCannotOverflowInBatch(child, LongType) == (precision <= 9))
      val maxSum = (BigInt(10).pow(precision) - 1) * BigInt(Int.MaxValue)
      assert((maxSum <= BigInt(Long.MaxValue)) == (precision <= 9))
    }
    val child = GpuUnscaledValue(AttributeReference("v", DecimalType(7, 2))())
    assert(!GpuExpandAggregateExec.sumCannotOverflowInBatch(child, IntegerType))
  }
}
