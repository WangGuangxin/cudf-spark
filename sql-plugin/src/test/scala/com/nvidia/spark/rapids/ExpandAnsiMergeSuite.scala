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

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum}
import org.apache.spark.sql.types.{IntegerType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class ExpandAnsiMergeSuite extends AnyFunSuite {
  test("ANSI branch buffer reuse applies preMerge and preserves null sums") {
    val key = AttributeReference("k", IntegerType)()
    val value = AttributeReference("v", LongType)()
    val sum = GpuBasicSum(value, LongType, failOnErrorOverride = true)
    val expression = GpuAggregateExpression(sum, Partial, isDistinct = false,
      filter = None, NamedExpression.newExprId)
    val conf = new SQLConf
    val helper = new AggHelper(Seq(key, value), Seq(key), Seq(expression),
      forceMerge = false, conf = conf, metrics = Map.empty)
    val merge = new AggHelper(Seq(key, value), Seq(key), Seq(expression),
      forceMerge = true, conf = conf, metrics = Map.empty)
    val attributes = Seq(key) ++ sum.aggBufferAttributes
    val identity = GpuBindReferences.bindGpuReferencesTiered(
      attributes, attributes, conf, Map.empty)
    val branch = ExpandAggregateBranch(helper, identity,
      Some(ExpandAggregateMerge(merge, identity, Set(0))))
    withResource(GpuColumnVector.from(ColumnVector.fromInts(1, 1, 2), IntegerType)) { keys =>
      withResource(GpuColumnVector.from(
          ColumnVector.fromBoxedLongs(10L, 20L, null), LongType)) { sums =>
        withResource(new ColumnarBatch(Array(keys.incRefCount(), sums.incRefCount()), 3)) {
          buffers =>
            withResource(branch(buffers, mergeBuffers = true)) { result =>
              val keyColumn = result.column(0).asInstanceOf[GpuColumnVector].getBase
              val valueColumn = result.column(1).asInstanceOf[GpuColumnVector].getBase
              withResource(keyColumn.copyToHost()) { k =>
                withResource(valueColumn.copyToHost()) { v =>
                  val observed = (0 until result.numRows()).map { i =>
                    k.getInt(i) -> (if (v.isNull(i)) None else Some(v.getLong(i)))
                  }.toMap
                  assert(observed == Map(1 -> Some(30L), 2 -> None))
                }
              }
            }
        }
      }
    }
  }
}
