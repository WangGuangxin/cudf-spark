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

import ai.rapids.cudf.{ColumnVector, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.GpuRetryOOM
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.rapids.GpuAdd
import org.apache.spark.sql.rapids.execution.{HashJoinIterator, JoinBuildSideSelection, JoinBuildSideStats, JoinOptions, JoinStrategy}
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class HashJoinIteratorSuite extends RmmSparkRetrySuiteBase {
  private val attr = AttributeReference("key", IntegerType, nullable = false)()
  private val ref = GpuBoundReference(0, IntegerType, nullable = false)(attr.exprId, attr.name)
  private val options = JoinOptions(JoinStrategy.AUTO, JoinBuildSideSelection.FIXED,
    targetSize = 1024, logCardinalityEnabled = false, sizeEstimateThreshold = 0.75)

  private case class CountingKey(retryProjection: Boolean) extends GpuLeafExpression {
    var evaluations = 0
    override def dataType = IntegerType
    override def nullable: Boolean = false

    override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
      evaluations += 1
      if (retryProjection && evaluations == 1) {
        throw new GpuRetryOOM("retry build key projection")
      }
      GpuAdd(ref, GpuLiteral(1, IntegerType), failOnError = false)().columnarEval(batch)
    }
  }

  private def newBatch(values: Int*): LazySpillableColumnarBatch = {
    withResource(ColumnVector.fromInts(values: _*)) { column =>
      withResource(new Table(column)) { table =>
        withResource(GpuColumnVector.from(table, Array[DataType](IntegerType))) { batch =>
          LazySpillableColumnarBatch(batch, "test_join_data")
        }
      }
    }
  }

  private def checkJoin(
      stats: Option[JoinBuildSideStats] = None,
      retryProjection: Boolean = false,
      retryJoin: Boolean = false,
      spillBetweenBatches: Boolean = false,
      targetSize: Long = 1024): Unit = {
    val key = CountingKey(retryProjection)
    withResource(newBatch(1, 2, 3)) { built =>
      withResource(Seq(newBatch(2, 3, 4), newBatch(2, 3, 4))) { stream =>
        var joinAttempts = 0
        val iterator = new HashJoinIterator(built, Seq(key), stats, stream.iterator, Seq(ref),
          Seq(attr), options.copy(targetSize = targetSize), Inner, GpuBuildLeft,
          compareNullsEqual = false, None, NoopMetric, NoopMetric) {
          override protected def joinGathererLeftRight(
              leftKeys: Table,
              leftData: LazySpillableColumnarBatch,
              rightKeys: Table,
              rightData: LazySpillableColumnarBatch): Option[JoinGatherer] = {
            joinAttempts += 1
            if (retryJoin && joinAttempts == 1) {
              throw new GpuRetryOOM("retry join after projecting build keys")
            }
            super.joinGathererLeftRight(leftKeys, leftData, rightKeys, rightData)
          }
        }
        withResource(iterator) { iter =>
          val results = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
          while (iter.hasNext) {
            withResource(iter.next()) { batch =>
              withResource(batch.column(0).asInstanceOf[GpuColumnVector].getBase.copyToHost()) {
                left =>
                  withResource(batch.column(1).asInstanceOf[GpuColumnVector]
                      .getBase.copyToHost()) { right =>
                    (0 until batch.numRows()).foreach { row =>
                      results += ((left.getInt(row), right.getInt(row)))
                    }
                  }
              }
            }
            if (spillBetweenBatches && results.size == 3) {
              assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
            }
          }
          assert(results.sorted.toSeq == Seq((1, 2), (1, 2), (2, 3), (2, 3), (3, 4), (3, 4)))
          assert(key.evaluations == (if (retryProjection) 2 else 1))
          val expectedAttempts = if (targetSize == 16) 4 else 2
          assert(joinAttempts == expectedAttempts + (if (retryJoin) 1 else 0))
          // Exhaustion closes the iterator-owned batches; a second close must also be safe.
          intercept[IllegalStateException] { built.getBatch }
          assert(SpillFramework.stores.deviceStore.numHandles == 0)
          assert(SpillFramework.stores.hostStore.numHandles == 0)
          assert(SpillFramework.stores.diskStore.numHandles == 0)
        }
      }
    }
  }

  test("reuse projected build keys for statistics and multiple stream batches") {
    checkJoin()
  }

  test("reuse projected build keys with precomputed statistics") {
    checkJoin(stats = Some(JoinBuildSideStats(1.0, isDistinct = true)))
  }

  test("retry failed build key projection before caching") {
    checkJoin(retryProjection = true)
  }

  test("reuse projected build keys after a join retry") {
    checkJoin(retryJoin = true)
  }

  test("reuse projected build keys across split stream batches") {
    checkJoin(targetSize = 16)
  }

  test("reuse projected build keys after spilling between stream batches") {
    checkJoin(spillBetweenBatches = true)
  }
}
