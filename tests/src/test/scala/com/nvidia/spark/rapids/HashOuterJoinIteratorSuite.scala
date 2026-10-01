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
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.GpuRetryOOM
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.{FullOuter, JoinType, LeftOuter, RightOuter}
import org.apache.spark.sql.rapids.GpuLessThan
import org.apache.spark.sql.rapids.execution.{HashJoinStreamSideIterator, HashOuterJoinIterator, JoinBuildSideSelection, JoinOptions, JoinStrategy, LazyCompiledCondition}
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class HashOuterJoinIteratorSuite extends RmmSparkRetrySuiteBase {
  private case class Row(key: Option[Int], value: Int)
  private val buildRows = Seq(Row(Some(1), 10), Row(Some(1), 20), Row(Some(2), 30),
    Row(Some(3), 40), Row(None, 50))
  private val streamRows = Seq(Row(Some(1), 15), Row(Some(5), 5), Row(None, 0),
    Row(Some(3), 35), Row(Some(6), 6))
  private val attrs = Seq(AttributeReference("key", IntegerType)(),
    AttributeReference("value", IntegerType, nullable = false)())
  private val ref = GpuBoundReference(0, IntegerType, nullable = true)(
    attrs.head.exprId, attrs.head.name)
  private val options = JoinOptions(JoinStrategy.AUTO, JoinBuildSideSelection.FIXED,
    targetSize = 1024, logCardinalityEnabled = false, sizeEstimateThreshold = 0.75)

  private def newBatch(rows: Seq[Row]): LazySpillableColumnarBatch = {
    val keys = rows.map(_.key.map(Int.box).orNull)
    withResource(ColumnVector.fromBoxedInts(keys: _*)) { key =>
      withResource(ColumnVector.fromInts(rows.map(_.value): _*)) { value =>
        withResource(new Table(key, value)) { table =>
          withResource(GpuColumnVector.from(table, Array[DataType](IntegerType, IntegerType))) {
            batch => LazySpillableColumnarBatch(batch, "outer_join_test")
          }
        }
      }
    }
  }

  private def collect(iter: Iterator[ColumnarBatch]): Seq[String] = {
    val rows = scala.collection.mutable.ArrayBuffer[String]()
    iter.foreach { batch =>
      withResource(batch) { _ =>
        withResource(GpuColumnVector.extractBases(batch).toSeq.safeMap(_.copyToHost())) { cols =>
          (0 until batch.numRows()).foreach { row =>
            rows += cols.map(c => if (c.isNull(row)) "null" else c.getInt(row).toString)
              .mkString(",")
          }
        }
      }
    }
    rows.toSeq
  }

  private def expected(joinType: JoinType, side: GpuBuildSide, conditional: Boolean,
      streamed: Seq[Row]): Seq[String] = {
    val (left, right) = if (side == GpuBuildLeft) (buildRows, streamed) else (streamed, buildRows)
    def matches(l: Row, r: Row): Boolean = l.key.nonEmpty && l.key == r.key &&
      (!conditional || l.value < r.value)
    def render(row: Option[Row]): Seq[String] = row.map(r =>
      Seq(r.key.map(_.toString).getOrElse("null"), r.value.toString))
      .getOrElse(Seq("null", "null"))
    val output = scala.collection.mutable.ArrayBuffer[String]()
    left.foreach { l =>
      val hits = right.filter(r => matches(l, r))
      hits.foreach(r => output += (render(Some(l)) ++ render(Some(r))).mkString(","))
      if (hits.isEmpty && (joinType == LeftOuter || joinType == FullOuter)) {
        output += (render(Some(l)) ++ render(None)).mkString(",")
      }
    }
    if (joinType == RightOuter || joinType == FullOuter) {
      right.filter(r => !left.exists(l => matches(l, r))).foreach { r =>
        output += (render(None) ++ render(Some(r))).mkString(",")
      }
    }
    output.toSeq.sorted
  }

  private def condition(enabled: Boolean): Option[LazyCompiledCondition] = {
    if (enabled) {
      val left = GpuBoundReference(1, IntegerType, nullable = false)(
        attrs(1).exprId, "left_value")
      val right = GpuBoundReference(3, IntegerType, nullable = false)(
        attrs(1).exprId, "right_value")
      Some(new LazyCompiledCondition(GpuLessThan(left, right), 2, 2))
    } else {
      None
    }
  }

  private val cases = Seq((FullOuter, GpuBuildLeft), (FullOuter, GpuBuildRight),
    (LeftOuter, GpuBuildLeft), (RightOuter, GpuBuildRight))

  cases.foreach { case (joinType, side) =>
    Seq(false, true).foreach { conditional =>
      test(s"$joinType $side conditional=$conditional preserves unmatched build rows") {
        withResource(newBatch(buildRows)) { built =>
          withResource(streamRows.grouped(3).toSeq.safeMap(newBatch)) { stream =>
            withResource(new HashOuterJoinIterator(joinType, built, Seq(ref), None, None,
                stream.iterator, Seq(ref), attrs, condition(conditional), options, side,
                compareNullsEqual = false, None, NoopMetric, NoopMetric)) { iter =>
              assert(collect(iter).sorted == expected(joinType, side, conditional, streamRows))
            }
          }
        }
        assert(SpillFramework.stores.deviceStore.numHandles == 0)
      }
    }
  }

  test("empty stream with an existing tracker emits every build row") {
    withResource(newBatch(buildRows)) { built =>
      withResource(OuterJoinTracker(buildRows.size)) { tracker =>
        withResource(new HashOuterJoinIterator(FullOuter, built, Seq(ref), None, Some(tracker),
            Iterator.empty, Seq(ref), attrs, None, options, GpuBuildLeft,
            compareNullsEqual = false, None, NoopMetric, NoopMetric)) { iter =>
          assert(collect(iter).sorted == expected(FullOuter, GpuBuildLeft, false, Seq.empty))
        }
      }
    }
  }

  test("join group tracker survives retry, transfer between iterators and spill") {
    withResource(newBatch(buildRows)) { built =>
      val firstRows = streamRows.take(3)
      val lastRows = streamRows.drop(3)
      withResource(newBatch(firstRows)) { first =>
        var attempts = 0
        val (initial, tracker) = withResource(new HashJoinStreamSideIterator(FullOuter,
            LazySpillableColumnarBatch.spillOnly(built), Seq(ref), None, None,
            Iterator(first), Seq(ref), attrs, None, options, GpuBuildLeft,
            compareNullsEqual = false, None, NoopMetric, NoopMetric) {
          override protected def joinGathererLeftRight(
              leftKeys: Table, leftData: LazySpillableColumnarBatch,
              rightKeys: Table, rightData: LazySpillableColumnarBatch): Option[JoinGatherer] = {
            attempts += 1
            val gatherer = super.joinGathererLeftRight(leftKeys, leftData, rightKeys, rightData)
            if (attempts == 1) {
              withResource(gatherer) { _ =>
                throw new GpuRetryOOM("retry after marking outer join matches")
              }
            } else {
              gatherer
            }
          }
        }) { iter =>
          (collect(iter), iter.releaseBuiltSideTracker())
        }
        assert(attempts == 2)
        withResource(tracker) { state =>
          assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
          withResource(newBatch(lastRows)) { last =>
            withResource(new HashOuterJoinIterator(FullOuter,
                LazySpillableColumnarBatch.spillOnly(built), Seq(ref), None, state,
                Iterator(last), Seq(ref), attrs, None, options, GpuBuildLeft,
                compareNullsEqual = false, None, NoopMetric, NoopMetric)) { iter =>
              assert((initial ++ collect(iter)).sorted ==
                expected(FullOuter, GpuBuildLeft, false, streamRows))
            }
          }
        }
      }
    }
  }
}
