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

import ai.rapids.cudf.{ColumnVector, GroupByAggregation, Rmm, Scalar, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{GpuRetryOOM, GpuSplitAndRetryOOM, RmmSpark}
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.aggregate.{CudfAggregate, GpuBasicSum, GpuCount,
  GpuDecimal128Sum}
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.rapids.metrics.source.MockTaskContext
import org.apache.spark.sql.types.{DataType, DecimalType, IntegerType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object ExpandAggregateRetrySuite {
  case class FailOnce(child: Expression, failure: RuntimeException) extends GpuUnaryExpression {
    private var failed = false
    override def dataType: DataType = child.dataType
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

class ExpandAggregateRetrySuite extends RmmSparkRetrySuiteBase {
  import ExpandAggregateRetrySuite.FailOnce

  private val rows = 100
  private val metrics = GpuHashAggregateMetrics(NoopMetric, NoopMetric, NoopMetric, NoopMetric,
    NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric,
    NoopMetric, NoopMetric)

  private def batch(rowCount: Int = rows, allNull: Boolean = false,
      uniqueKeys: Boolean = false): ColumnarBatch = {
    val keys = (0 until rowCount).map(i =>
      if (uniqueKeys) Int.box(i) else if (i % 5 == 0) null else Int.box(i % 4))
    val values = (0 until rowCount).map(i =>
      if (allNull || i % 6 == 0) null else Long.box(i.toLong - 30))
    withResource(new Table.TestBuilder().column(keys: _*).column(values: _*)
        .column((0 until rowCount).map(Int.box): _*).build()) { table =>
      GpuColumnVector.from(table, Array[DataType](IntegerType, LongType, IntegerType))
    }
  }

  private def branches: Seq[ExpandAggregateBranch] = {
    val (agg, expand) = ExpandAggregateFusionSuite.fixture()
    GpuExpandAggregateExec.bind(agg, expand.projections, expand.output, expand.child.output,
      new SQLConf(), Map.empty)
  }

  private type Result = (Option[Int], Long, Long, Option[Long], Option[Long], Option[Long], Long)

  private def read(batch: ColumnarBatch): Seq[Result] = {
    withResource((0 until batch.numCols()).safeMap(i =>
        batch.column(i).asInstanceOf[GpuColumnVector].copyToHost())) { columns =>
      (0 until batch.numRows()).map { i =>
        def long(c: Int): Option[Long] = if (columns(c).isNullAt(i)) None
          else Some(columns(c).getLong(i))
        (if (columns.head.isNullAt(i)) None else Some(columns.head.getInt(i)),
          columns(1).getLong(i), columns(2).getLong(i), long(3), long(4), long(5),
          columns(6).getLong(i))
      }
    }
  }

  private def expected(allNull: Boolean, uniqueKeys: Boolean = false): Map[(Option[Int], Long),
      (Long, Option[Long], Option[Long], Option[Long], Long)] = {
    (0 until rows).flatMap { i =>
      val key = if (uniqueKeys) Some(i) else if (i % 5 == 0) None else Some(i % 4)
      Seq((key, 0L, i), (None, 1L, i))
    }.groupBy { case (key, gid, _) => (key, gid) }.map { case (key, group) =>
      val values = group.map(_._3).filter(i => !allNull && i % 6 != 0).map(_.toLong - 30)
      (key, (values.length.toLong, if (values.isEmpty) None else Some(values.sum),
        if (values.isEmpty) None else Some(values.min),
        if (values.isEmpty) None else Some(values.max), group.length.toLong))
    }
  }

  private def assertResults(results: Seq[Result], allNull: Boolean = false,
      uniqueKeys: Boolean = false): Unit = {
    val combined = results.groupBy(r => (r._1, r._2)).map { case (key, pieces) =>
      val sums = pieces.flatMap(_._4)
      val mins = pieces.flatMap(_._5)
      val maxs = pieces.flatMap(_._6)
      (key, (pieces.map(_._3).sum, if (sums.isEmpty) None else Some(sums.sum),
        if (mins.isEmpty) None else Some(mins.min),
        if (maxs.isEmpty) None else Some(maxs.max), pieces.map(_._7).sum))
    }
    assert(combined == expected(allNull, uniqueKeys))
  }

  private def assertReleased(): Unit = {
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
  }

  override def afterEach(): Unit = {
    RmmSpark.getAndResetNumRetryThrow(1)
    RmmSpark.getAndResetNumSplitRetryThrow(1)
    super.afterEach()
  }

  for (fault <- Seq("none", "retry", "split"); afterFirst <- Seq(false, true)) {
    test(s"branch retry commits each partial once: $fault, afterFirst=$afterFirst") {
      val inputRows = new LocalGpuMetric
      val inputBatches = new LocalGpuMetric
      val branchBatches = new LocalGpuMetric
      val reused = new LocalGpuMetric
      withResource(new GpuExpandAggregateIterator(Iterator(batch()), branches, metrics,
          new LocalGpuMetric, inputRows, inputBatches, branchBatches, reused)) { iter =>
        val results = ArrayBuffer[Result]()
        var successfulBatches = 0
        if (afterFirst) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
          }
          successfulBatches += 1
        }
        fault match {
          case "retry" => RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case "split" => RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case _ =>
        }
        while (iter.hasNext) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
          }
          successfulBatches += 1
        }
        assertResults(results.toSeq)
        assert(inputRows.value == rows)
        assert(inputBatches.value == 1)
        assert(branchBatches.value == successfulBatches)
        assert(iter.hasUniqueKeys == (fault != "split"))
        if (fault == "split" && !afterFirst) assert(reused.value == 0)
        else assert(reused.value > 0)
        assertReleased()
      }
    }
  }

  for (fault <- Seq("none", "retry", "split"); allNull <- Seq(false, true)) {
    test(s"decimal and count buffers survive branch retries: $fault, allNull=$allNull") {
      val (agg, expand) = ExpandAggregateFusionSuite.fixture()
      val value = expand.child.output(1)
      val decimal = DecimalType(38, 0)
      val functions = Seq(
        GpuDecimal128Sum(GpuCast(value, decimal)(), decimal, failOnErrorOverride = false,
          forceWindowSumToNotBeReplaced = false),
        GpuCount(Seq(value), failOnError = false))
      val expressions = agg.aggregateExpressions.take(2).zip(functions).map { case (e, fn) =>
        e.copy(origAggregateFunction = fn)
      }
      val buffers = functions.flatMap(_.aggBufferAttributes)
      val modified = agg.copy(aggregateExpressions = expressions, aggregateAttributes = buffers,
        resultExpressions = agg.groupingExpressions ++ buffers)
      assert(GpuExpandAggregateExec.eligible(modified, expand))
      val bound = GpuExpandAggregateExec.bind(modified, expand.projections, expand.output,
        expand.child.output, new SQLConf(), Map.empty)
      assert(bound.forall(b => b.merge.isEmpty && !b.cacheResult))
      val reused = new LocalGpuMetric
      withResource(new GpuExpandAggregateIterator(Iterator(batch(allNull = allNull)), bound,
          metrics, new LocalGpuMetric, reusedBranchBatches = reused)) { iter =>
        fault match {
          case "retry" => RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case "split" => RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case _ =>
        }
        val results = ArrayBuffer[(Option[Int], Long, Option[Long], Boolean, Long)]()
        while (iter.hasNext) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch()) { cb =>
              withResource((0 until cb.numCols()).safeMap(i =>
                  cb.column(i).asInstanceOf[GpuColumnVector].copyToHost())) { columns =>
                (0 until cb.numRows()).foreach { i =>
                  val key = if (columns.head.isNullAt(i)) None else Some(columns.head.getInt(i))
                  val sum = if (columns(3).getBoolean(i) || columns(2).isNullAt(i)) None
                    else Some(columns(2).getDecimal(i, 38, 0).toJavaBigDecimal.longValueExact())
                  results += ((key, columns(1).getLong(i), sum, columns(3).getBoolean(i),
                    columns(4).getLong(i)))
                }
              }
            }
          }
        }
        val actual = results.groupBy(r => (r._1, r._2)).map { case (key, pieces) =>
          val sums = pieces.flatMap(_._3)
          (key, (if (sums.isEmpty) None else Some(sums.sum), pieces.forall(_._4),
            pieces.map(_._5).sum))
        }
        val expectedValues = expected(allNull).map { case (key, (count, sum, _, _, _)) =>
          (key, (sum, count == 0, count))
        }
        assert(actual == expectedValues)
        assert(reused.value == 0)
        assertReleased()
      }
    }
  }

  for (split <- Seq(false, true)) {
    test(s"late restore OOM recomputes only the remaining branch, split=$split") {
      val original = branches
      val branch = original(1)
      val tiers = branch.restore.exprTiers
      val failure = if (split) new GpuSplitAndRetryOOM("late branch split")
        else new GpuRetryOOM("late branch retry")
      val late = branch.copy(restore = GpuTieredProject(tiers.init :+
        tiers.last.updated(2, FailOnce(tiers.last(2), failure))))
      withResource(new GpuExpandAggregateIterator(Iterator(batch()), Seq(original.head, late),
          metrics, new LocalGpuMetric)) { iter =>
        val results = ArrayBuffer[Result]()
        while (iter.hasNext) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
          }
        }
        assertResults(results.toSeq)
        assertReleased()
      }
    }
  }

  for (split <- Seq(false, true); reuse <- Seq(false, true)) {
    test(s"late reduction failure releases columns: split=$split, reuse=$reuse") {
      val initialBytes = Rmm.getTotalBytesAllocated
      val bound = branches
      val original = if (reuse) bound else bound.updated(0, bound.head.copy(cacheResult = false))
      val helper = if (reuse) original(1).merge.get.helper else original(1).helper
      val delegate = helper.cudfAggregates(1)
      var failureThrown = false
      helper.cudfAggregates(1) = new CudfAggregate {
        private var failed = false
        override val name: String = delegate.name
        override def dataType: DataType = delegate.dataType
        override val groupByAggregate: GroupByAggregation = delegate.groupByAggregate
        override val reductionAggregate: ColumnVector => Scalar = column => {
          if (!failed) {
            failed = true
            failureThrown = true
            if (split) throw new GpuSplitAndRetryOOM("late reduction split")
            else throw new GpuRetryOOM("late reduction retry")
          }
          delegate.reductionAggregate(column)
        }
      }
      withResource(new GpuExpandAggregateIterator(Iterator(batch()), original, metrics,
          new LocalGpuMetric)) { iter =>
        val results = ArrayBuffer[Result]()
        while (iter.hasNext) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
          }
        }
        assertResults(results.toSeq)
        assert(failureThrown)
        assertReleased()
        assert(Rmm.getTotalBytesAllocated == initialBytes)
      }
    }
  }

  for (uniqueKeys <- Seq(false, true)) {
    test(s"partial buffer reuse requires observed compression: uniqueKeys=$uniqueKeys") {
      val reused = new LocalGpuMetric
      val sourceRows = new LocalGpuMetric
      withResource(new GpuExpandAggregateIterator(Iterator(batch(uniqueKeys = uniqueKeys)),
          branches, metrics, new LocalGpuMetric, reusedBranchBatches = reused,
          branchInputRows = sourceRows)) { iter =>
        val results = ArrayBuffer[Result]()
        while (iter.hasNext) {
          withResource(iter.next()) { sb =>
            withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
          }
        }
        assertResults(results.toSeq, uniqueKeys = uniqueKeys)
        assert(reused.value == (if (uniqueKeys) 0 else 1))
        assert(sourceRows.value == (if (uniqueKeys) rows * 2 else rows + 5))
        assert(iter.hasUniqueKeys)
        assertReleased()
      }
    }
  }

  for ((inputCount, split) <- Seq((1, false), (2, false), (1, true))) {
    test(s"merge bypass requires one unsplit input: inputs=$inputCount, split=$split") {
      val (agg, expand) = ExpandAggregateFusionSuite.fixture()
      val aggregateOps = new LocalGpuMetric
      val firstBatches = new LocalGpuMetric
      val currentMetrics = metrics.copy(numAggOps = aggregateOps)
      val input = Iterator.fill(inputCount)(batch())
      withResource(new GpuExpandAggregateIterator(input, branches, currentMetrics,
          new LocalGpuMetric, branchBatches = firstBatches)) { first =>
        assert(!first.hasUniqueKeys)
        assert(first.hasNext)
        if (split) {
          RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
        }
        withResource(new CloseableBufferedIterator(first)) { buffered =>
          withResource(new GpuMergeAggregateIterator(buffered, expand.output,
              agg.groupingExpressions, agg.aggregateExpressions, agg.aggregateAttributes,
              agg.resultExpressions, AggregateModeInfo(Seq(
                org.apache.spark.sql.catalyst.expressions.aggregate.Partial)), currentMetrics,
              agg.configuredTargetBatchSize, new SQLConf(), false, 0.85, new LocalGpuMetric,
              Map.empty, () => first.hasUniqueKeys)) { merge =>
            val results = ArrayBuffer[Result]()
            while (merge.hasNext) withResource(merge.next())(cb => results ++= read(cb))
            assert(results.map(r => (r._1, r._2)).distinct.length == results.length)
            val values = expected(allNull = false).map { case (key, (count, sum, min, max, all)) =>
              (key, (count * inputCount, sum.map(_ * inputCount), min, max, all * inputCount))
            }
            assert(results.map(r => ((r._1, r._2),
              (r._3, r._4, r._5, r._6, r._7))).toMap == values)
            val unique = inputCount == 1 && !split
            assert(first.hasUniqueKeys == unique)
            assert(aggregateOps.value == firstBatches.value + (if (unique) 0 else 1))
          }
        }
      }
      assertReleased()
    }
  }

  test("original spillable input survives between branches and is restored") {
    withResource(new GpuExpandAggregateIterator(Iterator(batch()), branches, metrics,
        new LocalGpuMetric)) { iter =>
      val results = ArrayBuffer[Result]()
      assert(iter.hasNext)
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
      withResource(iter.next()) { sb =>
        withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
      }
      // Materialization may leave the original cached on host rather than device.
      SpillFramework.stores.deviceStore.spill(Long.MaxValue)
      assert(SpillFramework.stores.hostStore.numHandles > 0)
      while (iter.hasNext) {
        withResource(iter.next()) { sb =>
          withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
        }
      }
      assertResults(results.toSeq)
      assertReleased()
    }
  }

  test("aggregate references to expanded keys use each branch's substituted NULL") {
    val (agg, expand) = ExpandAggregateFusionSuite.fixture()
    val key = expand.output(3)
    val functions = Seq(GpuCount(Seq(key), failOnError = false),
      GpuBasicSum(key, LongType, failOnErrorOverride = false))
    val expressions = agg.aggregateExpressions.take(2).zip(functions).map { case (e, fn) =>
      e.copy(origAggregateFunction = fn)
    }
    val buffers = functions.flatMap(_.aggBufferAttributes)
    val modified = agg.copy(aggregateExpressions = expressions, aggregateAttributes = buffers,
      resultExpressions = agg.groupingExpressions ++ buffers)
    assert(GpuExpandAggregateExec.eligible(modified, expand))
    val bound = GpuExpandAggregateExec.bind(modified, expand.projections, expand.output,
      expand.child.output, new SQLConf(), Map.empty)
    assert(bound(1).merge.isEmpty)
    withResource(new GpuExpandAggregateIterator(Iterator(batch()), bound, metrics,
        new LocalGpuMetric)) { iter =>
      while (iter.hasNext) {
        withResource(iter.next()) { sb =>
          withResource(sb.getColumnarBatch()) { cb =>
            withResource((0 until cb.numCols()).safeMap(i =>
                cb.column(i).asInstanceOf[GpuColumnVector].copyToHost())) { columns =>
              (0 until cb.numRows()).foreach { i =>
                if (columns(1).getLong(i) == 1L || columns.head.isNullAt(i)) {
                  assert(columns(2).getLong(i) == 0L)
                  assert(columns(3).isNullAt(i))
                } else {
                  val k = columns.head.getInt(i)
                  val count = (0 until rows).count(r => r % 5 != 0 && r % 4 == k)
                  assert(columns(2).getLong(i) == count)
                  assert(columns(3).getLong(i) == count.toLong * k)
                }
              }
            }
          }
        }
      }
      assertReleased()
    }
  }

  test("all-null measures retain count zero and null sum/min/max") {
    withResource(new GpuExpandAggregateIterator(Iterator(batch(allNull = true)), branches,
        metrics, new LocalGpuMetric)) { iter =>
      val results = ArrayBuffer[Result]()
      while (iter.hasNext) {
        withResource(iter.next()) { sb =>
          withResource(sb.getColumnarBatch())(cb => results ++= read(cb))
        }
      }
      assertResults(results.toSeq, allNull = true)
      assertReleased()
    }
  }

  test("zero-row batches and empty partitions never invoke branch reductions") {
    for (input <- Seq(Iterator.empty, Iterator(batch(0)))) {
      withResource(new GpuExpandAggregateIterator(input, branches, metrics,
          new LocalGpuMetric)) { iter =>
        assert(!iter.hasNext)
        intercept[NoSuchElementException](iter.next())
        assertReleased()
      }
    }
  }

  for (consumeFirst <- Seq(false, true)) {
    test(s"early close releases original and abandoned split pieces: $consumeFirst") {
      withResource(new GpuExpandAggregateIterator(Iterator(batch()), branches, metrics,
          new LocalGpuMetric)) { iter =>
        assert(iter.hasNext)
        if (consumeFirst) {
          RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          withResource(iter.next())(_ => ())
          assert(iter.hasNext)
        }
      }
      assertReleased()
    }
  }

  for (consumeFirst <- Seq(false, true)) {
    test(s"task completion releases retained input and split attempts: $consumeFirst") {
      val context = new MockTaskContext(taskAttemptId = 1, partitionId = 0)
      TrampolineUtil.setTaskContext(context)
      try {
        withResource(new GpuExpandAggregateIterator(Iterator(batch()), branches, metrics,
            new LocalGpuMetric)) { iter =>
          assert(iter.hasNext)
          if (consumeFirst) {
            RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
              RmmSpark.OomInjectionType.GPU.ordinal, 0)
            // Standalone RMM has no SparkEnv; callbacks were already registered above.
            TrampolineUtil.unsetTaskContext()
            try {
              withResource(iter.next())(_ => ())
            } finally {
              TrampolineUtil.setTaskContext(context)
            }
            assert(iter.hasNext)
          }
          context.markTaskComplete()
          assert(!iter.hasNext)
          assertReleased()
        }
      } finally {
        TrampolineUtil.unsetTaskContext()
        ScalableTaskCompletion.reset()
      }
    }
  }

  test("unexpected late failure closes allocations and does not hide the exception") {
    val original = branches
    val branch = original(1)
    val tiers = branch.restore.exprTiers
    val failure = new IllegalStateException("branch failure")
    val bad = branch.copy(restore = GpuTieredProject(tiers.init :+
      tiers.last.updated(2, FailOnce(tiers.last(2), failure))))
    withResource(new GpuExpandAggregateIterator(Iterator(batch()), Seq(original.head, bad),
        metrics, new LocalGpuMetric)) { iter =>
      withResource(iter.next())(_ => ())
      assert(intercept[IllegalStateException](iter.next()) eq failure)
      assertReleased()
    }
  }
}
