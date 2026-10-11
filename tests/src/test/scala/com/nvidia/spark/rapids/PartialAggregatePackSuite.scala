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

import ai.rapids.cudf.{Cuda, Rmm, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows, withRetry,
  withRetryNoSplit}
import com.nvidia.spark.rapids.jni.RmmSpark
import com.nvidia.spark.rapids.shims.GpuHashPartitioning

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{BoundReference, Murmur3Hash}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.types.{DataType, IntegerType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class PartialAggregatePackSuite extends RmmSparkRetrySuiteBase {
  override def beforeEach(): Unit = {
    super.beforeEach()
    SparkSession.builder().master("local[1]").appName(getClass.getSimpleName).getOrCreate()
  }

  override def afterEach(): Unit = {
    GpuShuffleEnv.shutdown()
    super.afterEach()
  }

  private val rows = 20003 // Also exercise grid-stride probing and partial mask words.
  private val keys = Array.tabulate[java.lang.Long](rows) { i =>
    if (i % 7 == 0) null else Long.box((i % 211 - 105).toLong)
  }
  private val values = Array.tabulate[java.lang.Long](rows) { i =>
    if (i % 9 == 0 || keys(i) == Long.box(0L)) null
    else Long.box(if (i % 13 == 0) Long.MaxValue else i.toLong - 10000L)
  }
  private val counts = Array.tabulate[java.lang.Long](rows)(i => Long.box(i % 11))
  private val operations = Array(0, 1, 2, 3)

  private def batch(longKey: Boolean): ColumnarBatch = {
    val builder = new Table.TestBuilder()
    if (longKey) builder.column(keys: _*)
    else builder.column(keys.map(k => if (k == null) null else Int.box(k.intValue())): _*)
    withResource(builder.column(values: _*).column(values: _*).column(values: _*)
        .column(counts: _*).build()) { table =>
      GpuColumnVector.from(table, Array[DataType](if (longKey) LongType else IntegerType,
        LongType, LongType, LongType, LongType))
    }
  }

  private def partitioner(partitions: Int, codec: String): GpuHashPartitioning = {
    val conf = new SQLConf()
    conf.setConfString(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, codec)
    SQLConf.withExistingConf(conf) {
      GpuShuffleEnv.init(new RapidsConf(conf))
      new GpuHashPartitioning(Seq.empty, partitions) {
        override def usesGPUShuffle: Boolean = true
        override def usesKudoGPUSlicing: Boolean = false
      }
    }
  }

  private def expected = keys.indices.groupBy(i => Option(keys(i)).map(_.longValue())).map {
      case (key, indices) =>
        val nonNull = indices.flatMap(i => Option(values(i)).map(_.longValue()))
        (key, Seq(if (nonNull.isEmpty) None else Some(nonNull.foldLeft(0L)(_ + _)),
          if (nonNull.isEmpty) None else Some(nonNull.min),
          if (nonNull.isEmpty) None else Some(nonNull.max),
          Some(indices.map(i => counts(i).longValue()).sum)))
    }

  private def readOutput(output: Array[(ColumnarBatch, Int)], partitions: Int,
      longKey: Boolean): Map[Option[Long], Seq[Option[Long]]] = {
    val found = scala.collection.mutable.Map[Option[Long], Seq[Option[Long]]]()
    val cpuHash = Murmur3Hash(Seq(BoundReference(0,
      if (longKey) LongType else IntegerType, nullable = true)), 42)
    output.foreach { case (cb, p) =>
      val table = cb.column(0).asInstanceOf[GpuPackedTableColumn].getContiguousTable.getTable
      withResource((0 until table.getNumberOfColumns)
          .map(c => table.getColumn(c).copyToHost())) { columns =>
        for (row <- 0 until cb.numRows()) {
          val key = if (columns(0).isNull(row)) None
          else Some(if (longKey) columns(0).getLong(row) else columns(0).getInt(row).toLong)
          assert(!found.contains(key), s"Duplicate group $key")
          val value: Any = if (longKey) key.map(Long.box).orNull
          else key.map(k => Int.box(k.toInt)).orNull
          assert(Math.floorMod(cpuHash.eval(InternalRow(value)).asInstanceOf[Int], partitions) == p)
          found(key) = (1 to 4).map { c =>
            if (columns(c).isNull(row)) None else Some(columns(c).getLong(row))
          }
        }
      }
    }
    found.toMap
  }

  private def check(output: Array[(ColumnarBatch, Int)], partitions: Int,
      longKey: Boolean): Unit = assert(readOutput(output, partitions, longKey) == expected)

  for (longKey <- Seq(false, true); partitions <- Seq(1, 2, 32, 256)) {
    test(s"merge and pack key64=$longKey partitions=$partitions") {
      val hp = partitioner(partitions, "none")
      val output = hp.partialAggregateAndPack(batch(longKey), operations)
      withResource(output.map(_._1)) { _ => check(output, partitions, longKey) }
    }
  }

  test("retry after hash state or packed output allocation releases every temporary") {
    val hp = partitioner(32, "none")
    for (skip <- Seq(0, 4, 12, 24)) {
      val before = Rmm.getTotalBytesAllocated
      withResource(SpillableColumnarBatch(batch(longKey = true),
          SpillPriorities.ACTIVE_BATCHING_PRIORITY)) { input =>
        RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
          RmmSpark.OomInjectionType.GPU.ordinal, skip)
        val output = withRetryNoSplit(input) { attempt =>
          hp.partialAggregateAndPack(attempt.getColumnarBatch(), operations)
        }
        withResource(output.map(_._1)) { _ => check(output, 32, longKey = true) }
      }
      Cuda.DEFAULT_STREAM.sync()
      assert(Rmm.getTotalBytesAllocated == before)
    }
  }

  test("split retry emits valid partial states that merge to the same result") {
    val hp = partitioner(32, "none")
    val before = Rmm.getTotalBytesAllocated
    withResource(SpillableColumnarBatch(batch(longKey = true),
        SpillPriorities.ACTIVE_BATCHING_PRIORITY)) { input =>
      RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId)
      val parts = withRetry(input, splitSpillableInHalfByRows) { attempt =>
        hp.partialAggregateAndPack(attempt.getColumnarBatch(), operations)
      }.map { output =>
        withResource(output.map(_._1)) { _ => readOutput(output, 32, longKey = true) }
      }.toVector
      assert(parts.length > 1)
      val combined = parts.flatMap(_.toSeq).groupBy(_._1).map { case (key, entries) =>
        val columns = (0 until operations.length).map { c =>
          val values = entries.flatMap(_._2(c))
          if (values.isEmpty) None
          else Some(c match {
            case 1 => values.min
            case 2 => values.max
            case _ => values.foldLeft(0L)(_ + _)
          })
        }
        (key, columns)
      }
      assert(combined == expected)
    }
    Cuda.DEFAULT_STREAM.sync()
    assert(Rmm.getTotalBytesAllocated == before)
  }

  test("INT64 keys retain their upper bits and extrema in grouping and Spark hashing") {
    val hp = partitioner(32, "none")
    val extremes = Array[java.lang.Long](Long.MinValue, Long.MaxValue, -1L, 0L,
      1L << 32, Long.MinValue, null, null)
    val ones = Array.fill[java.lang.Long](extremes.length)(1L)
    withResource(new Table.TestBuilder().column(extremes: _*)
        .column(ones: _*).column(ones: _*).column(ones: _*).column(ones: _*).build()) {
      table =>
        val input = GpuColumnVector.from(table, Array.fill[DataType](5)(LongType))
        val output = hp.partialAggregateAndPack(input, operations)
        withResource(output.map(_._1)) { _ =>
          val counts = extremes.groupBy(k => Option(k).map(_.longValue())).map {
            case (key, copies) => (key, Seq(Some(copies.length.toLong), Some(1L), Some(1L),
              Some(copies.length.toLong)))
          }
          assert(readOutput(output, 32, longKey = true) == counts)
        }
    }
  }

  test("distinct keys exercise collision probing at the maximum hash-table load") {
    val hp = partitioner(32, "none")
    val unique = Array.tabulate[java.lang.Integer](65536)(i => Int.box(i - 32768))
    val values = unique.map(k => Long.box(k.longValue()))
    val counts = Array.fill[java.lang.Long](unique.length)(1L)
    withResource(new Table.TestBuilder().column(unique: _*).column(values: _*)
        .column(values: _*).column(values: _*).column(counts: _*).build()) { table =>
      val input = GpuColumnVector.from(table, Array[DataType](IntegerType,
        LongType, LongType, LongType, LongType))
      val output = hp.partialAggregateAndPack(input, operations)
      withResource(output.map(_._1)) { _ =>
        val expected = unique.map { key =>
          val value = key.longValue()
          (Some(value), Seq(Some(value), Some(value), Some(value), Some(1L)))
        }.toMap
        assert(readOutput(output, 32, longKey = false) == expected)
      }
    }
  }

  test("empty buffers produce no shuffle output") {
    val hp = partitioner(4, "none")
    withResource(new Table.TestBuilder().column(Array.empty[java.lang.Long]: _*)
        .column(Array.empty[java.lang.Long]: _*).build()) { table =>
      val empty = GpuColumnVector.from(table, Array[DataType](LongType, LongType))
      assert(hp.partialAggregateAndPack(empty, Array(0)).isEmpty)
    }
  }
}
