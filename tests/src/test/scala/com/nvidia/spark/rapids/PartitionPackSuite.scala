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

import java.math.{BigInteger, RoundingMode}

import ai.rapids.cudf.{ColumnVector, Cuda, Rmm, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.jni.{PartitionPack, RmmSpark}
import com.nvidia.spark.rapids.shims.GpuHashPartitioning

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{BoundReference, ExprId, HiveHash, Murmur3Hash}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.types.{ArrayType, BooleanType, DataType, DecimalType, IntegerType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String

class PartitionPackSuite extends RmmSparkRetrySuiteBase {
  override def beforeEach(): Unit = {
    super.beforeEach()
    SparkSession.builder().master("local[1]").appName(getClass.getSimpleName).getOrCreate()
  }

  override def afterEach(): Unit = {
    GpuShuffleEnv.shutdown()
    super.afterEach()
  }

  private val rows = 97 // Cross several mask words and leave a partial final word.
  private val strings = Array.tabulate[String](rows) { i =>
    if (i % 7 == 0) null else if (i % 5 == 0) "" else s"中文🙂-$i"
  }
  private val types: Array[DataType] = Array(IntegerType, StringType, StringType,
    BooleanType, DecimalType(38, 3))

  private def buildBatch(): ColumnarBatch = {
    withResource(new Table.TestBuilder()
      .column(Array.tabulate[java.lang.Integer](rows)(i => i): _*)
      .column(strings: _*)
      .column(Array.fill[String](rows)(null): _*)
      .column(Array.tabulate[java.lang.Boolean](rows) { i =>
        if (i % 11 == 0) null else java.lang.Boolean.valueOf(i % 2 == 0)
      }: _*)
      .decimal128Column(-3, RoundingMode.UNNECESSARY,
        Array.tabulate[BigInteger](rows) { i =>
          if (i % 13 == 0) null else new BigInteger(s"123456789012345678901234$i")
        }: _*)
      .build()) { table => GpuColumnVector.from(table, types) }
  }

  private def partitioner(enabled: Boolean, partitions: Int, emptyKeys: Boolean,
      mode: HashMode = Murmur3Mode, expectFusion: Boolean = true,
      gpu: Boolean = true): GpuHashPartitioning = {
    val conf = new SQLConf()
    conf.setConfString(RapidsConf.SHUFFLE_PARTITION_PACK_ENABLED.key, enabled.toString)
    conf.setConfString(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, "none")
    SQLConf.withExistingConf(conf) {
      GpuShuffleEnv.init(new RapidsConf(conf))
      val keys = if (emptyKeys) {
        Seq.empty
      } else {
        Seq(GpuBoundReference(1, StringType, nullable = true)(ExprId(1), "key"))
      }
      new GpuHashPartitioning(keys, partitions, mode) {
        override def usesGPUShuffle: Boolean = gpu
        override def usesKudoGPUSlicing: Boolean = false
        override def partitionInternalAndClose(
            batch: ColumnarBatch): (Array[Int], Array[GpuColumnVector]) = {
          assert(!enabled || !expectFusion,
            "The enabled, supported payload must take the native path")
          super.partitionInternalAndClose(batch)
        }
      }
    }
  }

  private def checkOutput(input: ColumnarBatch, output: Array[(ColumnarBatch, Int)],
      partitions: Int, emptyKeys: Boolean, mode: HashMode): Unit = {
    val cpuKeys = if (emptyKeys) Seq.empty else Seq(BoundReference(0, StringType, true))
    val cpuHash = mode match {
      case Murmur3Mode => Murmur3Hash(cpuKeys, GpuHashPartitioningBase.DEFAULT_HASH_SEED)
      case HiveMode => HiveHash(cpuKeys)
      case other => fail(s"Unexpected mode $other")
    }
    val seen = scala.collection.mutable.ArrayBuffer[Int]()
    assert(output.map(_._2).sameElements(0 until partitions))
    withResource(GpuColumnVector.from(input)) { original =>
      output.foreach { case (batch, partition) =>
        assert(GpuPackedTableColumn.isBatchPacked(batch))
        val actual = batch.column(0).asInstanceOf[GpuPackedTableColumn]
          .getContiguousTable.getTable
        withResource(actual.getColumn(0).copyToHost()) { ids =>
          val indices = Array.tabulate(batch.numRows())(i => ids.getInt(i))
          indices.foreach { id =>
            seen.append(id)
            val value = if (strings(id) == null) null else UTF8String.fromString(strings(id))
            assert(Math.floorMod(cpuHash.eval(InternalRow(value)).asInstanceOf[Int],
              partitions) == partition)
          }
          withResource(ColumnVector.fromInts(indices: _*)) { map =>
            withResource(original.gather(map)) { expected =>
              for (c <- types.indices) {
                val column = actual.getColumn(c)
                assert(column.getType == expected.getColumn(c).getType)
                assert(column.getNullCount == expected.getColumn(c).getNullCount)
                if (batch.numRows() > 0) {
                  withResource(expected.getColumn(c).equalToNullAware(column)) { equal =>
                    withResource(equal.all()) { all => assert(all.getBoolean) }
                  }
                }
              }
            }
          }
        }
      }
    }
    assert(seen.sorted == (0 until rows))
  }

  for (enabled <- Seq(false, true); partitions <- Seq(1, 2, 32, 128);
       emptyKeys <- Seq(false, true); mode <- Seq(Murmur3Mode, HiveMode)) {
    test(s"partition pack enabled=$enabled partitions=$partitions empty=$emptyKeys mode=$mode") {
      val hp = partitioner(enabled, partitions, emptyKeys, mode)
      withResource(buildBatch()) { input =>
        val output = hp.columnarEvalAny(GpuColumnVector.incRefCounts(input))
          .asInstanceOf[Array[(ColumnarBatch, Int)]]
        withResource(output.map(_._1)) { _ =>
          checkOutput(input, output, partitions, emptyKeys, mode)
        }
      }
    }
  }

  test("zero rows produce no shuffle batches") {
    val hp = partitioner(true, 4, emptyKeys = false)
    withResource(ColumnVector.fromStrings()) { column =>
      withResource(new Table(column)) { table =>
        val batch = GpuColumnVector.from(table, Array[DataType](StringType))
        assert(hp.columnarEvalAny(batch).asInstanceOf[Array[(ColumnarBatch, Int)]].isEmpty)
      }
    }
  }

  test("nested payloads use the existing packed-table path") {
    val hp = partitioner(true, 4, emptyKeys = true, expectFusion = false)
    withResource(ColumnVector.fromLists(
        new ai.rapids.cudf.HostColumnVector.ListType(true,
          new ai.rapids.cudf.HostColumnVector.BasicType(true, ai.rapids.cudf.DType.INT32)),
        java.util.Arrays.asList[Integer](1, 2))) { column =>
      withResource(new Table(column)) { table =>
        assert(!PartitionPack.isSupported(table))
        val input = GpuColumnVector.from(table, Array[DataType](ArrayType(IntegerType)))
        val output = hp.columnarEvalAny(input).asInstanceOf[Array[(ColumnarBatch, Int)]]
        withResource(output.map(_._1)) { _ =>
          assert(output.map(_._1.numRows()).sum == 1)
          val batch = output(Math.floorMod(GpuHashPartitioningBase.DEFAULT_HASH_SEED, 4))._1
          val actual = batch.column(0).asInstanceOf[GpuPackedTableColumn]
            .getContiguousTable.getTable
          withResource(actual.getColumn(0).copyToHost()) { host =>
            assert(host.getList(0) == java.util.Arrays.asList[Integer](1, 2))
          }
        }
      }
    }
  }

  test("the switch preserves the CPU shuffle endpoint") {
    val hp = partitioner(true, 4, emptyKeys = false, expectFusion = false, gpu = false)
    val output = hp.columnarEvalAny(buildBatch()).asInstanceOf[Array[(ColumnarBatch, Int)]]
    withResource(output.map(_._1)) { _ =>
      assert(output.map(_._1.numRows()).sum == rows)
      assert(output.forall(_._1.column(0).isInstanceOf[SlicedGpuColumnVector]))
    }
  }

  test("native allocation failures retry without losing rows or leaking packed buffers") {
    val hp = partitioner(true, 32, emptyKeys = false)
    // Delay injection until after hash and control allocations, including partial output buffers.
    for (skip <- Seq(0, 8, 20)) {
      val before = Rmm.getTotalBytesAllocated
      withResource(buildBatch()) { input =>
        withResource(SpillableColumnarBatch(GpuColumnVector.incRefCounts(input),
            SpillPriorities.ACTIVE_BATCHING_PRIORITY)) { spillable =>
          RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, skip)
          val output = withRetryNoSplit(spillable) { attempt =>
            hp.columnarEvalAny(attempt.getColumnarBatch())
              .asInstanceOf[Array[(ColumnarBatch, Int)]]
          }
          withResource(output.map(_._1)) { _ =>
            checkOutput(input, output, 32, emptyKeys = false, Murmur3Mode)
          }
        }
      }
      Cuda.DEFAULT_STREAM.sync()
      assert(Rmm.getTotalBytesAllocated == before)
    }
  }
}
