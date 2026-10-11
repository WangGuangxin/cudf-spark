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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, OutputStream}
import java.time.Instant

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

import ai.rapids.cudf.{ColumnVector, Cuda, Rmm, RmmAllocationMode}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsConf.ShuffleKudoMode
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.kudo.DumpOption
import com.nvidia.spark.rapids.shims.GpuHashPartitioning
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, BoundReference}
import org.apache.spark.sql.catalyst.expressions.{Murmur3Hash, NamedExpression, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuLessThan, GpuShuffleEnv}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum, GpuCount}
import org.apache.spark.sql.types.{IntegerType, LongType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Resident-input cost attribution, not a fusion implementation or a shuffle-writer benchmark.
 * Uses the existing FilterProject, coalescing, full partial-aggregate and shuffle primitives.
 * CPU/GPU Kudo include serialization into a counting sink; packed buffers remain on device.
 * Correctness checks and input construction run outside the measured ranges.
 */
object OperatorFusionCostBenchmark {
  private val TotalRows = 128 * 1024
  private val TargetBytes = 1024L * 1024 * 1024

  private case class Case(id: Int, kind: String, survival: Int, cardinality: Int,
      strings: Boolean, batches: Int, partitions: Int, path: String, phases: Set[String])

  private class CountingSink extends OutputStream {
    var bytes: Long = 0
    override def write(value: Int): Unit = { bytes += 1 }
    override def write(values: Array[Byte], offset: Int, length: Int): Unit = { bytes += length }
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 4 || args.length == 5,
      "Arguments: casesCsv iterations rmmLogFile captureProfiler [ASYNC|NONE]")
    val iterations = args(1).toInt
    val capture = args(3).toBoolean
    val allocator = if (args.length == 5) args(4) else "ASYNC"
    require(Set("ASYNC", "NONE").contains(allocator))
    require(iterations > 0)
    val cases = withResource(Source.fromFile(args(0))) { source =>
      source.getLines().filter(line => line.nonEmpty && !line.startsWith("#"))
        .zipWithIndex.map { case (line, id) =>
          val fields = line.split(",")
          require(fields.length == 8)
          Case(id, fields(0), fields(1).toInt, fields(2).toInt, fields(3).toBoolean,
            fields(4).toInt, fields(5).toInt, fields(6), fields(7).split(":").toSet)
        }.toArray
    }
    val conf = new SparkConf().set(RapidsConf.SHUFFLE_COMPRESSION_CODEC.key, "none")
    val spark = SparkSession.builder().master("local[1]").appName(getClass.getSimpleName)
      .config(conf).getOrCreate()
    val allocationMode = if (allocator == "ASYNC") RmmAllocationMode.CUDA_ASYNC
      else RmmAllocationMode.CUDA_DEFAULT
    val poolBytes = if (allocator == "ASYNC") 512L * 1024 * 1024 else 0L
    Rmm.initialize(allocationMode,
      if (args(2) == "-") null else Rmm.logTo(new File(args(2))), poolBytes)
    println(s"ENV allocator=$allocator totalRows=$TotalRows warmup=5 iterations=$iterations")
    try {
      SpillFramework.initialize(new RapidsConf(conf))
      GpuShuffleEnv.init(new RapidsConf(conf))
      cases.foreach(c => runCase(c, iterations, capture))
    } finally {
      GpuShuffleEnv.shutdown()
      SpillFramework.shutdown()
      Rmm.shutdown()
      spark.stop()
    }
  }

  private def runCase(c: Case, iterations: Int, capture: Boolean): Unit = {
    require(Set("filter", "shuffle").contains(c.kind))
    require(c.survival >= 0 && c.survival <= 100 && c.cardinality > 0 &&
      c.cardinality <= TotalRows && c.batches > 0 && TotalRows % c.batches == 0 &&
      c.partitions > 0 && Set("cpu", "gpu", "packed").contains(c.path))
    val keyType = if (c.strings) StringType else IntegerType
    val attrs = Seq(AttributeReference("key", keyType)(),
      AttributeReference("value", LongType)(), AttributeReference("selector", IntegerType)())
    val aggInput = attrs.take(2)
    val sqlConf = new SQLConf()
    val random = new Random(42)
    val selectors = Array.fill(TotalRows)(random.nextInt(10000))
    val threshold = c.survival * 100
    val keyStrings = if (c.strings) {
      Array.tabulate(c.cardinality)(i => f"$i%032d")
    } else {
      Array.empty[String]
    }
    val expected = mutable.Map.empty[String, (Long, Long)]
    (0 until TotalRows).foreach { row =>
      if (c.kind == "shuffle" || selectors(row) < threshold) {
        val key = (row % c.cardinality).toString
        val (count, sum) = expected.getOrElse(key, (0L, 0L))
        expected(key) = (count + 1, sum + row % 17)
      }
    }
    val inputs = (0 until c.batches).safeMap { batchIndex =>
      val size = TotalRows / c.batches
      val start = batchIndex * size
      val columns = Seq(0, 1, 2).safeMap { column =>
        val vector = column match {
          case 0 if c.strings =>
            ColumnVector.fromStrings(Array.tabulate(size)(i =>
              keyStrings((start + i) % c.cardinality)): _*)
          case 0 => ColumnVector.fromInts(Array.tabulate(size)(i =>
            (start + i) % c.cardinality): _*)
          case 1 => ColumnVector.fromLongs(Array.tabulate(size)(i =>
            ((start + i) % 17).toLong): _*)
          case 2 => ColumnVector.fromInts(selectors.slice(start, start + size): _*)
        }
        GpuColumnVector.from(vector, attrs(column).dataType)
      }
      new ColumnarBatch(columns.toArray, size)
    }
    withResource(inputs) { _ =>
      val condition = GpuLessThan(attrs(2), GpuLiteral(threshold, IntegerType))
      withResource(new GpuFilterProject(condition, aggInput, attrs, false,
          sqlConf, Map.empty)) { filter =>
        val projection = GpuBindReferences.bindGpuReferencesTiered(aggInput, attrs,
          sqlConf, Map.empty)
        def projected(): Iterator[ColumnarBatch] = inputs.iterator.map { batch =>
          if (c.kind == "filter") filter(batch) else projection.project(batch)
        }
        def coalesced(): Iterator[ColumnarBatch] = new GpuCoalesceIterator(projected(),
          aggInput.map(_.dataType).toArray, TargetSize(TargetBytes), NoopMetric, NoopMetric,
          NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, "fusionCostBenchmark")
        val groups = Seq(attrs.head)
        val functions = Seq(GpuCount(Seq(attrs(1)), failOnError = false),
          GpuBasicSum(attrs(1), LongType, failOnErrorOverride = false))
        val expressions = functions.map(fn => GpuAggregateExpression(fn, Partial,
          isDistinct = false, None, NamedExpression.newExprId))
        val buffers = functions.flatMap(_.aggBufferAttributes)
        require(buffers.length == 2)
        val output = groups ++ buffers
        val mode = AggregateModeInfo(Seq(Partial))
        val finalRefs = GpuAggFinalPassIterator.setupReferences(groups, expressions, buffers,
          output, mode, Map.empty)
        val boundGroups = GpuBindReferences.bindGpuReferencesTiered(groups, aggInput,
          sqlConf, Map.empty)
        val helper = new AggHelper(aggInput, groups, expressions, forceMerge = false,
          conf = sqlConf, metrics = Map.empty)
        val estimatedInput = aggInput.map(a =>
          GpuBatchUtils.estimateGpuMemory(a.dataType, a.nullable, TotalRows)).sum
        val estimatedPreProject = helper.preStepBound.outputExprs.map(e =>
          GpuBatchUtils.estimateGpuMemory(e.dataType, true, TotalRows)).sum
        val preProjectGrowth = estimatedPreProject.toDouble / estimatedInput
        val aggConf = new RapidsConf(sqlConf)
        val allowSinglePass = aggConf.allowSinglePassPartialSortAgg && preProjectGrowth > 1.1
        val singlePassTasks = new LocalGpuMetric
        val metrics = GpuHashAggregateMetrics(NoopMetric, NoopMetric, NoopMetric, NoopMetric,
          NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric,
          singlePassTasks, NoopMetric)
        def aggregate(input: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = {
          new DynamicGpuPartialAggregateIterator(input, aggInput, groups, boundGroups,
            expressions, buffers, output, mode, preProjectGrowth, alreadySorted = false,
            groups.map(SortOrder(_, Ascending)),
            finalRefs, TargetBytes, metrics, sqlConf, forceSinglePassAgg = false,
            allowSinglePassAgg = allowSinglePass, allowNonFullyAggregatedOutput = true,
            skipAggPassReductionRatio = 0.85, allMetrics = Map.empty)
        }
        def retain(iter: Iterator[ColumnarBatch]): ArrayBuffer[ColumnarBatch] = {
          closeOnExcept(new ArrayBuffer[ColumnarBatch]) { batches =>
            while (iter.hasNext) batches += iter.next()
            batches
          }
        }
        def consume(iter: Iterator[ColumnarBatch]): Unit = {
          while (iter.hasNext) withResource(iter.next())(_ => ())
        }
        def measure(phase: String)(run: => Unit): Unit = {
          if (c.phases.contains(phase)) {
            (0 until 5).foreach { _ => run; Cuda.DEFAULT_STREAM.sync() }
            Rmm.resetScopedMaximumBytesAllocated()
            if (capture) Cuda.profilerStart()
            val rangeStart = Instant.now()
            val times = try {
              (0 until iterations).map { _ =>
                val start = System.nanoTime()
                run
                Cuda.DEFAULT_STREAM.sync()
                (System.nanoTime() - start) / 1e6
              }.sorted
            } finally {
              if (capture) Cuda.profilerStop()
            }
            val rangeEnd = Instant.now()
            def epochMicros(time: Instant): Long =
              time.getEpochSecond * 1000000L + time.getNano / 1000
            println(s"RANGE case=${c.id} phase=$phase " +
              s"startEpochUs=${epochMicros(rangeStart)} endEpochUs=${epochMicros(rangeEnd)}")
            println(s"RESULT case=${c.id} kind=${c.kind} survival=${c.survival} " +
              s"cardinality=${c.cardinality} strings=${c.strings} batches=${c.batches} " +
              s"partitions=${c.partitions} path=${c.path} phase=$phase " +
              s"iterations=$iterations medianMs=${times(iterations / 2)} " +
              s"minMs=${times.head} maxMs=${times.last} " +
              s"peakExtraBytes=${Rmm.getScopedMaximumBytesAllocated}")
          }
        }
        withResource(retain(coalesced())) { selected =>
          withResource(retain(aggregate(selected.iterator.map(GpuColumnVector.incRefCounts)))) {
            partial =>
              validate(partial.toSeq, c.strings, expected.toMap)
              val selectedRows = selected.map(_.numRows().toLong).sum
              val outputRows = partial.map(_.numRows().toLong).sum
              val outputBytes = partial.map(GpuColumnVector.getTotalDeviceMemoryUsed).sum
              println(s"GEOMETRY case=${c.id} inputRows=$TotalRows selectedRows=$selectedRows " +
                s"outputRows=$outputRows outputBytes=$outputBytes outputBatches=${partial.size} " +
                s"preProjectGrowth=$preProjectGrowth allowSinglePass=$allowSinglePass " +
                s"singlePassTasks=${singlePassTasks.value}")
              if (c.kind == "filter") {
                if (c.phases.contains("predicate") || c.phases.contains("gather")) {
                  val predicate = GpuBindReferences.bindGpuReferencesTiered(Seq(condition),
                    attrs, sqlConf, Map.empty)
                  measure("predicate") {
                    inputs.foreach { batch =>
                      withResource(GpuFilter.computeCheckedFilterMask(predicate, batch))(_ => ())
                    }
                  }
                  withResource(inputs.safeMap { batch =>
                    withResource(predicate.project(batch)) { mask =>
                      mask.column(0).asInstanceOf[GpuColumnVector].getBase.incRefCount()
                    }
                  }) { masks =>
                    measure("gather") {
                      inputs.zip(masks).foreach { case (batch, mask) =>
                        withResource(projection.project(batch)) { projected =>
                          if (c.survival < 100) {
                            withResource(GpuColumnVector.from(projected)) { table =>
                              withResource(table.filter(mask))(_ => ())
                            }
                          }
                        }
                      }
                    }
                  }
                }
                measure("filter") { consume(coalesced()) }
                measure("agg") {
                  consume(aggregate(selected.iterator.map(GpuColumnVector.incRefCounts)))
                }
                measure("pipeline") { consume(aggregate(coalesced())) }
              } else {
                val hash = GpuHashPartitioning(Seq(GpuBoundReference(0, keyType, true)(
                  attrs.head.exprId, attrs.head.name)),
                  c.partitions)
                val slicer = new GpuPartitioning {
                  override val numPartitions: Int = c.partitions
                  override def usesGPUShuffle: Boolean = c.path == "packed"
                  override def usesKudoGPUSlicing: Boolean = c.path == "gpu"
                  override def usesMultiThreadedShuffle: Boolean = false
                }
                val serializer = new GpuColumnarBatchSerializer(
                  Map.empty[String, GpuMetric].withDefaultValue(NoopMetric),
                  output.map(_.dataType).toArray,
                  if (c.path == "gpu") ShuffleKudoMode.GPU else ShuffleKudoMode.CPU,
                  useKudo = true, kudoMeasureBufferCopy = false).newInstance()
                def pack(indexes: Array[Int], batch: ColumnarBatch,
                    check: Boolean): Unit = {
                  val columns = GpuColumnVector.extractColumns(
                    GpuColumnVector.incRefCounts(batch))
                  val splits = slicer.sliceInternalGpuOrCpuAndClose(batch.numRows(),
                    indexes, columns)
                  withResource(splits.map(_._1)) { _ =>
                    var payloadBytes = 0L
                    val recovered = new ArrayBuffer[ColumnarBatch]
                    withResource(recovered) { _ =>
                      splits.foreach { case (split, partitionId) =>
                        val firstRecovered = recovered.size
                        if (c.path == "packed") {
                          require(GpuPackedTableColumn.isBatchPacked(split))
                          if (check) {
                            payloadBytes += split.column(0).asInstanceOf[GpuPackedTableColumn]
                              .getContiguousTable.getBuffer.getLength
                          }
                          if (check && split.numRows() > 0) {
                            val table = split.column(0).asInstanceOf[GpuPackedTableColumn]
                              .getContiguousTable.getTable
                            recovered += GpuColumnVector.from(table, output.map(_.dataType).toArray)
                          }
                        } else if (check) {
                          withResource(new ByteArrayOutputStream()) { sink =>
                            withResource(serializer.serializeStream(sink)) { stream =>
                              stream.writeValue(split)
                            }
                            payloadBytes += sink.size()
                            withResource(serializer.deserializeStream(
                                new ByteArrayInputStream(sink.toByteArray))) { stream =>
                              val serialized = stream.asKeyValueIterator.map(_._2)
                                .asInstanceOf[Iterator[ColumnarBatch]]
                              val readOption = CoalesceReadOption(kudoEnabled = true,
                                kudoMode = ShuffleKudoMode.GPU, kudoDebugMode = DumpOption.Never,
                                kudoDebugDumpPrefix = None, useAsync = false)
                              val decoded = GpuShuffleCoalesceUtils.getGpuShuffleCoalesceIterator(
                                serialized, TargetBytes, output.map(_.dataType).toArray,
                                readOption,
                                Map.empty[String, GpuMetric].withDefaultValue(NoopMetric))
                              while (decoded.hasNext) recovered += decoded.next()
                            }
                          }
                        } else {
                          val sink = new CountingSink
                          withResource(serializer.serializeStream(sink)) { stream =>
                            stream.writeValue(split)
                          }
                          require(sink.bytes > 0)
                        }
                        if (check) {
                          val cpuHash = Murmur3Hash(Seq(BoundReference(0, keyType, true)),
                            GpuHashPartitioningBase.DEFAULT_HASH_SEED)
                          recovered.drop(firstRecovered).foreach { cb =>
                            withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) {
                              keys =>
                                (0 until cb.numRows()).foreach { row =>
                                  val key = if (c.strings) {
                                    keys.getUTF8String(row)
                                  } else {
                                    keys.getInt(row)
                                  }
                                  val pid = Math.floorMod(
                                    cpuHash.eval(InternalRow(key)).asInstanceOf[Int], c.partitions)
                                  require(pid == partitionId, s"Wrong shuffle partition $pid")
                                }
                            }
                          }
                        }
                      }
                      if (check) {
                        validate(recovered.toSeq, c.strings, expected.toMap)
                        println(s"PACK_GEOMETRY case=${c.id} payloadBytes=$payloadBytes " +
                          s"buffers=${splits.length} " +
                          s"nonEmptyPartitions=${splits.count(_._1.numRows() > 0)}")
                      }
                    }
                  }
                }
                // The target permits one output batch for this fixed resident-input geometry.
                require(partial.size == 1)
                val (indexes, columns) = hash.partitionInternalAndClose(
                  GpuColumnVector.incRefCounts(partial.head))
                withResource(new ColumnarBatch(columns.map(c => c:
                    org.apache.spark.sql.vectorized.ColumnVector), partial.head.numRows())) {
                  partitioned =>
                    pack(indexes, partitioned, check = true)
                    measure("agg") { consume(aggregate(coalesced())) }
                    measure("partition") {
                      partial.foreach { batch =>
                        val (_, cols) = hash.partitionInternalAndClose(
                          GpuColumnVector.incRefCounts(batch))
                        withResource(cols)(_ => ())
                      }
                    }
                    measure("pack") { pack(indexes, partitioned, check = false) }
                    def prepare(batch: ColumnarBatch): Unit = {
                      val (parts, cols) = hash.partitionInternalAndClose(batch)
                      withResource(new ColumnarBatch(cols.map(c => c:
                          org.apache.spark.sql.vectorized.ColumnVector), outputRows.toInt)) {
                        cb => pack(parts, cb, check = false)
                      }
                    }
                    measure("prepare") {
                      partial.foreach(b => prepare(GpuColumnVector.incRefCounts(b)))
                    }
                    measure("pipeline") {
                      aggregate(coalesced()).foreach(prepare)
                    }
                }
              }
          }
        }
      }
    }
    require(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  private def validate(batches: Seq[ColumnarBatch], strings: Boolean,
      expected: Map[String, (Long, Long)]): Unit = {
    val actual = mutable.Map.empty[String, (Long, Long)]
    batches.foreach { batch =>
      withResource(GpuColumnVector.extractColumns(batch).safeMap(_.copyToHost())) { columns =>
        (0 until batch.numRows()).foreach { row =>
          val key = if (strings) columns(0).getUTF8String(row).toString.toInt.toString
            else columns(0).getInt(row).toString
          require(!actual.contains(key), s"Duplicate partial key $key")
          actual(key) = (columns(1).getLong(row), columns(2).getLong(row))
        }
      }
    }
    require(actual.toMap == expected,
      s"Partial COUNT/SUM mismatch: ${actual.size}/${expected.size}")
  }
}
