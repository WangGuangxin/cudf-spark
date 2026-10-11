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

import java.io.File

import ai.rapids.cudf.{BinaryOp, ColumnVector, Cuda, DType, Rmm, RmmAllocationMode, Scalar}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuBasicSum, GpuCount}
import org.apache.spark.sql.types.{IntegerType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/** Manual partial-stage benchmark, including Expand coalescing and the existing merge path. */
object ExpandAggregateFusionBenchmark {
  def main(args: Array[String]): Unit = {
    require(args.length == 5 || args.length == 6,
      "Arguments: enabled branches cardinality inputColumns rmmLogFile [inputBatches]")
    val enabled = args(0).toBoolean
    val branchCount = args(1).toInt
    val cardinality = args(2).toInt
    val width = args(3).toInt
    val inputBatches = if (args.length == 6) args(5).toInt else 1
    val rows = 128 * 1024
    val iterations = 20
    val warmups = 30
    val dimensions = branchCount - 1
    val targetSize = 1024L * 1024 * 1024
    require(branchCount >= 2 && branchCount <= 8 && width > dimensions &&
      cardinality > 0 && cardinality <= rows && rows % cardinality == 0 && inputBatches > 0)
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT,
      if (args(4) == "-") null else Rmm.logTo(new File(args(4))), 0)
    try {
      SpillFramework.initialize(new RapidsConf(new SparkConf()))
      val conf = new SQLConf()
      val attrs = (0 until width).map(i => AttributeReference(s"c$i",
        if (i == dimensions) LongType else IntegerType)())
      val columns = attrs.zipWithIndex.safeMap { case (attr, i) =>
        if (i == dimensions) {
          withResource(Scalar.fromLong(1L)) { one =>
            GpuColumnVector.from(ColumnVector.fromScalar(one, rows), LongType)
          }
        } else {
          withResource(Scalar.fromInt(0)) { start =>
            withResource(ColumnVector.sequence(start, rows)) { sequence =>
              withResource(Scalar.fromInt(cardinality)) { modulus =>
                GpuColumnVector.from(sequence.binaryOp(BinaryOp.MOD, modulus, DType.INT32),
                  attr.dataType)
              }
            }
          }
        }
      }
      withResource(new ColumnarBatch(columns.toArray, rows)) { batch =>
        val keys = (0 until dimensions).map(i => AttributeReference(s"k$i", IntegerType)())
        val gid = AttributeReference("gid", LongType, nullable = false)()
        val output = Seq(attrs(dimensions)) ++ keys :+ gid
        val projections = (0 until branchCount).map { branch =>
          Seq(attrs(dimensions)) ++ keys.indices.map { i =>
            if (i < dimensions - branch) attrs(i) else GpuLiteral(null, IntegerType)
          } :+ GpuLiteral(branch.toLong, LongType)
        }
        val functions = Seq(GpuCount(Seq(attrs(dimensions)), failOnError = false),
          GpuBasicSum(attrs(dimensions), LongType, failOnErrorOverride = false))
        val expressions = functions.map(fn => GpuAggregateExpression(fn, Partial,
          isDistinct = false, None, NamedExpression.newExprId))
        val groups = keys :+ gid
        val buffers = functions.flatMap(_.aggBufferAttributes)
        val expand = GpuExpandExec(projections, output,
          ExpandAggregateFusionSuite.TestLeaf(attrs))()
        val agg = GpuHashAggregateExec(None, groups, expressions, buffers, groups ++ buffers,
          expand, targetSize, 1.0, forceSinglePassAgg = false, allowSinglePassAgg = false,
          allowNonFullyAggregatedOutput = false, skipAggPassReductionRatio = 0.85)
        require(GpuExpandAggregateExec.eligible(agg, expand))
        val branches = GpuExpandAggregateExec.bind(agg, projections, output, attrs, conf, Map.empty)
        val boundExpand = projections.map(p => GpuBindReferences.bindGpuReferencesTiered(
          p, attrs, conf, Map.empty))
        val mode = AggregateModeInfo(Seq(Partial))
        val finalRefs = GpuAggFinalPassIterator.setupReferences(groups, expressions, buffers,
          groups ++ buffers, mode, Map.empty)
        val boundGroup = GpuBindReferences.bindGpuReferencesTiered(groups, output, conf, Map.empty)
        val metrics = GpuHashAggregateMetrics(NoopMetric, NoopMetric, NoopMetric, NoopMetric,
          NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric,
          NoopMetric, NoopMetric)
        val expandMetrics = Seq(NUM_INPUT_ROWS, NUM_INPUT_BATCHES, NUM_OUTPUT_ROWS,
          NUM_OUTPUT_BATCHES, OP_TIME_LEGACY).map(_ -> NoopMetric).toMap[String, GpuMetric]
        def consume(iter: Iterator[ColumnarBatch]): Unit = {
          var outputRows = 0
          while (iter.hasNext) {
            withResource(iter.next())(cb => outputRows += cb.numRows())
          }
          require(outputRows == dimensions * cardinality + 1)
          Cuda.DEFAULT_STREAM.sync()
          require(SpillFramework.stores.deviceStore.numHandles == 0)
        }
        var branchRows = 0L
        var reusedBatches = 0L
        def run(): Unit = {
          val input = Iterator.fill(inputBatches)(GpuColumnVector.incRefCounts(batch))
          if (enabled) {
            val expandedRows = new LocalGpuMetric
            val sourceRows = new LocalGpuMetric
            val reused = new LocalGpuMetric
            // Mirrors GpuExpandAggregateExec.childrenCoalesceGoal: the fused node coalesces
            // the unexpanded input instead of Expand's expanded output.
            val coalescedInput = new GpuCoalesceIterator(input,
              attrs.map(_.dataType).toArray, TargetSize(targetSize), NoopMetric, NoopMetric,
              NoopMetric, NoopMetric, NoopMetric, NoopMetric, NoopMetric,
              "expandAggregateBenchmark")
            withResource(new GpuExpandAggregateIterator(coalescedInput, branches, metrics,
                expandedRows, reusedBranchBatches = reused,
                branchInputRows = sourceRows)) { first =>
              withResource(new CloseableBufferedIterator(first)) { buffered =>
                withResource(new GpuMergeAggregateIterator(buffered, output, groups,
                    expressions, buffers, groups ++ buffers, mode, metrics, targetSize,
                    conf, false, 0.85, expandedRows, Map.empty,
                    () => first.hasUniqueKeys)) { merge =>
                  consume(GpuAggFinalPassIterator.makeIter(merge, finalRefs, metrics))
                }
              }
            }
            branchRows = sourceRows.value
            reusedBatches = reused.value
          } else {
            val expanded = new GpuExpandIterator(boundExpand, expandMetrics, input)
            val coalesced = new GpuCoalesceIterator(expanded, output.map(_.dataType).toArray,
              TargetSize(targetSize), NoopMetric, NoopMetric, NoopMetric, NoopMetric,
              NoopMetric, NoopMetric, NoopMetric, "expandAggregateBenchmark")
            consume(new DynamicGpuPartialAggregateIterator(coalesced, output, groups,
              boundGroup, expressions, buffers, groups ++ buffers, mode, 1.0,
              alreadySorted = false, Seq.empty, finalRefs, targetSize, metrics, conf,
              forceSinglePassAgg = false, allowSinglePassAgg = false,
              allowNonFullyAggregatedOutput = false, skipAggPassReductionRatio = 0.85,
              allMetrics = Map.empty))
          }
        }
        (0 until warmups).foreach(_ => run())
        Rmm.resetScopedMaximumBytesAllocated()
        Cuda.profilerStart()
        try {
          val times = (0 until iterations).map { _ =>
            val start = System.nanoTime()
            run()
            (System.nanoTime() - start) / 1e6
          }.sorted
          println(s"enabled=$enabled branches=$branchCount cardinality=$cardinality " +
            s"width=$width rows=$rows inputBatches=$inputBatches iterations=$iterations " +
            s"medianMs=${times(iterations / 2)} " +
            s"peakExtraBytes=${Rmm.getScopedMaximumBytesAllocated} " +
            s"branchInputRows=$branchRows reusedBranchBatches=$reusedBatches")
        } finally {
          Cuda.profilerStop()
        }
      }
    } finally {
      SpillFramework.shutdown()
      Rmm.shutdown()
    }
  }
}
