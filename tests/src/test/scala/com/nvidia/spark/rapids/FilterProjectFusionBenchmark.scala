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

import ai.rapids.cudf.{ColumnVector, Cuda, Rmm, RmmAllocationMode, Scalar}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuAdd, GpuGreaterThan}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

/** Manual operator microbenchmark; excludes scan, scheduling, and spill/retry overhead. */
object FilterProjectFusionBenchmark {
  def main(args: Array[String]): Unit = {
    require(args.length == 5,
      "Arguments: enabled passPercent inputColumns arithmeticDepth rmmLogFile")
    val enabled = args(0).toBoolean
    val passPercent = args(1).toInt
    val width = args(2).toInt
    val depth = args(3).toInt
    val rows = 1024 * 1024
    val iterations = 20
    require(width >= 2 && passPercent >= 0 && passPercent <= 100 && depth >= 1)
    val allocationLog = if (args(4) == "-") null else Rmm.logTo(new File(args(4)))
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, allocationLog, 0)
    try {
      val attrs = (0 until width).map(i => AttributeReference(s"c$i", IntegerType)())
      val columns = attrs.safeMap { _ =>
        withResource(Scalar.fromInt(0)) { start =>
          GpuColumnVector.from(ColumnVector.sequence(start, rows), IntegerType)
        }
      }
      withResource(new ColumnarBatch(columns.toArray, rows)) { input =>
        val condition = GpuGreaterThan(attrs.head,
          GpuLiteral(rows - rows * passPercent / 100 - 1, IntegerType))
        val arithmetic = (0 until depth).foldLeft[Expression](attrs(1)) { (expr, _) =>
          GpuAdd(expr, GpuLiteral(1, IntegerType), failOnError = false)()
        }
        val projects = Seq(attrs.head, GpuAlias(arithmetic, "computed")())
        val conf = new SQLConf()
        val predicate = GpuBindReferences.bindGpuReferencesTiered(
          Seq(condition), attrs, conf, Map.empty)
        val project = GpuBindReferences.bindGpuReferencesTiered(projects, attrs, conf, Map.empty)
        withResource(new GpuFilterProject(condition, projects, attrs, false,
            conf, Map.empty)) { fused =>
          def run(): Unit = {
            val result = if (enabled) {
              fused(input)
            } else {
              withResource(GpuFilter(input, predicate)) { filtered => project.project(filtered) }
            }
            withResource(result) { cb =>
              require(cb.numRows() == rows * passPercent / 100)
              Cuda.DEFAULT_STREAM.sync()
            }
          }
          (0 until 5).foreach(_ => run())
          Rmm.resetScopedMaximumBytesAllocated()
          Cuda.profilerStart()
          try {
            val times = (0 until iterations).map { _ =>
              val start = System.nanoTime()
              run()
              (System.nanoTime() - start) / 1e6
            }.sorted
            println(s"enabled=$enabled passPercent=$passPercent width=$width depth=$depth " +
              s"rows=$rows iterations=$iterations medianMs=${times(iterations / 2)} " +
              s"peakExtraBytes=${Rmm.getScopedMaximumBytesAllocated}")
          } finally {
            Cuda.profilerStop()
          }
        }
      }
    } finally {
      Rmm.shutdown()
    }
  }
}
