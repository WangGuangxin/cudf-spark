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

import ai.rapids.cudf.{BinaryOp, ColumnVector, Cuda, DType, Rmm, RmmAllocationMode, Scalar, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._

/**
 * Cost attribution for fusing a filter into an explode, with no plan rule and no Spark.
 *
 * Eager  : explode the payload with the list, then filter. Every payload column is replicated
 *          k times and the rows the predicate rejects are thrown away afterwards.
 * Fused  : explode only a row index with the list, filter the index, then gather the payload
 *          once. Replication costs 4 bytes per output row instead of the payload width.
 *
 * Both paths produce the same rows. Input construction and correctness checks stay outside the
 * measured ranges.
 */
object GenerateFilterCostBenchmark {

  private def intSequence(rows: Int): ColumnVector =
    withResource(Scalar.fromInt(0))(ColumnVector.sequence(_, rows))

  /** Values spread over [0, 1000) so a threshold gives a predictable, non-contiguous survival. */
  private def values(total: Int): ColumnVector =
    withResource(intSequence(total)) { seq =>
      withResource(Scalar.fromInt(1000)) { modulus =>
        seq.binaryOp(BinaryOp.MOD, modulus, DType.INT32)
      }
    }

  private def listColumn(rows: Int, k: Int): ColumnVector = {
    withResource(values(rows * k)) { child =>
      withResource(intSequence(rows + 1)) { seq =>
        withResource(Scalar.fromInt(k)) { width =>
          withResource(seq.binaryOp(BinaryOp.MUL, width, DType.INT32)) { offsets =>
            child.makeListFromOffsets(rows.toLong, offsets)
          }
        }
      }
    }
  }

  private def payloadColumns(rows: Int, longs: Int, strings: Boolean): Array[ColumnVector] = {
    val numeric = (0 until longs).safeMap { i =>
      withResource(intSequence(rows)) { seq =>
        withResource(Scalar.fromInt(i + 3)) { factor =>
          seq.binaryOp(BinaryOp.MUL, factor, DType.INT64)
        }
      }
    }
    closeOnExcept(numeric) { _ =>
      if (!strings) numeric.toArray
      else (numeric :+ withResource(intSequence(rows))(_.castTo(DType.STRING))).toArray
    }
  }

  /** Replicate every payload column, then discard the rejected rows. */
  private def eager(payload: Array[ColumnVector], list: ColumnVector, threshold: Int): Long = {
    val genOffset = payload.length
    withResource(new Table(payload :+ list: _*)) { input =>
      withResource(input.explode(genOffset)) { exploded =>
        withResource(Scalar.fromInt(threshold)) { limit =>
          withResource(exploded.getColumn(genOffset).lessThan(limit)) { mask =>
            withResource(exploded.filter(mask))(_.getRowCount)
          }
        }
      }
    }
  }

  /** Replicate a row index, discard rejected rows, then gather the payload once. */
  private def fused(payload: Array[ColumnVector], list: ColumnVector, threshold: Int,
      rows: Int): Long = {
    withResource(intSequence(rows)) { index =>
      withResource(new Table(index, list)) { input =>
        withResource(input.explode(1)) { exploded =>
          withResource(Scalar.fromInt(threshold)) { limit =>
            withResource(exploded.getColumn(1).lessThan(limit)) { mask =>
              withResource(exploded.filter(mask)) { survivors =>
                withResource(new Table(payload: _*)) { payloadTable =>
                  withResource(payloadTable.gather(survivors.getColumn(0))) { gathered =>
                    val columns = (0 until gathered.getNumberOfColumns)
                      .map(gathered.getColumn) :+ survivors.getColumn(1)
                    withResource(new Table(columns: _*))(_.getRowCount)
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 6,
      "Arguments: explodedRows k selectivity payloadLongs strings iterations")
    val explodedRows = args(0).toInt
    val k = args(1).toInt
    val selectivity = args(2).toDouble
    val payloadLongs = args(3).toInt
    val strings = args(4).toBoolean
    val iterations = args(5).toInt
    val rows = math.max(1, explodedRows / k)
    val threshold = math.max(1, (selectivity * 1000).toInt)

    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 0L)
    try {
      withResource(listColumn(rows, k)) { list =>
        withResource(payloadColumns(rows, payloadLongs, strings)) { payload =>
          val columns = payload.toArray
          val eagerRows = eager(columns, list, threshold)
          val fusedRows = fused(columns, list, threshold, rows)
          require(eagerRows == fusedRows, s"row mismatch $eagerRows vs $fusedRows")

          def measure(run: => Long): Double = {
            (0 until 3).foreach { _ => run; Cuda.DEFAULT_STREAM.sync() }
            val times = (0 until iterations).map { _ =>
              val start = System.nanoTime()
              run
              Cuda.DEFAULT_STREAM.sync()
              (System.nanoTime() - start) / 1e6
            }.sorted
            times(times.length / 2)
          }

          // Interleave so any drift hits both paths equally.
          val eagerTimes = new scala.collection.mutable.ArrayBuffer[Double]()
          val fusedTimes = new scala.collection.mutable.ArrayBuffer[Double]()
          (0 until 3).foreach { _ =>
            eagerTimes += measure(eager(columns, list, threshold))
            fusedTimes += measure(fused(columns, list, threshold, rows))
          }
          val e = eagerTimes.sorted.apply(1)
          val f = fusedTimes.sorted.apply(1)
          // scalastyle:off println
          println(f"RESULT explodedRows=$explodedRows k=$k sel=$selectivity%.2f " +
            f"payloadLongs=$payloadLongs strings=$strings inRows=$rows outRows=$eagerRows " +
            f"eagerMs=$e%.2f fusedMs=$f%.2f speedup=${e / f}%.2fx " +
            f"delta=${(f - e) / e * 100}%+.1f%%")
          // scalastyle:on println
        }
      }
    } finally {
      Rmm.shutdown()
    }
  }
}
