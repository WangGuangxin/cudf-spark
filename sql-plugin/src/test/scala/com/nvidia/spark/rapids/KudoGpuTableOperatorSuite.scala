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

import org.scalatest.funsuite.AnyFunSuite

class KudoGpuTableOperatorSuite extends AnyFunSuite {
  private val Chunk = KudoGpuTableOperator.H2D_CHUNK_BYTES

  /** Start offsets of tables of the given sizes, with the total as the last element. */
  private def starts(sizes: Long*): Array[Long] = sizes.scanLeft(0L)(_ + _).toArray

  private def assertCovers(ranges: Seq[(Int, Int)], numTables: Int): Unit = {
    assert(ranges.head._1 == 0)
    assert(ranges.last._2 == numTables)
    ranges.zip(ranges.drop(1)).foreach { case (a, b) => assert(a._2 == b._1) }
    ranges.foreach { case (from, until) => assert(until > from) }
  }

  test("small concatenations are one range") {
    assert(KudoGpuTableOperator.splitRanges(starts(10, 20, 30)) == Seq((0, 3)))
    assert(KudoGpuTableOperator.splitRanges(starts(Chunk - 1)) == Seq((0, 1)))
  }

  test("large concatenations split into at most four ranges of about equal bytes") {
    val r = KudoGpuTableOperator.splitRanges(starts(Seq.fill(64)(Chunk / 4): _*))
    assert(r == Seq((0, 16), (16, 32), (32, 48), (48, 64)))
    val two = KudoGpuTableOperator.splitRanges(starts(Seq.fill(8)(Chunk / 4): _*))
    assert(two == Seq((0, 4), (4, 8)))
  }

  test("ranges cover every table once even when sizes are uneven") {
    Seq(
      Seq(10 * Chunk, 1L, 1L),
      Seq(1L, 1L, 10 * Chunk),
      Seq(3 * Chunk, 0L, 0L, 3 * Chunk),
      Seq(5 * Chunk)).foreach { sizes =>
      val r = KudoGpuTableOperator.splitRanges(starts(sizes: _*))
      assertCovers(r, sizes.length)
      assert(r.length <= 4)
    }
  }

  test("no tables gives one empty range") {
    assert(KudoGpuTableOperator.splitRanges(Array(0L)) == Seq((0, 0)))
  }
}
