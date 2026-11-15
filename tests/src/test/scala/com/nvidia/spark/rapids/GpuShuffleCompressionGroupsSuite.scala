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

import scala.util.Random

import org.scalatest.funsuite.AnyFunSuite

class GpuShuffleCompressionGroupsSuite extends AnyFunSuite {
  import GpuShuffleFrame.compressionGroups

  test("groups count partition chunk padding and preserve order") {
    val chunk = 64 * 1024
    assert(compressionGroups(Array(65537L, 65536L, 65536L), chunk, 2L * chunk).toList ==
      List(0 until 1, 1 until 3))
  }

  test("empty and raw-only inputs need no compression capacity") {
    assert(!compressionGroups(Array.empty[Long], 65536).hasNext)
    assert(compressionGroups(Array(0L, 1L, 4095L), 65536, 1).toList == List(0 until 3))
    intercept[NoSuchElementException] {
      compressionGroups(Array.empty[Long], 65536).next()
    }
  }

  test("an oversized partition runs alone without dropping the following partition") {
    assert(compressionGroups(Array(65536L, 262144L, 65536L), 65536, 131072).toList ==
      List(0 until 1, 1 until 2, 2 until 3))
  }

  test("groups cover all partitions and bound padded bytes except a single oversized partition") {
    val random = new Random(42)
    val chunk = 65536
    val budget = 4L * 1024 * 1024
    (0 until 200).foreach { _ =>
      val sizes = Array.fill(random.nextInt(100))(random.nextInt(8 * 1024 * 1024).toLong)
      val groups = compressionGroups(sizes, chunk, budget).toList
      assert(groups.flatMap(_.toList) == sizes.indices.toList)
      groups.foreach { group =>
        val padded = group.map { i =>
          if (sizes(i) < GpuShuffleFrame.MIN_COMPRESS_BYTES) 0L
          else ((sizes(i) - 1) / chunk + 1) * chunk
        }.sum
        assert(padded <= budget || group.length == 1)
      }
    }
  }
}
