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

package org.apache.spark.sql.rapids.execution

import java.util.concurrent.{Callable, CyclicBarrier, Executors, TimeUnit}

import com.nvidia.spark.rapids.{GpuBoundReference, GpuBuildLeft, GpuBuildRight}
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.ExprId
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

class JoinChainPreparationSuite extends AnyFunSuite {
  private def ref(i: Int): GpuBoundReference =
    GpuBoundReference(i, IntegerType, nullable = true)(ExprId(i), s"c$i")

  test("Q21-shaped projections drop fact and orders maps after the orders probe") {
    val stages = Seq(
      JoinChainStage(Seq(ref(0)), Seq(1), Seq(0, 1), GpuBuildLeft, false),
      JoinChainStage(Seq(ref(0)), Seq(2), Seq(1, 2, 3), GpuBuildRight, false),
      JoinChainStage(Seq(ref(0)), Seq(1), Seq(0, 1), GpuBuildRight, false))
    assert(GpuJoinChain.liveSourcesAfterStages(2, Seq(3, 1, 1), stages) ==
      Seq(Set(0, 1), Set(1), Set(1, 3)))
  }

  test("dimension-derived keys preserve a source until its last use") {
    val stages = Seq(
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildRight, false),
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(3), GpuBuildLeft, false),
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(1), GpuBuildRight, false))
    assert(GpuJoinChain.liveSourcesAfterStages(2, Seq(3, 2, 2), stages) ==
      Seq(Set(1), Set(2), Set(2, 3)))
  }

  test("semi stages do not add a payload source and final columns remain live") {
    val stages = Seq(
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildRight, false),
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(3, 1), GpuBuildRight, false,
        isLeftSemi = true),
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildLeft, false))
    assert(GpuJoinChain.liveSourcesAfterStages(2, Seq(3, 2, 2), stages) ==
      Seq(Set(0, 1), Set(0, 1), Set(0, 1, 3)))
  }

  private class TrackedBatch(rows: Int) extends ColumnarBatch(Array.empty, rows) {
    var released = false
    override def close(): Unit = {
      released = true
      super.close()
    }
  }

  test("empty stream batches are released and the first live batch is preserved") {
    val empty = new TrackedBatch(0)
    val live = new TrackedBatch(3)
    withResource(GpuJoinChain.bufferNonemptyStream(Iterator(empty, live))) { stream =>
      assert(empty.released)
      assert(stream.hasNext)
      assert(stream.head eq live)
      assert(!live.released)
    }
    assert(live.released)
    val last = new TrackedBatch(0)
    withResource(GpuJoinChain.bufferNonemptyStream(Iterator(last))) { stream =>
      assert(!stream.hasNext)
    }
    assert(last.released)
  }

  test("looking ahead does not consume the first live batch or close it after transfer") {
    val live = new TrackedBatch(2)
    withResource(GpuJoinChain.bufferNonemptyStream(Iterator(live))) { stream =>
      withResource(stream.next()) { batch =>
        assert(batch eq live)
        assert(!stream.hasNext)
        assert(!live.released)
      }
    }
    assert(live.released)
  }

  test("metadata does not mix column positions or null equality") {
    val cache = new JoinChainBuildKeyStatsCache
    val unequal = JoinChainBuildKeyStats(unique = false, maxMatches = 1)
    val equal = JoinChainBuildKeyStats(unique = false, maxMatches = 3)
    val distinct = JoinChainBuildKeyStats(unique = true, maxMatches = 1)
    assert(cache.getOrCompute(Seq(0, 1), false)(unequal) == unequal)
    assert(cache.getOrCompute(Seq(0, 1), true)(equal) == equal)
    assert(cache.getOrCompute(Seq(1, 0), false)(distinct) == distinct)
    assert(cache.getOrCompute(Seq(0, 1), false) {
      fail("cached classification should not run again")
    } == unequal)
    val otherBroadcast = new JoinChainBuildKeyStatsCache
    assert(otherBroadcast.getOrCompute(Seq(0, 1), false)(distinct) == distinct)
  }

  test("failed classification is not published") {
    val cache = new JoinChainBuildKeyStatsCache
    intercept[IllegalStateException] {
      cache.getOrCompute(Seq(0), false)(throw new IllegalStateException("failed attempt"))
    }
    val valid = JoinChainBuildKeyStats(unique = true, maxMatches = 1)
    assert(cache.getOrCompute(Seq(0), false)(valid) == valid)
    assert(cache.getOrCompute(Seq(0), false)(fail("classification already cached")) == valid)
  }

  test("concurrent cache misses compute outside the monitor") {
    val cache = new JoinChainBuildKeyStatsCache
    val entered = new CyclicBarrier(2)
    val expected = JoinChainBuildKeyStats(unique = false, maxMatches = 3)
    val pool = Executors.newFixedThreadPool(2)
    try {
      val tasks = (0 until 2).map { _ =>
        pool.submit(new Callable[JoinChainBuildKeyStats] {
          override def call(): JoinChainBuildKeyStats = cache.getOrCompute(Seq(0), false) {
            entered.await(5, TimeUnit.SECONDS)
            expected
          }
        })
      }
      tasks.foreach(task => assert(task.get(10, TimeUnit.SECONDS) == expected))
      assert(cache.getOrCompute(Seq(0), false)(fail("result should be published")) == expected)
    } finally {
      pool.shutdownNow()
    }
  }

  test("releasing the broadcast invalidates its metadata") {
    val cache = new JoinChainBuildKeyStatsCache
    val old = JoinChainBuildKeyStats(unique = true, maxMatches = 1)
    val fresh = JoinChainBuildKeyStats(unique = false, maxMatches = 2)
    cache.getOrCompute(Seq(0), false)(old)
    cache.clear()
    assert(cache.getOrCompute(Seq(0), false)(fresh) == fresh)
  }

  test("capacity bounds retained metadata without changing uncached results") {
    val cache = new JoinChainBuildKeyStatsCache(maxEntries = 1)
    val first = JoinChainBuildKeyStats(unique = true, maxMatches = 1)
    val second = JoinChainBuildKeyStats(unique = false, maxMatches = 5)
    cache.getOrCompute(Seq(0), false)(first)
    var attempts = 0
    for (_ <- 0 until 2) {
      assert(cache.getOrCompute(Seq(1), false) {
        attempts += 1
        second
      } == second)
    }
    assert(attempts == 2)
    assert(cache.getOrCompute(Seq(0), false)(fail("first entry should be retained")) == first)
  }
}
