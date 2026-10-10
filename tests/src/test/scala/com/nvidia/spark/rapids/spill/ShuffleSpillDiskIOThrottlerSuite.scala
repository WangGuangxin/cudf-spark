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

package com.nvidia.spark.rapids.spill

import java.util.concurrent.TimeUnit

import scala.collection.mutable.ArrayBuffer

import org.scalatest.funsuite.AnyFunSuite

class ShuffleSpillDiskIOThrottlerSuite extends AnyFunSuite {
  test("zero bandwidth disables throttling") {
    val sleeps = new ArrayBuffer[Long]()
    val limiter = new ThroughputLimiter(0L, () => 0L, nanos => {
      sleeps += nanos
      ()
    })

    limiter.acquire(Long.MaxValue)

    assert(sleeps.isEmpty)
  }

  test("limiter paces aggregate reservations") {
    var now = 0L
    val sleeps = new ArrayBuffer[Long]()
    val limiter = new ThroughputLimiter(
      bytesPerSecond = 100L,
      nanoTime = () => now,
      sleepNanos = nanos => {
        sleeps += nanos
        now += nanos
      })

    limiter.acquire(25L)
    limiter.acquire(25L)
    limiter.acquire(50L)

    assert(sleeps === Seq(
      TimeUnit.MILLISECONDS.toNanos(250),
      TimeUnit.MILLISECONDS.toNanos(250),
      TimeUnit.MILLISECONDS.toNanos(500)))
  }

  test("limiter does not add delay when underlying IO is slower") {
    val sleeps = new ArrayBuffer[Long]()
    val limiter = new ThroughputLimiter(
      bytesPerSecond = 100L,
      nanoTime = () => 0L,
      sleepNanos = nanos => {
        sleeps += nanos
        ()
      })

    limiter.acquire(25L, TimeUnit.SECONDS.toNanos(1))

    assert(sleeps.isEmpty)
  }

  test("shared limit combines concurrent read and write reservations") {
    val sleeps = new ArrayBuffer[Long]()
    val limiter = new ShuffleSpillBandwidthLimiter(
      readBytesPerSecond = 0L,
      writeBytesPerSecond = 0L,
      totalBytesPerSecond = 100L,
      nanoTime = () => 0L,
      sleepNanos = sleeps += _)

    val readResult = limiter.throttleRead(50L, 0L)
    val writeResult = limiter.throttleWrite(50L, 0L)

    assert(readResult.waitNanos === TimeUnit.MILLISECONDS.toNanos(500))
    assert(writeResult.waitNanos === TimeUnit.SECONDS.toNanos(1))
    assert(sleeps === Seq(readResult.waitNanos, writeResult.waitNanos))
  }

  test("directional limits remain independent when shared limit is disabled") {
    val sleeps = new ArrayBuffer[Long]()
    val limiter = new ShuffleSpillBandwidthLimiter(
      readBytesPerSecond = 100L,
      writeBytesPerSecond = 100L,
      totalBytesPerSecond = 0L,
      nanoTime = () => 0L,
      sleepNanos = sleeps += _)

    val readResult = limiter.throttleRead(50L, 0L)
    val writeResult = limiter.throttleWrite(50L, 0L)

    assert(readResult.waitNanos === TimeUnit.MILLISECONDS.toNanos(500))
    assert(writeResult.waitNanos === TimeUnit.MILLISECONDS.toNanos(500))
  }
}
