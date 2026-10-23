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

import com.nvidia.spark.rapids.RapidsConf

import org.apache.spark.SparkEnv

private[spill] trait ShuffleSpillDiskIOThrottler {
  def throttleRead(numBytes: Long, ioTimeNanos: Long): ThrottleResult
  def throttleWrite(numBytes: Long, ioTimeNanos: Long): ThrottleResult
}

private[spill] case class ThrottleResult(waitNanos: Long, queueDepth: Long)

private[spill] object ExecutorShuffleSpillDiskIOThrottler
    extends ShuffleSpillDiskIOThrottler {
  private lazy val limiter = {
    val conf = Option(SparkEnv.get)
      .map(env => new RapidsConf(env.conf))
      .getOrElse(new RapidsConf(Map.empty[String, String]))
    new ShuffleSpillBandwidthLimiter(
      conf.shuffleSpillReadBandwidth,
      conf.shuffleSpillWriteBandwidth,
      conf.shuffleSpillTotalBandwidth)
  }

  override def throttleRead(numBytes: Long, ioTimeNanos: Long): ThrottleResult =
    limiter.throttleRead(numBytes, ioTimeNanos)

  override def throttleWrite(numBytes: Long, ioTimeNanos: Long): ThrottleResult =
    limiter.throttleWrite(numBytes, ioTimeNanos)
}

/**
 * Coordinates directional limits with an optional shared read/write limit. All reservations are
 * made under one monitor so concurrent reads and writes compete for the same total bandwidth.
 */
private[spill] class ShuffleSpillBandwidthLimiter(
    readBytesPerSecond: Long,
    writeBytesPerSecond: Long,
    totalBytesPerSecond: Long,
    nanoTime: () => Long = () => System.nanoTime(),
    sleepNanos: Long => Unit = nanos => TimeUnit.NANOSECONDS.sleep(nanos))
    extends ShuffleSpillDiskIOThrottler {
  require(readBytesPerSecond >= 0, "readBytesPerSecond must be non-negative")
  require(writeBytesPerSecond >= 0, "writeBytesPerSecond must be non-negative")
  require(totalBytesPerSecond >= 0, "totalBytesPerSecond must be non-negative")

  private var nextReadNanos = nanoTime()
  private var nextWriteNanos = nextReadNanos
  private var nextTotalNanos = nextReadNanos
  private var outstandingReservations = 0L

  // Without any limit there is nothing to reserve. Skip the executor-wide monitor, which every
  // partial-file read and write would otherwise contend on.
  private val unlimited =
    readBytesPerSecond == 0 && writeBytesPerSecond == 0 && totalBytesPerSecond == 0

  override def throttleRead(numBytes: Long, ioTimeNanos: Long): ThrottleResult = {
    throttle(numBytes, ioTimeNanos, read = true)
  }

  override def throttleWrite(numBytes: Long, ioTimeNanos: Long): ThrottleResult = {
    throttle(numBytes, ioTimeNanos, read = false)
  }

  private def throttle(numBytes: Long, ioTimeNanos: Long, read: Boolean): ThrottleResult = {
    if (numBytes <= 0 || unlimited) {
      return ThrottleResult(0L, 0L)
    }

    val (waitNanos, queueDepth) = synchronized {
      val now = nanoTime()
      val directionalRate = if (read) readBytesPerSecond else writeBytesPerSecond
      val directionalNext = if (read) nextReadNanos else nextWriteNanos
      val (directionalFinish, directionalWait) = reserve(
        now, directionalNext, directionalRate, numBytes, ioTimeNanos)
      val (totalFinish, totalWait) = reserve(
        now, nextTotalNanos, totalBytesPerSecond, numBytes, ioTimeNanos)

      if (read) {
        nextReadNanos = directionalFinish
      } else {
        nextWriteNanos = directionalFinish
      }
      nextTotalNanos = totalFinish
      val wait = math.max(directionalWait, totalWait)
      if (wait > 0) {
        outstandingReservations += 1
      }
      (wait, outstandingReservations)
    }

    if (waitNanos > 0) {
      try {
        sleepNanos(waitNanos)
      } finally {
        synchronized {
          outstandingReservations -= 1
        }
      }
    }
    ThrottleResult(waitNanos, if (waitNanos > 0) queueDepth else 0L)
  }

  private def reserve(
      now: Long,
      nextAvailable: Long,
      bytesPerSecond: Long,
      numBytes: Long,
      ioTimeNanos: Long): (Long, Long) = {
    if (bytesPerSecond == 0) {
      (nextAvailable, 0L)
    } else {
      val duration = math.max(1L,
        math.ceil(numBytes.toDouble * TimeUnit.SECONDS.toNanos(1) / bytesPerSecond).toLong)
      val remainingDuration = math.max(0L, duration - math.max(0L, ioTimeNanos))
      val start = math.max(now, nextAvailable)
      val finish = saturatedAdd(start, remainingDuration)
      (finish, math.max(0L, finish - now))
    }
  }

  private def saturatedAdd(left: Long, right: Long): Long = {
    if (left > 0 && right > Long.MaxValue - left) Long.MaxValue else left + right
  }
}

/**
 * Executor-wide pacing limiter. Reservations are serialized, while callers sleep outside the
 * monitor. If the underlying I/O is already slower than the configured rate, no extra delay is
 * added. A bandwidth of zero disables throttling.
 */
private[spill] class ThroughputLimiter(
    bytesPerSecond: Long,
    nanoTime: () => Long = () => System.nanoTime(),
    sleepNanos: Long => Unit = nanos => TimeUnit.NANOSECONDS.sleep(nanos)) {
  require(bytesPerSecond >= 0, "bytesPerSecond must be non-negative")

  private var nextAvailableNanos: Long = nanoTime()

  def acquire(numBytes: Long, ioTimeNanos: Long = 0L): Long = {
    if (bytesPerSecond > 0 && numBytes > 0) {
      val waitNanos = synchronized {
        val now = nanoTime()
        val start = math.max(now, nextAvailableNanos)
        val duration = math.max(1L,
          math.ceil(numBytes.toDouble * TimeUnit.SECONDS.toNanos(1) / bytesPerSecond).toLong)
        val remainingDuration = math.max(0L, duration - math.max(0L, ioTimeNanos))
        nextAvailableNanos = saturatedAdd(start, remainingDuration)
        math.max(0L, nextAvailableNanos - now)
      }
      if (waitNanos > 0) {
        sleepNanos(waitNanos)
      }
      waitNanos
    } else {
      0L
    }
  }

  private def saturatedAdd(left: Long, right: Long): Long = {
    if (left > 0 && right > Long.MaxValue - left) Long.MaxValue else left + right
  }
}
