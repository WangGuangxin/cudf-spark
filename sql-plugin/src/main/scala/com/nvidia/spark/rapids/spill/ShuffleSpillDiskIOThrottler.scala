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
  def throttleRead(numBytes: Long): Unit
  def throttleWrite(numBytes: Long): Unit
}

private[spill] object ExecutorShuffleSpillDiskIOThrottler
    extends ShuffleSpillDiskIOThrottler {
  private lazy val limiters = {
    val conf = Option(SparkEnv.get)
      .map(env => new RapidsConf(env.conf))
      .getOrElse(new RapidsConf(Map.empty[String, String]))
    (new ThroughputLimiter(conf.shuffleSpillReadBandwidth),
      new ThroughputLimiter(conf.shuffleSpillWriteBandwidth))
  }

  override def throttleRead(numBytes: Long): Unit = limiters._1.acquire(numBytes)

  override def throttleWrite(numBytes: Long): Unit = limiters._2.acquire(numBytes)
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

  def acquire(numBytes: Long): Unit = {
    if (bytesPerSecond > 0 && numBytes > 0) {
      val waitNanos = synchronized {
        val now = nanoTime()
        val start = math.max(now, nextAvailableNanos)
        val duration = math.max(1L,
          math.ceil(numBytes.toDouble * TimeUnit.SECONDS.toNanos(1) / bytesPerSecond).toLong)
        nextAvailableNanos = saturatedAdd(start, duration)
        math.max(0L, start - now)
      }
      if (waitNanos > 0) {
        sleepNanos(waitNanos)
      }
    }
  }

  private def saturatedAdd(left: Long, right: Long): Long = {
    if (left > 0 && right > Long.MaxValue - left) Long.MaxValue else left + right
  }
}
