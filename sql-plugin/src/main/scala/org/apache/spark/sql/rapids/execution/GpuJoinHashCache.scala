/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

import ai.rapids.cudf.{GatherMap, HashJoin, Table}
import com.nvidia.spark.Retryable
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.jni.DistinctHashJoin
import com.nvidia.spark.rapids.spill.{SpillableJoinHashHandle, SpillFramework}

/** Own one reusable hash state; pressure or a failed attempt may discard it without losing keys. */
private[execution] class GpuJoinHashCache(distinct: Boolean, compareNullsEqual: Boolean)
    extends AutoCloseable with Retryable {
  private var cached: Option[SpillableJoinHashHandle] = None
  private[execution] var buildCount: Long = 0

  private def withState[T](keys: Table)(fn: AutoCloseable => T): T = {
    cached.flatMap(_.withState(fn)) match {
      case Some(result) => result
      case None =>
        close()
        val state = if (distinct) new DistinctHashJoin(keys, compareNullsEqual)
          else new HashJoin(keys, compareNullsEqual)
        // Hash slots plus row metadata. This is a spill-planning estimate, not an exact size.
        val bytes = math.max(1L, keys.getRowCount * 32L)
        val handle = closeOnExcept(state)(new SpillableJoinHashHandle(_, bytes))
        closeOnExcept(handle) { _ =>
          // Register while borrowed so a concurrent spiller cannot evict before the first probe.
          handle.withState { resource =>
            SpillFramework.stores.deviceStore.track(handle)
            cached = Some(handle)
            buildCount += 1
            fn(resource)
          }.get
        }
    }
  }

  def innerJoin(keys: Table, probe: Table): Array[GatherMap] = withState(keys) {
    case hash: HashJoin => probe.innerJoinGatherMaps(hash)
    case hash: DistinctHashJoin => hash.innerJoin(probe)
  }

  def leftJoin(keys: Table, probe: Table): Array[GatherMap] = withState(keys) {
    case hash: HashJoin => probe.leftJoinGatherMaps(hash)
    case _ => throw new IllegalStateException("paired left join requires a regular hash state")
  }

  def leftDistinctJoin(keys: Table, probe: Table): GatherMap = withState(keys) {
    case hash: DistinctHashJoin => hash.leftJoin(probe)
    case _ => throw new IllegalStateException("single-map left join requires distinct keys")
  }

  override def checkpoint(): Unit = ()
  override def restore(): Unit = close()
  override def close(): Unit = withResource(cached)(_ => cached = None)
}
