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
package com.nvidia.spark.rapids.spill

import com.nvidia.spark.rapids.Arm.withResource

/** A rebuildable device resource. Eviction discards it after the store's CUDA synchronization. */
class SpillableJoinHashHandle(
    state: AutoCloseable,
    override val approxSizeInBytes: Long) extends DeviceSpillableHandle[AutoCloseable] {
  private[spill] override var dev: Option[AutoCloseable] = Some(state)
  private var inUse = false

  def withState[T](fn: AutoCloseable => T): Option[T] = {
    val borrowed = synchronized {
      if (closed || spilling || dev.isEmpty) None else {
        require(!inUse, "concurrent hash probes are not supported")
        inUse = true
        dev
      }
    }
    borrowed.map { resource =>
      try {
        fn(resource)
      } finally {
        val dispose = synchronized {
          inUse = false
          closed
        }
        if (dispose) doClose()
      }
    }
  }

  private[spill] override def spillable: Boolean = synchronized {
    super.spillable && !closed && !spilling && !inUse
  }

  override def spill(): Long = synchronized {
    if (spillable) {
      spilling = true
      approxSizeInBytes
    } else 0L
  }

  private[spill] override def doClose(): Unit = {
    SpillFramework.removeFromDeviceStore(this)
    synchronized {
      // A close racing a probe retires the resource when the borrower releases it.
      if (!inUse) withResource(dev)(_ => dev = None)
    }
  }

  override def releaseSpilled(): Unit = {
    doClose()
    synchronized { spilling = false }
  }
}
