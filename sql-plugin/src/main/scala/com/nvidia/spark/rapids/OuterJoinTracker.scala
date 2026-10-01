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

import ai.rapids.cudf.{ColumnVector, ColumnView}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.jni.JoinPrimitives

/** A spillable matched-row bitmap, owned by one outer join build group at a time. */
class OuterJoinTracker private(
    val numRows: Int,
    private var bitmap: Option[SpillableBuffer]) extends AutoCloseable {
  def sizeInBytes: Long = bitmap.get.sizeInBytes

  def update(gatherIndices: ColumnView): Unit = {
    if (gatherIndices.getRowCount > 0) {
      val previous = bitmap.get
      // Materialization pins the device buffer. A spilled handle retains an immutable snapshot,
      // so retire it before mutation and register the updated buffer as the new spill source.
      withResource(previous.getDeviceBuffer()) { buffer =>
        withResource(previous) { _ =>
          bitmap = None
        }
        try {
          JoinPrimitives.updateOuterJoinTracker(buffer, gatherIndices, numRows)
        } finally {
          // Matching is monotonic: a retry can safely repeat any already completed atomic ORs.
          // SpillableBuffer synchronizes the default stream before making the buffer spillable.
          buffer.incRefCount()
          val updated = closeOnExcept(buffer) { owned =>
            SpillableBuffer(owned, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
          }
          closeOnExcept(updated) { spillable =>
            bitmap = Some(spillable)
          }
        }
      }
    }
  }

  def unmatchedMask(): ColumnVector = {
    withResource(bitmap.get.getDeviceBuffer()) { buffer =>
      JoinPrimitives.outerJoinUnmatchedMask(buffer, numRows)
    }
  }

  override def close(): Unit = {
    withResource(bitmap) { _ =>
      bitmap = None
    }
  }
}

object OuterJoinTracker {
  def apply(numRows: Int): OuterJoinTracker = {
    val bitmap = closeOnExcept(JoinPrimitives.createOuterJoinTracker(numRows)) { buffer =>
      SpillableBuffer(buffer, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
    }
    closeOnExcept(bitmap) { spillable =>
      new OuterJoinTracker(numRows, Some(spillable))
    }
  }
}
