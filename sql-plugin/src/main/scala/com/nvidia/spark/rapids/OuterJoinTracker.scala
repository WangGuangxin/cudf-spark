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

import ai.rapids.cudf.{ColumnVector, ColumnView, Scalar, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.jni.JoinPrimitives

import org.apache.spark.sql.types.{BooleanType, DataType}

/** Matched-row tracking, owned by one outer join build group at a time. */
trait OuterJoinTracker extends AutoCloseable {
  def numRows: Int
  def sizeInBytes: Long
  def update(gatherIndices: ColumnView): Unit
  def unmatchedMask(): ColumnVector
}

private class BitmapOuterJoinTracker(
    val numRows: Int,
    private var bitmap: Option[SpillableBuffer]) extends OuterJoinTracker {
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

/** Baseline filter/scatter tracking with one boolean per build row. */
private class BooleanOuterJoinTracker(
    val numRows: Int,
    private var tracking: Option[SpillableColumnarBatch]) extends OuterJoinTracker {
  override def sizeInBytes: Long = tracking.get.sizeInBytes

  private def validIndices(gatherIndices: ColumnView): Table = {
    withResource(gatherIndices.copyToColumnVector()) { indices =>
      withResource(new Table(indices)) { indexTable =>
        withResource(Scalar.fromInt(Int.MinValue)) { invalid =>
          withResource(gatherIndices.notEqualTo(invalid)) { valid =>
            indexTable.filter(valid)
          }
        }
      }
    }
  }

  private def updatedTracking(indices: ColumnView): SpillableColumnarBatch = {
    val table = withResource(tracking.get.getColumnarBatch()) { batch =>
      GpuColumnVector.from(batch)
    }
    withResource(table) { current =>
      withResource(Scalar.fromBool(false)) { matched =>
        withResource(Table.scatter(Array(matched), indices, current)) { updated =>
          SpillableColumnarBatch(GpuColumnVector.from(updated, Array[DataType](BooleanType)),
            SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        }
      }
    }
  }

  override def update(gatherIndices: ColumnView): Unit = {
    if (gatherIndices.getRowCount > 0) {
      withResource(validIndices(gatherIndices)) { filtered =>
        val next = updatedTracking(filtered.getColumn(0))
        closeOnExcept(next) { _ =>
          withResource(tracking) { _ => tracking = Some(next) }
        }
      }
    }
  }

  override def unmatchedMask(): ColumnVector = {
    withResource(tracking.get.getColumnarBatch()) { batch =>
      batch.column(0).asInstanceOf[GpuColumnVector].getBase.incRefCount()
    }
  }

  override def close(): Unit = withResource(tracking)(_ => tracking = None)
}

object OuterJoinTracker {
  def apply(numRows: Int, useBitmap: Boolean = true): OuterJoinTracker = {
    if (useBitmap) {
      val bitmap = closeOnExcept(JoinPrimitives.createOuterJoinTracker(numRows)) { buffer =>
        SpillableBuffer(buffer, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
      }
      closeOnExcept(bitmap) { spillable =>
        new BitmapOuterJoinTracker(numRows, Some(spillable))
      }
    } else {
      val tracking = withResource(Scalar.fromBool(true)) { unmatched =>
        withResource(ColumnVector.fromScalar(unmatched, numRows)) { mask =>
          withResource(new Table(mask)) { table =>
            SpillableColumnarBatch(GpuColumnVector.from(table, Array[DataType](BooleanType)),
              SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
          }
        }
      }
      closeOnExcept(tracking) { spillable =>
        new BooleanOuterJoinTracker(numRows, Some(spillable))
      }
    }
  }
}
