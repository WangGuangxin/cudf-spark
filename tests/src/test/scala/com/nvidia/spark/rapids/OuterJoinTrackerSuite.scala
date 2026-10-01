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

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.jni.RmmSpark
import com.nvidia.spark.rapids.spill.SpillFramework

class OuterJoinTrackerSuite extends RmmSparkRetrySuiteBase {
  private def assertUnmatched(tracker: OuterJoinTracker, matched: Set[Int]): Unit = {
    withResource(tracker.unmatchedMask()) { mask =>
      withResource(mask.copyToHost()) { host =>
        (0 until tracker.numRows).foreach { row =>
          assert(host.getBoolean(row) == !matched.contains(row), s"row $row")
        }
      }
    }
  }

  test("bitmap stays compact and accumulates matches across updates") {
    withResource(OuterJoinTracker(65)) { tracker =>
      assert(tracker.sizeInBytes == 12)
      withResource(ColumnVector.fromInts(0, 31, 31, 32, Int.MinValue, 65)) { indices =>
        tracker.update(indices)
        tracker.update(indices)
      }
      assertUnmatched(tracker, Set(0, 31, 32))
      assert(SpillFramework.stores.deviceStore.numHandles == 1)
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  test("updates after host and disk spill retire old snapshots") {
    withResource(OuterJoinTracker(65)) { tracker =>
      withResource(ColumnVector.fromInts(1, 32)) { indices => tracker.update(indices) }
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
      assert(SpillFramework.stores.hostStore.spill(Long.MaxValue) > 0)
      withResource(ColumnVector.fromInts(2, 64)) { indices => tracker.update(indices) }
      assertUnmatched(tracker, Set(1, 2, 32, 64))
      // Spill the updated state again: restoring must not reuse the pre-update snapshot.
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
      assertUnmatched(tracker, Set(1, 2, 32, 64))
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
    assert(SpillFramework.stores.diskStore.numHandles == 0)
  }

  test("restore can retry before modifying a spilled bitmap") {
    withResource(OuterJoinTracker(65)) { tracker =>
      withResource(ColumnVector.fromInts(1)) { indices => tracker.update(indices) }
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
      withResource(ColumnVector.fromInts(2)) { indices =>
        RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1)
        withRetryNoSplit { tracker.update(indices) }
      }
      assertUnmatched(tracker, Set(1, 2))
      assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
    }
  }

  test("zero rows and empty maps remain valid") {
    Seq(0, 33).foreach { rows =>
      withResource(OuterJoinTracker(rows)) { tracker =>
        withResource(ColumnVector.fromInts()) { indices => tracker.update(indices) }
        assertUnmatched(tracker, Set.empty)
      }
    }
  }
}
