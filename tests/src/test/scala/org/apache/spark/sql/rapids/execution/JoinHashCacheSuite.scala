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

import ai.rapids.cudf.{ColumnVector, Table}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{withRestoreOnRetry, withRetryNoSplit}
import com.nvidia.spark.rapids.jni.RmmSpark
import com.nvidia.spark.rapids.spill.{SpillableJoinHashHandle, SpillFramework}

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, ExprId}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.types.{DataType, IntegerType}

class JoinHashCacheSuite extends RmmSparkRetrySuiteBase {
  for (distinct <- Seq(false, true)) {
    test(s"hash state reuse, eviction and retry distinct=$distinct") {
      withResource(ColumnVector.fromInts(1, 2, 3)) { build =>
        withResource(new Table(build)) { keys =>
          withResource(ColumnVector.fromInts(3, 1, 9, 3)) { stream =>
            withResource(new Table(stream)) { probe =>
              withResource(new GpuJoinHashCache(distinct, false)) { cache =>
                def check(): Unit = withResource(cache.innerJoin(keys, probe)) { maps =>
                  assert(maps(0).getRowCount == 3)
                  withResource(maps(0).toColumnView(0, 3)) { left =>
                    withResource(maps(1).toColumnView(0, 3)) { right =>
                      withResource(left.copyToColumnVector()) { lc =>
                        withResource(right.copyToColumnVector()) { rc =>
                          withResource(lc.copyToHost()) { lh =>
                            withResource(rc.copyToHost()) { rh =>
                              val rows = (0 until 3).map(i => (lh.getInt(i), rh.getInt(i)))
                              assert(rows.sorted == Seq((0, 2), (1, 0), (3, 2)))
                            }
                          }
                        }
                      }
                    }
                  }
                }
                check()
                check()
                assert(cache.buildCount == 1)
                assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
                check()
                assert(cache.buildCount == 2)
                cache.checkpoint()
                RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
                  RmmSpark.OomInjectionType.GPU.ordinal, 0)
                withRetryNoSplit {
                  withRestoreOnRetry(cache) { check() }
                }
                assert(cache.buildCount == 3)
              }
            }
          }
        }
      }
      assert(SpillFramework.stores.deviceStore.numHandles == 0)
    }
  }

  for (distinct <- Seq(false, true); side <- Seq(GpuBuildLeft, GpuBuildRight);
      selection <- Seq(JoinBuildSideSelection.AUTO, JoinBuildSideSelection.FIXED,
        JoinBuildSideSelection.SMALLEST)) {
    test(s"iterator hash reuse distinct=$distinct side=$side selection=$selection") {
      def batch(keys: Seq[Int], payload: Seq[Int]): LazySpillableColumnarBatch = {
        withResource(ColumnVector.fromInts(keys: _*)) { kc =>
          withResource(ColumnVector.fromInts(payload: _*)) { pc =>
            withResource(new Table(kc, pc)) { table =>
              withResource(GpuColumnVector.from(table, Array[DataType](IntegerType, IntegerType))) {
                data => LazySpillableColumnarBatch(data, "hash_reuse_test")
              }
            }
          }
        }
      }
      val bk = if (distinct) Seq(1, 2) else Seq(1, 1, 2)
      val bp = if (distinct) Seq(100, 200) else Seq(100, 101, 200)
      withResource(batch(bk, bp)) { build =>
        withResource((0 until 4).map(_ => batch(Seq(1, 2, 9), Seq(10, 20, 90)))) { stream =>
          val attrs = Seq(AttributeReference("key", IntegerType)(),
            AttributeReference("payload", IntegerType)())
          val key = GpuBoundReference(0, IntegerType, false)(ExprId(0), "key")
          val options = JoinOptions(JoinStrategy.HASH_ONLY, selection, 1024 * 1024, false, 0.75)
          withResource(new HashJoinIterator(build, Seq(key),
              Some(JoinBuildSideStats(1.0, distinct)), stream.iterator, Seq(key), attrs,
              options, Inner, side, false, None, NoopMetric, NoopMetric)) { iterator =>
            val result = iterator.flatMap { cb =>
              withResource(cb) { output =>
                val bi = if (side == GpuBuildLeft) 1 else 3
                val si = if (side == GpuBuildLeft) 3 else 1
                withResource(output.column(bi).asInstanceOf[GpuColumnVector]
                    .getBase.copyToHost()) { bh =>
                  withResource(output.column(si).asInstanceOf[GpuColumnVector]
                      .getBase.copyToHost()) { sh =>
                    (0 until output.numRows()).map(row => (bh.getInt(row), sh.getInt(row)))
                  }
                }
              }
            }.toVector
            val expected = bp.map(value => (value, if (value < 200) 10 else 20))
            assert(result.sorted == (0 until 4).flatMap(_ => expected).sorted)
            // Ties under SMALLEST may select the streamed table; do not cache that input.
            val cached = distinct || selection != JoinBuildSideSelection.SMALLEST ||
              side == GpuBuildRight
            assert(iterator.hashBuildCount == (if (cached) 1 else 0))
          }
        }
      }
      assert(SpillFramework.stores.deviceStore.numHandles == 0)
    }
  }

  test("borrowed hash resources cannot be evicted and close once") {
    var closes = 0
    val resource = new AutoCloseable {
      override def close(): Unit = closes += 1
    }
    withResource(new SpillableJoinHashHandle(resource, 128)) { handle =>
      SpillFramework.stores.deviceStore.track(handle)
      handle.withState { _ =>
        assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) == 0)
        assert(closes == 0)
      }
      assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) == 128)
      assert(closes == 1)
      assert(handle.withState(_ => ()).isEmpty)
    }
    assert(closes == 1)
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }
  test("closing a borrowed hash retires it after the probe") {
    var closes = 0
    val resource = new AutoCloseable {
      override def close(): Unit = closes += 1
    }
    withResource(new SpillableJoinHashHandle(resource, 128)) { handle =>
      SpillFramework.stores.deviceStore.track(handle)
      handle.withState { _ =>
        withResource(handle) { _ => () }
        assert(closes == 0)
      }
      assert(closes == 1)
    }
    assert(closes == 1)
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

}
