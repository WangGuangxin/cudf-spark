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
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{GpuRetryOOM, GpuSplitAndRetryOOM, RmmSpark}
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, ExprId}
import org.apache.spark.sql.functions.{broadcast, col}
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class BroadcastJoinChainSuite extends RmmSparkRetrySuiteBase {
  private type Row = Seq[Integer]
  private val options = JoinOptions(JoinStrategy.AUTO, JoinBuildSideSelection.FIXED,
    1024 * 1024, false, 0.75)
  private def ref(i: Int): GpuBoundReference =
    GpuBoundReference(i, IntegerType, nullable = true)(ExprId(i), s"c$i")

  private def batch(rows: Seq[Row], width: Int): ColumnarBatch = {
    withResource((0 until width).safeMap { i =>
      ColumnVector.fromBoxedInts(rows.map(_(i)): _*)
    }) { columns =>
      withResource(new Table(columns: _*)) { table =>
        GpuColumnVector.from(table, Array.fill[DataType](width)(IntegerType))
      }
    }
  }

  private val dims: Seq[Seq[Row]] = Seq(
    Seq(Seq[Integer](1, 10, 100), Seq[Integer](2, 20, 200), Seq[Integer](null, 30, 300)),
    Seq(Seq[Integer](10, 7), Seq[Integer](20, 8), Seq[Integer](30, 9), Seq[Integer](40, 9)),
    Seq(Seq[Integer](7, 700), Seq[Integer](8, 800), Seq[Integer](9, 900)))
  private def stages: Seq[JoinChainStage] = Seq(
    JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildRight, false),
    JoinChainStage(Seq(ref(0)), Seq(0), Seq(3, 1, 4), GpuBuildLeft, false),
    JoinChainStage(Seq(ref(0)), Seq(0), Seq(1, 3, 4, 2), GpuBuildRight, false))

  // Independent row oracle, including reordered projections and a later key from a dimension.
  private def expected(fact: Seq[Row], ss: Seq[JoinChainStage],
      dimensionRows: Seq[Seq[Row]] = dims): Seq[Row] = {
    ss.zip(dimensionRows).foldLeft(fact) { case (rows, (stage, dim)) =>
      rows.flatMap { original =>
        val row = stage.streamProjection.map(original)
        val matches = dim.filter { build =>
          stage.streamKeyOrdinals.zip(stage.buildKeys).forall { case (s, b) =>
            val l = row(s)
            val r = build(b.asInstanceOf[GpuBoundReference].ordinal)
            if (l == null || r == null) stage.nullsEqual && l == r else l == r
          }
        }
        if (stage.isLeftSemi) {
          if (matches.nonEmpty) Seq(row) else Seq.empty
        } else {
          matches.map { build =>
            if (stage.buildSide == GpuBuildRight) row ++ build else build ++ row
          }
        }
      }
    }
  }

  private def run(fact: Seq[Row], ss: Seq[JoinChainStage],
      action: Int => Unit = _ => (), targetSize: Long = options.targetSize,
      inputBatchRows: Int = 120, dimensionRows: Seq[Seq[Row]] = dims,
      retryKeys: Boolean = false, expectedHashBuilds: Option[Long] = None,
      reuseHash: Boolean = true, retryStats: Boolean = false): Seq[Row] = {
    val builds = dimensionRows.zip(dims).take(ss.length).safeMap { case (rows, shape) =>
      val width = rows.headOption.map(_.size).getOrElse(shape.head.size)
      withResource(batch(rows, width))(LazySpillableColumnarBatch(_, "test_dim"))
    }
    withResource(builds) { _ =>
      if (retryKeys) {
        RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
          RmmSpark.OomInjectionType.GPU.ordinal, 0)
      }
      withResource(GpuJoinChain.buildKeys(builds, ss)) { keys =>
        if (retryStats) {
          RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
        }
        val classified = GpuJoinChain.classifyBuildKeys(keys, ss)
        val input = fact.grouped(inputBatchRows).map(batch(_, 2))
        val attrs = (0 until 2).map(i => AttributeReference(s"c$i", IntegerType)())
        val result = withResource(new JoinChainIterator(input, attrs, builds, keys, classified,
            options.copy(targetSize = targetSize, hashReuseEnabled = reuseHash)) {
          override protected def beforeProbe(stage: Int): Unit = action(stage)
        }) { iterator =>
          val outputRows = iterator.flatMap { output =>
            withResource(output) { gpu =>
              withResource(GpuColumnVector.extractBases(gpu).toSeq.safeMap(_.copyToHost())) {
                columns =>
                  (0 until gpu.numRows()).map { row =>
                    columns.map(c => if (c.isNull(row)) null else Integer.valueOf(c.getInt(row)))
                  }
              }
            }
          }.toVector
          expectedHashBuilds.foreach(count => assert(iterator.hashBuildCount == count))
          outputRows
        }
        assert(SpillFramework.stores.deviceStore.numHandles == 0)
        assert(SpillFramework.stores.hostStore.numHandles == 0)
        result
      }
    }
  }

  private def sorted(rows: Seq[Row]): Seq[String] = rows.map(_.mkString(",")).sorted
  private val fact = Seq(Seq[Integer](1, 11), Seq[Integer](2, 22),
    Seq[Integer](3, 33), Seq[Integer](null, 44), Seq[Integer](1, 55))

  private def semiScenario(indices: Set[Int], nullsEqual: Boolean):
      (Seq[JoinChainStage], Seq[Seq[Row]]) = {
    val dimensions = Seq(dims.head,
      Seq(Seq[Integer](1, 7), Seq[Integer](2, 8), Seq[Integer](null, 9)),
      Seq(Seq[Integer](1, 700), Seq[Integer](2, 800), Seq[Integer](null, 900)))
    var width = 2
    val ss = dimensions.indices.map { i =>
      val semi = indices.contains(i)
      val stage = JoinChainStage(Seq(ref(0)), Seq(0), 0 until width, GpuBuildRight,
        nullsEqual, isLeftSemi = semi)
      if (!semi) width += dimensions(i).head.size
      stage
    }
    val builds = dimensions.zipWithIndex.map { case (rows, i) =>
      if (indices.contains(i)) rows ++ rows else rows
    }
    (ss, builds)
  }

  for (semi <- Seq(Set(0), Set(1), Set(2), Set(0, 1), Set(0, 2), Set(1, 2), Set(0, 1, 2));
      nullsEqual <- Seq(false, true)) {
    test(s"semi stages $semi preserve stream duplicates and null equality=$nullsEqual") {
      val (ss, builds) = semiScenario(semi, nullsEqual)
      val rows = (0 until 20).flatMap(_ => fact)
      val result = run(rows, ss, targetSize = 64, inputBatchRows = 5,
        dimensionRows = builds, expectedHashBuilds = Some(ss.length.toLong))
      assert(sorted(result) == sorted(expected(rows, ss, builds)))
    }
  }

  test("semi stages preserve build-left layout and reordered projections") {
    val (_, builds) = semiScenario(Set(0, 2), nullsEqual = false)
    val ss = Seq(
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildRight, false,
        isLeftSemi = true),
      JoinChainStage(Seq(ref(0)), Seq(0), Seq(0, 1), GpuBuildLeft, false),
      JoinChainStage(Seq(ref(0)), Seq(1), Seq(3, 2, 1, 0), GpuBuildRight, false,
        isLeftSemi = true))
    assert(sorted(run(fact, ss, dimensionRows = builds)) ==
      sorted(expected(fact, ss, builds)))
  }

  for (nullsEqual <- Seq(false, true)) {
    test(s"semi multi-column keys deduplicate complete tuples null equality=$nullsEqual") {
      val rows = Seq(Seq[Integer](1, 10), Seq[Integer](1, 10), Seq[Integer](2, 20),
        Seq[Integer](1, null), Seq[Integer](null, 30), Seq[Integer](null, null))
      val unique = dims.head ++ Seq(Seq[Integer](1, null, 500),
        Seq[Integer](null, null, 600))
      val builds = Seq(unique ++ unique)
      val ss = Seq(JoinChainStage(Seq(ref(0), ref(1)), Seq(0, 1), Seq(0, 1),
        GpuBuildRight, nullsEqual, isLeftSemi = true))
      assert(sorted(run(rows, ss, dimensionRows = builds)) ==
        sorted(expected(rows, ss, builds)))
      assert(run(rows, ss, dimensionRows = Seq(Seq.empty)).isEmpty)
      assert(run(Seq.empty, ss, dimensionRows = builds).isEmpty)
    }
  }

  test("semi build key deduplication retries and hashes are reusable") {
    val (ss, builds) = semiScenario(Set(0, 1, 2), nullsEqual = true)
    val result = run(fact, ss, dimensionRows = builds, retryKeys = true,
      expectedHashBuilds = Some(ss.length.toLong))
    assert(sorted(result) == sorted(expected(fact, ss, builds)))
    assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
  }

  test("semi placeholders survive composed-map spilling") {
    val (ss, builds) = semiScenario(Set(0, 2), nullsEqual = false)
    var spilled = 0L
    val result = run(fact, ss, action = stage => {
      if (stage == 1) {
        spilled += SpillFramework.stores.deviceStore.spill(Long.MaxValue)
        SpillFramework.stores.hostStore.spill(Long.MaxValue)
      }
    }, dimensionRows = builds)
    assert(spilled > 0)
    assert(sorted(result) == sorted(expected(fact, ss, builds)))
  }

  for (split <- Seq(false, true)) {
    test(s"semi partial probe OOM restores maps and counters split=$split") {
      val (rawStages, builds) = semiScenario(Set(0, 2), nullsEqual = false)
      val metrics = rawStages.map(_ => new LocalGpuMetric)
      val ss = rawStages.zip(metrics).map { case (stage, metric) =>
        stage.copy(outputRows = metric)
      }
      val rows = (0 until 48).flatMap(_ => fact)
      var injected = false
      val result = run(rows, ss, action = stage => {
        if (stage == 1 && !injected) {
          injected = true
          if (split) throw new GpuSplitAndRetryOOM("semi chain test")
          else throw new GpuRetryOOM("semi chain test")
        }
      }, inputBatchRows = 240, dimensionRows = builds)
      assert(injected)
      assert(sorted(result) == sorted(expected(rows, ss, builds)))
      assert(metrics.head.value == expected(rows, ss.take(1), builds.take(1)).size)
      assert(metrics.last.value == result.size)
    }
  }

  private val duplicateDims = Seq(
    dims.head ++ Seq(Seq[Integer](1, 20, 101), Seq[Integer](null, 30, 301)),
    dims(1) ++ Seq(Seq[Integer](10, 8), Seq[Integer](20, 7)),
    dims(2) ++ Seq(Seq[Integer](7, 701), Seq[Integer](8, 801)))

  for (nullsEqual <- Seq(false, true); reuse <- Seq(false, true)) {
    test(s"non-unique maps retain all payload matches nulls=$nullsEqual reuse=$reuse") {
      val ss = stages.map(_.copy(nullsEqual = nullsEqual))
      val rows = (0 until 3).flatMap(_ => fact)
      val result = run(rows, ss, dimensionRows = duplicateDims, inputBatchRows = 5,
        reuseHash = reuse, expectedHashBuilds = Some(if (reuse) 3L else 9L))
      assert(sorted(result) == sorted(expected(rows, ss, duplicateDims)))
    }
  }

  test("non-unique fanout splits stream batches before probing") {
    val rows = (0 until 10).flatMap(_ => fact)
    var probes = 0
    val result = run(rows, stages, dimensionRows = duplicateDims, targetSize = 320,
      inputBatchRows = rows.size, action = stage => if (stage == 0) probes += 1)
    assert(probes > 1)
    assert(sorted(result) == sorted(expected(rows, stages, duplicateDims)))
  }

  test("non-unique maps and cached hashes survive host and disk spilling") {
    var spilled = 0L
    val result = run(fact, stages, dimensionRows = duplicateDims, action = stage => {
      if (stage == 1) {
        spilled += SpillFramework.stores.deviceStore.spill(Long.MaxValue)
        SpillFramework.stores.hostStore.spill(Long.MaxValue)
      }
    })
    assert(spilled > 0)
    assert(sorted(result) == sorted(expected(fact, stages, duplicateDims)))
  }

  test("recursive fanout splitting bounds pending spillable batches") {
    val rows = Seq.fill(128)(Seq[Integer](1, 11))
    var probes = 0
    val result = run(rows, stages, dimensionRows = duplicateDims, targetSize = 320,
      inputBatchRows = rows.size, action = stage => {
        if (stage == 0) {
          probes += 1
          // Depth-first splits retain only a few siblings at each level, rather than all
          // 128 one-row inputs. Include build/key/hash handles in this generous limit.
          assert(SpillFramework.stores.deviceStore.numHandles < 64)
        }
      })
    assert(probes == rows.size)
    assert(sorted(result) == sorted(expected(rows, stages, duplicateDims)))
  }

  for (split <- Seq(false, true)) {
    test(s"non-unique partial probe retries small inputs without duplicate metrics split=$split") {
      val metrics = stages.map(_ => new LocalGpuMetric)
      val ss = stages.zip(metrics).map { case (stage, metric) => stage.copy(outputRows = metric) }
      var injected = false
      val result = run(fact, ss, dimensionRows = duplicateDims, action = stage => {
        if (stage == 1 && !injected) {
          injected = true
          if (split) throw new GpuSplitAndRetryOOM("non-unique chain test")
          else throw new GpuRetryOOM("non-unique chain test")
        }
      })
      assert(injected)
      assert(sorted(result) == sorted(expected(fact, ss, duplicateDims)))
      assert(metrics.head.value == expected(fact, ss.take(1), duplicateDims.take(1)).size)
      assert(metrics.last.value == result.size)
    }
  }

  for (nullsEqual <- Seq(false, true)) {
    test(s"non-unique complete key tuples preserve null equality=$nullsEqual") {
      val ss = stages.updated(0, stages.head.copy(buildKeys = Seq(ref(0), ref(1)),
        streamKeyOrdinals = Seq(0, 1), nullsEqual = nullsEqual))
      val builds = duplicateDims.updated(0, duplicateDims.head ++ dims.head)
      val rows = Seq(Seq[Integer](1, 10), Seq[Integer](null, 30), Seq[Integer](2, 20))
      assert(sorted(run(rows, ss, dimensionRows = builds)) ==
        sorted(expected(rows, ss, builds)))
    }
  }

  test("non-unique inner stages compose with deduplicated semi stages") {
    val (ss, builds) = semiScenario(Set(1), nullsEqual = true)
    val duplicates = builds.map(rows => rows ++ rows)
    assert(sorted(run(fact, ss, dimensionRows = duplicates)) ==
      sorted(expected(fact, ss, duplicates)))
    assert(run(fact, stages, dimensionRows = duplicateDims.updated(1, Seq.empty)).isEmpty)
    assert(run(Seq(Seq[Integer](99, 1)), stages, dimensionRows = duplicateDims).isEmpty)
  }

  test("duplicate null tuples only multiply null-equal probes") {
    val rows = Seq.fill(5)(Seq[Integer](null, 30, 300)) ++ dims.head.take(2)
    withResource(batch(rows, 3))(original =>
      withResource(LazySpillableColumnarBatch(original, "test_dim")) { built =>
        withResource(GpuJoinChain.buildKeys(Seq(built), stages.take(1))) { keys =>
          val unequal = GpuJoinChain.classifyBuildKeys(keys, stages.take(1)).head
          val equal = GpuJoinChain.classifyBuildKeys(keys,
            Seq(stages.head.copy(nullsEqual = true))).head
          assert(!unequal.buildKeysUnique)
          assert(unequal.maxBuildMatches == 1)
          assert(!equal.buildKeysUnique)
          assert(equal.maxBuildMatches == 5)
        }
      })
  }

  test("non-unique fanout classification retries with spillable build keys") {
    assert(sorted(run(fact, stages, dimensionRows = duplicateDims, retryStats = true)) ==
      sorted(expected(fact, stages, duplicateDims)))
    assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
  }

  test("composite fanout excludes any null component for null-unequal joins") {
    val rows = Seq.fill(5)(Seq[Integer](1, null, 100)) ++
      Seq.fill(3)(Seq[Integer](null, 20, 200))
    val ss = Seq(stages.head.copy(buildKeys = Seq(ref(0), ref(1)),
      streamKeyOrdinals = Seq(0, 1)))
    withResource(batch(rows, 3))(original =>
      withResource(LazySpillableColumnarBatch(original, "test_dim")) { built =>
        withResource(GpuJoinChain.buildKeys(Seq(built), ss)) { keys =>
          assert(GpuJoinChain.classifyBuildKeys(keys, ss).head.maxBuildMatches == 1)
          val equal = GpuJoinChain.classifyBuildKeys(keys,
            Seq(ss.head.copy(nullsEqual = true))).head
          assert(equal.maxBuildMatches == 5)
        }
      })
  }

  test("fanout bounds saturate without overflowing or accepting excessive expansion") {
    val huge = stages.map(_.copy(buildKeysUnique = false, maxBuildMatches = Int.MaxValue))
    assert(GpuJoinChain.mapWorkPerRow(huge) == Long.MaxValue)
    assert(GpuJoinChain.maxMapRows(Long.MaxValue) == Int.MaxValue.toLong)
  }

  test("each join stage builds one hash state across stream batches") {
    val rows = (0 until 20).flatMap(_ => fact)
    assert(sorted(run(rows, stages, inputBatchRows = 5,
      expectedHashBuilds = Some(stages.size.toLong))) == sorted(expected(rows, stages)))
  }

  test("join chains rebuild each stage hash for every batch when reuse is disabled") {
    val rows = (0 until 20).flatMap(_ => fact)
    assert(sorted(run(rows, stages, inputBatchRows = fact.size, reuseHash = false,
      expectedHashBuilds = Some(20L * stages.size))) == sorted(expected(rows, stages)))
  }

  test("composed maps preserve projections, build-left order and multiple input batches") {
    val rows = (0 until 50).flatMap(_ => fact)
    assert(sorted(run(rows, stages, targetSize = 64)) == sorted(expected(rows, stages)))
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  test("null equality and empty output") {
    val ss = stages.updated(0, stages.head.copy(nullsEqual = true))
    val result = run(fact, ss)
    assert(result.size == 4)
    assert(sorted(result) == sorted(expected(fact, ss)))
    assert(run(Seq(Seq[Integer](99, 1)), stages).isEmpty)
    assert(run(Seq.empty, stages).isEmpty)
    assert(run(fact, stages, dimensionRows = dims.updated(1, Seq.empty)).isEmpty)
  }

  for (nullsEqual <- Seq(false, true)) {
    test(s"multi-column keys preserve null equality=$nullsEqual") {
      val ss = stages.updated(0, stages.head.copy(buildKeys = Seq(ref(0), ref(1)),
        streamKeyOrdinals = Seq(0, 1), nullsEqual = nullsEqual))
      val rows = Seq(Seq[Integer](1, 10), Seq[Integer](1, null), Seq[Integer](null, 30),
        Seq[Integer](2, 20), Seq[Integer](1, 11))
      val result = run(rows, ss)
      assert(result.size == (if (nullsEqual) 3 else 2))
      assert(sorted(result) == sorted(expected(rows, ss)))
    }
  }

  test("partial maps restore after spilling to host and disk") {
    var spilled = 0L
    val result = run(fact, stages, stage => {
      if (stage == 1) {
        spilled += SpillFramework.stores.deviceStore.spill(Long.MaxValue)
        SpillFramework.stores.hostStore.spill(Long.MaxValue)
      }
    })
    assert(spilled > 0)
    assert(sorted(result) == sorted(expected(fact, stages)))
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  test("broadcast key stats reuse preserves current stage metrics") {
    val cache = new JoinChainBuildKeyStatsCache
    val classifications = new LocalGpuMetric
    val hits = new LocalGpuMetric
    withResource(batch(dims.head, dims.head.head.size)) { payload =>
      withResource(LazySpillableColumnarBatch(payload, "test_stats_build")) { built =>
        withResource(GpuJoinChain.buildKeys(Seq(built), stages.take(1))) { keys =>
          val first = GpuJoinChain.classifyBuildKey(keys.head, stages.head, Some(cache),
            classifications, hits)
          val currentRows = new LocalGpuMetric
          val current = stages.head.copy(outputRows = currentRows, streamProjection = Seq(1, 0))
          val second = GpuJoinChain.classifyBuildKey(keys.head, current, Some(cache),
            classifications, hits)
          assert(second.buildKeysUnique == first.buildKeysUnique)
          assert(second.maxBuildMatches == first.maxBuildMatches)
          assert(second.outputRows eq currentRows)
          assert(second.streamProjection == current.streamProjection)
          assert(classifications.value == 1)
          assert(hits.value == 1)
        }
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
  }

  test("cached build keys own separate device buffers") {
    withResource(batch(dims.head, 3)) { original =>
      withResource(LazySpillableColumnarBatch(original, "test_dim")) { built =>
        withResource(GpuJoinChain.buildKeys(Seq(built), stages.take(1))) { keys =>
          withResource(GpuColumnVector.extractBases(original)(0).getData) { payload =>
            withResource(GpuColumnVector.extractBases(keys.head.getBatch)(0).getData) { key =>
              assert(payload.getAddress != key.getAddress)
            }
          }
        }
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  test("identity build keys borrow their payload through spilling and key close") {
    val rows = Seq(Seq[Integer](1, 10), Seq[Integer](2, 20), Seq[Integer](null, 30))
    val stage = stages.head.copy(buildKeys = Seq(ref(0), ref(1)))
    val owned = withResource(batch(rows, 2))(LazySpillableColumnarBatch(_, "key_only_build"))
    withResource(owned) { built =>
      withResource(GpuJoinChain.buildKeys(Seq(built), Seq(stage))) { keys =>
        assert(keys.head.getBatch eq built.getBatch)
        keys.head.checkpoint()
        assert(SpillFramework.stores.deviceStore.spill(Long.MaxValue) > 0)
        SpillFramework.stores.hostStore.spill(Long.MaxValue)
        keys.head.restore()
        withResource(GpuColumnVector.extractBases(keys.head.getBatch)(1).copyToHost()) { col =>
          assert(col.getInt(2) == 30)
        }
      }
      // Closing borrowed keys may spill, but must leave the payload owner usable.
      withResource(GpuColumnVector.extractBases(built.getBatch)(0).copyToHost()) { col =>
        assert(col.getInt(0) == 1)
        assert(col.isNull(2))
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
  }

  test("reordered complete keys detach and semi complete keys still deduplicate") {
    val rows = Seq(Seq[Integer](1, 10), Seq[Integer](1, 10), Seq[Integer](null, 30))
    withResource(batch(rows, 2)) { original =>
      withResource(LazySpillableColumnarBatch(original, "test_full_key_build")) { built =>
        val reordered = stages.head.copy(buildKeys = Seq(ref(1), ref(0)))
        withResource(GpuJoinChain.buildKeys(Seq(built), Seq(reordered))) { keys =>
          withResource(GpuColumnVector.extractBases(original)(1).getData) { payload =>
            withResource(GpuColumnVector.extractBases(keys.head.getBatch)(0).getData) { key =>
              assert(payload.getAddress != key.getAddress)
            }
          }
          withResource(GpuColumnVector.extractBases(keys.head.getBatch)(0).copyToHost()) { col =>
            assert(col.getInt(0) == 10)
            assert(col.getInt(2) == 30)
          }
        }
        val semi = stages.head.copy(buildKeys = Seq(ref(0), ref(1)), isLeftSemi = true)
        withResource(GpuJoinChain.buildKeys(Seq(built), Seq(semi))) { keys =>
          assert(keys.head.numRows == 2)
          assert(built.numRows == 3)
        }
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
  }

  for (reuse <- Seq(false, true); nullsEqual <- Seq(false, true)) {
    test(s"key-only builds survive repeated probes and spilling reuse=$reuse nulls=$nullsEqual") {
      val dimensions = Seq(Seq(Seq[Integer](1), Seq[Integer](2), Seq[Integer](null)),
        Seq(Seq[Integer](1), Seq[Integer](1), Seq[Integer](2), Seq[Integer](null)))
      val ss = Seq.fill(2)(stages.head.copy(nullsEqual = nullsEqual))
      val rows = fact ++ fact
      var spilled = 0L
      val result = run(rows, ss, inputBatchRows = 2, dimensionRows = dimensions,
        reuseHash = reuse, action = stage => {
          if (stage == 1) {
            spilled += SpillFramework.stores.deviceStore.spill(Long.MaxValue)
            SpillFramework.stores.hostStore.spill(Long.MaxValue)
          }
        })
      assert(spilled > 0)
      assert(sorted(result) == sorted(expected(rows, ss, dimensions)))
    }
  }

  test("build key copies retry with spillable payloads") {
    assert(sorted(run(fact, stages, retryKeys = true)) == sorted(expected(fact, stages)))
    assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
  }

  for (duplicates <- Seq(false, true)) {
    test(s"early close releases input, payloads and maps duplicates=$duplicates") {
      val dimensions = if (duplicates) duplicateDims else dims
      val builds = dimensions.safeMap { rows =>
        withResource(batch(rows, rows.head.size))(LazySpillableColumnarBatch(_, "test_dim"))
      }
      withResource(builds) { _ =>
        withResource(GpuJoinChain.buildKeys(builds, stages)) { keys =>
          withResource(new CloseableBufferedIterator(Iterator(batch(fact, 2)))) { input =>
            input.head
            val attrs = (0 until 2).map(i => AttributeReference(s"c$i", IntegerType)())
            withResource(new JoinChainIterator(input, attrs, builds, keys,
                GpuJoinChain.classifyBuildKeys(keys, stages),
                options.copy(targetSize = if (duplicates) 320 else options.targetSize))) {
              iterator =>
              assert(iterator.hasNext)
            }
            assert(!input.hasNext)
          }
        }
      }
      assert(SpillFramework.stores.deviceStore.numHandles == 0)
      assert(SpillFramework.stores.hostStore.numHandles == 0)
    }
  }

  for (split <- Seq(false, true)) {
    test(s"partial probe OOM restores indices, split=$split") {
      val rows = (0 until 48).flatMap(_ => fact)
      val metrics = stages.map(_ => new LocalGpuMetric)
      val ss = stages.zip(metrics).map { case (stage, metric) =>
        stage.copy(outputRows = metric)
      }
      var injected = false
      val result = run(rows, ss, stage => {
        if (stage == 1 && !injected) {
          injected = true
          if (split) throw new GpuSplitAndRetryOOM("chain test")
          else throw new GpuRetryOOM("chain test")
        }
      }, inputBatchRows = 240)
      assert(injected)
      assert(sorted(result) == sorted(expected(rows, ss)))
      assert(metrics.head.value == 144)
      assert(metrics.last.value == expected(rows, ss).size)
      assert(SpillFramework.stores.deviceStore.numHandles == 0)
    }
  }
}

class BroadcastJoinChainQuerySuite extends SparkQueryCompareTestSuite {
  private def query(spark: SparkSession, duplicates: Boolean,
      buildLeft: Boolean = false): DataFrame = {
    val fact = spark.range(0, 200).selectExpr(
      "id AS k", "id * 13 AS payload", "id * 17 AS extra")
    val dim1 = spark.range(0, if (duplicates) 160 else 80).selectExpr(
      (if (duplicates) "id % 80 AS k1" else "id AS k1"), "id % 40 AS next_key")
    val dim2 = spark.range(0, 30).selectExpr("id AS k2", "id * 7 AS value2")
    val dim3 = spark.range(0, 25).selectExpr("id AS k3", "id * 9 AS value3")
    val first = (if (buildLeft) broadcast(dim1).join(fact, col("k") === col("k1"))
      else fact.join(broadcast(dim1), col("k") === col("k1")))
      .selectExpr("next_key AS lookup", "payload", "extra", "k AS original")
    val second = if (buildLeft) broadcast(dim2).join(first, col("lookup") === col("k2"))
      else first.join(broadcast(dim2), col("lookup") === col("k2"))
    if (buildLeft) broadcast(dim3).join(second, col("k2") === col("k3"))
      else second.join(broadcast(dim3), col("k2") === col("k3"))
  }

  for (aqe <- Seq(false, true); duplicates <- Seq(false, true);
      buildLeft <- Seq(false, true)) {
    val conf = new SparkConf().set("spark.sql.adaptive.enabled", aqe.toString)
      .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    testSparkResultsAreEqual(
      s"join chain AQE=$aqe duplicate keys=$duplicates buildLeft=$buildLeft",
      spark => query(spark, duplicates, buildLeft), conf, sort = true)(identity)
  }

  testSparkResultsAreEqual("join chain gathers string and nested payloads in small batches",
    spark => {
      val fact = spark.range(0, 200).selectExpr("id AS k",
        "concat(cast(id AS STRING), repeat('x', 200)) AS payload", "array(id, id + 1) AS nested")
      val dim1 = spark.range(0, 180).selectExpr("id AS d1")
      val dim2 = spark.range(0, 160).selectExpr("id AS d2")
      fact.join(broadcast(dim1), col("k") === col("d1"))
        .join(broadcast(dim2), col("k") === col("d2"))
    }, new SparkConf().set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.autoBroadcastJoinThreshold", "-1")
      .set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, "1024"), sort = true)(identity)

  test("reference projections and narrow chains participate; computed projections use eager") {
    val conf = new SparkConf().set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.autoBroadcastJoinThreshold", "-1")
    withGpuSparkSession(spark => {
      val df = query(spark, duplicates = false)
      df.collect()
      val joins = PlanUtils.findOperators(df.queryExecution.executedPlan,
        _.isInstanceOf[GpuBroadcastHashJoinExecBase])
        .map(_.asInstanceOf[GpuBroadcastHashJoinExecBase])
      assert(joins.exists(GpuJoinChain.chainLength(_) == 3))

      val computed = df.selectExpr("k3 + 1 AS computed", "payload")
        .join(broadcast(spark.range(0, 25)), col("computed") === col("id"))
      computed.collect()
      val outer = PlanUtils.findOperators(computed.queryExecution.executedPlan,
        _.isInstanceOf[GpuBroadcastHashJoinExecBase]).head
        .asInstanceOf[GpuBroadcastHashJoinExecBase]
      assert(GpuJoinChain.chainLength(outer) == 1)

      val narrow = spark.range(0, 40).selectExpr("id AS k")
        .join(broadcast(spark.range(0, 30).selectExpr("id AS d1")), col("k") === col("d1"))
        .join(broadcast(spark.range(0, 20).selectExpr("id AS d2")), col("k") === col("d2"))
      narrow.collect()
      val narrowJoin = PlanUtils.findOperators(narrow.queryExecution.executedPlan,
        _.isInstanceOf[GpuBroadcastHashJoinExecBase]).head
        .asInstanceOf[GpuBroadcastHashJoinExecBase]
      assert(GpuJoinChain.chainLength(narrowJoin) == 2)
    }, conf)
  }
}
