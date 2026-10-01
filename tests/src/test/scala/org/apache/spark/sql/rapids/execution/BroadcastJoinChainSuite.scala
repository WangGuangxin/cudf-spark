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
  private def expected(fact: Seq[Row], ss: Seq[JoinChainStage]): Seq[Row] = {
    ss.zip(dims).foldLeft(fact) { case (rows, (stage, dim)) =>
      rows.flatMap { original =>
        val row = stage.streamProjection.map(original)
        dim.filter { build =>
          stage.streamKeyOrdinals.zip(stage.buildKeys).forall { case (s, b) =>
            val l = row(s)
            val r = build(b.asInstanceOf[GpuBoundReference].ordinal)
            if (l == null || r == null) stage.nullsEqual && l == r else l == r
          }
        }.map { build =>
          if (stage.buildSide == GpuBuildRight) row ++ build else build ++ row
        }
      }
    }
  }

  private def run(fact: Seq[Row], ss: Seq[JoinChainStage],
      action: Int => Unit = _ => (), targetSize: Long = options.targetSize,
      inputBatchRows: Int = 120, dimensionRows: Seq[Seq[Row]] = dims,
      retryKeys: Boolean = false): Seq[Row] = {
    val builds = dimensionRows.zip(dims).safeMap { case (rows, shape) =>
      withResource(batch(rows, shape.head.size))(LazySpillableColumnarBatch(_, "test_dim"))
    }
    withResource(builds) { _ =>
      if (retryKeys) {
        RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
          RmmSpark.OomInjectionType.GPU.ordinal, 0)
      }
      withResource(GpuJoinChain.buildKeys(builds, ss)) { keys =>
        val input = fact.grouped(inputBatchRows).map(batch(_, 2))
        val attrs = (0 until 2).map(i => AttributeReference(s"c$i", IntegerType)())
        val result = withResource(new JoinChainIterator(input, attrs, builds, keys, ss,
            options.copy(targetSize = targetSize)) {
          override protected def beforeProbe(stage: Int): Unit = action(stage)
        }) { iterator =>
          iterator.flatMap { output =>
            withResource(output) { gpu =>
              withResource(GpuColumnVector.extractBases(gpu).toSeq.safeMap(_.copyToHost())) {
                columns =>
                  (0 until gpu.numRows()).map { row =>
                    columns.map(c => if (c.isNull(row)) null else Integer.valueOf(c.getInt(row)))
                  }
              }
            }
          }.toVector
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

  test("build key copies retry with spillable payloads") {
    assert(sorted(run(fact, stages, retryKeys = true)) == sorted(expected(fact, stages)))
    assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
  }

  test("early close releases buffered input, payloads and composed maps") {
    val builds = dims.safeMap { rows =>
      withResource(batch(rows, rows.head.size))(LazySpillableColumnarBatch(_, "test_dim"))
    }
    withResource(builds) { _ =>
      withResource(GpuJoinChain.buildKeys(builds, stages)) { keys =>
        withResource(new CloseableBufferedIterator(Iterator(batch(fact, 2)))) { input =>
          input.head
          val attrs = (0 until 2).map(i => AttributeReference(s"c$i", IntegerType)())
          withResource(new JoinChainIterator(input, attrs, builds, keys, stages,
              options)) { iterator =>
            assert(iterator.hasNext)
          }
          assert(!input.hasNext)
        }
      }
    }
    assert(SpillFramework.stores.deviceStore.numHandles == 0)
    assert(SpillFramework.stores.hostStore.numHandles == 0)
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

  test("reference projections participate; computed projections and narrow chains use eager") {
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
      assert(GpuJoinChain.chainLength(narrowJoin) == 1)
    }, conf)
  }
}
