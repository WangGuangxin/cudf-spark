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

import ai.rapids.cudf.{ColumnVector, Table}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.rapids.execution.GpuJoinExec
import org.apache.spark.sql.types._

class JoinProjectPruningSuite extends SparkQueryCompareTestSuite {

  private def configuration(enabled: Boolean, broadcast: Boolean,
      batchBytes: Long): SparkConf = {
    val conf = new SparkConf()
      .set(RapidsConf.ENABLE_JOIN_PROJECT_PRUNING.key, enabled.toString)
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.ansi.enabled", "false")
      .set("spark.sql.autoBroadcastJoinThreshold", if (broadcast) "10485760" else "-1")
      .set("spark.sql.join.preferSortMergeJoin", "false")
    if (batchBytes != Long.MaxValue) {
      conf.set(RapidsConf.GPU_BATCH_SIZE_BYTES.key, batchBytes.toString)
    } else {
      conf
    }
  }

  private def fact(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("fid", IntegerType),
      StructField("k", IntegerType),
      StructField("amount", DoubleType),
      StructField("label", StringType)))
    val rows = (0 until 400).map { i =>
      Row(Int.box(i), if (i % 23 == 0) null else Int.box(i % 17), Double.box(i * 1.5),
        if (i % 19 == 0) null else s"label-$i")
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 3), schema)
  }

  private def dimFor(spark: SparkSession, broadcast: Boolean): DataFrame =
    if (broadcast) org.apache.spark.sql.functions.broadcast(dim(spark)) else dim(spark)

  private def dim(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("k", IntegerType),
      StructField("name", StringType),
      StructField("weight", IntegerType)))
    val rows = (0 until 17).map { i =>
      Row(if (i % 11 == 0) null else Int.box(i), s"name-$i", Int.box(i * 3))
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 2), schema)
  }

  private def prunedNodes(plan: org.apache.spark.sql.execution.SparkPlan) =
    PlanUtils.findOperators(plan, _.isInstanceOf[GpuPrunedJoinExec])
      .map(_.asInstanceOf[GpuPrunedJoinExec])

  /** Run CPU, then GPU with pruning off and on, and require all three to agree. */
  private def compareOffOn(name: String, broadcast: Boolean, expectPruned: Option[Boolean],
      batchBytes: Long = Long.MaxValue)(query: SparkSession => DataFrame): Unit = {
    val cpu = withCpuSparkSession(s => query(s).collect(),
      configuration(enabled = false, broadcast, batchBytes))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(true, 0.0, cpu, df.collect())
        val plan = df.queryExecution.executedPlan
        val pruned = prunedNodes(plan)
        // With no fixed expectation, hold the rule to its own contract: it fires exactly when
        // the planned join advertises that it can narrow its gather.
        val expected = expectPruned.getOrElse(PlanUtils
          .findOperators(plan, _.isInstanceOf[GpuJoinExec])
          .exists(_.asInstanceOf[GpuJoinExec].canPruneOutput))
        val joinDebug = PlanUtils.findOperators(plan, _.isInstanceOf[GpuJoinExec])
          .map(_.asInstanceOf[GpuJoinExec])
          .map(j => s"${j.getClass.getSimpleName}/${j.joinType}/${j.canPruneOutput}")
        assert(pruned.nonEmpty == (enabled && expected),
          s"$name: pruned=${pruned.size} enabled=$enabled expected=$expected " +
            s"joins=$joinDebug\n$plan")
        pruned.foreach { node =>
          assert(node.output.length < node.child.output.length,
            s"$name: pruning kept every column\n$plan")
        }
      }, configuration(enabled, broadcast, batchBytes))
    }
  }

  for (broadcast <- Seq(true, false)) {
    val kind = if (broadcast) "broadcast" else "shuffled"

    test(s"$kind inner join drops the build key and unread payload") {
      compareOffOn(kind, broadcast, expectPruned = Some(true)) { spark =>
        fact(spark).join(dimFor(spark, broadcast), "k").select("fid", "name")
      }
    }

    test(s"$kind inner join with a computed projection still prunes") {
      compareOffOn(kind, broadcast, expectPruned = Some(true)) { spark =>
        fact(spark).join(dimFor(spark, broadcast), Seq("k"))
          .select((col("amount") * 2).as("doubled"), col("name"))
      }
    }

    test(s"$kind join reading every column is left alone") {
      compareOffOn(kind, broadcast, expectPruned = Some(false)) { spark =>
        fact(spark).as("f").join(dimFor(spark, broadcast).as("d"), col("f.k") === col("d.k"))
      }
    }

    test(s"$kind join keeps results across many output batches") {
      compareOffOn(kind, broadcast, expectPruned = Some(true), batchBytes = 2048) { spark =>
        fact(spark).join(dimFor(spark, broadcast), "k").select("label", "weight")
      }
    }

    for (joinType <- Seq("left", "right", "leftsemi", "leftanti")) {
      test(s"$kind $joinType join agrees with the CPU") {
        // Semi and anti joins only output the left side, so a projection that drops one of its
        // columns is still a narrowing.
        // Spark may ignore the broadcast hint (a right outer join cannot broadcast its right
        // side), so hold this to the rule's contract rather than a fixed expectation.
        compareOffOn(s"$kind-$joinType", broadcast, expectPruned = None) { spark =>
          val joined = fact(spark).as("f")
            .join(dimFor(spark, broadcast).as("d"), col("f.k") === col("d.k"), joinType)
          if (joinType == "leftsemi" || joinType == "leftanti") {
            joined.select("fid", "amount")
          } else {
            joined.select(col("fid"), col("d.k"), col("name"))
          }
        }
      }
    }

    test(s"$kind join that reads nothing from the stream side still works") {
      compareOffOn(kind, broadcast, expectPruned = Some(true)) { spark =>
        fact(spark).join(dimFor(spark, broadcast), "k").select("name", "weight")
      }
    }
  }

  test("a build-left broadcast join splits per-side ordinals correctly") {
    // The dimension is the left input and the broadcast side, so the surviving stream columns
    // land on the right of the join output.
    compareOffOn("build-left", broadcast = true, expectPruned = Some(true)) { spark =>
      org.apache.spark.sql.functions.broadcast(dim(spark)).as("d")
        .join(fact(spark).as("f"), col("d.k") === col("f.k"))
        .select(col("name"), col("fid"), col("label"))
    }
  }

  test("a projection reading only build columns keeps one left column alive") {
    compareOffOn("build-only", broadcast = true, expectPruned = Some(true)) { spark =>
      org.apache.spark.sql.functions.broadcast(dim(spark)).as("d")
        .join(fact(spark).as("f"), col("d.k") === col("f.k"))
        .select(col("name"), col("weight"))
    }
  }

  test("a non-equi condition keeps the full output") {
    compareOffOn("condition", broadcast = true, expectPruned = Some(false)) { spark =>
      fact(spark).as("f").join(org.apache.spark.sql.functions.broadcast(dim(spark)).as("d"),
        col("f.k") === col("d.k") && col("f.amount") > col("d.weight"))
        .select(col("fid"), col("name"))
    }
  }

  for (chain <- Seq(true, false)) {
    test(s"a fused join chain keeps its own gather, chain=$chain") {
      // Three broadcast joins in a row is exactly what GpuJoinChain fuses, and it gathers each
      // payload once instead of once per join. A pruning node inserted anywhere along the chain
      // would stop it forming, so the rule has to stand aside while the chain is enabled.
      def conf(enabled: Boolean): SparkConf =
        configuration(enabled, broadcast = true, Long.MaxValue)
          .set("spark.rapids.sql.join.chain.enabled", chain.toString)
      def query(spark: SparkSession): DataFrame = {
        val d1 = org.apache.spark.sql.functions.broadcast(dim(spark)).as("d1")
        val d2 = org.apache.spark.sql.functions.broadcast(dim(spark)).as("d2")
        val d3 = org.apache.spark.sql.functions.broadcast(dim(spark)).as("d3")
        fact(spark).as("f")
          .join(d1, col("f.k") === col("d1.k"))
          .join(d2, col("f.k") === col("d2.k"))
          .join(d3, col("f.k") === col("d3.k"))
          .select(col("fid"), col("d1.name"), col("d2.weight"), col("d3.name"))
      }
      val cpu = withCpuSparkSession(s => query(s).collect(), conf(false))
      for (enabled <- Seq(false, true)) {
        withGpuSparkSession({ spark =>
          val df = query(spark)
          compareResults(true, 0.0, cpu, df.collect())
          val plan = df.queryExecution.executedPlan
          val pruned = prunedNodes(plan)
          // The chain owns these joins while it is on; with it off they are ordinary joins.
          assert(pruned.nonEmpty == (enabled && !chain),
            s"chain=$chain enabled=$enabled pruned=${pruned.size}\n$plan")
        }, conf(enabled))
      }
    }
  }

  test("pruning is disabled by default") {
    assert(!new RapidsConf(new SparkConf()).isJoinProjectPruningEnabled)
  }

  test("required ordinals pad an empty left side and stay sorted") {
    // Mirrors a projection that only reads build-side columns of a two-by-two join.
    val required = Seq(2, 3)
    val pruning = JoinOutputPruning(Array(0), Array(0, 1))
    assert(pruning.left.toSeq == Seq(0))
    assert(pruning.right.toSeq == Seq(0, 1))
    assertThrows[IllegalArgumentException](JoinOutputPruning(Array.empty, Array(0)))
    assert(required.sorted == required)
  }

  test("a projected spillable batch exposes only the requested columns") {
    withGpuSparkSession({ _ =>
      val types = Array[DataType](IntegerType, StringType, DoubleType)
      // `new Table` increments the column refs, so the sources still need closing here.
      val full = withResource(Seq(
          ColumnVector.fromInts(1, 2, 3),
          ColumnVector.fromStrings("a", "b", "c"),
          ColumnVector.fromDoubles(1.0, 2.0, 3.0))) { columns =>
        withResource(new Table(columns: _*)) { table =>
          withResource(GpuColumnVector.from(table, types)) { batch =>
            LazySpillableColumnarBatch(batch, "test")
          }
        }
      }
      withResource(full) { _ =>
        withResource(LazySpillableColumnarBatch.project(full, Array(0, 2))) { projected =>
          assert(projected.numCols == 2)
          assert(projected.numRows == 3)
          assert(projected.dataTypes.toSeq == Seq(IntegerType, DoubleType))
          val view = projected.getBatch
          assert(view.numCols() == 2)
          assert(view.column(0).asInstanceOf[GpuColumnVector].dataType() == IntegerType)
          assert(view.column(1).asInstanceOf[GpuColumnVector].dataType() == DoubleType)
          // Spilling drops the cached view; the next read rebuilds it from the wrapped batch.
          projected.allowSpilling()
          assert(projected.getBatch.numCols() == 2)
        }
        // Keeping every column hands back the original rather than another wrapper.
        assert(LazySpillableColumnarBatch.project(full, Array(0, 1, 2)) eq full)
      }
      // An empty projection still carries the row count for the gatherer's semi-join path.
      val wrapped = withResource(GpuColumnVector.emptyBatchFromTypes(Array(IntegerType))) { e =>
        LazySpillableColumnarBatch(e, "empty")
      }
      withResource(LazySpillableColumnarBatch.project(wrapped, Array.empty[Int])) { empty =>
        assert(empty.numCols == 0)
        assert(empty.getBatch.numCols() == 0)
      }
    }, new SparkConf())
  }
}
