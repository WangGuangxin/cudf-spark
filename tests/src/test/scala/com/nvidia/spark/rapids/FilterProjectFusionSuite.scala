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

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeReference,
  NamedExpression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.functions._
import org.apache.spark.sql.rapids.{GpuAdd, GpuAnd, GpuGreaterThan}
import org.apache.spark.sql.types._

class FilterProjectFusionSuite extends SparkQueryCompareTestSuite {
  private case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Planning-only leaf")
    override def outputOrdering: Seq[SortOrder] = Seq(SortOrder(output.head, Ascending))
    override def outputPartitioning: HashPartitioning = HashPartitioning(Seq(output.head), 1)
  }

  private def configuration(enabled: Boolean, ast: Boolean): SparkConf = new SparkConf()
    .set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, enabled.toString)
    .set(RapidsConf.ENABLE_PROJECT_AST.key, ast.toString)
    .set("spark.sql.adaptive.enabled", "false")
    .set("spark.sql.ansi.enabled", "false")

  private def data(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("a", IntegerType),
      StructField("b", IntegerType),
      StructField("payload", StringType),
      StructField("nested", ArrayType(IntegerType))))
    val rows = (0 until 100).map { i =>
      Row(if (i % 7 == 0) null else Int.box(i), i * 2,
        if (i % 5 == 0) null else s"payload-$i",
        if (i % 11 == 0) null else Seq(if (i % 13 == 0) null else Int.box(i), Int.box(i + 1)))
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 2), schema)
  }

  for (ast <- Seq(false, true); threshold <- Seq(-1, 40, 101)) {
    test(s"CPU/off/on agree with nullable selection, AST=$ast, threshold=$threshold") {
      def query(spark: SparkSession): DataFrame = data(spark)
        .filter(col("a").isNull || col("a") > threshold)
        .select((col("b") + lit(3)).as("sum"), col("a").as("first"),
          col("a").as("duplicate"), lit(17).as("constant"))
      val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false, ast))
      for (enabled <- Seq(false, true)) {
        withGpuSparkSession({ spark =>
          val df = query(spark)
          val result = df.collect()
          compareResults(false, 0.0, cpu, result)
          val fused = PlanUtils.findOperators(df.queryExecution.executedPlan,
            _.isInstanceOf[GpuFilterProjectExec])
          assert(fused.nonEmpty == enabled, df.queryExecution.executedPlan.toString)
          if (enabled) {
            assert(fused.forall(_.asInstanceOf[GpuFilterProjectExec].useAst == ast))
          }
        }, configuration(enabled, ast))
      }
    }
  }

  test("string and nested payloads remain direct outputs") {
    def query(spark: SparkSession): DataFrame = data(spark)
      .filter(col("a") > 40)
      .select((col("b") + 1).as("sum"), col("payload"), col("nested"),
        col("payload").as("duplicate"))
    val cpu = withCpuSparkSession(s => query(s).collect())
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(false, 0.0, cpu, df.collect())
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuFilterProjectExec]).nonEmpty == enabled)
      }, configuration(enabled, ast = true))
    }
  }

  for (expression <- Seq("rand(10)", "cast(payload as int)", "b / 2")) {
    test(s"unsupported or side-effecting projection falls back: $expression") {
      val conf = configuration(enabled = true, ast = false)
        .set("spark.sql.ansi.enabled", "true")
      withGpuSparkSession({ spark =>
        val df = data(spark).filter(col("a") > 40).selectExpr(expression)
        assert(PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuFilterProjectExec]).isEmpty)
      }, conf)
    }
  }

  test("fusion is disabled by default") {
    assert(!new RapidsConf(new SparkConf()).isFilterProjectFusionEnabled)
  }

  test("rewrite preserves output attributes and single-batch planning contracts") {
    val a = AttributeReference("a", IntegerType)()
    val child = TestLeaf(Seq(a))
    val filter = GpuFilterExec(GpuAnd(GpuIsNotNull(a),
      GpuGreaterThan(a, GpuLiteral(0, IntegerType))), child)()
    val projects = List[NamedExpression](GpuAlias(filter.output.head, "renamed")(),
      GpuAlias(GpuAdd(a, GpuLiteral(1, IntegerType), failOnError = false)(), "sum")())
    val project = GpuProjectExec(projects, filter)
    val fused = GpuFilterProjectExec.fuse(project)
    assert(fused.isInstanceOf[GpuFilterProjectExec])
    // Scan-pruning rules for ordinary projections must not drop predicate-only columns.
    assert(GpuProjectExecLike.unapply(fused).isEmpty)
    assert(fused.output == project.output)
    assert(fused.outputOrdering == project.outputOrdering)
    assert(fused.outputPartitioning == project.outputPartitioning)
    assert(fused.children == Seq(child))
    val noSplit = project.copy(enablePreSplit = false)
    assert(GpuFilterProjectExec.fuse(noSplit) eq noSplit)
    val noCoalesce = project.copy(child = filter.copy()(coalesceAfter = false))
    assert(GpuFilterProjectExec.fuse(noCoalesce) eq noCoalesce)
  }

  test("throwing ANSI arithmetic and unsupported predicates are not eligible") {
    val a = AttributeReference("a", IntegerType)()
    val condition = GpuGreaterThan(a, GpuLiteral(0, IntegerType))
    val throwing = GpuAlias(
      GpuAdd(a, GpuLiteral(1, IntegerType), failOnError = true)(), "sum")()
    assert(!GpuFilterProjectExec.eligible(condition, Seq(throwing)))
    val s = AttributeReference("s", StringType)()
    assert(!GpuFilterProjectExec.eligible(GpuIsNotNull(s), Seq(a)))
  }
}
