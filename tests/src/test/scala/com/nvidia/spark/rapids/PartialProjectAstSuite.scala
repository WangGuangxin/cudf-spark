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
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

class PartialProjectAstSuite extends SparkQueryCompareTestSuite {
  private def configuration(enabled: Boolean): SparkConf = new SparkConf()
    .set(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, enabled.toString)
    .set(RapidsConf.ENABLE_PROJECT_AST.key, "false")
    .set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, "false")
    .set("spark.sql.adaptive.enabled", "false")
    .set("spark.sql.ansi.enabled", "false")

  private def data(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("a", IntegerType), StructField("b", IntegerType),
      StructField("payload", StringType), StructField("nested", ArrayType(IntegerType)),
      StructField("f", DoubleType)))
    val values = Seq(Int.MinValue, -1, 0, 1, Int.MaxValue)
    val floating = Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity, -0.0, 1.0)
    val rows = (0 until 100).map { i =>
      Row(if (i % 7 == 0) null else Int.box(values(i % values.size)), i % 5,
        if (i % 11 == 0) null else s"payload-$i", Seq(i, null),
        if (i % 13 == 0) null else Double.box(floating(i % floating.size)))
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 2), schema)
  }

  private def projections(df: DataFrame): Seq[GpuTieredProject] = {
    val projects = PlanUtils.findOperators(df.queryExecution.executedPlan,
      _.isInstanceOf[GpuProjectExec]).map(_.asInstanceOf[GpuProjectExec])
    projects.map { p =>
      GpuBindReferences.bindGpuReferencesTiered(p.projectList, p.child.output,
        SQLConf.get, Map.empty)
    }
  }

  private def astExpressions(projects: Seq[GpuTieredProject]): Seq[GpuPartialProjectAst] =
    projects.flatMap(_.exprTiers.flatten).flatMap(_.collect {
      case a: GpuPartialProjectAst => a
    })

  for (tiered <- Seq(false, true); combined <- Seq(false, true)) {
    test(s"mixed outputs preserve CPU semantics: tiered=$tiered, combined=$combined") {
      def query(s: SparkSession): DataFrame = data(s).selectExpr(
        "(a * 2 + b) * 3 as x", "a * 2 + b as shared", "substring(payload, 2, 3) as text",
        "a is null or b > 1 as predicate", "a / b as ratio", "f > 1 as float_predicate",
        "nested", "payload", "payload as duplicate", "17 as constant")
      val config = configuration(false)
        .set(RapidsConf.ENABLE_TIERED_PROJECT.key, tiered.toString)
        .set(RapidsConf.ENABLE_COMBINED_EXPRESSIONS.key, combined.toString)
      val cpu = withCpuSparkSession(s => query(s).collect(), config)
      for (enabled <- Seq(false, true)) {
        withGpuSparkSession({ spark =>
          val df = query(spark)
          compareResults(false, 0.0, cpu, df.collect())
          val bound = projections(df)
          assert(bound.nonEmpty)
          assert(astExpressions(bound).nonEmpty == enabled)
          if (enabled) {
            // Unsupported whole outputs must not be turned into AST islands.
            assert(astExpressions(bound).forall(_.child.deterministic))
            assert(astExpressions(bound).forall(a => a.dataType != DoubleType))
            assert(astExpressions(bound).forall(a => a.dataType != StringType))
          }
          if (tiered) {
            assert(bound.exists(_.exprTiers.size > 1), "shared arithmetic should retain CSE")
          }
          assert(df.schema("text").dataType == StringType)
          assert(df.schema("duplicate").nullable)
        }, config.clone().set(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, enabled.toString))
      }
    }
  }

  test("ANSI arithmetic stays ordinary while an eligible predicate uses AST") {
    def query(s: SparkSession): DataFrame = data(s).selectExpr(
      "b + 1 as checked", "b > 2 as predicate", "substring(payload, 1, 2) as text")
    val config = configuration(true).set("spark.sql.ansi.enabled", "true")
    val cpu = withCpuSparkSession(s => query(s).collect(), config)
    withGpuSparkSession({ spark =>
      val df = query(spark)
      compareResults(false, 0.0, cpu, df.collect())
      val ast = astExpressions(projections(df))
      assert(ast.nonEmpty)
      assert(ast.forall(_.dataType == BooleanType), "ANSI add must not use AST")
    }, config)
  }

  test("conditional divide by zero retains branch evaluation in ANSI mode") {
    def query(s: SparkSession): DataFrame = data(s).selectExpr(
      "case when b = 0 then 0.0 else b / b end as guarded",
      "b is not null and b > 2 as predicate", "substring(payload, 1, 2) as text")
    val config = configuration(true).set("spark.sql.ansi.enabled", "true")
    val cpu = withCpuSparkSession(s => query(s).collect(), config)
    withGpuSparkSession({ spark =>
      val df = query(spark)
      compareResults(false, 0.0, cpu, df.collect())
      assert(astExpressions(projections(df)).nonEmpty)
    }, config)
  }

  test("floating AST arithmetic preserves NaN, infinity, signed zero and null") {
    def query(s: SparkSession): DataFrame = data(s).selectExpr(
      "f * 2 + 1 as arithmetic", "f * 2 as signed_zero", "f > 0 as comparison",
      "substring(payload, 1, 2) as text")
    val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        val result = df.collect()
        compareResults(false, 0.0, cpu, result)
        cpu.zip(result).foreach { case (expected, actual) =>
          if (!expected.isNullAt(1) && expected.getDouble(1) == 0.0) {
            assert(java.lang.Double.doubleToRawLongBits(expected.getDouble(1)) ==
              java.lang.Double.doubleToRawLongBits(actual.getDouble(1)))
          }
        }
        val ast = astExpressions(projections(df))
        assert(ast.nonEmpty == enabled)
        assert(ast.forall(_.dataType == DoubleType), "floating comparison must remain ordinary")
      }, configuration(enabled))
    }
  }

  test("ANSI overflow is still reported with partial AST enabled") {
    def query(s: SparkSession): DataFrame = data(s).selectExpr(
      "a + 1 as overflow", "b > 2 as predicate", "substring(payload, 1, 2) as text")
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = query(spark)
        assert(astExpressions(projections(df)).nonEmpty == enabled)
        val error = intercept[Exception](df.collect())
        assert(error.toString.contains("overflow"), error.toString)
      }, configuration(enabled).set("spark.sql.ansi.enabled", "true"))
    }
  }

  test("seeded nondeterministic output agrees with GPU baseline") {
    def query(s: SparkSession): DataFrame = data(s).selectExpr(
      "rand(10) as random", "b * 2 + 1 as arithmetic", "substring(payload, 1, 2) as text")
    val baseline = withGpuSparkSession(s => query(s).collect(), configuration(false))
    withGpuSparkSession({ spark =>
      val df = query(spark)
      compareResults(false, 0.0, baseline, df.collect())
      assert(astExpressions(projections(df)).nonEmpty)
    }, configuration(true))
  }

  for (threshold <- Seq(-1, 2, 10)) {
    test(s"partial AST composes with filter/project fusion: threshold=$threshold") {
      def query(s: SparkSession): DataFrame = data(s).filter(col("b") > threshold)
        .selectExpr("b * 2 + 1 as arithmetic", "payload", "payload as duplicate")
      val cpu = withCpuSparkSession(s => query(s).collect(), configuration(false))
      for (enabled <- Seq(false, true)) {
        withGpuSparkSession({ spark =>
          val df = query(spark)
          compareResults(false, 0.0, cpu, df.collect())
          val fused = PlanUtils.findOperators(df.queryExecution.executedPlan,
            _.isInstanceOf[GpuFilterProjectExec]).map(_.asInstanceOf[GpuFilterProjectExec])
          assert(fused.nonEmpty)
          val ast = fused.flatMap { p =>
            val bound = GpuFilterProject.bind(p.condition, p.projectList, p.child.output,
              p.useAst, SQLConf.get, Map.empty)
            astExpressions(Seq(bound.tiered))
          }
          assert(ast.nonEmpty == enabled)
        }, configuration(enabled).set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, "true"))
      }
    }
  }

  test("legacy all-AST switch takes precedence and partial switch is independent") {
    for (allAst <- Seq(false, true); partial <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = data(spark).selectExpr("b * 2 + 1 as arithmetic")
        df.collect()
        val legacy = PlanUtils.findOperators(df.queryExecution.executedPlan,
          _.isInstanceOf[GpuProjectAstExec])
        assert(legacy.nonEmpty == allAst)
        assert(astExpressions(projections(df)).nonEmpty == (partial && !allAst))
      }, configuration(partial).set(RapidsConf.ENABLE_PROJECT_AST.key, allAst.toString))
    }
  }

  test("partial AST is disabled by default") {
    assert(!new RapidsConf(Map.empty[String, String]).get(RapidsConf.ENABLE_PROJECT_PARTIAL_AST))
  }
}
