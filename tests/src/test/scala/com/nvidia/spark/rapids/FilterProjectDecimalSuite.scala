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
import org.apache.spark.sql.rapids.{GpuDecimalDivide, GpuDecimalMultiply, GpuGreaterThan}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

class FilterProjectDecimalSuite extends SparkQueryCompareTestSuite {
  private val decimalTypes = Seq(DecimalType(9, 2), DecimalType(18, 6), DecimalType(38, 10))

  private def configuration(enabled: Boolean, ansi: Boolean): SparkConf = new SparkConf()
    .set(RapidsConf.ENABLE_FILTER_PROJECT_FUSION.key, enabled.toString)
    .set(RapidsConf.ENABLE_PROJECT_AST.key, "false")
    .set(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "false")
    .set("spark.sql.adaptive.enabled", "false")
    .set("spark.sql.ansi.enabled", ansi.toString)

  private def data(spark: SparkSession, dt: DecimalType): DataFrame = {
    val schema = StructType(Seq(StructField("id", IntegerType, nullable = false),
      StructField("price", dt), StructField("factor", DecimalType(9, 4)),
      StructField("payload", StringType), StructField("unused", StringType)))
    val unit = BigDecimal(1) / BigDecimal(10).pow(dt.scale)
    val maximum = BigDecimal(10).pow(dt.precision - dt.scale) - unit
    val values = Seq(None, Some(BigDecimal(0)), Some(unit), Some(-unit),
      Some(BigDecimal("1.125")), Some(BigDecimal("-1.125")), Some(maximum), Some(-maximum),
      Some(BigDecimal(1) + 5 * unit), Some(-BigDecimal(1) - 5 * unit))
    val factors = Seq(BigDecimal(0), BigDecimal("0.0125"), BigDecimal("-0.0125"))
    val rows = (0 until 96).map { i =>
      val value = values(i % values.size).map(_.setScale(dt.scale,
        BigDecimal.RoundingMode.HALF_UP).bigDecimal).orNull
      Row(i, value, if (i % 7 == 0) null else factors(i % factors.size).bigDecimal,
        if (i % 11 == 0) null else s"payload-$i", "unused-wide-payload" * 10)
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 2), schema)
  }

  private def fused(df: DataFrame): Seq[GpuFilterProjectExec] =
    PlanUtils.findOperators(df.queryExecution.executedPlan,
      _.isInstanceOf[GpuFilterProjectExec]).map(_.asInstanceOf[GpuFilterProjectExec])

  private def compareVariants(query: SparkSession => DataFrame,
      expectFusion: Boolean = true, ansi: Boolean = false,
      settings: SparkConf = new SparkConf()): Unit = {
    val cpu = withCpuSparkSession(s => query(s).collect(),
      settings.clone().set("spark.sql.ansi.enabled", ansi.toString))
    for (enabled <- Seq(false, true)) {
      val conf = configuration(enabled, ansi)
      settings.getAll.foreach { case (key, value) => conf.set(key, value) }
      withGpuSparkSession({ spark =>
        val df = query(spark)
        compareResults(false, 0.0, cpu, df.collect())
        val operators = fused(df)
        assert(operators.nonEmpty == (enabled && expectFusion),
          df.queryExecution.executedPlan.toString)
        assert(operators.forall(!_.useAst), "decimal evaluation must retain tiered projection")
        if (enabled && expectFusion &&
            conf.get(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "false") == "true") {
          val usesPartialAst = operators.exists { p =>
            val bound = GpuFilterProject.bind(p.condition, p.projectList, p.child.output,
              p.useAst, SQLConf.get, Map.empty)
            bound.tiered.exprTiers.flatten.exists(_.exists(_.isInstanceOf[GpuPartialProjectAst]))
          }
          assert(usesPartialAst, "eligible integer output should still use partial AST")
        }
      }, conf)
    }
  }

  for (dt <- decimalTypes; condition <- Seq(
      "price > 0", "price is null", "price is not null and price <= 0",
      "price is null or price >= 0", "price = 0 or price != 0")) {
    test(s"decimal predicate and direct outputs: $dt, $condition") {
      compareVariants(s => data(s, dt).filter(condition)
        .selectExpr("price", "price as duplicate", "payload"))
    }
  }

  for (dt <- decimalTypes; threshold <- Seq(-1, 40, 100)) {
    test(s"decimal arithmetic, casts and CSE agree with CPU: $dt, threshold=$threshold") {
      val smaller = DecimalType(dt.precision - 1, math.max(0, dt.scale - 1)).sql
      compareVariants(s => data(s, dt).filter(col("id") > threshold).selectExpr(
        "price", "price as duplicate", "price + price as shared",
        "(price + price) * factor as reused", "price - price as difference",
        "price * factor as product", "price / factor as quotient",
        s"cast(price as $smaller) as rounded", "cast(price as bigint) as integral",
        s"cast(2.345 as ${dt.sql}) as constant", "payload"))
    }
  }

  for (precisionLoss <- Seq(false, true); tiered <- Seq(false, true)) {
    test(s"decimal precision-loss and tiered settings: loss=$precisionLoss, tiered=$tiered") {
      compareVariants(s => data(s, DecimalType(18, 6)).filter("price > factor")
        .selectExpr("price * factor as product", "price / factor as quotient", "payload"),
        settings = new SparkConf()
          .set("spark.sql.decimalOperations.allowPrecisionLoss", precisionLoss.toString)
          .set(RapidsConf.ENABLE_TIERED_PROJECT.key, tiered.toString))
    }
  }

  test("decimal predicate can contain safe arithmetic and scale alignment") {
    compareVariants(s => data(s, DecimalType(18, 6)).filter("price * factor > factor + factor")
      .selectExpr("cast(price / factor as decimal(12, 3)) as value", "payload"))
  }

  test("integer arithmetic still uses partial AST alongside decimal outputs") {
    compareVariants(s => data(s, DecimalType(18, 6)).filter("price > 0")
      .selectExpr("id * 2 + 1 as integer_value", "price * factor as decimal_value", "payload"),
      settings = new SparkConf().set(RapidsConf.ENABLE_PROJECT_PARTIAL_AST.key, "true"))
  }

  test("decimal predicates with matching operand types support ANSI mode") {
    compareVariants(s => data(s, DecimalType(9, 4)).filter("price > factor")
      .selectExpr("price", "payload"), ansi = true)
  }

  for (expression <- Seq("price + price", "price - price", "price * factor",
      "price / factor", "cast(price as decimal(3, 1))")) {
    test(s"throwing decimal computations retain ordinary operators: $expression") {
      withGpuSparkSession({ spark =>
        val df = data(spark, DecimalType(38, 10)).filter("id > 40")
          .selectExpr(s"$expression as value", "payload")
        assert(fused(df).isEmpty, df.queryExecution.executedPlan.toString)
      }, configuration(enabled = true, ansi = true))
    }
  }

  test("ANSI errors on rows removed by the filter are not evaluated") {
    compareVariants(s => data(s, DecimalType(38, 10)).filter("id = 2")
      .selectExpr("cast(price as decimal(3, 1)) as narrowed", "price / factor as quotient"),
      expectFusion = false, ansi = true)
  }

  test("ANSI decimal overflow remains an error with fusion enabled") {
    for (enabled <- Seq(false, true)) {
      withGpuSparkSession({ spark =>
        val df = data(spark, DecimalType(38, 10)).filter("id = 6")
          .selectExpr("cast(price as decimal(3, 1)) as narrowed")
        assert(fused(df).isEmpty)
        intercept[Exception](df.collect())
      }, configuration(enabled, ansi = true))
    }
  }

  test("explicit decimal error guards do not rely on shim side-effect annotations") {
    val dt = DecimalType(18, 6)
    val a = GpuLiteral(Decimal(1), dt)
    val b = GpuLiteral(Decimal(0), dt)
    val condition = GpuGreaterThan(a, b)
    val throwing = Seq(
      GpuCheckOverflow(a, dt, nullOnOverflow = false),
      GpuCast(a, DecimalType(3, 1), ansiMode = true)(),
      GpuDecimalMultiply(a, b, dt, failOnError = true),
      GpuDecimalDivide(a, b, dt, failOnError = false, failOnDivideByZero = true),
      GpuDecimalDivide(a, b, dt, failOnError = true, failOnDivideByZero = false))
    throwing.foreach { expr =>
      assert(!GpuFilterProjectExec.eligible(condition, Seq(expr)), expr.toString)
      assert(!GpuFilterProjectExec.eligible(GpuGreaterThan(expr, b), Seq(a)), expr.toString)
    }
    assert(!GpuFilterProjectExec.eligible(condition,
      Seq(GpuCast(GpuLiteral(UTF8String.fromString("invalid"), StringType), dt)())))
  }
}
