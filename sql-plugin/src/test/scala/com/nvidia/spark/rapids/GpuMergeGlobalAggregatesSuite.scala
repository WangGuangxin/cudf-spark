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

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, LogicalPlan}
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class GpuMergeGlobalAggregatesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var session: TrampolineConnectShims.SparkSession = _
  private var ownsSession = false
  private var previousSettings = Map.empty[String, Option[String]]
  private val table = "merge_global_aggregate_input"

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    ownsSession = SparkSession.getDefaultSession.isEmpty
    previousSettings = SparkSession.getDefaultSession.map { existing =>
      Seq("spark.sql.ansi.enabled", "spark.rapids.sql.enabled").map { key =>
        key -> existing.conf.getOption(key)
      }.toMap
    }.getOrElse(Map.empty)
    session = TrampolineConnectShims.getBuilder().master("local[1]")
      .appName("GpuMergeGlobalAggregatesSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.ansi.enabled", "true")
      .config("spark.rapids.sql.enabled", "false").getOrCreate()
    session.sql("""SELECT * FROM VALUES
      (1, CAST(1.25 AS DECIMAL(7,2)), 10, 'bad'),
      (2, CAST(1.25 AS DECIMAL(7,2)), 20, '2'),
      (3, CAST(3.50 AS DECIMAL(7,2)), 30, '3'),
      (4, CAST(NULL AS DECIMAL(7,2)), 40, '4'),
      (5, CAST(5.75 AS DECIMAL(7,2)), NULL, '5'),
      (NULL, CAST(8.00 AS DECIMAL(7,2)), 60, '6')
      AS t(k, price, cost, label)""").createOrReplaceTempView(table)
  }

  override protected def afterAll(): Unit = {
    try {
      if (session != null) {
        session.catalog.dropTempView(table)
        session.catalog.dropTempView(table + "_other")
        session.catalog.dropTempView(table + "_overflow")
        if (ownsSession) {
          session.stop()
        } else {
          previousSettings.foreach {
            case (key, Some(value)) => session.conf.set(key, value)
            case (key, None) => session.conf.unset(key)
          }
        }
      }
    } finally {
      super.afterAll()
    }
  }

  private def aggregates(plan: LogicalPlan): Int =
    plan.collect { case _: Aggregate => 1 }.sum

  private def check(query: String, branches: Int): LogicalPlan = {
    val original = session.sql(query)
    val analyzed = original.queryExecution.analyzed
    assert(aggregates(analyzed) == branches)
    val merged = GpuMergeGlobalAggregates(analyzed)
    assert(aggregates(merged) == 1, merged.treeString)
    assert(GpuMergeGlobalAggregates(merged).fastEquals(merged))
    assert(merged.output == analyzed.output)
    val actual = TrampolineConnectShims.createDataFrame(session, merged)
    assert(actual.schema == original.schema)
    assert(actual.collect().toSeq == original.collect().toSeq)
    merged
  }

  test("six filtered AVG/count/count-distinct branches retain decimal and NULL semantics") {
    val predicates = Seq("k BETWEEN 0 AND 2", "k BETWEEN 2 AND 4", "k = 5", "k IS NULL",
      "k > 100", "k IS NOT NULL")
    val inputs = predicates.zipWithIndex.map { case (predicate, index) =>
      s"""(SELECT AVG(price) AS a$index, COUNT(price) AS c$index,
        COUNT(DISTINCT price) AS d$index FROM $table
        WHERE $predicate AND (price BETWEEN 1 AND 1+4 OR cost BETWEEN 10 AND 20+50)) b$index"""
    }
    check("SELECT * FROM " + inputs.mkString(","), 6)
  }

  test("empty and all-NULL branches still produce exactly one row") {
    check(s"""SELECT * FROM
      (SELECT AVG(price) a, COUNT(price) c, COUNT(DISTINCT price) d
       FROM $table WHERE k > 100) x,
      (SELECT AVG(price) b, COUNT(price) e, COUNT(DISTINCT price) f
       FROM $table WHERE k = 4) y""", 2)
    check(s"""SELECT * FROM
      (SELECT SUM(price) a, COUNT(*) c FROM $table WHERE k > 100) x,
      (SELECT MIN(price) b, MAX(price) d FROM $table WHERE k > 200) y""", 2)
  }

  test("common input predicates and existing aggregate FILTER clauses are both preserved") {
    check(s"""SELECT * FROM
      (SELECT COUNT(DISTINCT price) FILTER (WHERE cost > 10) a,
        SUM(price) FILTER (WHERE k < 4) b
       FROM $table WHERE cost >= 0 AND k BETWEEN 1 AND 4) x,
      (SELECT COUNT(DISTINCT price) FILTER (WHERE cost < 40) c,
        AVG(price) FILTER (WHERE price IS NOT NULL) d
       FROM $table WHERE cost >= 0 AND k BETWEEN 2 AND 5) y""", 2)
  }

  test("transparent aliases and projections map different relation ExprIds to the same input") {
    check(s"""SELECT * FROM
      (SELECT SUM(p) a, COUNT(DISTINCT p) c FROM
        (SELECT price p, k q FROM $table) x WHERE q < 4) b,
      (SELECT MAX(other_price) d, COUNT(*) e FROM
        (SELECT k other_key, price other_price FROM $table) y WHERE other_key > 1) z""", 2)
  }

  test("ANSI decimal overflow is preserved and an excluded overflow branch remains harmless") {
    val maximum = "9" * 38
    session.sql(s"""SELECT * FROM VALUES
      (1, CAST('$maximum' AS DECIMAL(38,0))),
      (2, CAST('$maximum' AS DECIMAL(38,0))) AS t(k, value)""")
      .createOrReplaceTempView(table + "_overflow")
    val source = table + "_overflow"
    check(s"""SELECT * FROM
      (SELECT SUM(value) a FROM $source WHERE k < 0) x,
      (SELECT COUNT(*) b FROM $source WHERE k > 0) y""", 2)
    val original = session.sql(s"""SELECT * FROM
      (SELECT SUM(value) a FROM $source WHERE k > 0) x,
      (SELECT COUNT(*) b FROM $source WHERE k < 0) y""")
    val merged = GpuMergeGlobalAggregates(original.queryExecution.analyzed)
    assert(aggregates(merged) == 1)
    Seq(original, TrampolineConnectShims.createDataFrame(session, merged)).foreach { frame =>
      val error = intercept[Exception] { frame.collect() }
      assert(error.toString.toLowerCase.contains("overflow"), error.toString)
    }
  }

  test("throwing casts and nondeterministic inputs or predicates are not merged") {
    val queries = Seq(
      s"""SELECT * FROM
        (SELECT SUM(CAST(label AS BIGINT)) a FROM $table WHERE k < 0) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT SUM(price) a FROM $table WHERE CAST(label AS BIGINT) > 0) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT SUM(price) a FROM $table WHERE rand() > 0.5) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT SUM(price) a FROM $table WHERE rand() BETWEEN 0 AND 1) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT SUM(price) a FROM $table WHERE CAST(label AS BIGINT) BETWEEN 0 AND 1) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT SUM(random_value) a FROM
          (SELECT rand() random_value, k FROM $table) q WHERE k < 4) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""",
      s"""SELECT * FROM
        (SELECT COUNT(*) a FROM (SELECT CAST(label AS BIGINT) q FROM $table) t
         WHERE q > 0) x,
        (SELECT COUNT(*) b FROM $table WHERE k > 0) y""")
    queries.foreach { query =>
      val original = session.sql(query).queryExecution.analyzed
      assert(GpuMergeGlobalAggregates(original).fastEquals(original), query)
    }
  }

  test("grouped aggregates, HAVING, outer joins and different relations are not merged") {
    session.sql("""SELECT * FROM VALUES
      (99, CAST(99.00 AS DECIMAL(7,2)), 99, '99')
      AS t(k, price, cost, label)""").createOrReplaceTempView(table + "_other")
    val queries = Seq(
      s"SELECT * FROM (SELECT k, SUM(price) a FROM $table GROUP BY k) x, " +
        s"(SELECT COUNT(*) b FROM $table) y",
      s"SELECT * FROM (SELECT SUM(price) a FROM $table HAVING COUNT(*) > 0) x, " +
        s"(SELECT COUNT(*) b FROM $table) y",
      s"SELECT * FROM (SELECT SUM(price) a FROM $table) x LEFT JOIN " +
        s"(SELECT COUNT(*) b FROM $table) y ON true",
      s"SELECT * FROM (SELECT SUM(price) a FROM $table) x, " +
        s"(SELECT COUNT(*) b FROM ${table}_other) y")
    queries.foreach { query =>
      val original = session.sql(query).queryExecution.analyzed
      assert(GpuMergeGlobalAggregates(original).fastEquals(original), query)
    }
  }

  test("post-hoc rule reads the live SQL_ENABLED setting rather than caching the first query") {
    val original = session.sql(s"""SELECT * FROM
      (SELECT SUM(price) a FROM $table WHERE k < 4) x,
      (SELECT COUNT(*) b FROM $table WHERE k > 0) y""").queryExecution.analyzed
    val rule = GpuPostHocResolutionOverrides(session)
    try {
      assert(rule(original).fastEquals(original))
      session.conf.set("spark.rapids.sql.enabled", "true")
      assert(aggregates(rule(original)) == 1)
      session.conf.set("spark.rapids.sql.enabled", "false")
      assert(rule(original).fastEquals(original))
    } finally {
      session.conf.set("spark.rapids.sql.enabled", "false")
    }
  }
}
