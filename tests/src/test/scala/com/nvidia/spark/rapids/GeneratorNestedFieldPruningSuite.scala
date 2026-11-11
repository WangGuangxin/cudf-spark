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

import java.io.File
import java.nio.file.Files

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.expressions.ArraysZip
import org.apache.spark.sql.execution.FileSourceScanExec

/**
 * CPU-only checks for [[GeneratorNestedFieldPruning]]: every query returns the same rows as
 * with nested schema pruning disabled (which also disables the rule), and the scan reads only
 * the referenced struct fields when the generated struct is used field by field only.
 */
class GeneratorNestedFieldPruningSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var dir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      // keep FileSourceScanExec visible in executedPlan
      .config("spark.sql.adaptive.enabled", "false")
      .withExtensions { e =>
        e.injectPostHocResolutionRule(_ => GeneratorNestedFieldPruning)
        e.injectOptimizerRule(_ => GeneratorNestedFieldPruning)
      }
      .getOrCreate()
    dir = Files.createTempDirectory("gen-nested-pruning").toFile
    spark.sql(
      """select * from values
        | (1, array(named_struct('a', 1, 'b', 'x', 'c', 1.5d),
        |           named_struct('a', 2, 'b', null, 'c', 2.5d))),
        | (2, array(named_struct('a', 3, 'b', 'y', 'c', 3.5d),
        |           cast(null as struct<a:int,b:string,c:double>))),
        | (3, cast(array() as array<struct<a:int,b:string,c:double>>)),
        | (4, cast(null as array<struct<a:int,b:string,c:double>>)),
        | (5, array(named_struct('a', null, 'b', 'z', 'c', null)))
        | as t(k, arr)""".stripMargin)
      .write.mode("overwrite").parquet(dir.getAbsolutePath)
    spark.read.parquet(dir.getAbsolutePath).createOrReplaceTempView("t")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
      if (dir != null) org.apache.commons.io.FileUtils.deleteDirectory(dir)
    } finally {
      super.afterAll()
    }
  }

  private def run(sql: String, prune: Boolean): (Seq[Row], Seq[String], Boolean) = {
    spark.conf.set("spark.sql.optimizer.nestedSchemaPruning.enabled", prune.toString)
    try {
      val df = spark.sql(sql)
      val rows = df.collect().toSeq
      val schemas = df.queryExecution.executedPlan.collect {
        case s: FileSourceScanExec => s.requiredSchema.simpleString
      }
      val zipped = df.queryExecution.optimizedPlan.exists(_.expressions.exists(
        _.exists(_.isInstanceOf[ArraysZip])))
      (rows, schemas, zipped)
    } finally {
      spark.conf.unset("spark.sql.optimizer.nestedSchemaPruning.enabled")
    }
  }

  private def check(sql: String, expectPruned: Boolean, expectedFields: Set[String] = Set.empty)
      : Unit = {
    val (expected, _, zippedOff) = run(sql, prune = false)
    val (actual, schemas, zipped) = run(sql, prune = true)
    assert(!zippedOff)
    assert(actual == expected)
    assert(zipped == expectPruned, s"rewrite expected=$expectPruned, plan schemas: $schemas")
    if (expectPruned) {
      val read = schemas.flatMap(s => "struct<([^<>]*)>>".r.findAllMatchIn(s).flatMap(
        _.group(1).split(",").map(_.split(":").head))).toSet
      assert(read == expectedFields, s"read schemas: $schemas")
    }
  }

  test("two of three fields after explode") {
    check("select k, s.a, s.b from (select k, explode(arr) as s from t) order by 1, 2, 3",
      expectPruned = true, Set("a", "b"))
  }

  test("explode_outer keeps null and empty arrays") {
    check("select k, s.a, s.c from (select k, explode_outer(arr) as s from t) order by 1, 2, 3",
      expectPruned = true, Set("a", "c"))
  }

  test("posexplode and posexplode_outer") {
    check("select k, p, s.a, s.b from (select k, posexplode(arr) as (p, s) from t) order by 1, 2",
      expectPruned = true, Set("a", "b"))
    check("select k, p, s.b, s.c from (select k, posexplode_outer(arr) as (p, s) from t) " +
      "order by 1, 2", expectPruned = true, Set("b", "c"))
  }

  test("fields used by a filter above the generate") {
    check("select k, s.a from (select k, explode(arr) as s from t) where s.a < s.c order by 1, 2",
      expectPruned = true, Set("a", "c"))
  }

  test("fields used inside EXISTS / NOT EXISTS subqueries") {
    check("select k from t t1 where exists (select * from (select k, explode(arr) as s from t) " +
      "t2 where t2.k = t1.k and t2.s.a > t2.s.c) order by 1", expectPruned = true, Set("a", "c"))
    check("select k from t t1 where not exists (select * from (select k, explode(arr) as s " +
      "from t) t2 where t2.k = t1.k and t2.s.b = 'x' and t2.s.a = 1) order by 1",
      expectPruned = true, Set("a", "b"))
  }

  test("whole struct observed: no rewrite") {
    Seq(
      "select k, s from (select k, explode(arr) as s from t) order by 1, s.a",
      "select k, s is null, s.a from (select k, explode(arr) as s from t) order by 1, 3",
      "select count(*) from (select distinct s from (select explode(arr) as s from t))",
      "select s.a, count(*) from (select explode(arr) as s from t) group by s order by 1",
      "select count(*) from (select s from (select explode(arr) as s from t) union " +
        "select s from (select explode(arr) as s from t))",
      "select k, to_json(s) from (select k, explode(arr) as s from t) order by 1, 2"
    ).foreach(q => check(q, expectPruned = false))
  }

  test("single field below a filter: explode(arr.f) without a zip") {
    val q = "select k from (select k, explode(arr) as s from t) where s.b = 'x' order by 1"
    val (expected, unpruned, _) = run(q, prune = false)
    val (actual, schemas, zipped) = run(q, prune = true)
    assert(actual == expected)
    assert(!zipped)
    assert(unpruned.exists(_.contains("struct<a:int,b:string,c:double>")))
    assert(schemas.exists(_.contains("array<struct<b:string>>")), s"read schemas: $schemas")
  }

  test("all fields used: no rewrite") {
    check("select s.a, s.b, s.c from (select explode(arr) as s from t) order by 1, 2, 3",
      expectPruned = false)
  }
}
