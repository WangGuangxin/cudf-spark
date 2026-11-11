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
import org.apache.spark.sql.catalyst.expressions.ArrayExists
import org.apache.spark.sql.catalyst.plans.{LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, LogicalPlan}

/**
 * CPU-only checks for [[GeneratorArrayPrefilter]]: results are the same as with the rule
 * excluded, and the array-level prefilter is added (once) only where it is valid.
 */
class GeneratorArrayPrefilterSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var dir: File = _
  private val ruleName = GeneratorArrayPrefilter.ruleName

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .withExtensions(_.injectOptimizerRule(_ => GeneratorArrayPrefilter))
      .getOrCreate()
    dir = Files.createTempDirectory("gen-array-prefilter").toFile
    spark.sql(
      """select * from values
        | (1, array(named_struct('a', 1, 'b', 'x'), named_struct('a', 2, 'b', null))),
        | (2, array(named_struct('a', 3, 'b', 'y'), cast(null as struct<a:int,b:string>))),
        | (3, cast(array() as array<struct<a:int,b:string>>)),
        | (4, cast(null as array<struct<a:int,b:string>>)),
        | (5, array(named_struct('a', null, 'b', 'z'))),
        | (6, array(named_struct('a', 6, 'b', 'w')))
        | as t(k, arr)""".stripMargin)
      .write.mode("overwrite").parquet(dir.getAbsolutePath)
    // a file source, so that filters are not evaluated at planning time on a local relation
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

  private def prefilters(p: LogicalPlan): Seq[Filter] = p.collect {
    case f: Filter if f.condition.exists(_.isInstanceOf[ArrayExists]) => f
  }

  private def run(sql: String, enabled: Boolean): (Seq[Row], LogicalPlan) = {
    if (!enabled) spark.conf.set("spark.sql.optimizer.excludedRules", ruleName)
    try {
      val df = spark.sql(sql)
      (df.collect().toSeq, df.queryExecution.optimizedPlan)
    } finally {
      spark.conf.unset("spark.sql.optimizer.excludedRules")
    }
  }

  /** Runs with and without the rule, checks the results match and returns the new plan. */
  private def check(sql: String, expectedPrefilters: Int): LogicalPlan = {
    val (expected, offPlan) = run(sql, enabled = false)
    val (actual, plan) = run(sql, enabled = true)
    assert(prefilters(offPlan).isEmpty)
    assert(actual == expected)
    assert(prefilters(plan).size == expectedPrefilters, plan.treeString)
    plan
  }

  test("filter on the exploded struct") {
    check("select k, s.a from (select k, explode(arr) as s from t) where s.a > 1 order by 1, 2",
      expectedPrefilters = 1)
  }

  test("null-tolerant predicates and posexplode") {
    check("select k, s.b from (select k, explode(arr) as s from t) " +
      "where s.b is null or s.a = 2 order by 1, 2", expectedPrefilters = 1)
    check("select k, p from (select k, posexplode(arr) as (p, s) from t) " +
      "where s.a < 3 order by 1, 2", expectedPrefilters = 1)
  }

  test("not added for outer explode, position-only or non-generated references") {
    check("select k, s.a from (select k, explode_outer(arr) as s from t) " +
      "where s.a > 1 order by 1, 2", expectedPrefilters = 0)
    check("select k, p from (select k, posexplode(arr) as (p, s) from t) where p = 1 order by 1",
      expectedPrefilters = 0)
    check("select k, s.a from (select k, explode(arr) as s from t) where s.a > k order by 1",
      expectedPrefilters = 0)
    check("select k from (select k, explode(arr) as s from t) where s is not null order by 1",
      expectedPrefilters = 0)
  }

  test("prefilter is pushed below semi / anti joins from EXISTS / NOT EXISTS") {
    val plan = check(
      """select t1.k, t1.s.a from (select k, explode(arr) as s from t) t1
        |where t1.s.a > 1
        |  and exists (select * from t t2 where t2.k = t1.k)
        |  and not exists (select * from t t3 where t3.k = t1.k + 100)
        |order by 1, 2""".stripMargin, expectedPrefilters = 1)
    val joins = plan.collect { case j @ Join(_, _, LeftSemi | LeftAnti, _, _) => j }
    assert(joins.nonEmpty)
    assert(joins.forall(j => prefilters(j.left).size == 1), plan.treeString)
  }
}
