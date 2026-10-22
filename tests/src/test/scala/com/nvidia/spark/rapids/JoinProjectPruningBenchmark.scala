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

import scala.collection.mutable

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.execution.{QueryExecution, SparkPlan}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.util.QueryExecutionListener

/**
 * End-to-end A/B for join output pruning on a star schema. Interleaves the OFF and ON runs in
 * one session so data, caches and GPU state are shared, and reports the plugin's own join time
 * alongside wall time so a wall-time change can be attributed to the gather.
 */
object JoinProjectPruningBenchmark {

  private val captured = new mutable.ArrayBuffer[SparkPlan]()

  private class Capture extends QueryExecutionListener {
    override def onSuccess(name: String, qe: QueryExecution, duration: Long): Unit =
      captured.synchronized(captured += qe.executedPlan)
    override def onFailure(name: String, qe: QueryExecution, exception: Exception): Unit = ()
  }

  private def metric(plan: SparkPlan, name: String): Long =
    plan.collect { case p => p.metrics.get(name).map(_.value).getOrElse(0L) }.sum

  private def session(): SparkSession = SparkSession.builder()
    .master("local[1]")
    .appName("JoinProjectPruningBenchmark")
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    .config("spark.rapids.sql.enabled", "true")
    .config("spark.rapids.sql.explain", "NONE")
    .config("spark.rapids.sql.metrics.level", "DEBUG")
    .config("spark.rapids.sql.concurrentGpuTasks", "1")
    .config("spark.rapids.memory.gpu.allocFraction", "0.5")
    .config("spark.rapids.memory.gpu.minAllocFraction", "0")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.ansi.enabled", "false")
    .config("spark.sql.autoBroadcastJoinThreshold", "10485760")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()

  /** fact(fid, k0..k[joins-1], p0..p[width-1]) joined to one small dimension per key. */
  private def fact(spark: SparkSession, rows: Long, joins: Int, width: Int,
      stringPayload: Boolean, dimRows: Int, parts: Int, stringKeys: Boolean): DataFrame = {
    val keys = (0 until joins).map { i =>
      val raw = s"pmod(xxhash64(id, ${i + 1}), $dimRows)"
      if (stringKeys) s"concat('key-00000-', cast($raw as string)) k$i"
      else s"cast($raw as int) k$i"
    }
    val payload = (0 until width).map { i =>
      if (stringPayload) s"concat('v', cast(pmod(id, 1000) + $i as string)) p$i"
      else s"cast(id * ${i + 3} as long) p$i"
    }
    spark.range(0, rows, 1, parts).selectExpr(Seq("id fid") ++ keys ++ payload: _*)
  }

  private def dim(spark: SparkSession, i: Int, dimRows: Int, stringKeys: Boolean): DataFrame = {
    val key = if (stringKeys) s"concat('key-00000-', cast(id as string)) dk$i"
      else s"cast(id as int) dk$i"
    spark.range(0, dimRows.toLong, 1, 1)
      .selectExpr(key, s"concat('name', cast(id as string)) dn$i")
  }

  private def query(spark: SparkSession, rows: Long, joins: Int, width: Int,
      stringPayload: Boolean, dimRows: Int, parts: Int, stringKeys: Boolean): DataFrame = {
    var df = fact(spark, rows, joins, width, stringPayload, dimRows, parts, stringKeys)
    (0 until joins).foreach { i =>
      df = df.join(broadcast(dim(spark, i, dimRows, stringKeys)), col(s"k$i") === col(s"dk$i"))
    }
    val payload = (0 until width).map(i => col(s"p$i"))
    val names = (0 until joins).map(i => col(s"dn$i"))
    df.select(payload ++ names: _*)
  }

  private def run(spark: SparkSession, df: => DataFrame): (Long, SparkPlan) = {
    captured.synchronized(captured.clear())
    val start = System.nanoTime()
    df.write.format("noop").mode("overwrite").save()
    val elapsed = System.nanoTime() - start
    // The execution listener runs on the async listener bus, so the plan arrives after the
    // action returns. Waiting here keeps the measured window free of the poll.
    val deadline = System.currentTimeMillis() + 30000
    while (captured.synchronized(captured.isEmpty) && System.currentTimeMillis() < deadline) {
      Thread.sleep(5)
    }
    val plan = captured.synchronized {
      require(captured.nonEmpty, "no query execution was captured")
      captured.last
    }
    (elapsed, plan)
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 8,
      "Arguments: rows joins payloadColumns stringPayload iterations toggle fixedOther stringKeys")
    val rows = args(0).toLong
    val joins = args(1).toInt
    val width = args(2).toInt
    val stringPayload = args(3).toBoolean
    val iterations = args(4).toInt
    // Which switch is the variable, and what the other one is pinned to. Pruning bypasses the
    // join chain, so leaving the chain free would compare two optimizations at once.
    val toggle = args(5)
    val fixedOther = args(6).toBoolean
    val stringKeys = args(7).toBoolean
    require(toggle == "pruning" || toggle == "chain", "toggle must be pruning or chain")
    val variable = if (toggle == "pruning") RapidsConf.ENABLE_JOIN_PROJECT_PRUNING.key
      else "spark.rapids.sql.join.chain.enabled"
    val fixed = if (toggle == "pruning") "spark.rapids.sql.join.chain.enabled"
      else RapidsConf.ENABLE_JOIN_PROJECT_PRUNING.key
    val dimRows = 100000
    val parts = 8

    val spark = session()
    spark.sparkContext.setLogLevel("ERROR")
    spark.listenerManager.register(new Capture)

    spark.conf.set(fixed, fixedOther.toString)
    def once(enabled: Boolean): (Long, SparkPlan) = {
      spark.conf.set(variable, enabled.toString)
      run(spark, query(spark, rows, joins, width, stringPayload, dimRows, parts, stringKeys))
    }

    // Warm up both paths before timing anything.
    (0 until 3).foreach { _ => once(false); once(true) }

    val off = new mutable.ArrayBuffer[Long]()
    val on = new mutable.ArrayBuffer[Long]()
    var offPlan: SparkPlan = null
    var onPlan: SparkPlan = null
    (0 until iterations).foreach { _ =>
      val a = once(false)
      off += a._1
      offPlan = a._2
      val b = once(true)
      on += b._1
      onPlan = b._2
    }

    def median(v: mutable.ArrayBuffer[Long]): Double = {
      val s = v.sorted
      (if (s.length % 2 == 1) s(s.length / 2).toDouble
       else (s(s.length / 2 - 1) + s(s.length / 2)) / 2.0) / 1e6
    }

    val prunedNodes = onPlan.collect { case p: GpuPrunedJoinExec => p }
    val offJoin = metric(offPlan, "joinTime") / 1e6
    val onJoin = metric(onPlan, "joinTime") / 1e6
    val offRows = metric(offPlan, "numOutputRows")
    val onRows = metric(onPlan, "numOutputRows")
    val offMedian = median(off)
    val onMedian = median(on)
    val offMin = off.min / 1e6
    val onMin = on.min / 1e6
    def pct(a: Double, b: Double): Double = (b - a) / math.max(a, 1e-9) * 100

    // scalastyle:off println
    println("RESULT " +
      s"rows=$rows joins=$joins width=$width string=$stringPayload stringKeys=$stringKeys " +
      s"iters=$iterations " +
      s"toggle=$toggle $fixed=$fixedOther " +
      s"prunedNodes=${prunedNodes.size} " +
      s"kept=${prunedNodes.map(p => s"${p.output.length}/${p.child.output.length}")
        .mkString(",")} " +
      s"offRows=$offRows onRows=$onRows " +
      f"offMs=$offMedian%.1f onMs=$onMedian%.1f delta=${pct(offMedian, onMedian)}%+.2f%% " +
      f"offMinMs=$offMin%.1f onMinMs=$onMin%.1f minDelta=${pct(offMin, onMin)}%+.2f%% " +
      f"offJoinMs=$offJoin%.1f onJoinMs=$onJoin%.1f joinDelta=${pct(offJoin, onJoin)}%+.2f%%")
    println(s"OFF_SAMPLES ${off.map(v => f"${v / 1e6}%.1f").mkString(",")}")
    println(s"ON_SAMPLES ${on.map(v => f"${v / 1e6}%.1f").mkString(",")}")
    // scalastyle:on println
    spark.stop()
  }
}
