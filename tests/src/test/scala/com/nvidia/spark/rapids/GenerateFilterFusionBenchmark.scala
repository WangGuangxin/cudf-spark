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
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.util.QueryExecutionListener

/**
 * End-to-end A/B for generate/filter fusion. Interleaves the OFF and ON runs in one session and
 * reports the plugin's own op time next to wall time, so a wall time change can be attributed
 * to the explode rather than to the surrounding scan.
 */
object GenerateFilterFusionBenchmark {

  private val captured = new mutable.ArrayBuffer[SparkPlan]()

  private class Capture extends QueryExecutionListener {
    override def onSuccess(name: String, qe: QueryExecution, duration: Long): Unit =
      captured.synchronized(captured += qe.executedPlan)
    override def onFailure(name: String, qe: QueryExecution, exception: Exception): Unit = ()
  }

  private def metric(plan: SparkPlan, name: String): Long =
    plan.collect { case p => p.metrics.get(name).map(_.value).getOrElse(0L) }.sum

  /** Op time of just the operators this fusion replaces, so the surrounding plan cancels out. */
  private def operatorOpTime(plan: SparkPlan, names: Set[String]): Long =
    plan.collect {
      case p if names.contains(p.getClass.getSimpleName) =>
        p.metrics.get("opTimeLegacy").map(_.value).getOrElse(0L)
    }.sum

  private def session(): SparkSession = SparkSession.builder()
    .master("local[1]")
    .appName("GenerateFilterFusionBenchmark")
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    .config("spark.rapids.sql.enabled", "true")
    .config("spark.rapids.sql.explain", "NONE")
    .config("spark.rapids.sql.metrics.level", "DEBUG")
    .config("spark.rapids.sql.concurrentGpuTasks", "1")
    .config("spark.rapids.memory.gpu.allocFraction", "0.5")
    .config("spark.rapids.memory.gpu.minAllocFraction", "0")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()

  private def query(spark: SparkSession, rows: Long, k: Int, threshold: Int,
      payloadLongs: Int, strings: Boolean, parts: Int): DataFrame = {
    val payload = (0 until payloadLongs).map(i => s"id * ${i + 3} p$i") ++
      (if (strings) Seq("concat('payload-value-', cast(pmod(id, 1000) as string)) s0") else Nil)
    val arr = s"transform(sequence(0, ${k - 1}), v -> cast(pmod(id + v, 1000) as int)) arr"
    val base = spark.range(0, rows, 1, parts).selectExpr(Seq("id") ++ payload :+ arr: _*)
    val kept = payload.map(p => col(p.split(" ").last))
    base.select(kept :+ explode(col("arr")).as("x"): _*).filter(col("x") < threshold)
  }

  private def run(spark: SparkSession, df: => DataFrame): (Long, SparkPlan) = {
    captured.synchronized(captured.clear())
    val start = System.nanoTime()
    df.write.format("noop").mode("overwrite").save()
    val elapsed = System.nanoTime() - start
    val deadline = System.currentTimeMillis() + 30000
    while (captured.synchronized(captured.isEmpty) && System.currentTimeMillis() < deadline) {
      Thread.sleep(5)
    }
    (elapsed, captured.synchronized(captured.last))
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 6,
      "Arguments: rows k selectivity payloadLongs strings iterations")
    val rows = args(0).toLong
    val k = args(1).toInt
    val selectivity = args(2).toDouble
    val payloadLongs = args(3).toInt
    val strings = args(4).toBoolean
    val iterations = args(5).toInt
    val threshold = math.max(1, (selectivity * 1000).toInt)

    val spark = session()
    spark.sparkContext.setLogLevel("ERROR")
    spark.listenerManager.register(new Capture)

    def once(enabled: Boolean): (Long, SparkPlan) = {
      spark.conf.set(RapidsConf.ENABLE_GENERATE_FILTER_FUSION.key, enabled.toString)
      spark.conf.set(RapidsConf.GENERATE_FILTER_FUSION_MIN_BYTES.key, "1")
      run(spark, query(spark, rows, k, threshold, payloadLongs, strings, 8))
    }

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

    val fused = onPlan.collect { case p: GpuGenerateFilterExec => p }.size
    val offOp = operatorOpTime(offPlan, Set("GpuGenerateExec", "GpuFilterExec")) / 1e6
    val onOp = operatorOpTime(onPlan, Set("GpuGenerateFilterExec")) / 1e6
    val offTotal = metric(offPlan, "opTimeLegacy") / 1e6
    val offMedian = median(off)
    val onMedian = median(on)
    def pct(a: Double, b: Double): Double = (b - a) / math.max(a, 1e-9) * 100

    // scalastyle:off println
    println("RESULT " +
      s"rows=$rows k=$k sel=$selectivity payloadLongs=$payloadLongs strings=$strings " +
      s"iters=$iterations fusedNodes=$fused " +
      f"offMs=$offMedian%.1f onMs=$onMedian%.1f delta=${pct(offMedian, onMedian)}%+.1f%% " +
      f"genFilterOffMs=$offOp%.1f genFilterOnMs=$onOp%.1f " +
      f"opDelta=${pct(offOp, onOp)}%+.1f%% " +
      f"shareOfPlanOpTime=${offOp / math.max(offTotal, 1e-9) * 100}%.0f%%")
    println(s"OFF ${off.map(v => f"${v / 1e6}%.0f").mkString(",")}")
    println(s"ON  ${on.map(v => f"${v / 1e6}%.0f").mkString(",")}")
    // scalastyle:on println
    spark.stop()
  }
}
