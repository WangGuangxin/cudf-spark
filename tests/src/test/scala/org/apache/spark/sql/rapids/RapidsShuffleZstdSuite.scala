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

package org.apache.spark.sql.rapids

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.io.ZStdCompressionCodec
import org.apache.spark.serializer.{JavaSerializer, SerializerManager}
import org.apache.spark.storage.ShuffleBlockId

class RapidsShuffleZstdSuite extends AnyFunSuite {
  private val conf = new SparkConf(false)
    .set("spark.shuffle.compress", "true")
    .set("spark.io.compression.codec", "zstd")
  private val serializerManager = new SerializerManager(new JavaSerializer(conf), conf)
  private val sparkCodec = new ZStdCompressionCodec(conf)
  private val blockId = ShuffleBlockId(0, 0L, 0)
  private lazy val in = RapidsShuffleLz4.inputWrapper(serializerManager)
  private lazy val out = RapidsShuffleLz4.outputWrapper(serializerManager, conf)

  private def readAll(is: InputStream): Array[Byte] = {
    val bos = new ByteArrayOutputStream()
    val buf = new Array[Byte](8192)
    var n = is.read(buf)
    while (n >= 0) {
      bos.write(buf, 0, n)
      n = is.read(buf)
    }
    is.close()
    bos.toByteArray
  }

  private def data(size: Int): Array[Byte] = {
    val rnd = new scala.util.Random(size)
    Array.tabulate[Byte](size)(i => if (i % 3 == 0) rnd.nextInt().toByte else (i % 7).toByte)
  }

  /** Writes `bytes` as `parts` streams opened by `wrap`, one after another into one buffer. */
  private def write(bytes: Array[Byte], wrap: OutputStream => OutputStream,
      parts: Int = 1): Array[Byte] = {
    val bos = new ByteArrayOutputStream()
    val step = math.max(1, (bytes.length + parts - 1) / parts)
    var off = 0
    var first = true
    // An empty input is still written as one (empty) stream.
    while (first || off < bytes.length) {
      first = false
      val os = wrap(bos)
      val len = math.min(step, bytes.length - off)
      os.write(bytes, off, len)
      os.close()
      off += len
    }
    bos.toByteArray
  }

  test("detects a plain zstd serializer manager") {
    assert(RapidsShuffleZstd.isPlainZstd(serializerManager))
    val lz4 = conf.clone.set("spark.io.compression.codec", "lz4")
    assert(!RapidsShuffleZstd.isPlainZstd(new SerializerManager(new JavaSerializer(lz4), lz4)))
    val off = conf.clone.set("spark.shuffle.compress", "false")
    assert(!RapidsShuffleZstd.isPlainZstd(new SerializerManager(new JavaSerializer(off), off)))
  }

  test("streams are compatible with Spark's zstd codec") {
    Seq(0, 1, 100, 600, 65536, 200000, 1 << 20, 5 << 20).foreach { size =>
      val bytes = data(size)
      val hint = size.toLong
      val results = Seq(
        "Spark writes, read here" ->
          readAll(in(blockId, new ByteArrayInputStream(write(bytes,
            sparkCodec.compressedOutputStream)))),
        "written here, Spark reads" ->
          readAll(sparkCodec.compressedInputStream(new ByteArrayInputStream(
            write(bytes, out(blockId, _, hint))))),
        "written and read here" ->
          readAll(in(blockId, new ByteArrayInputStream(write(bytes, out(blockId, _, hint))))),
        "concatenated records read here" ->
          readAll(in(blockId, new ByteArrayInputStream(write(bytes, out(blockId, _, 0L), 3)))),
        "concatenated records, Spark reads" ->
          readAll(sparkCodec.compressedContinuousInputStream(new ByteArrayInputStream(
            write(bytes, out(blockId, _, 7L), 3)))))
      results.foreach { case (name, result) =>
        assert(java.util.Arrays.equals(result, bytes), s"$name, $size bytes")
      }
    }
  }

  test("nested streams on one thread do not share buffers") {
    val b1 = new ByteArrayOutputStream()
    val b2 = new ByteArrayOutputStream()
    val o1 = out(blockId, b1, 10L)
    val o2 = out(blockId, b2, 10L)
    o1.write(Array.fill[Byte](1000)(1))
    o2.write(Array.fill[Byte](1000)(2))
    o2.close()
    o1.close()
    assert(readAll(in(blockId, new ByteArrayInputStream(b1.toByteArray))).forall(_ == 1))
    assert(readAll(in(blockId, new ByteArrayInputStream(b2.toByteArray))).forall(_ == 2))
  }

  test("a corrupted frame does not decode to the original bytes") {
    val bytes = data(5000)
    val compressed = write(bytes, out(blockId, _, 5000L))
    compressed(compressed.length / 2) = (compressed(compressed.length / 2) ^ 0x55).toByte
    val decoded = try {
      Some(readAll(in(blockId, new ByteArrayInputStream(compressed))))
    } catch {
      case _: Exception => None
    }
    assert(!decoded.exists(java.util.Arrays.equals(_, bytes)))
  }
}
