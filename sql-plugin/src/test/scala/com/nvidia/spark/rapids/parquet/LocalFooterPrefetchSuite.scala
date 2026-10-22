/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.parquet

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.Files

import com.nvidia.spark.rapids.{PerfIO, PerfIOConf}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.fileio.hadoop.HadoopInputFile
import com.nvidia.spark.rapids.jni.fileio.{RapidsInputFile, SeekableInputStream}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.hadoop.ParquetFileWriter.MAGIC
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf

class LocalFooterPrefetchSuite extends AnyFunSuite {
  PerfIO.initS3Client(PerfIOConf(new SparkConf(false)
    .set("spark.rapids.perfio.s3.enabled", "false")
    .set("spark.rapids.perfio.gcs.enabled", "false")
    .set("spark.rapids.perfio.suffixPrefetchSize", "16k")))

  private class CountingInput(delegate: RapidsInputFile) extends RapidsInputFile {
    var opens = 0
    var closes = 0
    override def path(): String = delegate.path()
    override def getLength: Long = delegate.getLength
    override def open(): SeekableInputStream = {
      opens += 1
      val stream = delegate.open()
      new SeekableInputStream {
        override def getPos: Long = stream.getPos
        override def seek(offset: Long): Unit = stream.seek(offset)
        override def read(): Int = stream.read()
        override def read(bytes: Array[Byte], offset: Int, length: Int): Int =
          stream.read(bytes, offset, length)
        override def close(): Unit = {
          closes += 1
          withResource(stream)(_ => ())
        }
      }
    }
  }

  private def checkFooter(footerLength: Int, expectedOpens: Int): Unit = {
    val footer = Array.tabulate(footerLength)(i => (i * 13).toByte)
    val length = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(footerLength).array()
    val trailer = length ++ MAGIC
    val bytes = MAGIC ++ Array.fill[Byte](128)(42) ++ footer ++ trailer
    val file = Files.createTempFile("rapids-footer-prefetch-", ".parquet")
    try {
      Files.write(file, bytes)
      val input = new CountingInput(HadoopInputFile.create(new Path(file.toUri),
        new Configuration(false)))
      withResource(ParquetFooterUtils.readFooterBufferFromInputFile(input,
          new Path(file.toUri))) { result =>
        val observed = new Array[Byte](result.getLength.toInt)
        result.getBytes(observed, 0, 0, observed.length)
        assert(observed.sameElements(MAGIC ++ footer ++ trailer))
      }
      assert(input.opens == expectedOpens)
      assert(input.closes == expectedOpens)
    } finally {
      Files.delete(file)
    }
  }

  test("local footer in the prefetched suffix needs one stream and preserves framing") {
    checkFooter(1024, expectedOpens = 1)
  }

  test("footer larger than suffix uses a second bounded read with identical framing") {
    checkFooter(32 * 1024, expectedOpens = 2)
  }

  test("encrypted footer magic fails and closes the prefetched stream once") {
    val file = Files.createTempFile("rapids-footer-encrypted-", ".parquet")
    try {
      Files.write(file, MAGIC ++ Array.fill[Byte](128)(0) ++
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(128).array() ++
        "PARE".getBytes(java.nio.charset.StandardCharsets.US_ASCII))
      val input = new CountingInput(HadoopInputFile.create(new Path(file.toUri),
        new Configuration(false)))
      val error = intercept[RuntimeException] {
        withResource(ParquetFooterUtils.readFooterBufferFromInputFile(input,
          new Path(file.toUri)))(_ => ())
      }
      assert(error.getMessage.contains("encrypted Parquet"))
      assert(input.opens == 1)
      assert(input.closes == 1)
    } finally {
      Files.delete(file)
    }
  }
}
