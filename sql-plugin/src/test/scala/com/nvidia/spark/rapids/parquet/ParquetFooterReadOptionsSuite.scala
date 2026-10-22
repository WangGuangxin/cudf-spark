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

package com.nvidia.spark.rapids.parquet

import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._

import com.nvidia.spark.rapids.Arm.withResource
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.example.data.simple.SimpleGroupFactory
import org.apache.parquet.format.converter.ParquetMetadataConverter
import org.apache.parquet.hadoop.{ParquetFileReader, ParquetWriter}
import org.apache.parquet.hadoop.example.ExampleParquetWriter
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.io.{DelegatingSeekableInputStream, InputFile, SeekableInputStream}
import org.apache.parquet.schema.MessageTypeParser
import org.scalatest.funsuite.AnyFunSuite

@scala.annotation.nowarn("msg=method readFooter in class ParquetFileReader is deprecated")
class ParquetFooterReadOptionsSuite extends AnyFunSuite {
  test("existing configuration preserves footer schema, statistics and row-group filters") {
    val directory = Files.createTempDirectory("footer-options-")
    val file = directory.resolve("data.parquet")
    val conf = new Configuration(false)
    val schema = MessageTypeParser.parseMessageType(
      "message root { required int32 id; optional int32 amount (DECIMAL(7,2)); }")
    val groups = new SimpleGroupFactory(schema)
    try {
      val writer: ParquetWriter[org.apache.parquet.example.data.Group] =
        ExampleParquetWriter.builder(new Path(file.toUri))
          .withConf(conf).withType(schema).withDictionaryEncoding(false)
          .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
          .withRowGroupSize(128L).build()
      withResource(writer) { out =>
        (0 until 300).foreach { i =>
          out.write(groups.newGroup().append("id", i).append("amount", i * 3))
        }
      }
      val bytes = Files.readAllBytes(file)
      val filters = Seq(ParquetMetadataConverter.NO_FILTER,
        ParquetMetadataConverter.SKIP_ROW_GROUPS,
        ParquetMetadataConverter.range(0, bytes.length / 2))
      filters.foreach { filter =>
        val expected = ParquetFileReader.readFooter(new ByteArrayInputFile(bytes), filter)
        val actual = ParquetFooterUtils.readFooter(new ByteArrayInputFile(bytes), conf, filter)
        val converter = new ParquetMetadataConverter()
        assert(converter.toParquetMetadata(1, actual) ==
          converter.toParquetMetadata(1, expected))
      }
    } finally {
      withResource(Files.newDirectoryStream(directory)) { files =>
        files.iterator().asScala.foreach(Files.delete)
      }
      Files.delete(directory)
    }
  }

  test("malformed footer closes the input stream") {
    val closed = new AtomicInteger()
    val bytes = new Array[Byte](16)
    val input = new InputFile {
      override def getLength: Long = bytes.length
      override def newStream(): SeekableInputStream = {
        val delegate = new ByteArrayInputFile(bytes).newStream()
        new DelegatingSeekableInputStream(delegate) {
          override def getPos: Long = delegate.getPos
          override def seek(position: Long): Unit = delegate.seek(position)
          override def close(): Unit = {
            closed.incrementAndGet()
            super.close()
          }
        }
      }
    }
    intercept[Exception] {
      ParquetFooterUtils.readFooter(input, new Configuration(false),
        ParquetMetadataConverter.NO_FILTER)
    }
    assert(closed.get() == 1)
  }

}
