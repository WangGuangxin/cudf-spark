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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream}

import ai.rapids.cudf.{BaseDeviceMemoryBuffer, Cuda, DeviceMemoryBuffer, HostMemoryBuffer, Rmm, RmmAllocationMode, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.kudo.{KudoSerializer, KudoTableHeader}
import com.nvidia.spark.rapids.spill.SpillFramework
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{DataType, LongType, StringType}

class GpuShuffleFrameSuite extends AnyFunSuite with BeforeAndAfterAll {
  import GpuShuffleFrame._

  private val dataTypes: Array[DataType] = Array(LongType, StringType)
  private val numRows = 20000

  override def beforeAll(): Unit = {
    super.beforeAll()
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024)
    SpillFramework.initialize(new RapidsConf(new SparkConf))
  }

  override def afterAll(): Unit = {
    super.afterAll()
    SpillFramework.shutdown()
    Rmm.shutdown()
  }

  /** A compressible table: repeating longs and a few distinct strings. */
  private def buildTable(): Table = {
    val longs = (0 until numRows).map(i => java.lang.Long.valueOf((i % 97).toLong))
    val strs = (0 until numRows).map(i => if (i % 13 == 0) null else s"value-${i % 11}")
    new Table.TestBuilder().column(longs: _*).column(strs: _*).build()
  }

  /** The table serialized by kudo: (header bytes, body bytes). */
  private def kudoBytes(table: Table): (Array[Byte], Array[Byte]) = {
    val out = new ByteArrayOutputStream()
    val hostColumns = (0 until table.getNumberOfColumns).map(table.getColumn(_).copyToHost())
    withResource(hostColumns) { cols =>
      new KudoSerializer(GpuColumnVector.from(dataTypes))
        .writeToStreamWithMetrics(cols.toArray, out, 0, table.getRowCount.toInt)
    }
    val bytes = out.toByteArray
    val header = KudoTableHeader.readFrom(
      new DataInputStream(new ByteArrayInputStream(bytes))).get()
    val hdrLen = header.getSerializedSize
    (bytes.take(hdrLen), bytes.slice(hdrLen, hdrLen + header.getTotalDataLen))
  }

  /** A RAW frame as GpuPartitioning writes it when the CPU compresses. */
  private def rawFrame(hdr: Array[Byte], body: Array[Byte]): HostMemoryBuffer = {
    val frame = HostMemoryBuffer.allocate(FRAME_HEADER_BYTES + hdr.length + body.length)
    writeFrameHeader(frame, 0, CODEC_RAW, body.length, body.length, System.nanoTime())
    frame.setBytes(FRAME_HEADER_BYTES, hdr, 0, hdr.length)
    frame.setBytes(FRAME_HEADER_BYTES + hdr.length, body, 0, body.length)
    frame
  }

  private def bytesOf(buf: HostMemoryBuffer): Array[Byte] = {
    val b = new Array[Byte](buf.getLength.toInt)
    buf.getBytes(b, 0, 0, b.length)
    b
  }

  private def readFrames(bytes: Array[Byte]): Seq[KudoSerializedTableColumn] = {
    val it = new FramedKudoBatchIterator(
      new DataInputStream(new ByteArrayInputStream(bytes)), NoopMetric)
    it.map(_._2.column(0).asInstanceOf[KudoSerializedTableColumn]).toList
  }

  test("CPU-compressed frame is read back on the CPU and concatenated on the GPU") {
    withResource(buildTable()) { table =>
      val (hdr, body) = kudoBytes(table)
      assert(body.length >= MIN_COMPRESS_BYTES)
      val out = new ByteArrayOutputStream()
      val written = withResource(rawFrame(hdr, body))(writeFrame(_, out))
      val bytes = out.toByteArray
      assert(written == bytes.length)
      assert(bytes.length < FRAME_HEADER_BYTES + hdr.length + body.length, "not compressed")
      val cols = readFrames(bytes ++ bytes)
      try {
        assert(cols.size == 2)
        cols.foreach { c =>
          assert(c.spillableKudoTable.isCompressed)
          withResource(c.spillableKudoTable.makeKudoTable) { kt =>
            assert(bytesOf(kt.getBuffer).sameElements(body))
          }
        }
        // GPU decompression in the shuffle coalesce: two copies of the table
        withResource(new KudoGpuTableOperator(dataTypes).concat(cols.toArray)) { actual =>
          withResource(Table.concatenate(table, table)) { both =>
            withResource(GpuColumnVector.from(both, dataTypes)) { expected =>
              TestUtils.compareBatches(expected, actual)
            }
          }
        }
      } finally {
        cols.foreach(_.close())
      }
    }
  }

  test("GPU-compressed body is decompressed on the CPU") {
    withResource(buildTable()) { table =>
      val (_, body) = kudoBytes(table)
      val payload = withResource(HostMemoryBuffer.allocate(body.length)) { h =>
        h.setBytes(0, body, 0, body.length)
        val dev = DeviceMemoryBuffer.allocate(body.length)
        dev.copyFromHostBuffer(h)
        // compress() closes its inputs
        withResource(compressor().compress(Array[BaseDeviceMemoryBuffer](dev),
            Cuda.DEFAULT_STREAM)) { comp =>
          val ph = HostMemoryBuffer.allocate(comp(0).getLength)
          ph.copyFromDeviceBuffer(comp(0))
          ph
        }
      }
      withResource(payload) { p =>
        assert(p.getLength < body.length)
        withResource(cpuDecompressHost(p, body.length)) { raw =>
          assert(bytesOf(raw).sameElements(body))
        }
      }
    }
  }

  test("small bodies and row-count-only tables stay raw") {
    val hdr = Array.tabulate[Byte](32)(_.toByte)
    val body = Array.fill[Byte](MIN_COMPRESS_BYTES - 1)(7)
    val out = new ByteArrayOutputStream()
    withResource(rawFrame(hdr, body))(writeFrame(_, out))
    val bytes = out.toByteArray
    assert(bytes.length == FRAME_HEADER_BYTES + hdr.length + body.length)
    assert(java.nio.ByteBuffer.wrap(bytes).getInt(4) == CODEC_RAW)

    val rowsOnly = new ByteArrayOutputStream()
    KudoSerializer.writeRowCountToStream(rowsOnly, 42)
    val frame = rawHeaderOnlyFrame(rowsOnly.toByteArray, rowsOnly.size())
    val cols = readFrames(frame)
    try {
      assert(cols.size == 1)
      assert(cols.head.spillableKudoTable.header.getNumRows == 42)
      assert(!cols.head.spillableKudoTable.isCompressed)
    } finally {
      cols.foreach(_.close())
    }
  }
}
