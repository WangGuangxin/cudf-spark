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

package com.nvidia.spark.rapids.fileio.hadoop

import java.io.IOException
import java.net.URI
import java.util.{Arrays, Collections}

import ai.rapids.cudf.HostMemoryBuffer
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.fileio.RapidsInputFile.CopyRange
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSInputStream, Path, RawLocalFileSystem}
import org.scalatest.funsuite.AnyFunSuite

/** Captures the actual byte array passed to the filesystem, not just the read length. */
class VectoredCopyTrackingFileSystem extends RawLocalFileSystem {
  override def getUri: URI = URI.create("copytracking:///")

  override def open(path: Path, bufferSize: Int): FSDataInputStream = {
    VectoredCopyTrackingFileSystem.opens += 1
    new FSDataInputStream(new FSInputStream {
      private var position = 0
      override def getPos: Long = position
      override def seek(offset: Long): Unit = position = offset.toInt
      override def seekToNewSource(offset: Long): Boolean = false
      override def read(): Int = {
        if (position == VectoredCopyTrackingFileSystem.data.length) -1 else {
          val value = VectoredCopyTrackingFileSystem.data(position) & 0xff
          position += 1
          value
        }
      }
      override def read(bytes: Array[Byte], offset: Int, length: Int): Int = {
        VectoredCopyTrackingFileSystem.capacities :+= bytes.length
        VectoredCopyTrackingFileSystem.buffers :+= bytes
        if (VectoredCopyTrackingFileSystem.failNextRead) {
          VectoredCopyTrackingFileSystem.failNextRead = false
          throw new IOException("injected read failure")
        }
        val available = VectoredCopyTrackingFileSystem.data.length - position
        if (available == 0) -1 else {
          val count = math.min(length, available)
          System.arraycopy(VectoredCopyTrackingFileSystem.data, position, bytes, offset, count)
          position += count
          val hook = VectoredCopyTrackingFileSystem.afterRead
          VectoredCopyTrackingFileSystem.afterRead = None
          hook.foreach(_())
          count
        }
      }
      override def close(): Unit = VectoredCopyTrackingFileSystem.closes += 1
    })
  }
}

object VectoredCopyTrackingFileSystem {
  val data: Array[Byte] = Array.tabulate(256)(_.toByte)
  var opens = 0
  var closes = 0
  var capacities = Vector.empty[Int]
  var buffers = Vector.empty[Array[Byte]]
  var failNextRead = false
  var afterRead = Option.empty[() => Unit]

  def reset(): Unit = {
    opens = 0
    closes = 0
    capacities = Vector.empty
    buffers = Vector.empty
    failNextRead = false
    afterRead = None
  }
}

class HadoopVectoredCopySuite extends AnyFunSuite {
  private def input(cap: Int): HadoopInputFile = {
    VectoredCopyTrackingFileSystem.reset()
    val conf = new Configuration(false)
    conf.setClass("fs.copytracking.impl", classOf[VectoredCopyTrackingFileSystem],
      classOf[org.apache.hadoop.fs.FileSystem])
    conf.setBoolean("fs.copytracking.impl.disable.cache", true)
    conf.setInt("parquet.read.allocation.size", cap)
    HadoopInputFile.create(new Path("copytracking:///data"), conf)
  }

  test("small disjoint ranges use bounded buffers and copy to correct offsets") {
    val file = input(8 * 1024 * 1024)
    withResource(HostMemoryBuffer.allocate(64)) { output =>
      file.readVectored(output, Arrays.asList(new CopyRange(3, 7, 0), new CopyRange(20, 13, 25)))
      val first = new Array[Byte](7)
      val second = new Array[Byte](13)
      output.getBytes(first, 0, 0, first.length)
      output.getBytes(second, 0, 25, second.length)
      assert(first.sameElements(VectoredCopyTrackingFileSystem.data.slice(3, 10)))
      assert(second.sameElements(VectoredCopyTrackingFileSystem.data.slice(20, 33)))
      assert(VectoredCopyTrackingFileSystem.capacities.size == 2)
      assert(VectoredCopyTrackingFileSystem.capacities.forall(_ <= 8 * 1024 * 1024))
      assert(VectoredCopyTrackingFileSystem.opens == 1)
      assert(VectoredCopyTrackingFileSystem.closes == 1)
    }
  }

  test("ranges larger than the configured cap retain chunked reads") {
    val file = input(32)
    withResource(HostMemoryBuffer.allocate(80)) { output =>
      file.readVectored(output, Collections.singletonList(new CopyRange(11, 80, 0)))
      val bytes = new Array[Byte](80)
      output.getBytes(bytes, 0, 0, bytes.length)
      assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(11, 91)))
      assert(VectoredCopyTrackingFileSystem.capacities == Vector(32, 32, 32))
      assert(VectoredCopyTrackingFileSystem.closes == 1)
    }
  }

  test("empty and out of bounds requests do not open an input stream") {
    val file = input(32)
    withResource(HostMemoryBuffer.allocate(8)) { output =>
      file.readVectored(output, Collections.emptyList[CopyRange]())
      intercept[IllegalArgumentException] {
        file.readVectored(output, Collections.singletonList(new CopyRange(0, 9, 0)))
      }
      assert(VectoredCopyTrackingFileSystem.opens == 0)
      assert(VectoredCopyTrackingFileSystem.closes == 0)
    }
  }

  test("scratch arrays are reused across files without exceeding the configured cap") {
    withResource(HostMemoryBuffer.allocate(80)) { output =>
      val first = input(80)
      first.readVectored(output, Collections.singletonList(new CopyRange(0, 80, 0)))
      val scratch = VectoredCopyTrackingFileSystem.buffers.head
      val second = input(80)
      second.readVectored(output, Collections.singletonList(new CopyRange(11, 80, 0)))
      assert(VectoredCopyTrackingFileSystem.buffers.head eq scratch)
      val bytes = new Array[Byte](80)
      output.getBytes(bytes, 0, 0, bytes.length)
      assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(11, 91)))

      val smaller = input(32)
      smaller.readVectored(output, Collections.singletonList(new CopyRange(20, 80, 0)))
      assert(VectoredCopyTrackingFileSystem.capacities == Vector(32, 32, 32))
      output.getBytes(bytes, 0, 0, bytes.length)
      assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(20, 100)))
    }
  }

  test("reentrant reads cannot overwrite a loaned scratch array") {
    val file = input(32)
    withResource(HostMemoryBuffer.allocate(32)) { outer =>
      withResource(HostMemoryBuffer.allocate(32)) { inner =>
        VectoredCopyTrackingFileSystem.afterRead = Some(() =>
          file.readVectored(inner, Collections.singletonList(new CopyRange(100, 32, 0))))
        file.readVectored(outer, Collections.singletonList(new CopyRange(3, 32, 0)))
        val buffers = VectoredCopyTrackingFileSystem.buffers
        assert(buffers.size == 2)
        assert(!(buffers.head eq buffers.last))
        val bytes = new Array[Byte](32)
        outer.getBytes(bytes, 0, 0, bytes.length)
        assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(3, 35)))
        inner.getBytes(bytes, 0, 0, bytes.length)
        assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(100, 132)))
      }
    }
  }

  test("a failed read closes its stream and returns scratch for a subsequent successful read") {
    val file = input(80)
    withResource(HostMemoryBuffer.allocate(80)) { output =>
      VectoredCopyTrackingFileSystem.failNextRead = true
      val error = intercept[IOException] {
        file.readVectored(output, Collections.singletonList(new CopyRange(0, 80, 0)))
      }
      assert(error.getMessage == "injected read failure")
      val scratch = VectoredCopyTrackingFileSystem.buffers.head
      assert(VectoredCopyTrackingFileSystem.closes == 1)
      file.readVectored(output, Collections.singletonList(new CopyRange(40, 80, 0)))
      assert(VectoredCopyTrackingFileSystem.buffers.last eq scratch)
      assert(VectoredCopyTrackingFileSystem.closes == 2)
      val bytes = new Array[Byte](80)
      output.getBytes(bytes, 0, 0, bytes.length)
      assert(bytes.sameElements(VectoredCopyTrackingFileSystem.data.slice(40, 120)))
    }
  }
}
