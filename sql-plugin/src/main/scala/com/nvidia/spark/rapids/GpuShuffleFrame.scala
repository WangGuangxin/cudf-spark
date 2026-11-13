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

import java.io.{DataInputStream, OutputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{BaseDeviceMemoryBuffer, Cuda, DeviceMemoryBuffer, HostMemoryBuffer}
import ai.rapids.cudf.nvcomp.{BatchedCompressor, BatchedDecompressor, BatchedLZ4Compressor, BatchedLZ4Decompressor}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.jni.kudo.KudoTableHeader

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Framed, self-compressed kudo shuffle data for the GPU kudo serializer, with the compression
 * placed on the GPU (nvcomp) or on the CPU (shuffle writer threads), chosen per batch.
 *
 * Every partition slice written by the GPU kudo serializer becomes one frame:
 *   [MAGIC, codec, chunkSize, bodyLen, payloadLen] (big-endian int32), createdNanos (int64),
 *   the kudo table header (uncompressed), and the payload.
 *   - CODEC_RAW: payload is the raw kudo body (payloadLen == bodyLen, 0 for row-count-only tables)
 *   - CODEC_LZ4: payload is the body in nvcomp batched LZ4 layout: numChunks int64 little-endian
 *     compressed chunk sizes, then the LZ4 blocks; chunk i holds body bytes
 *     [i * chunkSize, min((i + 1) * chunkSize, bodyLen)). It is only used when smaller than the
 *     body, so a kudo buffer shorter than the header's totalDataLen is compressed.
 *
 * The same layout is produced by nvcomp on the GPU (GpuPartitioning) and by lz4-java on the CPU
 * (KudoGpuSerializerInstance, in the shuffle writer threads). It is expanded on the GPU when the
 * shuffle coalesce concatenates the tables (KudoGpuTableOperator), or on the CPU for host
 * consumers (SpillableKudoTable.makeKudoTable).
 *
 * Placement ADAPTIVE compresses a batch on the GPU while the CPU compression queue delay (time
 * from the frame's creation to a writer thread picking it up, smoothed) exceeds the configured
 * limit, and on the CPU otherwise; one batch in PROBE_INTERVAL still goes to the CPU so that the
 * delay keeps being measured.
 */
object GpuShuffleFrame extends Logging {
  val MAGIC: Int = 0x4B5A4652 // "KZFR"
  val FRAME_HEADER_BYTES: Int = 28
  val CODEC_RAW = 0
  val CODEC_LZ4 = 1
  /** bytes copied from the start of a slice to read its kudo header size */
  val HEADER_PROBE_BYTES: Int = 1024
  /** bodies smaller than this are not compressed */
  val MIN_COMPRESS_BYTES: Int = 4096
  /** bodies larger than this are not compressed on the CPU (bounds writer-thread memory) */
  val MAX_CPU_COMPRESS_BYTES: Long = 256L << 20
  val PROBE_INTERVAL: Int = 8

  import RapidsConf.ShuffleKudoCompressionPlacement._

  // --- settings (read on the executor) ---------------------------------------------------------
  private lazy val sparkConf = Option(SparkEnv.get).map(_.conf)
  private lazy val rapidsConf = sparkConf.map(new RapidsConf(_))
    .getOrElse(new RapidsConf(Map.empty[String, String]))

  lazy val placement: RapidsConf.ShuffleKudoCompressionPlacement.Value = {
    val p = rapidsConf.shuffleKudoCompressionPlacement
    if (p != NONE && sparkConf.exists(_.getBoolean("spark.shuffle.compress", true))) {
      logWarning(s"${RapidsConf.SHUFFLE_KUDO_COMPRESSION_PLACEMENT.key}=$p needs " +
        "spark.shuffle.compress=false; falling back to NONE")
      NONE
    } else {
      p
    }
  }
  lazy val enabled: Boolean = placement != NONE
  lazy val chunkSize: Int = rapidsConf.shuffleKudoCompressionChunkSize.toInt
  lazy val maxQueueDelayNs: Long =
    rapidsConf.shuffleKudoCompressionAdaptiveMaxQueueDelayMs.toLong * 1000000L

  // --- adaptive placement state (per executor) -------------------------------------------------
  /** smoothed CPU compression queue delay in ns */
  @volatile private var queueDelayNs: Double = 0.0
  private val batchCounter = new AtomicLong(0)
  val gpuCompressedBatches = new AtomicLong(0)
  val cpuBatches = new AtomicLong(0)

  /** Called once per partitioned batch: compress this batch on the GPU? */
  def compressOnGpu(): Boolean = {
    val gpu = placement match {
      case GPU => true
      case ADAPTIVE =>
        queueDelayNs > maxQueueDelayNs &&
          batchCounter.incrementAndGet() % PROBE_INTERVAL != 0
      case _ => false
    }
    val total = if (gpu) gpuCompressedBatches.incrementAndGet() + cpuBatches.get()
      else cpuBatches.incrementAndGet() + gpuCompressedBatches.get()
    if (placement == ADAPTIVE && total % 500 == 0) {
      logInfo(s"shuffle compression placement: ${gpuCompressedBatches.get()} batches on GPU, " +
        s"${cpuBatches.get()} on CPU, CPU queue delay ${(queueDelayNs / 1e6).toLong} ms")
    }
    gpu
  }

  /** Called by the writer threads with the time a RAW frame waited for CPU compression. */
  def recordQueueDelay(ns: Long): Unit = {
    // exponentially weighted; a race between writer threads only loses a sample
    queueDelayNs = queueDelayNs * 0.8 + ns.toDouble * 0.2
  }

  def compressor(): BatchedCompressor = new BatchedLZ4Compressor(chunkSize, 64L << 20)

  def decompressor(): BatchedDecompressor = new BatchedLZ4Decompressor(chunkSize)

  def writeFrameHeader(hb: HostMemoryBuffer, pos: Long, codec: Int, bodyLen: Long,
      payloadLen: Long, createdNanos: Long): Unit = {
    hb.setInt(pos, Integer.reverseBytes(MAGIC))
    hb.setInt(pos + 4, Integer.reverseBytes(codec))
    hb.setInt(pos + 8, Integer.reverseBytes(chunkSize))
    hb.setInt(pos + 12, Integer.reverseBytes(bodyLen.toInt))
    hb.setInt(pos + 16, Integer.reverseBytes(payloadLen.toInt))
    hb.setLong(pos + 20, java.lang.Long.reverseBytes(createdNanos))
  }

  /** Frame for a kudo table without a body (row-count-only): raw = serialized kudo header. */
  def rawHeaderOnlyFrame(raw: Array[Byte], len: Int): Array[Byte] = {
    val out = new Array[Byte](FRAME_HEADER_BYTES + len)
    ByteBuffer.wrap(out).putInt(MAGIC).putInt(CODEC_RAW).putInt(chunkSize).putInt(0).putInt(0)
      .putLong(System.nanoTime())
    System.arraycopy(raw, 0, out, FRAME_HEADER_BYTES, len)
    out
  }

  private lazy val lz4Factory = net.jpountz.lz4.LZ4Factory.fastestInstance()

  /**
   * Write one frame (a host slice produced by GpuPartitioning) to the shuffle stream. RAW frames
   * are LZ4-compressed here, on the CPU in the shuffle writer thread, unless the placement is GPU.
   * @return the number of bytes written
   */
  def writeFrame(frame: HostMemoryBuffer, out: OutputStream): Long = {
    val bb = frame.asByteBuffer().order(ByteOrder.BIG_ENDIAN) // frame header is big-endian
    val codec = bb.getInt(4)
    val bodyLen = bb.getInt(12)
    val payloadLen = bb.getInt(16)
    val created = bb.getLong(20)
    val hdrLen = frame.getLength.toInt - FRAME_HEADER_BYTES - payloadLen
    if (codec == CODEC_RAW && placement == ADAPTIVE) {
      recordQueueDelay(System.nanoTime() - created)
    }
    if (codec != CODEC_RAW || placement == GPU || bodyLen < MIN_COMPRESS_BYTES ||
        bodyLen > MAX_CPU_COMPRESS_BYTES) {
      copyOut(frame, 0, frame.getLength, out)
      frame.getLength
    } else {
      val bodyOff = FRAME_HEADER_BYTES + hdrLen
      val numChunks = (bodyLen + chunkSize - 1) / chunkSize
      val lz4 = lz4Factory.fastCompressor()
      val maxOut = 8 * numChunks + numChunks * lz4.maxCompressedLength(chunkSize)
      val arr = compressBuffer(maxOut)
      val src = frame.asByteBuffer()
      val dst = ByteBuffer.wrap(arr).order(ByteOrder.LITTLE_ENDIAN)
      var pos = 8 * numChunks
      var i = 0
      while (i < numChunks) {
        val off = i * chunkSize
        val len = math.min(chunkSize, bodyLen - off)
        val clen = lz4.compress(src, bodyOff + off, len, dst, pos, maxOut - pos)
        dst.putLong(i * 8, clen.toLong)
        pos += clen
        i += 1
      }
      if (pos >= bodyLen) {
        copyOut(frame, 0, frame.getLength, out)
        frame.getLength
      } else {
        val head = new Array[Byte](FRAME_HEADER_BYTES)
        ByteBuffer.wrap(head).putInt(MAGIC).putInt(CODEC_LZ4).putInt(chunkSize)
          .putInt(bodyLen).putInt(pos).putLong(created)
        out.write(head)
        copyOut(frame, FRAME_HEADER_BYTES, hdrLen, out)
        out.write(arr, 0, pos)
        FRAME_HEADER_BYTES.toLong + hdrLen + pos
      }
    }
  }

  /** Reused per writer thread for outputs up to this size; larger ones are allocated per call. */
  private val MAX_REUSED_COMPRESS_BUFFER = 32 << 20
  private val compressBuffers = new ThreadLocal[Array[Byte]]

  private def compressBuffer(size: Int): Array[Byte] = {
    val cached = compressBuffers.get()
    if (cached != null && cached.length >= size) {
      cached
    } else {
      val b = new Array[Byte](size)
      if (size <= MAX_REUSED_COMPRESS_BUFFER) compressBuffers.set(b)
      b
    }
  }

  private def copyOut(buf: HostMemoryBuffer, start: Long, len: Long, out: OutputStream): Unit = {
    val temp = new Array[Byte](math.min(64 * 1024L, math.max(len, 1L)).toInt)
    var at = start
    var remaining = len
    while (remaining > 0) {
      val n = math.min(remaining, temp.length.toLong).toInt
      buf.getBytes(temp, 0, at, n)
      out.write(temp, 0, n)
      at += n
      remaining -= n
    }
  }

  /**
   * Assemble kudo tables, some of whose bodies are compressed, into the contiguous device layout
   * that KudoGpuSerializer.assembleFromDeviceRaw reads: every table's header followed by its raw
   * body. The host staging buffer only holds what is stored (headers, raw bodies and compressed
   * bodies, packed), so a single host-to-device copy moves the compressed bytes; the GPU then
   * scatters headers and raw bodies into place with one batched copy and decompresses the
   * compressed bodies directly into their place.
   * @param headerSizes, storedLens, bodyLens per table: kudo header size, stored data length
   *                    (compressed when shorter than the body), raw body length
   * @param fill copies table i's header and stored data into the staging buffer at an offset
   * @return (device data, host offsets of every table plus the end)
   */
  def assembleOnGpu(headerSizes: Array[Long], storedLens: Array[Long], bodyLens: Array[Long],
      fill: (Int, HostMemoryBuffer, Long) => Unit): (DeviceMemoryBuffer, Array[Long]) = {
    val n = headerSizes.length
    val stagedSize = (0 until n).map(i => headerSizes(i) + storedLens(i)).sum
    val dataOffsets = new Array[Long](n + 1)
    (0 until n).foreach(i => dataOffsets(i + 1) = dataOffsets(i) + headerSizes(i) + bodyLens(i))
    val copyDst = new ArrayBuffer[Long](); val copySrc = new ArrayBuffer[Long]()
    val copyLen = new ArrayBuffer[Long]()
    val decIdx = new ArrayBuffer[Int](); val decSrc = new ArrayBuffer[Long]()
    withResource(HostMemoryBuffer.allocate(math.max(1L, stagedSize))) { staged =>
      var s = 0L
      (0 until n).foreach { i =>
        fill(i, staged, s)
        val compressed = storedLens(i) < bodyLens(i)
        copyDst += dataOffsets(i); copySrc += s
        copyLen += headerSizes(i) + (if (compressed) 0L else storedLens(i))
        if (compressed) {
          decIdx += i; decSrc += s + headerSizes(i)
        }
        s += headerSizes(i) + storedLens(i)
      }
      withResource(DeviceMemoryBuffer.allocate(math.max(1L, stagedSize))) { stagedDev =>
        stagedDev.copyFromHostBuffer(staged)
        closeOnExcept(DeviceMemoryBuffer.allocate(math.max(1L, dataOffsets(n)))) { dataDev =>
          Cuda.multiBufferCopyAsync(copyDst.map(_ + dataDev.getAddress).toArray,
            copySrc.map(_ + stagedDev.getAddress).toArray, copyLen.toArray, Cuda.DEFAULT_STREAM)
          if (decIdx.nonEmpty) {
            withResource(decIdx.map { i =>
              dataDev.slice(dataOffsets(i) + headerSizes(i), bodyLens(i))
            }.toArray) { outs =>
              val ins: Array[BaseDeviceMemoryBuffer] = decIdx.indices.map { k =>
                stagedDev.slice(decSrc(k), storedLens(decIdx(k))): BaseDeviceMemoryBuffer
              }.toArray
              // decompressAsync closes the inputs
              decompressor().decompressAsync(ins, outs.map(o => o: BaseDeviceMemoryBuffer),
                Cuda.DEFAULT_STREAM)
            }
          }
          Cuda.DEFAULT_STREAM.sync()
          (dataDev, dataOffsets)
        }
      }
    }
  }

  /** CPU decompression of an LZ4 frame payload held in a host buffer into a new host buffer. */
  def cpuDecompressHost(payload: HostMemoryBuffer, outLen: Int): HostMemoryBuffer = {
    closeOnExcept(HostMemoryBuffer.allocate(outLen, false)) { out =>
      val src = payload.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
      val dst = out.asByteBuffer()
      val lz4 = lz4Factory.safeDecompressor()
      val numChunks = ((outLen.toLong + chunkSize - 1) / chunkSize).toInt
      var s = numChunks * 8
      var i = 0
      while (i < numChunks) {
        val clen = src.getLong(i * 8).toInt
        val dOff = i * chunkSize
        val dLen = math.min(chunkSize, outLen - dOff)
        val n = lz4.decompress(src, s, clen, dst, dOff, dLen)
        if (n != dLen) {
          throw new IllegalStateException(s"chunk $i decompressed to $n bytes, expected $dLen")
        }
        s += clen
        i += 1
      }
      out
    }
  }
}

/**
 * Reads GpuShuffleFrame frames. A compressed body is kept as the kudo table's host buffer
 * (shorter than totalDataLen) and expanded later, on the GPU by KudoGpuTableOperator.concat or
 * on the CPU by SpillableKudoTable.makeKudoTable.
 */
class FramedKudoBatchIterator(dIn: DataInputStream, deserTime: GpuMetric)
  extends BaseSerializedTableIterator {
  import GpuShuffleFrame._
  private[this] var pending: Option[(KudoTableHeader, Int)] = None
  private[this] var streamClosed = false

  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      dIn.close()
      streamClosed = true
    }
  }

  override def peekNextBatchSize(): Option[Long] = deserTime.ns {
    if (streamClosed) {
      None
    } else {
      if (pending.isEmpty) {
        val b0 = dIn.read()
        if (b0 < 0) {
          dIn.close()
          streamClosed = true
        } else {
          val magic = (b0 << 24) | (dIn.readUnsignedByte() << 16) |
            (dIn.readUnsignedByte() << 8) | dIn.readUnsignedByte()
          if (magic != MAGIC) {
            throw new IllegalStateException(f"bad shuffle frame magic 0x$magic%08x")
          }
          dIn.readInt() // codec: a payload shorter than the body is LZ4
          val chunk = dIn.readInt()
          if (chunk != chunkSize) {
            throw new IllegalStateException(s"shuffle frame chunk size $chunk != $chunkSize")
          }
          dIn.readInt() // bodyLen (== header totalDataLen)
          val payloadLen = dIn.readInt()
          dIn.readLong() // createdNanos
          val header = KudoTableHeader.readFrom(dIn).get()
          pending = Some((header, payloadLen))
        }
      }
      pending.map(_._1.getTotalDataLen.toLong)
    }
  }

  private def readNextBatch(): ColumnarBatch = deserTime.ns {
    val (header, payloadLen) = pending.get
    pending = None
    if (header.getNumColumns == 0) {
      if (payloadLen > 0) dIn.skipBytes(payloadLen)
      KudoSerializedTableColumn.from(header, null)
    } else {
      val buffer = HostMemoryBuffer.allocate(payloadLen, false)
      closeOnExcept(buffer) { _ =>
        buffer.copyFromStream(0, dIn, payloadLen)
        KudoSerializedTableColumn.from(header, buffer)
      }
    }
  }

  override def hasNext: Boolean = {
    peekNextBatchSize()
    pending.isDefined
  }

  override def next(): (Int, ColumnarBatch) = {
    if (!hasNext) {
      throw new NoSuchElementException("Walked off of the end...")
    }
    (0, readNextBatch())
  }
}
