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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, IOException,
  OutputStream, SequenceInputStream}

import scala.util.control.NonFatal

import com.github.luben.zstd.{Zstd, ZstdCompressCtx, ZstdDecompressCtx}

import org.apache.spark.SparkConf
import org.apache.spark.internal.config
import org.apache.spark.serializer.SerializerManager
import org.apache.spark.storage.ShuffleBlockId

/**
 * Shuffle block streams for Spark's zstd codec without a zstd stream per block.
 *
 * Spark's ZStdCompressionCodec opens a ZstdOutputStream or ZstdInputStream per stream, each with
 * its own native context and a buffer from a pool shared by the whole executor. The
 * multithreaded shuffle opens one stream per record on the write side and one per block on the
 * read side, so with thousands of partitions that setup dominates and threads queue on the pool.
 *
 * Here a record is buffered and compressed in one call with a per-thread ZstdCompressCtx into a
 * single standard zstd frame, which Spark's codec reads. A block is read into memory and each
 * frame is decompressed in one call with a per-thread ZstdDecompressCtx. A block this can't
 * decode that way (a frame without its content size, as Spark's stream writes, a dictionary,
 * a skippable frame, or a block larger than MAX_IN_MEMORY_BLOCK) is handed to the
 * SerializerManager's stream.
 */
object RapidsShuffleZstd {
  private val ZSTD_MAGIC = 0xFD2FB528
  // Records and blocks larger than this use Spark's streams instead of whole-buffer calls.
  private val MAX_IN_MEMORY_BLOCK = 64 << 20
  // Per-thread buffers larger than this are not kept after a record.
  private val MAX_RETAINED_BUFFER = 4 << 20

  /**
   * Whether `serializerManager` wraps shuffle streams in zstd without encryption. Compresses a
   * probe through it and looks for the zstd frame magic, so it agrees with the manager in use.
   */
  def isPlainZstd(serializerManager: SerializerManager): Boolean = {
    try {
      !serializerManager.encryptionEnabled && {
        val bytes = new ByteArrayOutputStream()
        val probe = serializerManager.wrapStream(ShuffleBlockId(0, 0L, 0), bytes)
        try {
          probe.write(1)
        } finally {
          probe.close()
        }
        val b = bytes.toByteArray
        b.length >= 4 && readIntLE(b, 0) == ZSTD_MAGIC
      }
    } catch {
      case NonFatal(_) => false
    }
  }

  private def readIntLE(b: Array[Byte], off: Int): Int =
    (b(off) & 0xFF) | ((b(off + 1) & 0xFF) << 8) | ((b(off + 2) & 0xFF) << 16) |
      ((b(off + 3) & 0xFF) << 24)

  private final class WriterState(level: Int) {
    val ctx: ZstdCompressCtx = new ZstdCompressCtx().setLevel(level).setContentSize(true)
      .setChecksum(false)
    var in: Array[Byte] = new Array[Byte](64 << 10)
    var out: Array[Byte] = new Array[Byte](64 << 10)
    var inUse = false
  }

  private val writerStates = new ThreadLocal[WriterState]()

  private def writerState(level: Int): WriterState = {
    var st = writerStates.get()
    if (st == null) {
      st = new WriterState(level)
      writerStates.set(st)
    }
    st
  }

  /**
   * Buffers what is written and, on close, writes it to `out` as one zstd frame and closes
   * `out`. Uses the calling thread's context and buffers unless they are already in use.
   */
  private final class RecordOutputStream(out: OutputStream, level: Int, sizeHint: Long)
      extends OutputStream {
    private val shared = {
      val st = writerState(level)
      if (st.inUse) null else { st.inUse = true; st }
    }
    private var buf: Array[Byte] = if (shared != null) shared.in else {
      new Array[Byte](math.max(1024L, math.min(sizeHint, MAX_RETAINED_BUFFER.toLong)).toInt)
    }
    private var count = 0
    private var closed = false

    private def ensure(extra: Int): Unit = {
      val needed = count.toLong + extra
      if (needed > buf.length) {
        if (needed > Int.MaxValue - 16) {
          throw new IOException(s"zstd shuffle record too large: $needed bytes")
        }
        val n = math.min(Int.MaxValue - 16L, math.max(needed, buf.length * 2L)).toInt
        buf = java.util.Arrays.copyOf(buf, n)
      }
    }

    override def write(b: Int): Unit = {
      ensure(1)
      buf(count) = b.toByte
      count += 1
    }

    override def write(b: Array[Byte], off: Int, len: Int): Unit = {
      ensure(len)
      System.arraycopy(b, off, buf, count, len)
      count += len
    }

    override def flush(): Unit = {}

    override def close(): Unit = {
      if (!closed) {
        closed = true
        try {
          val bound = Zstd.compressBound(count.toLong)
          val dst = if (shared != null && bound <= shared.out.length) shared.out else {
            new Array[Byte](bound.toInt)
          }
          val ctx = if (shared != null) shared.ctx else {
            new ZstdCompressCtx().setLevel(level).setContentSize(true).setChecksum(false)
          }
          try {
            val n = ctx.compressByteArray(dst, 0, dst.length, buf, 0, count)
            out.write(dst, 0, n)
          } finally {
            if (shared == null) {
              ctx.close()
            }
          }
          if (shared != null) {
            shared.in = if (buf.length <= MAX_RETAINED_BUFFER) buf else shared.in
            if (dst.length <= MAX_RETAINED_BUFFER && dst.length > shared.out.length) {
              shared.out = dst
            }
          }
        } finally {
          if (shared != null) {
            shared.inUse = false
          }
          out.close()
        }
      }
    }
  }

  /**
   * The output side: one zstd frame per stream at `spark.io.compression.zstd.level`. Falls back
   * to the SerializerManager for a record expected to be larger than MAX_IN_MEMORY_BLOCK.
   */
  def outputWrapper(
      serializerManager: SerializerManager,
      conf: SparkConf): (org.apache.spark.storage.BlockId, OutputStream, Long) => OutputStream = {
    val level = conf.get(config.IO_COMPRESSION_ZSTD_LEVEL)
    (blockId, out, sizeHint) =>
      if (sizeHint > MAX_IN_MEMORY_BLOCK) {
        serializerManager.wrapStream(blockId, out)
      } else {
        new RecordOutputStream(out, level, sizeHint)
      }
  }

  private val decompressCtxs = new ThreadLocal[ZstdDecompressCtx]()

  private def decompressCtx(): ZstdDecompressCtx = {
    var ctx = decompressCtxs.get()
    if (ctx == null) {
      ctx = new ZstdDecompressCtx()
      decompressCtxs.set(ctx)
    }
    ctx
  }

  /**
   * The end offset and content size of the zstd frame at `pos`, or None unless it is a frame
   * that records its content size and uses no dictionary.
   */
  private def parseFrame(src: Array[Byte], pos: Int, len: Int): Option[(Int, Long)] = {
    def has(p: Int, n: Int): Boolean = len - p >= n
    if (!has(pos, 5) || readIntLE(src, pos) != ZSTD_MAGIC) {
      None
    } else {
      val fhd = src(pos + 4) & 0xFF
      val singleSegment = (fhd >>> 5) & 1
      val hasChecksum = ((fhd >>> 2) & 1) == 1
      val dictIdFlag = fhd & 3
      // Frame_Content_Size field width from its flag (RFC 8878 section 3.1.1.1.1.1)
      val fcsBytes = (fhd >>> 6) match {
        case 0 => singleSegment
        case 1 => 2
        case 2 => 4
        case _ => 8
      }
      // A window descriptor byte follows the header descriptor unless single segment.
      val fcsPos = pos + 5 + (1 - singleSegment)
      if (dictIdFlag != 0 || fcsBytes == 0 || !has(fcsPos, fcsBytes)) {
        None
      } else {
        var fcs = 0L
        (0 until fcsBytes).foreach(i => fcs |= (src(fcsPos + i) & 0xFFL) << (8 * i))
        if (fcsBytes == 2) {
          fcs += 256 // a 2 byte field stores the size minus 256
        }
        var p = fcsPos + fcsBytes
        var ok = fcs >= 0
        var last = false
        while (ok && !last) {
          if (!has(p, 3)) {
            ok = false
          } else {
            // 3 byte block header: last flag, 2 bit block type, 21 bit block size
            val h = (src(p) & 0xFF) | ((src(p + 1) & 0xFF) << 8) | ((src(p + 2) & 0xFF) << 16)
            last = (h & 1) == 1
            val stored = ((h >>> 1) & 3) match {
              case 1 => 1 // RLE block: one byte repeated block size times
              case 3 => -1 // reserved
              case _ => h >>> 3 // raw or compressed
            }
            ok = stored >= 0 && has(p + 3, stored)
            p += 3 + math.max(stored, 0)
          }
        }
        if (ok && hasChecksum) {
          ok = has(p, 4)
          p += 4
        }
        if (ok) Some((p, fcs)) else None
      }
    }
  }

  /** (start, end, content size) of each frame in `src(0 until len)`, if all can be decoded. */
  private def parseFrames(src: Array[Byte], len: Int): Option[Seq[(Int, Int, Long)]] = {
    val frames = new scala.collection.mutable.ArrayBuffer[(Int, Int, Long)]()
    var pos = 0
    var total = 0L
    var ok = true
    while (ok && pos < len) {
      parseFrame(src, pos, len) match {
        case Some((end, size)) if total + size <= MAX_IN_MEMORY_BLOCK =>
          frames += ((pos, end, size))
          total += size
          pos = end
        case _ =>
          ok = false
      }
    }
    if (ok) Some(frames) else None
  }

  /** Reads up to `limit + 1` bytes of `in`, so a result longer than `limit` means "more". */
  private def readUpTo(in: InputStream, limit: Int): (Array[Byte], Int) = {
    var buf = new Array[Byte](math.min(math.max(in.available(), 1024), limit + 1))
    var count = 0
    var done = false
    while (!done && count <= limit) {
      if (count == buf.length) {
        buf = java.util.Arrays.copyOf(buf, math.min(limit + 1L, buf.length * 2L).toInt)
      }
      val n = in.read(buf, count, buf.length - count)
      if (n < 0) done = true else count += n
    }
    (buf, count)
  }

  /** The input side: decodes whole blocks of frames written with their content sizes. */
  def inputWrapper(
      serializerManager: SerializerManager): (org.apache.spark.storage.BlockId, InputStream) =>
      InputStream = {
    (blockId, in) => {
      val (src, len) = try {
        readUpTo(in, MAX_IN_MEMORY_BLOCK)
      } catch {
        case t: Throwable =>
          in.close()
          throw t
      }
      if (len > MAX_IN_MEMORY_BLOCK) {
        // Too large to decode in memory: stream it, starting with the bytes already read.
        serializerManager.wrapStream(blockId,
          new SequenceInputStream(new ByteArrayInputStream(src, 0, len), in))
      } else {
        in.close()
        parseFrames(src, len) match {
          case None =>
            serializerManager.wrapStream(blockId, new ByteArrayInputStream(src, 0, len))
          case Some(frames) =>
            val dst = new Array[Byte](frames.map(_._3).sum.toInt)
            val ctx = decompressCtx()
            var off = 0
            frames.foreach { case (start, end, size) =>
              val n = ctx.decompressByteArray(dst, off, size.toInt, src, start, end - start)
              if (n != size) {
                throw new IOException(
                  s"zstd frame in $blockId decompressed to $n bytes, expected $size")
              }
              off += n
            }
            new ByteArrayInputStream(dst)
        }
      }
    }
  }
}
