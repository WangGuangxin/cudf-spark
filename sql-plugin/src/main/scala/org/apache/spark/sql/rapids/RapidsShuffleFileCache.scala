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

import java.io.{ByteArrayInputStream, EOFException, InputStream, IOException}
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._
import scala.util.control.NonFatal

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.{ManagedBuffer, NioManagedBuffer}
import org.apache.spark.shuffle.IndexShuffleBlockResolver
import org.apache.spark.storage.ShuffleBlockId

/**
 * Executor-wide cache of open shuffle data and index files, for reading local and host-local
 * shuffle blocks.
 *
 * Spark's resolver opens a map's index file to look up a block's offsets and its data file to
 * read the block, for every block. With tens of thousands of partitions blocks are a few hundred
 * bytes, and each open and close also registers and unregisters a cleaner on a JVM-wide list, so
 * reducers spend their time opening files and waiting on that list's lock. Here each map's two
 * files are opened once per executor, offsets and data are read with positional reads, and the
 * channels are closed when the shuffle is unregistered.
 *
 * Only task threads use the cache, and only with Spark's own IndexShuffleBlockResolver: the
 * shuffle server, other threads and subclasses that may change how blocks are found go through
 * the resolver. Once `maxOpenFiles` channels are open, more files are not cached and their blocks
 * also go through the resolver.
 */
object RapidsShuffleFileCache extends Logging {
  // Two channels per map output. q06 of the stress suite (20000 maps) needs 40000 per executor.
  private val maxOpenFiles = 131072

  private case class Key(shuffleId: Int, mapId: Long, dirsKey: String)

  private final class MapFiles(val data: FileChannel, val index: FileChannel) {
    def close(): Unit = {
      try data.close() catch { case NonFatal(_) => }
      try index.close() catch { case NonFatal(_) => }
    }
  }

  private val cache = new ConcurrentHashMap[Key, MapFiles]()
  private val openFiles = new AtomicInteger(0)

  /**
   * The block's data, read through cached channels when possible, else from `resolver`.
   * @param dirs the local directories of the executor that wrote the block, or None for this one
   */
  def getBlockData(
      resolver: IndexShuffleBlockResolver,
      blockId: ShuffleBlockId,
      dirs: Option[Array[String]]): ManagedBuffer = {
    val files = if (TaskContext.get() != null &&
        resolver.getClass == classOf[IndexShuffleBlockResolver]) {
      mapFiles(resolver, blockId, dirs)
    } else {
      null
    }
    if (files == null) {
      resolver.getBlockData(blockId, dirs)
    } else {
      val offsets = ByteBuffer.allocate(16)
      readFully(files.index, offsets, blockId.reduceId * 8L)
      offsets.flip()
      val start = offsets.getLong()
      val end = offsets.getLong()
      new ChannelSegmentBuffer(files.data, start, end - start)
    }
  }

  private def mapFiles(
      resolver: IndexShuffleBlockResolver,
      blockId: ShuffleBlockId,
      dirs: Option[Array[String]]): MapFiles = {
    val key = Key(blockId.shuffleId, blockId.mapId, dirs.flatMap(_.headOption).getOrElse(""))
    val cached = cache.get(key)
    lazy val indexFile = resolver.getIndexFile(blockId.shuffleId, blockId.mapId, dirs)
    lazy val dataFile = resolver.getDataFile(blockId.shuffleId, blockId.mapId, dirs)
    if (cached != null) {
      cached
    } else if (indexFile == null || dataFile == null || !indexFile.isFile || !dataFile.isFile) {
      // Let the resolver handle, and report, anything but two existing files.
      null
    } else if (openFiles.addAndGet(2) > maxOpenFiles) {
      openFiles.addAndGet(-2)
      null
    } else {
      val opened = try {
        val index = FileChannel.open(indexFile.toPath, StandardOpenOption.READ)
        try {
          new MapFiles(FileChannel.open(dataFile.toPath, StandardOpenOption.READ), index)
        } catch {
          case t: Throwable =>
            index.close()
            throw t
        }
      } catch {
        case t: Throwable =>
          openFiles.addAndGet(-2)
          throw t
      }
      val raced = cache.putIfAbsent(key, opened)
      if (raced != null) {
        opened.close()
        openFiles.addAndGet(-2)
        raced
      } else {
        opened
      }
    }
  }

  /** Closes the channels of a shuffle's map outputs. Its blocks must no longer be read. */
  def removeShuffle(shuffleId: Int): Unit = {
    cache.keySet().asScala.filter(_.shuffleId == shuffleId).foreach { key =>
      val files = cache.remove(key)
      if (files != null) {
        files.close()
        openFiles.addAndGet(-2)
      }
    }
  }

  private def readFully(channel: FileChannel, buf: ByteBuffer, position: Long): Unit = {
    var pos = position
    while (buf.hasRemaining) {
      val n = channel.read(buf, pos)
      if (n < 0) {
        throw new EOFException(s"Unexpected end of shuffle file at $pos")
      }
      pos += n
    }
  }

  // Segments up to this size are read into memory in one positional read when opened as a stream.
  private val maxEagerSegment = 1 << 20

  /** A segment of a cached channel. Never closes the channel. */
  private final class ChannelSegmentBuffer(channel: FileChannel, offset: Long, length: Long)
      extends ManagedBuffer {
    override def size(): Long = length

    override def nioByteBuffer(): ByteBuffer = {
      if (length > Int.MaxValue) {
        throw new IOException(s"Shuffle block of $length bytes is too large to read into memory")
      }
      val buf = ByteBuffer.allocate(length.toInt)
      readFully(channel, buf, offset)
      buf.flip()
      buf
    }

    override def createInputStream(): InputStream = {
      if (length <= maxEagerSegment) {
        new ByteArrayInputStream(nioByteBuffer().array(), 0, length.toInt)
      } else {
        new ChannelSegmentInputStream(channel, offset, length)
      }
    }

    override def retain(): ManagedBuffer = this

    override def release(): ManagedBuffer = this

    override def convertToNetty(): AnyRef = new NioManagedBuffer(nioByteBuffer()).convertToNetty()
  }

  /** Reads a segment of a channel with positional reads. Never closes the channel. */
  private final class ChannelSegmentInputStream(channel: FileChannel, offset: Long, length: Long)
      extends InputStream {
    private var pos = offset
    private val end = offset + length

    override def read(): Int = {
      val one = new Array[Byte](1)
      if (read(one, 0, 1) < 0) -1 else one(0) & 0xff
    }

    override def read(b: Array[Byte], off: Int, len: Int): Int = {
      if (pos >= end) {
        -1
      } else if (len == 0) {
        0
      } else {
        val n = channel.read(ByteBuffer.wrap(b, off, math.min(len.toLong, end - pos).toInt), pos)
        if (n < 0) {
          throw new EOFException(s"Unexpected end of shuffle file at $pos")
        }
        pos += n
        n
      }
    }

    override def skip(n: Long): Long = {
      val skipped = math.max(0L, math.min(n, end - pos))
      pos += skipped
      skipped
    }

    override def available(): Int = math.min(end - pos, Int.MaxValue.toLong).toInt
  }
}
