/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.
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

import java.io.{ByteArrayOutputStream, InputStream, IOException, OutputStream}
import java.util.concurrent.{Callable, CompletableFuture, ConcurrentHashMap, ConcurrentLinkedQueue,
  ExecutionException, Executors, ExecutorService, Future, FutureTask, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong, AtomicReference}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.control.NonFatal

import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.NvtxRegistry
import com.nvidia.spark.rapids.RapidsConf
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.jni.kudo.OpenByteArrayOutputStream
import com.nvidia.spark.rapids.shuffle.{CatalogShuffleRequestHandler, RapidsShuffleServer, RapidsShuffleTransport}
import com.nvidia.spark.rapids.spill.SpillablePartialFileHandle

import net.jpountz.lz4.{LZ4BlockInputStream, LZ4BlockOutputStream, LZ4Factory}
import net.jpountz.xxhash.XXHashFactory

import org.apache.spark.{InterruptibleIterator, MapOutputTracker, ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.executor.ShuffleWriteMetrics
import org.apache.spark.internal.{config, Logging}
import org.apache.spark.io.CompressionCodec
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.serializer.SerializerManager
import org.apache.spark.shuffle.{ShuffleWriter, _}
import org.apache.spark.shuffle.api._
import org.apache.spark.shuffle.sort.SortShuffleManager
import org.apache.spark.shuffle.sort.io.{RapidsLocalDiskShuffleDataIO, RapidsLocalDiskShuffleMapOutputWriter}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.rapids.execution.GpuShuffleExchangeExecBase.{METRIC_DATA_READ_SIZE, METRIC_DATA_SIZE, METRIC_SHUFFLE_DESERIALIZATION_TIME, METRIC_SHUFFLE_READ_TIME, METRIC_THREADED_READER_DESER_WAIT_TIME, METRIC_THREADED_READER_IO_WAIT_TIME, METRIC_THREADED_READER_LIMITER_ACQUIRE_COUNT, METRIC_THREADED_READER_LIMITER_ACQUIRE_FAIL_COUNT, METRIC_THREADED_READER_LIMITER_PENDING_BLOCK_COUNT, METRIC_THREADED_WRITER_INPUT_FETCH_TIME, METRIC_THREADED_WRITER_LIMITER_WAIT_TIME, METRIC_THREADED_WRITER_SERIALIZATION_WAIT_TIME}
import org.apache.spark.sql.rapids.shims.{GpuShuffleBlockResolver, RapidsShuffleThreadedReader, RapidsShuffleThreadedWriter}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{RapidsShuffleBlockFetcherIterator, _}
import org.apache.spark.util.{CompletionIterator, Utils}
import org.apache.spark.util.collection.{ExternalSorter, OpenHashSet}

class GpuShuffleHandle[K, V](
    val wrapped: ShuffleHandle,
    override val dependency: GpuShuffleDependency[K, V, V])
  extends BaseShuffleHandle(wrapped.shuffleId, dependency) {

  override def toString: String = s"GPU SHUFFLE HANDLE $shuffleId"
}

class ShuffleHandleWithMetrics[K, V, C](
    shuffleId: Int,
    val metrics: Map[String, SQLMetric],
    override val dependency: GpuShuffleDependency[K, V, C])
  extends BaseShuffleHandle(shuffleId, dependency) {
}

abstract class GpuShuffleBlockResolverBase(
    val wrapped: IndexShuffleBlockResolver,
    catalog: ShuffleBufferCatalog)
  extends ShuffleBlockResolver with Logging {
  override def getBlockData(blockId: BlockId, dirs: Option[Array[String]]): ManagedBuffer = {
    // Get MultithreadedShuffleBufferCatalog dynamically since it may not be
    // initialized when the resolver is created
    val mtCatalogOpt = GpuShuffleEnv.getMultithreadedCatalog

    blockId match {
      case sbid: ShuffleBlockId =>
        // Check MultithreadedShuffleBufferCatalog for single partition blocks
        mtCatalogOpt.flatMap(_.getMergedBufferOption(sbid)) match {
          case Some(buffer) =>
            return buffer
          case None =>
        }

        // Check UCX/CACHE_ONLY catalog
        if (catalog != null && catalog.hasActiveShuffle(sbid.shuffleId)) {
          throw new IllegalStateException(s"The block $blockId is being managed by the catalog")
        }

        // Fall back to the files written to disk
        RapidsShuffleFileCache.getBlockData(wrapped, sbid, dirs)

      case sbbid: ShuffleBlockBatchId =>
        // ShuffleBlockBatchId contains multiple reduce partitions for batch fetch
        mtCatalogOpt match {
          case Some(mtc) if mtc.hasActiveShuffle(sbbid.shuffleId) =>
            return mtc.getMergedBatchBuffer(sbbid)
          case _ =>
        }

        // Check UCX/CACHE_ONLY catalog
        if (catalog != null && catalog.hasActiveShuffle(sbbid.shuffleId)) {
          throw new IllegalStateException(s"The block $blockId is being managed by the catalog")
        }
        wrapped.getBlockData(blockId, dirs)

      case _ =>
        throw new IllegalArgumentException(s"${blockId.getClass} $blockId "
          + "is not currently supported")
    }
  }

  override def stop(): Unit = wrapped.stop()
}

/**
 * The `ShuffleWriteMetricsReporter` is based on accumulators, which are not thread safe.
 * This class is a thin wrapper that adds synchronization, since these metrics will be written
 * by multiple threads.
 * @param wrapped
 */
class ThreadSafeShuffleWriteMetricsReporter(val wrapped: ShuffleWriteMetricsReporter)
  extends ShuffleWriteMetrics {

  def getWriteTime: Long = synchronized {
    TaskContext.get.taskMetrics().shuffleWriteMetrics.writeTime
  }

  override private[spark] def incBytesWritten(v: Long): Unit = synchronized {
    wrapped.incBytesWritten(v)
  }
  override private[spark] def incRecordsWritten(v: Long): Unit = synchronized {
    wrapped.incRecordsWritten(v)
  }
  override private[spark] def incWriteTime(v: Long): Unit = synchronized {
    wrapped.incWriteTime(v)
  }
  override private[spark] def decBytesWritten(v: Long): Unit = synchronized {
    wrapped.decBytesWritten(v)
  }
  override private[spark] def decRecordsWritten(v: Long): Unit = synchronized {
    wrapped.decRecordsWritten(v)
  }
}

object RapidsShuffleInternalManagerBase extends Logging {
  def unwrapHandle(handle: ShuffleHandle): ShuffleHandle = handle match {
    case gh: GpuShuffleHandle[_, _] => gh.wrapped
    case other => other
  }

  /**
   * Whether the shuffle's writers and readers use the shuffle buffer catalog: a GPU shuffle
   * whose writers do not fall back to Spark's sort shuffle.
   */
  def usesShuffleBufferCatalog(handle: ShuffleHandle): Boolean = handle match {
    case gpu: GpuShuffleHandle[_, _] => !gpu.dependency.checksumFallback
    case _ => false
  }

  // this is set by the executor on startup, when the MULTITHREADED
  // shuffle mode is utilized, as per these configs:
  //   spark.rapids.shuffle.multiThreaded.writer.threads
  //   spark.rapids.shuffle.multiThreaded.reader.threads
  private var writerPool: ExecutorService = _
  private var readerPool: ExecutorService = _
  private var mergerPool: ExecutorService = _

  private var mtShuffleInitialized: Boolean = false

  def queueWriteTask[T](task: FutureTask[T]): Future[T] = {
    writerPool.execute(task)
    task
  }

  /** Send a deserialization task to the shared reader pool. */
  def queueReadTask[T](task: Callable[T]): Future[T] = {
    readerPool.submit(task)
  }

  def executeMergerTask(task: Runnable): Unit = mergerPool.execute(task)

  private def shutdownNow(pool: ExecutorService): Unit = {
    pool.shutdownNow().asScala.foreach {
      case future: Future[_] => future.cancel(false)
      case _ =>
    }
  }

  private def awaitTermination(poolName: String, pool: ExecutorService): Unit = {
    var terminated = false
    try {
      terminated = pool.awaitTermination(5, TimeUnit.SECONDS)
    } catch {
      case ie: InterruptedException =>
        Thread.currentThread.interrupt()
        logWarning(s"Interrupted while waiting for thread pool ${poolName} to terminate", ie)
      case e: Throwable =>
        logWarning(s"Exception during shutdown while terminating pool ${poolName}", e)
    } finally {
      if (!terminated) {
        logWarning(s"Thread pool ${poolName} did not terminate within 5 seconds after shutdown")
      }
    }
  }

  def startThreadPoolIfNeeded(
      numWriterThreads: Int,
      numReaderThreads: Int): Unit = synchronized {
    if (!mtShuffleInitialized) {
      mtShuffleInitialized = true
      if (numWriterThreads > 0) {
        writerPool = Executors.newFixedThreadPool(numWriterThreads, new ThreadFactoryBuilder()
          .setNameFormat("rapids-shuffle-writer-%d")
          .setDaemon(true)
          .build())
        mergerPool = Executors.newFixedThreadPool(numWriterThreads, new ThreadFactoryBuilder()
          .setNameFormat("rapids-shuffle-merger-%d")
          .setDaemon(true)
          .build())
      }
      if (numReaderThreads > 0) {
        readerPool = Executors.newFixedThreadPool(numReaderThreads, new ThreadFactoryBuilder()
          .setNameFormat("rapids-shuffle-reader-%d")
          .setDaemon(true)
          .build())
      }
    }
  }

  def stopThreadPool(): Unit = synchronized {
    mtShuffleInitialized = false
    // Interrupt all pools first so workers receive the signal concurrently.
    if (writerPool != null) {
      shutdownNow(writerPool)
    }
    if (readerPool != null) {
      shutdownNow(readerPool)
    }
    if (mergerPool != null) {
      shutdownNow(mergerPool)
    }
    // Then wait for each pool to drain before releasing shared resources.
    if (writerPool != null) {
      awaitTermination("shuffle writer", writerPool)
      writerPool = null
    }
    if (readerPool != null) {
      awaitTermination("shuffle reader", readerPool)
      readerPool = null
    }
    if (mergerPool != null) {
      awaitTermination("shuffle merge", mergerPool)
      mergerPool = null
    }
  }
}

trait RapidsShuffleWriterShimHelper {
  def setChecksumIfNeeded(writer: DiskBlockObjectWriter, partition: Int): Unit = {
    // noop until Spark 3.2.0+
  }

  // Partition lengths, used for MapStatus, but also exposed in Spark 3.2.0+
  private var myPartitionLengths: Array[Long] = null

  // This is a Spark 3.2.0+ function, adding a default here for testing purposes
  def getPartitionLengths(): Array[Long] = myPartitionLengths

  def commitAllPartitions(writer: ShuffleMapOutputWriter, emptyChecksums: Boolean): Array[Long] = {
    myPartitionLengths = doCommitAllPartitions(writer, emptyChecksums)
    myPartitionLengths
  }

  def doCommitAllPartitions(writer: ShuffleMapOutputWriter, emptyChecksums: Boolean): Array[Long]
}

abstract class RapidsShuffleThreadedWriterBase[K, V](
    blockManager: BlockManager,
    handle: ShuffleHandleWithMetrics[K, V, V],
    mapId: Long,
    sparkConf: SparkConf,
    writeMetrics: ShuffleWriteMetricsReporter,
    maxBytesInFlight: Long,
    shuffleExecutorComponents: ShuffleExecutorComponents,
    numWriterThreads: Int)
  extends RapidsShuffleWriter[K, V]
    with RapidsShuffleWriterShimHelper {
  private val dep: ShuffleDependency[K, V, V] = handle.dependency
  private val shuffleId = dep.shuffleId
  private val partitioner = dep.partitioner
  private val numPartitions = partitioner.numPartitions
  private val serializer = dep.serializer.newInstance()
  private val fileBufferSize = sparkConf.get(config.SHUFFLE_FILE_BUFFER_SIZE).toInt * 1024
  private lazy val shuffleOutputWrapper =
    RapidsShuffleLz4.outputWrapper(blockManager.serializerManager, sparkConf)
  private val limiter = new BytesInFlightLimiter(maxBytesInFlight)
  private val limiterWaitTimeMetric =
    handle.metrics.get(METRIC_THREADED_WRITER_LIMITER_WAIT_TIME)
  private val serializationWaitTimeMetric =
    handle.metrics.get(METRIC_THREADED_WRITER_SERIALIZATION_WAIT_TIME)
  private val inputFetchTimeMetric =
    handle.metrics.get(METRIC_THREADED_WRITER_INPUT_FETCH_TIME)

  private var shuffleWriteRange: NvtxId = NvtxRegistry.THREADED_WRITER_WRITE.push()

  // Case class for tracking partial sorted files in multi-batch scenario
  private case class PartialFile(
      handle: SpillablePartialFileHandle,
      partitionLengths: Array[Long],
      mapOutputWriter: ShuffleMapOutputWriter)

  /**
   * Represents a single compressed record ready to be written to disk.
   * Each record has its own independent buffer, avoiding the 2GB limit issue
   * that occurs when multiple records share a single buffer.
   *
   * @param buffer The compressed data buffer (owned by this record, closed after writing)
   * @param compressedSize The actual size of compressed data in buffer
   * @param quotaToRelease The quota to release after writing to disk
   */
  private case class CompressedRecord(
    buffer: OpenByteArrayOutputStream,
    compressedSize: Long,
    quotaToRelease: AtomicLong)

  /**
   * A record waiting for a compression drainer: a slice of a GPU batch, its limiter quota and
   * the future the merger waits on. `owner` decides whether a drainer or a cleanup path closes
   * the slice.
   */
  private case class QueuedRecord(
    key: Any,
    value: Any,
    cb: ColumnarBatch,
    partitionId: Int,
    recordSize: Long,
    quotaToRelease: AtomicLong,
    batch: BatchState,
    future: CompletableFuture[CompressedRecord],
    owner: AtomicBoolean)

  /**
   * A queued record's compression future. Cancelling it (task cleanup) gives back the record's
   * remaining quota right away, as the done() hook of a per-record compression task did, so the
   * quota is free when the writer returns even if a drainer is still working on the record.
   */
  private class RecordFuture(quotaToRelease: AtomicLong)
    extends CompletableFuture[CompressedRecord] {
    override def cancel(mayInterruptIfRunning: Boolean): Boolean = {
      val cancelled = super.cancel(mayInterruptIfRunning)
      if (cancelled) {
        val toRelease = quotaToRelease.getAndSet(0)
        if (toRelease > 0) {
          limiter.release(toRelease)
        }
      }
      cancelled
    }
  }

  // Upper bound on writer-pool tasks one map task uses to drain its compression queue. A new
  // drainer is only submitted while fewer than this many are active, so a burst of small records
  // costs a few pool submissions instead of one per record.
  private val maxDrainers: Int = math.max(1, math.min(numWriterThreads, 4))

  /**
   * Cooperatively writes one GPU batch without occupying a merger thread while waiting for work.
   * At most one step is scheduled for this merger. A step drains all currently ready records and
   * yields when it reaches an empty queue or an unfinished compression future.
   */
  private class BatchMerger(
      writer: ShuffleMapOutputWriter,
      partitionRecords: ConcurrentHashMap[Int,
        ConcurrentLinkedQueue[Future[CompressedRecord]]],
      maxPartitionIdQueued: AtomicInteger) {
    val completionFuture = new CompletableFuture[Void]()

    private val scheduled = new AtomicBoolean(false)
    private val stepFuture = new AtomicReference[FutureTask[Void]]()
    private var currentPartitionToWrite = 0
    private var outputStream: OutputStream = _

    private sealed trait WorkState
    private case object Complete extends WorkState
    private case object NotReady extends WorkState
    private case object EmptyPartition extends WorkState
    private case class ReadyRecord(
        queue: ConcurrentLinkedQueue[Future[CompressedRecord]]) extends WorkState
    private case object FinishedPartition extends WorkState

    def schedule(): Unit = {
      if (!completionFuture.isDone && scheduled.compareAndSet(false, true)) {
        val task = new FutureTask[Void](new Callable[Void] {
          override def call(): Void = {
            runStep()
            null
          }
        })
        stepFuture.set(task)
        try {
          RapidsShuffleInternalManagerBase.executeMergerTask(task)
        } catch {
          case t: Throwable =>
            stepFuture.compareAndSet(task, null)
            scheduled.set(false)
            fail(t)
        }
      }
    }

    def cancel(): Unit = {
      completionFuture.cancel(true)
      Option(stepFuture.get()).foreach(_.cancel(true))
      synchronized {
        closeOutputStreamQuietly()
      }
    }

    private def runStep(): Unit = synchronized {
      try {
        var keepDraining = true
        while (keepDraining && !completionFuture.isDone) {
          currentWorkState match {
            case Complete =>
              completionFuture.complete(null)
              keepDraining = false
            case NotReady =>
              keepDraining = false
            case EmptyPartition =>
              // The producer has advanced beyond this partition without adding records.
              writer.getPartitionWriter(currentPartitionToWrite).openStream().close()
              currentPartitionToWrite += 1
            case ReadyRecord(recordQueue) =>
              if (outputStream == null) {
                outputStream = writer.getPartitionWriter(currentPartitionToWrite).openStream()
              }
              writeRecord(recordQueue)
            case FinishedPartition =>
              closeOutputStream()
              partitionRecords.remove(currentPartitionToWrite)
              currentPartitionToWrite += 1
          }
        }

        if (currentPartitionToWrite >= numPartitions) {
          completionFuture.complete(null)
        }
      } catch {
        case ee: ExecutionException => fail(ee.getCause)
        case ie: InterruptedException =>
          Thread.currentThread().interrupt()
          completionFuture.cancel(true)
          limiter.signalFailure(ie)
        case t: Throwable => fail(t)
      } finally {
        stepFuture.set(null)
        scheduled.set(false)
        if (completionFuture.isDone) {
          closeOutputStreamQuietly()
        }

        // Recheck after clearing scheduled to avoid losing work queued during the transition.
        if (!completionFuture.isDone && hasReadyWork) {
          schedule()
        }
      }
    }

    private def currentWorkState: WorkState = {
      if (currentPartitionToWrite >= numPartitions) {
        Complete
      } else {
        val maxQueued = maxPartitionIdQueued.get()
        if (currentPartitionToWrite > maxQueued) {
          NotReady
        } else {
          val recordQueue = partitionRecords.get(currentPartitionToWrite)
          if (recordQueue == null) {
            EmptyPartition
          } else {
            val future = recordQueue.peek()
            if (future != null && future.isDone) {
              ReadyRecord(recordQueue)
            } else if (future == null && currentPartitionToWrite < maxQueued) {
              FinishedPartition
            } else {
              NotReady
            }
          }
        }
      }
    }

    private def hasReadyWork: Boolean = currentWorkState != NotReady

    private def writeRecord(queue: ConcurrentLinkedQueue[Future[CompressedRecord]]): Unit = {
      var record: CompressedRecord = null
      var buffer: OpenByteArrayOutputStream = null
      try {
        record = queue.peek().get()
        buffer = record.buffer
        if (record.compressedSize > 0) {
          outputStream.write(buffer.getBuf, 0, record.compressedSize.toInt)
        }
        queue.poll()
      } finally {
        if (record != null) {
          val toRelease = record.quotaToRelease.getAndSet(0)
          if (toRelease > 0) {
            limiter.release(toRelease)
          }
        }
        if (buffer != null) {
          buffer.close()
        }
      }
    }

    private def closeOutputStream(): Unit = {
      if (outputStream != null) {
        outputStream.close()
        outputStream = null
      }
    }

    private def closeOutputStreamQuietly(): Unit = {
      try {
        closeOutputStream()
      } catch {
        case _: Exception =>
      }
    }

    private def fail(t: Throwable): Unit = {
      closeOutputStreamQuietly()
      completionFuture.completeExceptionally(t)
      // Signal the limiter so any producer blocked in acquireOrBlock wakes up and
      // throws, allowing the write loop to exit and cleanupBatch to release all quotas.
      limiter.signalFailure(t)
      // Also release quota for already-completed futures that writeRecord will never see.
      partitionRecords.values().asScala.foreach { recordQueue =>
        recordQueue.forEach { future =>
          if (future != null && future.isDone && !future.isCancelled) {
            try {
              val toRelease = future.get().quotaToRelease.getAndSet(0)
              if (toRelease > 0) {
                limiter.release(toRelease)
              }
            } catch {
              case _: Exception => // quota was released in the compression catch block
            }
          }
        }
      }
    }
  }

  /**
   * Encapsulates all state for processing one GPU batch in the multi-batch shuffle write.
   *
   * In multi-batch mode, each GPU batch gets its own BatchState with independent buffers,
   * futures, and a cooperative merger. This enables pipeline parallelism where:
   * - Main thread: processes records and queues compression tasks (non-blocking)
   * - Writer threads: execute compression tasks in parallel (each record gets its own buffer)
   * - Merger steps: write ready partitions sequentially and yield while waiting for work
   *
   * Key design: Each record uses an INDEPENDENT buffer to avoid the 2GB array limit.
   * When a partition has many records, instead of accumulating in one giant buffer,
   * each record's compressed data is in its own small buffer that gets written and
   * released immediately by a merger step.
   *
   * The merger writes partitions in order (0, 1, 2, ...) because Spark's
   * ShuffleMapOutputWriter requires sequential partition writes.
   *
   * @param batchId Unique identifier for this batch (for debugging/logging)
   * @param mapOutputWriter Shuffle output writer for this batch
   * @param partitionRecords Maps partitionId -> queue of compressed record futures.
   *                         Each future completes with an independent CompressedRecord.
   * @param maxPartitionIdQueued Highest partition ID that main thread has queued tasks for.
   *                             The merger uses this to know when a partition is complete.
   * @param merger Cooperative merger state and completion future.
   */
  private case class BatchState(
    batchId: Int,
    mapOutputWriter: ShuffleMapOutputWriter,
    partitionRecords: ConcurrentHashMap[Int,
      ConcurrentLinkedQueue[Future[CompressedRecord]]],
    maxPartitionIdQueued: AtomicInteger,
    merger: BatchMerger) {
    def mergerFuture: Future[_] = merger.completionFuture
    def scheduleMerger(): Unit = merger.schedule()
    def cancelMerger(): Unit = merger.cancel()
  }

  /**
   * Increment the reference count and get the memory size for a value.
   * This method handles ColumnarBatch values with SlicedGpuColumnVector or
   * SlicedSerializedColumnVector columns.
   *
   * @param value the value to process (typically a ColumnarBatch)
   * @return a tuple of (ColumnarBatch with incremented ref count, memory size)
   * @throws IllegalStateException if value is not a ColumnarBatch or contains
   *         unsupported column types
   */
  private def incRefCountAndGetSize(value: Any): (ColumnarBatch, Long) = {
    value match {
      case columnarBatch: ColumnarBatch =>
        if (columnarBatch.numCols() > 0) {
          columnarBatch.column(0) match {
            case _: SlicedGpuColumnVector =>
              (SlicedGpuColumnVector.incRefCount(columnarBatch),
                SlicedGpuColumnVector.getTotalHostMemoryUsed(columnarBatch))
            case _: SlicedSerializedColumnVector =>
              (SlicedSerializedColumnVector.incRefCount(columnarBatch),
                SlicedSerializedColumnVector.getTotalHostMemoryUsed(
                  columnarBatch))
            case other =>
              throw new IllegalStateException(
                s"Unexpected column type in ColumnarBatch: ${other.getClass.getName}. " +
                  "Expected SlicedGpuColumnVector or SlicedSerializedColumnVector.")
          }
        } else {
          (columnarBatch, 0L)
        }
      case other =>
        throw new IllegalStateException(
          s"Unexpected value type: ${if (other == null) "null" else other.getClass.getName}. " +
            "Expected ColumnarBatch.")
    }
  }

  /**
   * Create independent state for processing one GPU batch.
   * This allows multiple batches to be processed in pipeline without blocking.
   */
  private def createBatchState(
      batchId: Int,
      writer: ShuffleMapOutputWriter): BatchState = {

    // Each partition has a queue of compressed record futures.
    // Each record has its own independent buffer for memory isolation.
    val partitionRecords = new ConcurrentHashMap[Int,
      ConcurrentLinkedQueue[Future[CompressedRecord]]]()


    // maxPartitionIdQueued: Tracks the highest partition ID queued by main thread.
    //   - Main thread: updates via set() after adding futures
    //   - Merger step: reads via get() to check if current partition is complete
    //     (currentPartition < maxPartitionIdQueued means all data for currentPartition
    //     has been queued)
    val maxPartitionIdQueued = new AtomicInteger(-1)
    val merger = new BatchMerger(writer, partitionRecords, maxPartitionIdQueued)

    BatchState(
      batchId,
      writer,
      partitionRecords,
      maxPartitionIdQueued,
      merger)
  }

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    val mapOutputWriter = shuffleExecutorComponents.createMapOutputWriter(
      shuffleId,
      mapId,
      numPartitions)
    mapOutputWriters += mapOutputWriter  // Track for cleanup

    val partLengths = if (!records.hasNext) {
      val lengths = commitAllPartitions(mapOutputWriter, true)
      // An empty attempt follows the catalog's rules too: a map id published earlier keeps its
      // output and lengths, and a shuffle already cleaned up fails the task.
      GpuShuffleEnv.getMultithreadedCatalog.map { catalog =>
        catalog.publishMapOutputOrFail(shuffleId, mapId, new MapOutputSegments.Builder().build(),
          numPartitions)
      }.getOrElse(lengths)
    } else {
      writePartitionedGpuBatches(records, mapOutputWriter)
    }

    myMapStatus = Some(getMapStatus(blockManager.shuffleServerId, partLengths, mapId))

    if (shuffleWriteRange != null) {
      shuffleWriteRange.pop()
      shuffleWriteRange = null
    }
  }

  /**
   * Unified write path that handles both single batch and multi-batch tasks.
   * Uses streaming parallel processing with pipelined partition writing.
   *
   * Data flow for each record:
   * 1. ColumnarBatch (already copied to host memory, may be split from GPU batches based on
   *    spark.rapids.shuffle.partitioning.maxCpuBatchSize) -> Main thread acquires limiter quota
   * 2. Writer thread: serialize + compress -> OpenByteArrayOutputStream (JVM heap)
   * 3. Writer thread: release excess quota (recordSize - compressedSize)
   * 4. Merger step: heap buffer -> ShuffleMapOutputWriter (via SpillablePartialFileHandle)
   *    - If MEMORY_WITH_SPILL mode: data may stay in host memory until spill/commit
   *    - If FILE_ONLY mode or spilled: data goes to disk
   * 5. Merger step: release remaining quota after writing to output stream
   * 6. (Multi-batch only) Main thread: mergePartialFiles() combines all batch outputs into
   *    final shuffle file, reading from each SpillablePartialFileHandle sequentially
   *
   * Threading model (same for both scenarios):
   * - Main thread: Processes all records without blocking, queues compression tasks
   * - Merger steps: Run on a shared bounded pool, write ready partitions in order, and yield
   *   when the next compression task is incomplete
   * - Worker threads: Execute compression tasks in parallel
   *
   * Single batch: Cooperative merger steps write directly to the final output file
   *
   * Multi-batch: Detects partition ID decreasing (indicates new batch), creates
   * independent state for each batch (each with its own cooperative merger),
   * then merges all batch outputs into final file.
   */
  private def writePartitionedGpuBatches(
      records: Iterator[Product2[Any, Any]],
      mapOutputWriter: ShuffleMapOutputWriter): Array[Long] = {

    val serializerInstance = serializer
    var recordsWritten: Long = 0L

    // Track timing for metrics
    val writeStartTime = System.nanoTime()
    // Track total written size (compressed size)
    val totalCompressedSize = new AtomicLong(0L)
    var waitTimeOnLimiterNs: Long = 0L
    var inputFetchTimeNs: Long = 0L

    // Records waiting for compression. Instead of one writer-pool task per record, up to
    // maxDrainers tasks drain this queue and exit when it is empty. With thousands of reduce
    // partitions a GPU batch splits into tiny per-partition slices (q06: 20000 partitions,
    // ~0.5 KB per record); one FutureTask per slice made the shared pool queue the bottleneck.
    // Records are never held back: each one is queued as soon as it arrives, so the merger
    // keeps up even while the producer waits for the next GPU batch.
    val compressionQueue = new ConcurrentLinkedQueue[QueuedRecord]()
    val activeDrainers = new AtomicInteger(0)

    def tryClaimDrainerSlot(): Boolean = {
      var claimed = false
      var done = false
      while (!done) {
        val active = activeDrainers.get()
        if (active >= maxDrainers) {
          done = true
        } else if (activeDrainers.compareAndSet(active, active + 1)) {
          claimed = true
          done = true
        }
      }
      claimed
    }

    def releaseRecordQuota(record: QueuedRecord): Unit = {
      val toRelease = record.quotaToRelease.getAndSet(0)
      if (toRelease > 0) {
        limiter.release(toRelease)
      }
    }

    /** Closes the slice, gives back the quota and cancels a record nobody will compress. */
    def abandonRecord(record: QueuedRecord): Unit = {
      if (record.owner.compareAndSet(false, true)) {
        try { record.cb.close() } catch { case _: Exception => }
        releaseRecordQuota(record)
        record.future.cancel(false)
        record.batch.scheduleMerger()
      }
    }

    def abandonQueuedRecords(): Unit = {
      var record = compressionQueue.poll()
      while (record != null) {
        abandonRecord(record)
        record = compressionQueue.poll()
      }
    }

    def compressRecord(record: QueuedRecord): CompressedRecord = {
      withResource(record.cb) { _ =>
        // Create a new buffer for this record.
        // The buffer is closed by the merger thread after writing to disk.
        val buffer = new OpenByteArrayOutputStream()
        closeOnExcept(buffer) { _ =>
          // Serialize + compress + encryption to memory buffer
          val compressedOutputStream = shuffleOutputWrapper(
            ShuffleBlockId(shuffleId, mapId, record.partitionId), buffer, record.recordSize)
          val serializationStream = serializerInstance.serializeStream(compressedOutputStream)
          withResource(serializationStream) { serializer =>
            serializer.writeKey(record.key.asInstanceOf[Any])
            serializer.writeValue(record.value.asInstanceOf[Any])
          }
          // Track total written data size (compressed size)
          val compressedSize = buffer.getCount.toLong
          totalCompressedSize.addAndGet(compressedSize)
          // Release excess quota immediately after compression. Only compressedSize stays held
          // until the merger writes the buffer. A negative remainder means the quota was already
          // claimed by a cancel/failure path, so skip the release.
          val excessQuota = math.max(0L, record.recordSize - compressedSize)
          if (excessQuota > 0 && record.quotaToRelease.addAndGet(-excessQuota) >= 0) {
            limiter.release(excessQuota)
          }
          CompressedRecord(buffer, compressedSize, record.quotaToRelease)
        }
      }
    }

    def processRecord(record: QueuedRecord): Unit = {
      // The owner gate is shared with abandonRecord: whoever wins closes the slice.
      if (record.owner.compareAndSet(false, true)) {
        try {
          if (record.future.isDone) {
            // Cancelled while queued (task cleanup): skip the work.
            try { record.cb.close() } catch { case _: Exception => }
            releaseRecordQuota(record)
          } else {
            val compressed = compressRecord(record)
            if (!record.future.complete(compressed)) {
              // Cancelled while compressing: nobody will write this buffer.
              releaseRecordQuota(record)
              try { compressed.buffer.close() } catch { case _: Exception => }
            } else if (record.batch.merger.completionFuture.isCompletedExceptionally) {
              // The merger already failed and will never write or release this record.
              releaseRecordQuota(record)
            }
          }
        } catch {
          case t: Throwable =>
            releaseRecordQuota(record)
            val cause = t match {
              case ie: InterruptedException =>
                Thread.currentThread().interrupt()
                ie
              case other => other
            }
            record.future.completeExceptionally(new IOException(
              s"Failed compression task for shuffle $shuffleId, map $mapId, " +
                s"partition ${record.partitionId}", cause))
        } finally {
          record.batch.scheduleMerger()
        }
      }
    }

    def drain(): Unit = {
      var keepRunning = true
      while (keepRunning) {
        val record = compressionQueue.poll()
        if (record != null) {
          processRecord(record)
        } else {
          activeDrainers.decrementAndGet()
          // A producer may have queued a record after the poll while all slots looked busy.
          keepRunning = !compressionQueue.isEmpty && tryClaimDrainerSlot()
        }
      }
    }

    def ensureDrainer(): Unit = {
      if (tryClaimDrainerSlot()) {
        val drainer = new FutureTask[Void](new Callable[Void] {
          override def call(): Void = {
            drain()
            null
          }
        }) {
          override def done(): Unit = {
            if (isCancelled) {
              // Cancelled before it ran (pool shutdown): nothing will drain the queue.
              activeDrainers.decrementAndGet()
              abandonQueuedRecords()
            }
          }
        }
        try {
          RapidsShuffleInternalManagerBase.queueWriteTask(drainer)
        } catch {
          case t: Throwable =>
            activeDrainers.decrementAndGet()
            abandonQueuedRecords()
            throw t
        }
      }
    }

    // Multi-batch tracking
    val batchStates = new ArrayBuffer[BatchState]()
    val partialFiles = new ArrayBuffer[PartialFile]()
    var currentBatchId: Int = 0
    var previousMaxPartition: Int = -1
    var isMultiBatch: Boolean = false

    // Create initial batch state
    var currentBatch = createBatchState(currentBatchId, mapOutputWriter)

    try {
      var inputFetchStart = System.nanoTime()
      while (records.hasNext) {
        val record = records.next()
        inputFetchTimeNs += System.nanoTime() - inputFetchStart

        val key = record._1
        val value = record._2
        val reducePartitionId: Int = partitioner.getPartition(key)

        // Detect multi-batch: partition ID must be strictly increasing within a batch.
        // If current partition ID < previous max, it means we've jumped back to an earlier
        // partition, indicating a new upstream GPU batch. Note: we use < instead of <= because
        // consecutive identical partition IDs can occur in two scenarios:
        // 1. Reslicing: when a partition's data exceeds maxCpuBatchSize
        // 2. Data skew: multiple GPU batches each containing only the same partition's data
        // In both cases, merging them into a single shuffle batch is correct and more efficient
        // (fewer partial files, less merge overhead).
        if (reducePartitionId < previousMaxPartition) {
          if (!isMultiBatch) {
            isMultiBatch = true
            logDebug(s"Detected multi-batch scenario for shuffle $shuffleId, " +
              s"transitioning to pipeline mode")
          }

          // Signal current batch is complete by setting maxPartitionIdQueued to numPartitions.
          // This tells the merger thread that all partitions (0 to numPartitions-1) have been
          // queued, so it can finish writing remaining partitions without waiting.
          // Schedule the merger in case it yielded while waiting for more work.
          // Note: We don't block here - the merger runs in parallel while we start next batch.
          currentBatch.maxPartitionIdQueued.set(numPartitions)
          currentBatch.scheduleMerger()

          // Add to list for later finalization
          batchStates += currentBatch

          // Immediately create new batch and continue processing (pipeline!)
          currentBatchId += 1
          val newWriter = shuffleExecutorComponents.createMapOutputWriter(
            shuffleId,
            mapId,
            numPartitions)
          mapOutputWriters += newWriter  // Track for cleanup
          currentBatch = createBatchState(currentBatchId, newWriter)

          // Reset to -1 for new batch. This ensures the first record of the new batch
          // (with any valid partition ID >= 0) won't trigger another batch switch,
          // since reducePartitionId > -1 will always be true.
          previousMaxPartition = -1
        }

        recordsWritten += 1
        previousMaxPartition = math.max(previousMaxPartition, reducePartitionId)

        // Get or create record queue for this partition in current batch
        val recordQueue = currentBatch.partitionRecords.computeIfAbsent(reducePartitionId,
          _ => new ConcurrentLinkedQueue[Future[CompressedRecord]]())

        val (cb, recordSize) = incRefCountAndGetSize(value)

        // Acquire limiter and process compression task immediately
        val waitOnLimiterStart = System.nanoTime()
        closeOnExcept(cb) { _ =>
          limiter.acquireOrBlock(recordSize)
        }
        waitTimeOnLimiterNs += System.nanoTime() - waitOnLimiterStart

        val quota = new AtomicLong(recordSize)
        val queued = QueuedRecord(key, value, cb, reducePartitionId, recordSize,
          quota, currentBatch, new RecordFuture(quota), new AtomicBoolean(false))

        // The future goes into its partition queue before the record can be compressed, so a
        // drainer that completes it always finds it there when it schedules the merger.
        currentBatch.maxPartitionIdQueued.synchronized {
          recordQueue.add(queued.future)
          currentBatch.maxPartitionIdQueued.set(
            math.max(currentBatch.maxPartitionIdQueued.get(), reducePartitionId))
        }
        compressionQueue.add(queued)
        ensureDrainer()

        // Reset timer for next iteration's hasNext/next
        inputFetchStart = System.nanoTime()
      }
      // Account for the final hasNext call that returned false
      inputFetchTimeNs += System.nanoTime() - inputFetchStart

      // Mark end of last batch by setting maxPartitionIdQueued to numPartitions.
      // This signals the merger that all partitions have been queued.
      currentBatch.maxPartitionIdQueued.set(numPartitions)
      currentBatch.scheduleMerger()

      // Add last batch to list
      batchStates += currentBatch

      // Wait for all batches to complete (now they can finish in parallel!)
      var totalSerializationWaitTimeNs: Long = 0L
      batchStates.foreach { batch =>
        try {
          val waitStart = System.nanoTime()
          batch.mergerFuture.get()
          totalSerializationWaitTimeNs += System.nanoTime() - waitStart
        } catch {
          case ee: ExecutionException => throw ee.getCause
        }

        // CRITICAL: Preserve handle before any commit
        // commitAllPartitions() would flush/rename data, so we extract first
        val mtCatalog = GpuShuffleEnv.getMultithreadedCatalog
        if (isMultiBatch || mtCatalog.isDefined) {
          // For multi-batch or when using catalog mode, extract handle
          val (handle, partLengths) = extractHandleAndLengthsFromWriter(
            batch.mapOutputWriter)
          partialFiles += PartialFile(handle, partLengths, batch.mapOutputWriter)
        } else {
          // Single batch without catalog: commit normally
          commitAllPartitions(batch.mapOutputWriter, true)
        }
      }

      // Update write metrics (except writeTime which is calculated at the end)
      writeMetrics.incRecordsWritten(recordsWritten)
      writeMetrics.incBytesWritten(totalCompressedSize.get())
      limiterWaitTimeMetric.foreach(_ += waitTimeOnLimiterNs)
      serializationWaitTimeMetric.foreach(_ += totalSerializationWaitTimeNs)
      inputFetchTimeMetric.foreach(_ += inputFetchTimeNs)

    } finally {
      // Helper to cleanup a single batch
      def cleanupBatch(batch: BatchState): Unit = {
        // Cancel merger completion and any currently scheduled step.
        batch.cancelMerger()

        // Cancel pending futures and close their buffers
        batch.partitionRecords.values().asScala.foreach { recordQueue =>
          var future = recordQueue.poll()
          while (future != null) {
            future.cancel(true)
            // If future completed normally (merger never processed it), release the
            // remaining quota and close the buffer. Quota release comes first so that a
            // buffer close failure does not leave the limiter stranded.
            // Cancelled futures are handled by done().
            if (future.isDone && !future.isCancelled) {
              try {
                val record = future.get()
                val toRelease = record.quotaToRelease.getAndSet(0)
                if (toRelease > 0) {
                  limiter.release(toRelease)
                }
                try { record.buffer.close() } catch { case _: Exception => }
              } catch {
                case _: Exception => // future.get() threw (e.g. failed compression task)
              }
            }
            future = recordQueue.poll()
          }
        }
      }

      // Records that no drainer picked up still hold slices and quota.
      abandonQueuedRecords()

      // Cleanup all tracked batch states
      batchStates.foreach(cleanupBatch)

      // Also cleanup currentBatch if it was never added to batchStates
      // (exception occurred before batchStates += currentBatch)
      if (currentBatch != null && !batchStates.contains(currentBatch)) {
        cleanupBatch(currentBatch)
      }
    }

    // Track whether handles have been transferred to catalog or merged
    var handlesTransferred = false

    try {
      // Handle final output
      val mtCatalog = GpuShuffleEnv.getMultithreadedCatalog

      val result = mtCatalog match {
        case Some(catalog) =>
          // Store data in MultithreadedShuffleBufferCatalog instead of merging.
          // The catalog takes ownership of the handles.
          val lengths = storePartialFilesInCatalog(catalog, partialFiles.toSeq)
          handlesTransferred = true
          lengths
        case None =>
          // Fallback to original merge behavior
          if (isMultiBatch) {
            // Multi-batch: create NEW writer for final merge
            val finalMergeWriter = shuffleExecutorComponents.createMapOutputWriter(
              shuffleId,
              mapId,
              numPartitions)
            mapOutputWriters += finalMergeWriter

            finalMergeWriter match {
              case rapidsWriter: RapidsLocalDiskShuffleMapOutputWriter =>
                rapidsWriter.setForceFileOnlyMode()
              case _ =>
            }

            // mergePartialFiles closes handles in its finally block
            val lengths = mergePartialFiles(partialFiles.toSeq, finalMergeWriter)
            handlesTransferred = true
            lengths
          } else {
            getPartitionLengths
          }
      }

      // Update write time: total time from start minus input fetch time
      val totalWriteTime = System.nanoTime() - writeStartTime
      writeMetrics.incWriteTime(totalWriteTime - inputFetchTimeNs)
      result
    } finally {
      // Clean up handles if they weren't transferred to catalog or merged
      if (!handlesTransferred) {
        partialFiles.foreach { pf =>
          try {
            pf.handle.close()
          } catch {
            case e: Exception =>
              logWarning(s"Failed to close partial file handle during cleanup", e)
          }
        }
      }
    }
  }

  /**
   * Store partial files in MultithreadedShuffleBufferCatalog instead of merging.
   * This avoids the I/O cost of merging while keeping data in memory when possible.
   *
   * @param catalog the MultithreadedShuffleBufferCatalog to store data
   * @param partialFiles list of partial files from all batches
   * @return partition lengths of the output the catalog keeps for this map, which is an earlier
   *         attempt's if one was already published
   * @throws IllegalStateException if the shuffle was cleaned up before the publish; the catalog
   *                               has closed the handles by then
   */
  private def storePartialFilesInCatalog(
      catalog: MultithreadedShuffleBufferCatalog,
      partialFiles: Seq[PartialFile]): Array[Long] = {
    val output = partialFiles.foldLeft(new MapOutputSegments.Builder()) { (builder, pf) =>
      builder.addPartialFile(pf.handle, pf.partitionLengths)
    }.build()
    // Published whole, so readers never see part of a map output. The catalog owns the handles
    // from here and closes them itself if it does not keep this output.
    // Disk write savings are recorded by the reducer when reading the data
    catalog.publishMapOutputOrFail(shuffleId, mapId, output, numPartitions)
  }

  /**
   * Merge multiple partial sorted files into final output.
   * Each partial file contains data for all partitions (0 to N) from one GPU batch.
   * The merged file will have: partition 0 from all batches, partition 1 from all batches, etc.
   *
   * Layout of merged file:
   *   partition 0 data from partial file 0
   *   partition 0 data from partial file 1
   *   ...
   *   partition 0 data from partial file M
   *   partition 1 data from partial file 0
   *   partition 1 data from partial file 1
   *   ...
   */
  private def mergePartialFiles(
      partialFiles: Seq[PartialFile],
      finalWriter: ShuffleMapOutputWriter): Array[Long] = {

    try {
      // For each partition, copy data from all partial files in order
      // Note: Each partial file is read sequentially from beginning to end,
      // so no need to reset read position between partitions
      (0 until numPartitions).foreach { partitionId =>
        val partWriter = finalWriter.getPartitionWriter(partitionId)

        withResource(partWriter.openStream()) { os =>
          partialFiles.foreach { partialFile =>
            val partitionLength = partialFile.partitionLengths(partitionId)
            if (partitionLength > 0) {
              val handle = partialFile.handle

              // Read partition data sequentially
              // No reset needed - handle maintains read position automatically
              val temp = new Array[Byte](fileBufferSize)
              var remaining = partitionLength
              while (remaining > 0) {
                val bytesToRead = math.min(remaining, temp.length).toInt
                val bytesRead = handle.read(temp, 0, bytesToRead)
                if (bytesRead > 0) {
                  os.write(temp, 0, bytesRead)
                  remaining -= bytesRead
                } else {
                  throw new IOException(
                    s"EOF reading partition $partitionId " +
                      s"from partial file ${partialFiles.indexOf(partialFile)}, " +
                      s"expected $partitionLength bytes, got ${partitionLength - remaining}")
                }
              }
            }
          }
        }
      }
    } finally {
      // Cleanup partial file handles
      partialFiles.foreach { pf =>
        try {
          pf.handle.close()
        } catch {
          case e: Exception =>
            logWarning(s"Failed to close partial file handle during cleanup", e)
        }
      }
    }

    // Commit final merged output
    commitAllPartitions(finalWriter, true)
  }

  /**
   * Extract partial file handle and partitionLengths from ShuffleMapOutputWriter.
   * Since we always use RapidsLocalDiskShuffleMapOutputWriter, this is straightforward.
   */
  private def extractHandleAndLengthsFromWriter(writer: ShuffleMapOutputWriter):
  (SpillablePartialFileHandle, Array[Long]) = {
    writer match {
      case rapidsWriter: RapidsLocalDiskShuffleMapOutputWriter =>
        // finishWritePhase() will enable spill
        rapidsWriter.finishWritePhase()
        val handle = rapidsWriter.getPartialFileHandle().getOrElse {
          throw new IllegalStateException("RAPIDS writer should have a handle")
        }
        val lengths = rapidsWriter.getPartitionLengths()
        (handle, lengths)
      case _ =>
        throw new IllegalStateException(
          s"Unexpected writer type: ${writer.getClass.getName}. " +
            "RapidsShuffleManager should always use RapidsLocalDiskShuffleMapOutputWriter.")
    }
  }

  def getBytesInFlight: Long = limiter.getBytesInFlight
}

/**
 * LZ4 streams for the multithreaded shuffle that checksum with the pure Java XXHash32.
 *
 * Spark's LZ4 codec checksums every stream with the JNI StreamingXXHash32 when the native
 * library is loaded. That class has a finalizer, so each one is registered under the JVM-wide
 * finalizer lock when it is created, and the multithreaded shuffle creates one per record on
 * the write side and one per block on the read side: with tens of thousands of partitions,
 * writer and task threads queue on that lock. The streams built here use the same seed, block
 * size and format as Spark's LZ4CompressionCodec, so they read what Spark wrote and vice versa.
 * They are only used when the SerializerManager would wrap shuffle streams in plain LZ4
 * (compressed, not encrypted); anything else goes through the SerializerManager.
 */
object RapidsShuffleLz4 {
  // LZ4BlockOutputStream.DEFAULT_SEED, which Spark's LZ4CompressionCodec also uses
  private val DEFAULT_SEED = 0x9747b28c
  private lazy val lz4Factory = LZ4Factory.fastestInstance()
  private lazy val xxHashFactory = XXHashFactory.fastestJavaInstance()

  /**
   * Whether `serializerManager` wraps shuffle streams in LZ4 without encryption. Asks it to wrap
   * a probe stream rather than reading the configuration, so it agrees with the manager in use.
   */
  private def isPlainLz4(serializerManager: SerializerManager): Boolean = {
    try {
      !serializerManager.encryptionEnabled && {
        val probe = serializerManager.wrapStream(ShuffleBlockId(0, 0L, 0),
          new ByteArrayOutputStream())
        try {
          probe.isInstanceOf[LZ4BlockOutputStream]
        } finally {
          probe.close()
        }
      }
    } catch {
      case NonFatal(_) => false
    }
  }

  /** The equivalent of `serializerManager.wrapStream` for shuffle block input streams. */
  def inputWrapper(serializerManager: SerializerManager): (BlockId, InputStream) => InputStream = {
    if (isPlainLz4(serializerManager)) {
      (_, in) => new LZ4BlockInputStream(in, lz4Factory.fastDecompressor(),
        xxHashFactory.newStreamingHash32(DEFAULT_SEED).asChecksum,
        false) // same as Spark: do not stop at an empty block, streams can be concatenated
    } else if (RapidsShuffleZstd.isPlainZstd(serializerManager)) {
      RapidsShuffleZstd.inputWrapper(serializerManager)
    } else {
      serializerManager.wrapStream
    }
  }

  // Smallest LZ4 block size used for a small record, see blockSizeFor
  private val MIN_RECORD_BLOCK_SIZE = 1024

  /**
   * The LZ4 block size for a stream that will hold about `sizeHint` bytes. LZ4BlockOutputStream
   * allocates two buffers of about the block size per stream, and the multithreaded writer opens
   * one stream per record: with thousands of partitions records are a few hundred bytes, and
   * Spark's 32 KiB default made the writer allocate and zero about 65 KiB per record. A record
   * smaller than the configured block size gets the next power of two above its size plus some
   * headroom for serialization overhead, so it still compresses as a single block. Readers
   * take the block size from each block's header, so streams stay readable by Spark's codec.
   */
  private def blockSizeFor(sizeHint: Long, maxBlockSize: Int): Int = {
    val wanted = sizeHint + MIN_RECORD_BLOCK_SIZE
    if (sizeHint <= 0 || wanted >= maxBlockSize) {
      maxBlockSize
    } else {
      math.max(MIN_RECORD_BLOCK_SIZE, Integer.highestOneBit(wanted.toInt - 1) << 1)
    }
  }

  /**
   * The equivalent of `serializerManager.wrapStream` for shuffle block output streams. The
   * returned function also takes the expected uncompressed size of what will be written.
   */
  def outputWrapper(
      serializerManager: SerializerManager,
      conf: SparkConf): (BlockId, OutputStream, Long) => OutputStream = {
    if (isPlainLz4(serializerManager)) {
      val maxBlockSize = conf.get(config.IO_COMPRESSION_LZ4_BLOCKSIZE).toInt
      (_, out, sizeHint) => new LZ4BlockOutputStream(out, blockSizeFor(sizeHint, maxBlockSize),
        lz4Factory.fastCompressor(), xxHashFactory.newStreamingHash32(DEFAULT_SEED).asChecksum,
        false) // same as Spark: no sync flush
    } else if (RapidsShuffleZstd.isPlainZstd(serializerManager)) {
      RapidsShuffleZstd.outputWrapper(serializerManager, conf)
    } else {
      (blockId, out, _) => serializerManager.wrapStream(blockId, out)
    }
  }
}

class BytesInFlightLimiter(maxBytesInFlight: Long) {
  // Acquire and release are lock-free; the lock is only taken to block and to wake blocked
  // threads. Writer threads release quota once per record, so a monitor here is contended.
  private val inFlight = new AtomicLong(0L)
  private val lock = new Object
  // Number of threads in acquireOrBlock's slow path. Only changed while holding `lock`.
  @volatile private var waiters: Int = 0
  @volatile private var failureCause: Throwable = null

  def acquire(sz: Long): Boolean = {
    if (sz == 0) {
      true
    } else {
      var cur = inFlight.get()
      var acquired = false
      while (!acquired && (cur == 0 || sz + cur <= maxBytesInFlight)) {
        acquired = inFlight.compareAndSet(cur, cur + sz)
        if (!acquired) {
          cur = inFlight.get()
        }
      }
      acquired
    }
  }

  def acquireOrBlock(sz: Long): Unit = {
    if (failureCause != null || !acquire(sz)) {
      lock.synchronized {
        waiters += 1
        try {
          // A release that does not see this waiter decremented inFlight before reading
          // `waiters`, so the acquire below sees its quota.
          while (failureCause == null && !acquire(sz)) {
            lock.wait()
          }
        } finally {
          waiters -= 1
        }
        if (failureCause != null) {
          throw new IOException("Compression batch failed; quota acquisition aborted",
            failureCause)
        }
      }
    }
  }

  def signalFailure(t: Throwable): Unit = lock.synchronized {
    failureCause = t
    lock.notifyAll()
  }

  def release(sz: Long): Unit = {
    inFlight.addAndGet(-sz)
    if (waiters > 0) {
      lock.synchronized {
        lock.notifyAll()
      }
    }
  }

  def getBytesInFlight: Long = inFlight.get()
}

abstract class RapidsShuffleThreadedReaderBase[K, C](
    handle: ShuffleHandleWithMetrics[K, C, C],
    context: TaskContext,
    readMetrics: ShuffleReadMetricsReporter,
    maxBytesInFlight: Long,
    serializerManager: SerializerManager = SparkEnv.get.serializerManager,
    blockManager: BlockManager = SparkEnv.get.blockManager,
    mapOutputTracker: MapOutputTracker = SparkEnv.get.mapOutputTracker,
    canUseBatchFetch: Boolean = false,
    numReaderThreads: Int = 0)
  extends ShuffleReader[K, C] with Logging {

  case class GetMapSizesResult(
      blocksByAddress: Iterator[(BlockManagerId, collection.Seq[(BlockId, Long, Int)])],
      canEnableBatchFetch: Boolean)

  protected def getMapSizes: GetMapSizesResult

  private val GetMapSizesResult(blocksByAddress, canEnableBatchFetch) = getMapSizes

  // For spark versions 3.2.0+ `canEnableBatchFetch` can be false given merged
  // map output
  private val shouldBatchFetch = canUseBatchFetch && canEnableBatchFetch

  private val sqlMetrics = handle.metrics
  private val dep = handle.dependency
  private val deserializationTimeNs = sqlMetrics.get(METRIC_SHUFFLE_DESERIALIZATION_TIME)
  private val shuffleReadTimeNs = sqlMetrics.get(METRIC_SHUFFLE_READ_TIME)
  private val dataReadSize = sqlMetrics.get(METRIC_DATA_READ_SIZE)
  // New metrics for wall time breakdown
  private val ioWaitTimeNs = sqlMetrics.get(METRIC_THREADED_READER_IO_WAIT_TIME)
  private val deserWaitTimeNs = sqlMetrics.get(METRIC_THREADED_READER_DESER_WAIT_TIME)
  // Limiter metrics
  private val limiterAcquireCount =
    sqlMetrics.get(METRIC_THREADED_READER_LIMITER_ACQUIRE_COUNT)
  private val limiterAcquireFailCount =
    sqlMetrics.get(METRIC_THREADED_READER_LIMITER_ACQUIRE_FAIL_COUNT)
  private val limiterPendingBlockCount =
    sqlMetrics.get(METRIC_THREADED_READER_LIMITER_PENDING_BLOCK_COUNT)

  private var shuffleReadRange: NvtxId = NvtxRegistry.THREADED_READER_READ.push()

  private def closeShuffleReadRange(): Unit = {
    if (shuffleReadRange != null) {
      shuffleReadRange.pop()
      shuffleReadRange = null
    }
  }

  onTaskCompletion(context) {
    // should not be needed, but just in case
    closeShuffleReadRange()
  }

  private def fetchContinuousBlocksInBatch: Boolean = {
    val conf = SparkEnv.get.conf
    val serializerRelocatable = dep.serializer.supportsRelocationOfSerializedObjects
    val compressed = conf.get(config.SHUFFLE_COMPRESS)
    val codecConcatenation = if (compressed) {
      CompressionCodec.supportsConcatenationOfSerializedStreams(CompressionCodec.createCodec(conf))
    } else {
      true
    }
    val useOldFetchProtocol = conf.get(config.SHUFFLE_USE_OLD_FETCH_PROTOCOL)
    // SPARK-34790: Fetching continuous blocks in batch is incompatible with io encryption.
    val ioEncryption = conf.get(config.IO_ENCRYPTION_ENABLED)

    val doBatchFetch = shouldBatchFetch && serializerRelocatable &&
      (!compressed || codecConcatenation) && !useOldFetchProtocol && !ioEncryption
    if (shouldBatchFetch && !doBatchFetch) {
      logDebug("The feature tag of continuous shuffle block fetching is set to true, but " +
        "we can not enable the feature because other conditions are not satisfied. " +
        s"Shuffle compress: $compressed, serializer relocatable: $serializerRelocatable, " +
        s"codec concatenation: $codecConcatenation, use old shuffle fetch protocol: " +
        s"$useOldFetchProtocol, io encryption: $ioEncryption.")
    }
    doBatchFetch
  }


  class RapidsShuffleThreadedBlockIterator(
      fetcherIterator: RapidsShuffleBlockFetcherIterator,
      serializer: GpuColumnarBatchSerializer)
    extends Iterator[(Any, Any)] {
    private val queued = new LinkedBlockingQueue[(Any, Any)]
    // Each future deserializes a group of blocks and returns the ones that did not fit
    // in the limiter.
    private val futures = new mutable.Queue[Future[Seq[BlockState]]]()
    private val serializerInstance = serializer.newInstance()
    private val limiter = new BytesInFlightLimiter(maxBytesInFlight)
    // Limits on how many fetched blocks, and how many bytes of their first batches, one
    // reader pool task deserializes, so small blocks share a pool task while large ones
    // still spread across the pool.
    private val MAX_BLOCKS_PER_DESER_TASK = 64
    private val MAX_BYTES_PER_DESER_TASK = 4L << 20
    private val fallbackIter: Iterator[(Any, Any)] with AutoCloseable =
      if (numReaderThreads == 1) {
        // this is the non-optimized case, where we add metrics to capture the blocked
        // time and the deserialization time as part of the shuffle read time.
        new Iterator[(Any, Any)]() with AutoCloseable {
          private var currentIter: Iterator[(Any, Any)] = _
          private var currentStream: AutoCloseable = _
          override def hasNext: Boolean = fetcherIterator.hasNext || (
            currentIter != null && currentIter.hasNext)

          override def close(): Unit = {
            if (currentStream != null) {
              currentStream.close()
              currentStream = null
            }
          }

          override def next(): (Any, Any) = {
            val fetchTimeStart = System.nanoTime()
            var readBlockedTime = 0L
            if (currentIter == null || !currentIter.hasNext) {
              val readBlockedStart = System.nanoTime()
              val (_, stream) = fetcherIterator.next()
              readBlockedTime = System.nanoTime() - readBlockedStart
              // this is stored only to call close on it
              currentStream = stream
              currentIter = serializerInstance.deserializeStream(stream).asKeyValueIterator
            }
            val res = currentIter.next()
            val fetchTime = System.nanoTime() - fetchTimeStart
            deserializationTimeNs.foreach(_ += (fetchTime - readBlockedTime))
            shuffleReadTimeNs.foreach(_ += fetchTime)
            res
          }
        }
      } else {
        null
      }

    // Register a completion handler to close any queued cbs,
    // pending iterators, or futures
    onTaskCompletion(context) {
      // Cancel/join futures first so that no deserializeTask thread can call
      // queued.offer() after we drain the queue below.
      val futuresAndCancellations = futures.map { f =>
        val didCancel = try { f.cancel(true) } catch { case _: Exception => false }
        (f, didCancel)
      }

      // if we weren't able to cancel, we are going to make a best attempt at getting the future
      // and we are going to close it. The timeout is to prevent an (unlikely) infinite wait.
      // If we do timeout then this handler is going to throw.
      var failedFuture: Option[Throwable] = None
      futuresAndCancellations
        .filter { case (_, didCancel) => !didCancel }
        .foreach { case (future, _) =>
          try {
            // this could either be a successful future, or it finished with exception
            // the case when it will fail with exception is when the underlying stream is closed
            // as part of the shutdown process of the task.
            future.get(10, TimeUnit.MILLISECONDS).foreach(_.close())
          } catch {
            case t: Throwable =>
              // this is going to capture the first exception and not worry about others
              // because we probably don't want to spam the UI or log with an exception per
              // block we are fetching
              if (failedFuture.isEmpty) {
                failedFuture = Some(t)
              }
          }
        }
      futures.clear()

      // All futures are now done or cancelled — no more queued.offer() calls can arrive.
      // Safe to drain queued and pendingIts without a race.
      queued.forEach {
        case (_, cb:ColumnarBatch) => cb.close()
      }
      queued.clear()

      // close any materialized BlockState objects that are holding onto netty buffers or
      // file descriptors
      pendingIts.safeClose()
      pendingIts.clear()
      try {
        if (fallbackIter != null) {
          fallbackIter.close()
        }
      } catch {
        case t: Throwable =>
          if (failedFuture.isEmpty) {
            failedFuture = Some(t)
          } else {
            failedFuture.get.addSuppressed(t)
          }
      } finally {
        failedFuture.foreach { e =>
          throw e
        }
      }
    }

    override def hasNext: Boolean = {
      if (fallbackIter != null) {
        fallbackIter.hasNext
      } else {
        pendingIts.nonEmpty || futures.nonEmpty || queued.size() > 0 ||
          fetcherIterator.hasNext
      }
    }

    case class BlockState(
        blockId: BlockId,
        batchIter: BaseSerializedTableIterator,
        origStream: AutoCloseable)
      extends Iterator[(Any, Any)] with AutoCloseable {

      private var nextBatchSize = {
        var success = false
        try {
          val res = batchIter.peekNextBatchSize().getOrElse(0L)
          success = true
          res
        } finally {
          if (!success) {
            // we tried to read from a stream, but something happened
            // lets close it
            close()
          }
        }
      }

      def getNextBatchSize: Long = nextBatchSize

      override def hasNext: Boolean = batchIter.hasNext

      override def next(): (Any, Any) = {
        val nextBatch = batchIter.next()
        var success = false
        try {
          nextBatchSize = batchIter.peekNextBatchSize().getOrElse(0L)
          success = true
          nextBatch
        } finally {
          if (!success) {
            // the call to get a next header threw. We need to close `nextBatch`.
            nextBatch match {
              case (_, cb: ColumnarBatch) => cb.close()
            }
          }
        }
      }

      override def close(): Unit = {
        origStream.close() // make sure we call this on error
      }
    }

    private val pendingIts = new mutable.Queue[BlockState]()

    override def next(): (Any, Any) = {
      require(hasNext, "called next on an empty iterator")
      val res = NvtxRegistry.PARALLEL_DESERIALIZER_ITERATOR_NEXT {
        val result = if (fallbackIter != null) {
          fallbackIter.next()
        } else {
          var waitTime: Long = 0L
          var waitTimeStart: Long = 0L
          popFetchedIfAvailable()
          waitTime = 0L
          if (futures.nonEmpty) {
            NvtxRegistry.BATCH_WAIT {
              waitTimeStart = System.nanoTime()
              val pending = futures.dequeue().get // wait for one future
              val futureWaitThisCall = System.nanoTime() - waitTimeStart
              waitTime += futureWaitThisCall
              deserWaitTimeNs.foreach(_ += futureWaitThisCall)
              // if the future returned block states, we have more work to do
              pendingIts ++= pending
            }
          }

          if (pendingIts.nonEmpty) {
            // if we had pending iterators, we should try to see if now one can be handled
            popFetchedIfAvailable()
          }

          // We either have added futures and so will have items queued
          // or we already exhausted the fetchIterator and are just waiting
          // for our futures to finish. Either way, it's safe to block
          // here while we wait.
          waitTimeStart = System.nanoTime()
          val res = queued.take()
          var success = false
          try {
            val queueWaitThisCall = System.nanoTime() - waitTimeStart
            // limiter is now released immediately after deserialization in deserializeTask
            res match {
              case (_, _: ColumnarBatch) =>
                popFetchedIfAvailable()
              case _ => // do nothing
            }
            waitTime += queueWaitThisCall
            deserWaitTimeNs.foreach(_ += queueWaitThisCall)
            deserializationTimeNs.foreach(_ += waitTime)
            shuffleReadTimeNs.foreach(_ += waitTime)
            success = true
            res
          } finally {
            if (!success) {
              res match {
                case (_, cb: ColumnarBatch) => cb.close()
                case _ => // do nothing
              }
            }
          }
        }

        val uncompressedSize = result match {
          case (_, cb: ColumnarBatch) => SerializedTableColumn.getMemoryUsed(cb)
          case _ => 0 // TODO: do we need to handle other types here?
        }

        dataReadSize.foreach(_ += uncompressedSize)
        result
      }

      // if this is the last call, close our range
      if (!hasNext) {
        closeShuffleReadRange()
      }

      res
    }

    private def deserializeBlock(blockState: BlockState, acquiredSize: Long): Option[BlockState] = {
      var success = false
      // Track the size we need to release (starts with the pre-acquired size)
      var sizeToRelease = acquiredSize
      try {
        var currentBatchSize = blockState.getNextBatchSize
        var didFit = true
        while (blockState.hasNext && didFit) {
          val batch = blockState.next()
          queued.offer(batch)
          // peek at the next batch
          currentBatchSize = blockState.getNextBatchSize
          limiterAcquireCount.foreach(_ += 1)
          didFit = limiter.acquire(currentBatchSize)
          if (didFit) {
            // Successfully acquired, add to sizeToRelease for later release
            sizeToRelease += currentBatchSize
          } else {
            limiterAcquireFailCount.foreach(_ += 1)
          }
        }
        success = true
        if (!didFit) {
          Some(blockState)
        } else {
          None // no further batches
        }
      } finally {
        // Release limiter immediately after deserialization completes
        limiter.release(sizeToRelease)
        // Close blockState (Netty buffer) immediately if:
        // - failed (success = false), or
        // - all batches processed (success = true and returned None)
        if (!success || !blockState.hasNext) {
          blockState.close()
        }
      }
    }

    private def deserializeTask(blockState: BlockState, acquiredSize: Long): Unit =
      deserializeTasks(Seq((blockState, acquiredSize)))

    /**
     * Deserialize a group of blocks, in order, in one reader pool task. Each block's first
     * batch has already been acquired from the limiter. With many partitions a block is often
     * a single small batch, and submitting one pool task per block makes the pool's queue the
     * bottleneck.
     */
    private def deserializeTasks(blocks: Seq[(BlockState, Long)]): Unit = {
      futures += RapidsShuffleInternalManagerBase.queueReadTask(() => {
        val leftOvers = new ArrayBuffer[BlockState]()
        var done = 0
        var success = false
        try {
          blocks.foreach { case (blockState, acquiredSize) =>
            done += 1
            deserializeBlock(blockState, acquiredSize).foreach(leftOvers += _)
          }
          success = true
          leftOvers.toSeq
        } finally {
          if (!success) {
            // The failed block cleaned up after itself. Blocks after it never ran and
            // blocks before it that did not fit are not returned, so clean them up here.
            blocks.drop(done).foreach { case (blockState, acquiredSize) =>
              limiter.release(acquiredSize)
              blockState.close()
            }
            leftOvers.foreach(_.close())
          }
        }
      })
    }

    private def popFetchedIfAvailable(): Unit = {
      // If fetcherIterator is not exhausted, we try and get as many
      // ready results.
      if (pendingIts.nonEmpty) {
        var continue = true
        while(pendingIts.nonEmpty && continue) {
          val blockState = pendingIts.head
          // check if we can handle the head batch now
          val nextBatchSize = blockState.getNextBatchSize
          limiterAcquireCount.foreach(_ += 1)
          if (limiter.acquire(nextBatchSize)) {
            // kick off deserialization task
            pendingIts.dequeue()
            deserializeTask(blockState, nextBatchSize)
          } else {
            limiterAcquireFailCount.foreach(_ += 1)
            continue = false
          }
        }
      } else {
        if (fetcherIterator.hasNext) {
          NvtxRegistry.QUEUE_FETCHED {
            // `resultCount` is exposed from the fetcher iterator and if non-zero,
            // it means that there are pending results that need to be handled.
            // We max with 1, because there could be a race condition where
            // we are trying to get a batch and we haven't received any results
            // yet, we need to block on the fetch for this case so we have
            // something to return.
            var amountToDrain = Math.max(fetcherIterator.resultCount, 1)
            val fetchTimeStart = System.nanoTime()

            // We drain fetched results. That is, we push decode tasks
            // onto our queue until the results in the fetcher iterator
            // are all dequeued (the ones that were completed up until now).
            var readBlockedTime = 0L
            var didFit = true
            // blocks that fit, to deserialize together in one pool task
            val group = new ArrayBuffer[(BlockState, Long)]()
            var groupBytes = 0L
            def submitGroup(): Unit = if (group.nonEmpty) {
              deserializeTasks(group.toList)
              group.clear()
              groupBytes = 0L
            }
            try {
              while (amountToDrain > 0 && fetcherIterator.hasNext && didFit) {
                amountToDrain -= 1
                // fetch block time accounts for time spent waiting for streams.next()
                val readBlockedStart = System.nanoTime()
                val (blockId: BlockId, inputStream) = fetcherIterator.next()
                val ioWaitThisBlock = System.nanoTime() - readBlockedStart
                readBlockedTime += ioWaitThisBlock
                ioWaitTimeNs.foreach(_ += ioWaitThisBlock)

                val deserStream = serializerInstance.deserializeStream(inputStream)
                val batchIter = deserStream.asKeyValueIterator
                  .asInstanceOf[BaseSerializedTableIterator]
                val blockState = BlockState(blockId, batchIter, inputStream)
                // get the next known batch size (there could be multiple batches)
                val nextBatchSize = blockState.getNextBatchSize
                limiterAcquireCount.foreach(_ += 1)
                if (limiter.acquire(nextBatchSize)) {
                  // we can fit at least the first batch in this block
                  // add it to the group for a deserialization task
                  group += ((blockState, nextBatchSize))
                  groupBytes += nextBatchSize
                  if (group.length >= MAX_BLOCKS_PER_DESER_TASK ||
                      groupBytes >= MAX_BYTES_PER_DESER_TASK) {
                    submitGroup()
                  }
                } else {
                  // first batch didn't fit, put iterator aside and stop asking for results
                  // from the fetcher
                  limiterAcquireFailCount.foreach(_ += 1)
                  limiterPendingBlockCount.foreach(_ += 1)
                  pendingIts.enqueue(blockState)
                  didFit = false
                }
              }
            } finally {
              submitGroup()
            }
            // keep track of the overall metric which includes blocked time
            val fetchTime = System.nanoTime() - fetchTimeStart
            deserializationTimeNs.foreach(_ += (fetchTime - readBlockedTime))
            shuffleReadTimeNs.foreach(_ += fetchTime)
          }
        }
      }
    }
  }

  /** Read the combined key-values for this reduce task */
  override def read(): Iterator[Product2[K, C]] = {
    val wrappedStreams = RapidsShuffleBlockFetcherIterator.makeIterator(
      context,
      blockManager,
      SparkEnv.get,
      blocksByAddress,
      serializerManager,
      readMetrics,
      fetchContinuousBlocksInBatch)

    val recordIter = new RapidsShuffleThreadedBlockIterator(
      wrappedStreams,
      dep.serializer.asInstanceOf[GpuColumnarBatchSerializer])

    // Update the context task metrics for each record read.
    val metricIter = CompletionIterator[(Any, Any), Iterator[(Any, Any)]](
      recordIter.map { record =>
        readMetrics.incRecordsRead(1)
        record
      }, context.taskMetrics().mergeShuffleReadMetrics())

    // An interruptible iterator must be used here in order to support task cancellation
    val interruptibleIter = new InterruptibleIterator[(Any, Any)](context, metricIter)

    val aggregatedIter: Iterator[Product2[K, C]] = if (dep.aggregator.isDefined) {
      if (dep.mapSideCombine) {
        // We are reading values that are already combined
        val combinedKeyValuesIterator = interruptibleIter.asInstanceOf[Iterator[(K, C)]]
        dep.aggregator.get.combineCombinersByKey(combinedKeyValuesIterator, context)
      } else {
        // We don't know the value type, but also don't care -- the dependency *should*
        // have made sure its compatible w/ this aggregator, which will convert the value
        // type to the combined type C
        val keyValuesIterator = interruptibleIter.asInstanceOf[Iterator[(K, Nothing)]]
        dep.aggregator.get.combineValuesByKey(keyValuesIterator, context)
      }
    } else {
      interruptibleIter.asInstanceOf[Iterator[Product2[K, C]]]
    }

    // Sort the output if there is a sort ordering defined.
    val resultIter = dep.keyOrdering match {
      case Some(keyOrd: Ordering[K]) =>
        // Create an ExternalSorter to sort the data.
        val sorter =
          new ExternalSorter[K, C, C](context, ordering = Some(keyOrd), serializer = dep.serializer)
        sorter.insertAll(aggregatedIter)
        context.taskMetrics().incMemoryBytesSpilled(sorter.memoryBytesSpilled)
        context.taskMetrics().incDiskBytesSpilled(sorter.diskBytesSpilled)
        context.taskMetrics().incPeakExecutionMemory(sorter.peakMemoryUsedBytes)
        // Use completion callback to stop sorter if task was finished/cancelled.
        onTaskCompletion(context) {
          sorter.stop()
        }
        CompletionIterator[Product2[K, C], Iterator[Product2[K, C]]](sorter.iterator, sorter.stop())
      case None =>
        aggregatedIter
    }

    resultIter match {
      case _: InterruptibleIterator[Product2[K, C]] => resultIter
      case _ =>
        // Use another interruptible iterator here to support task cancellation as aggregator
        // or(and) sorter may have consumed previous interruptible iterator.
        new InterruptibleIterator[Product2[K, C]](context, resultIter)
    }
  }
}

class RapidsCachingWriter[K, V](
    blockManager: BlockManager,
    // Never keep a reference to the ShuffleHandle in the cache as it being GCed triggers
    // the data being released
    handle: GpuShuffleHandle[K, V],
    mapId: Long,
    metricsReporter: ShuffleWriteMetricsReporter,
    catalog: ShuffleBufferCatalog,
    rapidsShuffleServer: Option[RapidsShuffleServer],
    metrics: Map[String, SQLMetric])
  extends RapidsCachingWriterBase[K, V](blockManager, handle, mapId, rapidsShuffleServer, catalog) {

  private val uncompressedMetric: SQLMetric = metrics(METRIC_DATA_SIZE)

  // This is here for the special case where we have no columns like with the .count
  // case or when we have 0-byte columns. We pick 100 as an arbitrary number so that
  // we can shuffle these degenerate batches, which have valid metadata and should be
  // used on the reducer side for computation.
  private val DEGENERATE_PARTITION_BYTE_SIZE_DEFAULT: Long = 100L

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    // NOTE: This MUST NOT CLOSE the incoming batches because they are
    //       closed by the input iterator generated by GpuShuffleExchangeExec
    NvtxRegistry.RAPIDS_CACHING_WRITER_WRITE {
      var bytesWritten: Long = 0L
      var recordsWritten: Long = 0L
      records.foreach { p =>
        val partId = p._1.asInstanceOf[Int]
        val batch = p._2.asInstanceOf[ColumnarBatch]
        logDebug(s"Caching shuffle_id=${handle.shuffleId} map_id=$mapId, partId=$partId, "
          + s"batch=[num_cols=${batch.numCols()}, num_rows=${batch.numRows()}]")
        recordsWritten = recordsWritten + batch.numRows()
        var partSize: Long = 0
        val blockId = ShuffleBlockId(handle.shuffleId, mapId, partId)
        if (batch.numRows > 0 && batch.numCols > 0) {
          // Add the table to the shuffle store
          batch.column(0) match {
            case c: GpuPackedTableColumn =>
              val contigTable = c.getContiguousTable
              partSize = c.getTableBuffer.getLength
              uncompressedMetric += partSize
              writtenBufferIds += catalog.addContiguousTable(
                blockId,
                contigTable,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_TASK_PRIORITY)
            case c: GpuCompressedColumnVector =>
              partSize = c.getTableBuffer.getLength
              uncompressedMetric += c.getTableMeta.bufferMeta().uncompressedSize()
              writtenBufferIds += catalog.addCompressedBatch(
                blockId,
                batch,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_TASK_PRIORITY)
            case c =>
              throw new IllegalStateException(s"Unexpected column type: ${c.getClass}")
          }
          bytesWritten += partSize
          // if the size is 0 and we have rows, we are in a case where there are columns
          // but the type is such that there isn't a buffer in the GPU backing it.
          // For example, a Struct column without any members. We treat such a case as if it
          // were a degenerate table.
          if (partSize == 0 && batch.numRows() > 0) {
            sizes(partId) += DEGENERATE_PARTITION_BYTE_SIZE_DEFAULT
          } else {
            sizes(partId) += partSize
          }
        } else {
          // no device data, tracking only metadata
          val tableMeta = MetaUtils.buildDegenerateTableMeta(batch)
          writtenBufferIds += catalog.addDegenerateRapidsBuffer(
            blockId,
            tableMeta)

          // ensure that we set the partition size to the default in this case if
          // we have non-zero rows, so this degenerate batch is shuffled.
          if (batch.numRows > 0) {
            sizes(partId) += DEGENERATE_PARTITION_BYTE_SIZE_DEFAULT
          }
        }
      }
      metricsReporter.incBytesWritten(bytesWritten)
      metricsReporter.incRecordsWritten(recordsWritten)
      if (catalog.mapIdsCanRepeat) {
        catalog.commitMapOutput(handle.shuffleId, mapId, writtenBufferIds, sizes)
        // the catalog owns these buffers now, and a later stop(false) must not remove them
        writtenBufferIds.clear()
      }
    }
  }


  def getPartitionLengths(): Array[Long] = {
    throw new UnsupportedOperationException("TODO")
  }
}

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes. When configuring
 *       Apache Spark to use the RAPIDS shuffle manager,
 */
class RapidsShuffleInternalManagerBase(conf: SparkConf, val isDriver: Boolean)
  extends ShuffleManager with RapidsShuffleHeartbeatHandler with Logging
  with RapidsShuffleReaderShim with ProxyShuffleReaderDelegate {

  def getServerId: BlockManagerId = server.fold(blockManager.blockManagerId)(_.getId)

  override def addPeer(peer: BlockManagerId): Unit = {
    transport.foreach { t =>
      try {
        t.connect(peer)
      } catch {
        case ex: Exception =>
          // We ignore the exception after logging in this instance because
          // we may have a peer that doesn't exist anymore by the time `addPeer` is invoked
          // due to a heartbeat response from the driver, or the peer may have a temporary network
          // issue.
          //
          // This is safe because `addPeer` is only invoked due to a heartbeat that is used to
          // opportunistically hide cost of initializing transport connections. The transport
          // will re-try if it must fetch from this executor at a later time, in that case
          // a connection failure causes the tasks to fail.
          logWarning(s"Unable to connect to peer $peer, ignoring!", ex)
      }
    }
  }

  private val rapidsConf = new RapidsConf(conf)

  if (!isDriver && rapidsConf.isMultiThreadedShuffleManagerMode) {
    RapidsShuffleInternalManagerBase.startThreadPoolIfNeeded(
      rapidsConf.shuffleMultiThreadedWriterThreads,
      rapidsConf.shuffleMultiThreadedReaderThreads)
  }

  protected val wrapped = new SortShuffleManager(conf)

  private[this] val transportEnabledMessage =
    if (!rapidsConf.isUCXShuffleManagerMode) {
      if (rapidsConf.isCacheOnlyShuffleManagerMode) {
        "Transport disabled (local cached blocks only)"
      } else {
        val numWriteThreads = rapidsConf.shuffleMultiThreadedWriterThreads
        val numReadThreads = rapidsConf.shuffleMultiThreadedReaderThreads
        s"Multi-threaded shuffle mode " +
          s"(write threads=$numWriteThreads, read threads=$numReadThreads)"
      }
    } else {
      s"Transport enabled (remote fetches will use ${rapidsConf.shuffleTransportClassName}"
    }

  logWarning(s"Rapids Shuffle Plugin enabled. ${transportEnabledMessage}. To disable the " +
    s"RAPIDS Shuffle Manager set `${RapidsConf.SHUFFLE_MANAGER_ENABLED}` to false")

  //Many of these values like blockManager are not initialized when the constructor is called,
  // so they all need to be lazy values that are executed when things are first called

  // NOTE: this can be null in the driver side.
  protected lazy val env = SparkEnv.get
  protected lazy val blockManager = env.blockManager
  // Stable reasons to always fall back to SortShuffleManager, evaluated once at
  // first shuffle registration.
  protected lazy val shouldAlwaysFallBack = {
    val fallThroughReasons = new ListBuffer[String]()
    if (!rapidsConf.isMultiThreadedShuffleManagerMode) {
      if (GpuShuffleEnv.isExternalShuffleEnabled) {
        fallThroughReasons += "External Shuffle Service is enabled"
      }
      if (GpuShuffleEnv.isSparkAuthenticateEnabled) {
        fallThroughReasons += "Spark authentication is enabled"
      }
    }
    if (rapidsConf.isSqlExplainOnlyEnabled) {
      fallThroughReasons += "Plugin is in explain only mode"
    }
    if (fallThroughReasons.nonEmpty) {
      logWarning(s"Rapids Shuffle Plugin is falling back to SortShuffleManager " +
        s"because: ${fallThroughReasons.mkString(", ")}")
    }
    fallThroughReasons.nonEmpty
  }

  private val rowBasedChecksumFallbackLogged = new AtomicBoolean(false)

  private def shouldFallThroughForShuffle: Boolean = {
    val rowBasedChecksumFallback = GpuShuffleEnv.isRowBasedChecksumEnabled
    if (rowBasedChecksumFallback) {
      if (rowBasedChecksumFallbackLogged.compareAndSet(false, true)) {
        logWarning("Rapids Shuffle Plugin is falling back to SortShuffleManager because: " +
          "Detected order-independent checksum enabled " +
          "(spark.sql.shuffle.orderIndependentChecksum.enabled or enableFullRetryOnMismatch). " +
          "This Spark 4.1+ feature is not yet supported by Spark-Rapids.")
      }
    }
    shouldAlwaysFallBack || rowBasedChecksumFallback
  }

  private lazy val localBlockManagerId = blockManager.blockManagerId

  // Used to prevent stopping multiple times RAPIDS Shuffle Manager internals.
  // see the `stop` method
  private var stopped: Boolean = false

  // Code that expects the shuffle catalog to be initialized gets it this way,
  // with error checking in case we are in a bad state.
  protected def getCatalogOrThrow: ShuffleBufferCatalog =
    Option(GpuShuffleEnv.getCatalog).getOrElse(
      throw new IllegalStateException("The ShuffleBufferCatalog is not initialized but the " +
        "RapidsShuffleManager is configured"))

  protected lazy val resolver =
    if (shouldAlwaysFallBack) {
      wrapped.shuffleBlockResolver
    } else if (rapidsConf.isMultiThreadedShuffleManagerMode) {
      // MULTITHREADED mode: use GpuShuffleBlockResolver
      // mtCatalog will be fetched dynamically in getBlockData() since it may not be
      // initialized yet when this resolver is created
      new GpuShuffleBlockResolver(
        wrapped.shuffleBlockResolver.asInstanceOf[IndexShuffleBlockResolver],
        null) // No UCX catalog in MULTITHREADED mode
    } else { // we didn't fallback && we are using the UCX shuffle
      val catalog = GpuShuffleEnv.getCatalog
      if (catalog == null) {
        if (isDriver) {
          // this is an OK state to be in. It means we didn't fall back
          // (`shouldFallbackThroughOnEverything` is false) and this is just the driver
          // in a job with RapidsShuffleManager enabled. We want to just use the regular
          // shuffle block resolver here, since we don't do anything on the driver.
          wrapped.shuffleBlockResolver
        } else {
          // this would be bad: if we are an executor, didn't fallback, and RapidsShuffleManager
          // is enabled, we need to fail.
          throw new IllegalStateException(
            "An executor with RapidsShuffleManager is trying to use a ShuffleBufferCatalog " +
              "that isn't initialized."
          )
        }
      } else {
        // A driver in local mode with the RapidsShuffleManager enabled would go through this
        // else statement, because the "executor" is the driver, and isDriver=true, or
        // The regular case where the executor has RapidsShuffleManager enabled.
        // What these cases have in common is that `catalog` is defined.
        new GpuShuffleBlockResolver(wrapped.shuffleBlockResolver, catalog)
      }
    }

  private[this] lazy val transport: Option[RapidsShuffleTransport] = {
    if (rapidsConf.isUCXShuffleManagerMode && !isDriver) {
      Some(RapidsShuffleTransport.makeTransport(blockManager.shuffleServerId, rapidsConf))
    } else {
      None
    }
  }

  private[this] lazy val server: Option[RapidsShuffleServer] = {
    if (rapidsConf.isGPUShuffle && !isDriver) {
      val catalog = getCatalogOrThrow
      val requestHandler = new CatalogShuffleRequestHandler(catalog)
      val server = transport.get.makeServer(requestHandler)
      server.start()
      Some(server)
    } else {
      None
    }
  }

  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    // Always register with the wrapped handler so we can write to it ourselves if needed
    val orig = wrapped.registerShuffle(shuffleId, dependency)

    dependency match {
      case gpuDependency: GpuShuffleDependency[K, V, C] if gpuDependency.useGPUShuffle =>
        val gpuDep = gpuDependency.asInstanceOf[GpuShuffleDependency[K, V, V]]
        gpuDep.checksumFallback = shouldFallThroughForShuffle
        if (rapidsConf.isMultiThreadedShuffleManagerMode) orig
        else new GpuShuffleHandle(orig, gpuDep)
      case _ if shouldAlwaysFallBack ||
        rapidsConf.isMultiThreadedShuffleManagerMode => orig
      case _ => orig
    }
  }

  lazy val execComponents: Option[ShuffleExecutorComponents] = {
    // Check if user configured a different ShuffleDataIO plugin
    val configuredPlugin = conf.get("spark.shuffle.sort.io.plugin.class", "")
    val rapidsPlugin = "org.apache.spark.shuffle.sort.io.RapidsLocalDiskShuffleDataIO"

    if (configuredPlugin.nonEmpty && !configuredPlugin.endsWith("RapidsLocalDiskShuffleDataIO")) {
      throw new IllegalArgumentException(
        s"RapidsShuffleManager requires 'spark.shuffle.sort.io.plugin.class' to be " +
          s"'$rapidsPlugin' or unset, but found '$configuredPlugin'. " +
          s"Please update your configuration.")
    }

    val rapidsDataIO = new RapidsLocalDiskShuffleDataIO(conf)
    val executorComponents = rapidsDataIO.executor()

    val extraConfigs = conf.getAllWithPrefix(ShuffleDataIOUtils.SHUFFLE_SPARK_CONF_PREFIX).toMap
    executorComponents.initializeExecutor(
      conf.getAppId,
      SparkEnv.get.executorId,
      extraConfigs.asJava)
    Some(executorComponents)
  }

  /**
   * A mapping from shuffle ids to the task ids of mappers producing output for those shuffles.
   */
  protected val taskIdMapsForShuffle = new ConcurrentHashMap[Int, OpenHashSet[Long]]()

  private def trackMapTaskForCleanup(shuffleId: Int, mapId: Long): Unit = {
    // this uses OpenHashSet as it is copied from Spark
    val mapTaskIds = taskIdMapsForShuffle.computeIfAbsent(
      shuffleId, _ => new OpenHashSet[Long](16))
    mapTaskIds.synchronized {
      mapTaskIds.add(mapId)
    }
  }

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metricsReporter: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    handle match {
      case gpu: GpuShuffleHandle[_, _]
          if !RapidsShuffleInternalManagerBase.usesShuffleBufferCatalog(gpu) =>
        wrapped.getWriter(gpu.wrapped, mapId, context, metricsReporter)
      case gpu: GpuShuffleHandle[_, _] =>
        registerGpuShuffle(handle.shuffleId, context.taskAttemptId())
        new RapidsCachingWriter(
          env.blockManager,
          gpu.asInstanceOf[GpuShuffleHandle[K, V]],
          mapId,
          metricsReporter,
          getCatalogOrThrow,
          server,
          gpu.dependency.metrics)
      case handle: BaseShuffleHandle[_, _, _] =>
        handle.dependency match {
          case gpuDep: GpuShuffleDependency[_, _, _]
            if gpuDep.useMultiThreadedShuffle &&
              !gpuDep.checksumFallback &&
              rapidsConf.shuffleMultiThreadedWriterThreads > 0 =>
            // use the threaded writer if the number of threads specified is 1 or above,
            // with 0 threads we fallback to the Spark-provided writer.
            // Register shuffle with MultithreadedShuffleBufferCatalog
            registerGpuShuffle(handle.shuffleId, context.taskAttemptId())
            val handleWithMetrics = new ShuffleHandleWithMetrics(
              handle.shuffleId,
              gpuDep.metrics,
              // cast the handle with specific generic types due to type-erasure
              gpuDep.asInstanceOf[GpuShuffleDependency[K, V, V]])
            // we need to track this mapId so we can clean it up later on unregisterShuffle
            trackMapTaskForCleanup(handle.shuffleId, mapId)
            // in most scenarios, the pools have already started, except for local mode
            // here we try to start them if we see they haven't
            RapidsShuffleInternalManagerBase.startThreadPoolIfNeeded(
              rapidsConf.shuffleMultiThreadedWriterThreads,
              rapidsConf.shuffleMultiThreadedReaderThreads)
            new RapidsShuffleThreadedWriter[K, V](
              blockManager,
              handleWithMetrics,
              mapId,
              conf,
              new ThreadSafeShuffleWriteMetricsReporter(metricsReporter),
              rapidsConf.shuffleMultiThreadedMaxBytesInFlight,
              execComponents.get,
              rapidsConf.shuffleMultiThreadedWriterThreads)
          case _ =>
            wrapped.getWriter(handle, mapId, context, metricsReporter)
        }
      case _ =>
        wrapped.getWriter(handle, mapId, context, metricsReporter)
    }
  }

  def getReaderImpl[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    handle match {
      case gpuHandle: GpuShuffleHandle[_, _]
          if !RapidsShuffleInternalManagerBase.usesShuffleBufferCatalog(gpuHandle) =>
        ShuffleManagerShims.getReader(wrapped, gpuHandle.wrapped, startMapIndex, endMapIndex,
          startPartition, endPartition, context, metrics)
      case gpuHandle: GpuShuffleHandle[_, _] =>
        logInfo(s"Asking map output tracker for dependency ${gpuHandle.dependency}, " +
          s"map output sizes for: ${gpuHandle.shuffleId}, parts=$startPartition-$endPartition")
        if (gpuHandle.dependency.keyOrdering.isDefined) {
          // very unlikely, but just in case
          throw new IllegalStateException("A key ordering was requested for a gpu shuffle "
            + s"dependency ${gpuHandle.dependency.keyOrdering.get}, this is not supported.")
        }

        val blocksByAddress = NvtxRegistry.GET_MAP_SIZES_BY_EXEC_ID {
          SparkEnv.get.mapOutputTracker.getMapSizesByExecutorId(gpuHandle.shuffleId,
            startMapIndex, endMapIndex, startPartition, endPartition)
        }

        new RapidsCachingReader(rapidsConf, localBlockManagerId,
          blocksByAddress,
          context,
          metrics,
          transport,
          getCatalogOrThrow,
          gpuHandle.dependency.sparkTypes)
      case other: ShuffleHandle if
        rapidsConf.isMultiThreadedShuffleManagerMode
          && rapidsConf.shuffleMultiThreadedReaderThreads > 0 =>
        // we enable a multi-threaded reader in the case where we have 1 or
        // more threads and we have enbled the MULTITHREADED shuffle mode.
        // We special case the threads=1 case in the reader to behave like regular
        // spark, but this allows us to add extra metrics that Spark normally
        // doesn't look at while materializing blocks.
        val baseHandle = other.asInstanceOf[BaseShuffleHandle[K, C, C]]

        // we check that the dependency is a `GpuShuffleDependency` and if not
        // we go back to the regular path (e.g. is a GpuColumnarExchange?)
        // TODO: it may make sense to expand this code (and the writer code) to include
        //   regular Exchange nodes. For now this is being conservative and a few changes
        //   would need to be made to deal with missing metrics, for example, for a regular
        //   Exchange node.
        baseHandle.dependency match {
          case gpuDep: GpuShuffleDependency[K, C, C]
              if gpuDep.useMultiThreadedShuffle && !gpuDep.checksumFallback =>
            // We want to use batch fetch in the non-push shuffle case. Spark
            // checks for a config to see if batch fetch is enabled (this check), and
            // it also checks when getting (potentially merged) map status from
            // the MapOutputTracker.
            val canUseBatchFetch =
              SortShuffleManager.canUseBatchFetch(startPartition, endPartition, context)

            val shuffleHandleWithMetrics = new ShuffleHandleWithMetrics(
              baseHandle.shuffleId, gpuDep.metrics, gpuDep)
            // in most scenarios, the pools have already started, except for local mode
            // here we try to start them if we see they haven't
            RapidsShuffleInternalManagerBase.startThreadPoolIfNeeded(
              rapidsConf.shuffleMultiThreadedWriterThreads,
              rapidsConf.shuffleMultiThreadedReaderThreads)
            new RapidsShuffleThreadedReader(
              startMapIndex,
              endMapIndex,
              startPartition,
              endPartition,
              shuffleHandleWithMetrics,
              context,
              metrics,
              rapidsConf.shuffleMultiThreadedMaxBytesInFlight,
              canUseBatchFetch = canUseBatchFetch,
              numReaderThreads = rapidsConf.shuffleMultiThreadedReaderThreads)
          case _ =>
            val shuffleHandle = RapidsShuffleInternalManagerBase.unwrapHandle(other)
            ShuffleManagerShims.getReader(wrapped, shuffleHandle, startMapIndex, endMapIndex,
              startPartition, endPartition, context, metrics)
        }
      case other =>
        val shuffleHandle = RapidsShuffleInternalManagerBase.unwrapHandle(other)
        ShuffleManagerShims.getReader(wrapped, shuffleHandle, startMapIndex, endMapIndex,
          startPartition, endPartition, context, metrics)
    }
  }

  /**
   * Registers a shuffle for a writer task.
   *
   * @param writerTaskId the writer's task attempt id, which a reclaim's watermark is compared to
   */
  def registerGpuShuffle(shuffleId: Int, writerTaskId: Long): Unit = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      // Note that in local mode this can be called multiple times.
      logInfo(s"Registering shuffle $shuffleId")
      catalog.registerShuffle(shuffleId, writerTaskId)
    }
    // Also register with MultithreadedShuffleBufferCatalog if available
    GpuShuffleEnv.getMultithreadedCatalog.foreach { mtCatalog =>
      logInfo(s"Registering shuffle $shuffleId with multithreaded catalog")
      mtCatalog.registerShuffle(shuffleId)
    }
  }

  /**
   * Frees what the shuffle buffer catalog holds of a shuffle for writers at or below
   * `watermark`, and frees no further buffer once `isStopped` reads true. A catalog that is not
   * initialized yet holds nothing.
   */
  def reclaimGpuShuffle(
      shuffleId: Int,
      watermark: Long,
      isStopped: () => Boolean): ShuffleReclaimResult = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      val result = catalog.reclaimShuffle(shuffleId, watermark, isStopped)
      def line = s"Reclaimed shuffle $shuffleId up to task $watermark from shuffle buffer " +
        s"catalog: ${result.outcome}, ${result.buffers} buffers, ${result.bytes} bytes in " +
        s"${TimeUnit.NANOSECONDS.toMillis(result.elapsedNanos)} ms"
      // every executor is asked to reclaim every shuffle, so most hold none of it
      if (result.outcome == ShuffleReclaimOutcome.ABSENT) logDebug(line) else logInfo(line)
      result
    } else {
      ShuffleReclaimResult.absent(shuffleId, watermark)
    }
  }

  def unregisterGpuShuffle(shuffleId: Int): Unit = {
    GpuShuffleEnv.invalidateBroadcastAndThen(shuffleId) {
      val catalog = GpuShuffleEnv.getCatalog
      if (catalog != null) {
        logInfo(s"Unregistering shuffle $shuffleId from shuffle buffer catalog")
        catalog.unregisterShuffle(shuffleId)
      }
    }
    // For MultithreadedShuffleBufferCatalog:
    // Cleanup is triggered by ShuffleCleanupListener on job end, not here.
    // The ShuffleCleanupEndpoint polls the driver for shuffles to clean and calls
    // mtCatalog.unregisterShuffle on executors.
    //
    // Note: This method is called via GC-triggered ContextCleaner.doCleanupShuffle().
    // We do not register for cleanup here because:
    // 1. GC timing is unpredictable and often happens too late (at app shutdown)
    // 2. By that time, executors may already be shutting down
    // 3. ShuffleCleanupListener triggers cleanup proactively on job end
  }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    // Kept out of the catalog path that a reclaim shares, which in local mode runs in this JVM:
    // a reclaimed shuffle stays a candidate until Spark itself removes it.
    GpuShuffleRegistry.remove(shuffleId)
    RapidsShuffleFileCache.removeShuffle(shuffleId)
    var unregistered = false
    Utils.tryWithSafeFinally {
      unregisterGpuShuffle(shuffleId)
    } {
      // We need to remove old shuffle blocks when Spark GC's a shuffle Id upstream.
      // In order to do so, we need to find the IndexShuffleBlockResolver in use.
      // We have two scenarios:
      // 1) We could be running in some compatibility mode where IndexShuffleBlockResolver
      //    (which comes from Spark) is the resolver we are using.
      // 2) We are using our own GpuShuffleBlockResolver, which can keep data in its own
      //    internal catalog, and it will also use the block manager to write map output
      //    to disk.
      val isbr = shuffleBlockResolver match {
        case isbr: IndexShuffleBlockResolver => isbr
        case gpur: GpuShuffleBlockResolverBase => gpur.wrapped
        case _ =>
          throw new IllegalStateException(
            "unregisterShuffle called with unexpected resolver " +
              s"$shuffleBlockResolver and blocks left to be cleaned")
      }
      Option(taskIdMapsForShuffle.remove(shuffleId)).foreach { mapTaskIds =>
        mapTaskIds.synchronized {
          mapTaskIds.iterator.foreach { mapTaskId =>
            isbr.removeDataByMap(shuffleId, mapTaskId)
          }
        }
      }
      unregistered = wrapped.unregisterShuffle(shuffleId)
    }
    unregistered
  }

  def shuffleBlockResolver: ShuffleBlockResolver = resolver

  override def stop(): Unit = synchronized {
    Utils.tryWithSafeFinally {
      wrapped.stop()
    } {
      if (!stopped) {
        stopped = true
        GpuShuffleEnv.closeBroadcastCacheAndThen {
          Utils.tryWithSafeFinally {
            RapidsShuffleTransport.closeServerThenTransport(server, transport)
          } {
            if (rapidsConf.isMultiThreadedShuffleManagerMode) {
              RapidsShuffleInternalManagerBase.stopThreadPool()
            }
          }
        }
      }
    }
  }
}
