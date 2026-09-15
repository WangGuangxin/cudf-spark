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

/*** spark-rapids-shim-json-lines
{"spark": "351"}
spark-rapids-shim-json-lines ***/

package org.apache.spark.sql.rapids

import java.lang.reflect.Method
import java.util.{List => JList, Map => JMap, Optional => JOption}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.parquet.{GpuParquetPartitionReaderFactory, GpuParquetScan}
import com.nvidia.spark.rapids.shims.PartitionedFileUtilsShim

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.connector.expressions.{NamedReference, SortOrder}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.metric.{CustomMetric, CustomTaskMetric}
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.connector.read.partitioning.Partitioning
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.SerializableConfiguration

class PaimonProviderImpl extends PaimonProvider {
  override def getScans: Map[Class[_ <: Scan], ScanRule[_ <: Scan]] = {
    val scanClass = ShimReflectionUtils.loadClass(PaimonProvider.cpuScanClassName)
    val version = Option(scanClass.getPackage.getImplementationVersion)
    if (!version.contains(PaimonProvider.requiredVersion)) {
      Map.empty
    } else {
      val rule = new ScanRule[Scan](
        (scan, conf, parent, r) => new ScanMeta[Scan](scan, conf, parent, r) {
          private lazy val converted = Try(new GpuPaimonScan(scan, this.conf))

          override def supportsRuntimeFilters: Boolean = true

          override def tagSelfForGpu(): Unit = {
            GpuParquetScan.tagSupport(SparkSession.active, scan.readSchema(), this)
            converted match {
              case Success(gpuScan) => gpuScan.validationErrors.foreach(willNotWorkOnGpu)
              case Failure(e) =>
                willNotWorkOnGpu(s"error examining Paimon scan: ${e.getMessage}")
            }
          }

          override def convertToGpu(): GpuScan = converted.get
        },
        "Paimon 1.2.0 snapshot batch scan",
        ClassTag(scanClass))
      Map(scanClass.asSubclass(classOf[Scan]) -> rule)
    }
  }
}

private object PaimonReflection {
  private def zeroArgMethod(value: AnyRef, name: String): Method =
    value.getClass.getMethods.find(m => m.getName == name && m.getParameterCount == 0)
      .getOrElse(throw new NoSuchMethodException(s"${value.getClass.getName}.$name"))

  def invoke(value: AnyRef, name: String): AnyRef = zeroArgMethod(value, name).invoke(value)

  def bool(value: AnyRef, name: String): Boolean =
    invoke(value, name).asInstanceOf[java.lang.Boolean].booleanValue()

  def long(value: AnyRef, name: String): Long =
    invoke(value, name).asInstanceOf[java.lang.Long].longValue()

  def scalaSeq(value: AnyRef, name: String): Seq[AnyRef] =
    invoke(value, name).asInstanceOf[Seq[AnyRef]]

  def javaList(value: AnyRef, name: String): JList[AnyRef] =
    invoke(value, name).asInstanceOf[JList[AnyRef]]

  def optionalList(value: AnyRef, name: String): Option[JList[AnyRef]] = {
    val optional = invoke(value, name).asInstanceOf[JOption[JList[AnyRef]]]
    if (optional.isPresent) Some(optional.get()) else None
  }

  def stringMap(value: AnyRef, name: String): JMap[String, String] =
    invoke(value, name).asInstanceOf[JMap[String, String]]

  def partitionRow(split: AnyRef, table: AnyRef, requiredSchema: StructType): InternalRow = {
    if (requiredSchema.isEmpty) {
      InternalRow.empty
    } else {
      val tableSchema = invoke(table, "schema")
      val partitionType = invoke(tableSchema, "logicalPartitionType")
      val sparkTypeUtils = ShimReflectionUtils.loadClass(
        "org.apache.paimon.spark.SparkTypeUtils")
      val fromPaimon = sparkTypeUtils.getMethods.find(m =>
        m.getName == "fromPaimonRowType" && m.getParameterCount == 1).get
      val fullSchema = fromPaimon.invoke(null, partitionType).asInstanceOf[StructType]

      val converterClass = ShimReflectionUtils.loadClass(
        "org.apache.paimon.spark.data.SparkInternalRow")
      val create = converterClass.getMethods.find(m =>
        m.getName == "create" && m.getParameterCount == 1).get
      val converter = create.invoke(null, partitionType).asInstanceOf[AnyRef]
      val replace = converter.getClass.getMethods.find(m =>
        m.getName == "replace" && m.getParameterCount == 1).get
      val fullRow = replace.invoke(converter, invoke(split, "partition"))
        .asInstanceOf[InternalRow]

      new GenericInternalRow(requiredSchema.fields.map { field =>
        val index = fullSchema.fieldIndex(field.name)
        fullRow.get(index, fullSchema(index).dataType): Any
      })
    }
  }
}

private class GpuPaimonScan(
    val cpuScan: Scan,
    val rapidsConf: RapidsConf)
  extends GpuScan
  with SupportsReportStatistics
  with SupportsRuntimeV2Filtering
  with SupportsReportPartitioning
  with SupportsReportOrdering {

  private val filterableScan = cpuScan.asInstanceOf[SupportsRuntimeV2Filtering]
  private val partitioningScan = cpuScan.asInstanceOf[SupportsReportPartitioning]
  private val orderingScan = cpuScan.asInstanceOf[SupportsReportOrdering]
  private val statisticsScan = cpuScan.asInstanceOf[SupportsReportStatistics]

  private lazy val table = PaimonReflection.invoke(cpuScan, "table")
  private lazy val cpuBatch = cpuScan.toBatch
  private lazy val partitions = cpuBatch.planInputPartitions()
  private lazy val partitionKeys = PaimonReflection.javaList(table, "partitionKeys")
    .asScala.map(_.toString).toSet

  lazy val validationErrors: Seq[String] = {
    val errors = Seq.newBuilder[String]
    val fileStoreTableClass = ShimReflectionUtils.loadClass(
      "org.apache.paimon.table.FileStoreTable")
    if (!fileStoreTableClass.isInstance(table)) {
      errors += "Only Paimon file-store tables are supported on GPU"
    }
    if (!PaimonReflection.javaList(table, "primaryKeys").isEmpty) {
      errors += "Paimon primary-key tables require merge semantics and are not supported on GPU"
    }
    val options = PaimonReflection.stringMap(table, "options")
    if (Option(options.get("file.format")).exists(!_.equalsIgnoreCase("parquet"))) {
      errors += "Paimon GPU scans require file.format=parquet"
    }
    if (PaimonReflection.scalaSeq(cpuBatch, "metadataColumns").nonEmpty) {
      errors += "Paimon metadata and changelog columns are not supported on GPU"
    }
    partitions.foreach { partition =>
      PaimonReflection.scalaSeq(partition, "splits").foreach { split =>
        if (split.getClass.getName != "org.apache.paimon.table.source.DataSplit") {
          errors += s"Paimon split type ${split.getClass.getName} is not supported on GPU"
        } else {
          if (!PaimonReflection.javaList(split, "beforeFiles").isEmpty) {
            errors += "Paimon splits with beforeFiles are not supported on GPU"
          }
          if (PaimonReflection.optionalList(split, "beforeDeletionFiles").exists(!_.isEmpty) ||
              PaimonReflection.optionalList(split, "deletionFiles").exists(!_.isEmpty)) {
            errors += "Paimon deletion files and deletion vectors are not supported on GPU"
          }
          if (PaimonReflection.bool(split, "isStreaming")) {
            errors += "Paimon streaming and changelog splits are not supported on GPU"
          }
          if (!PaimonReflection.bool(split, "rawConvertible")) {
            errors += "Paimon split requires a merge reader and is not supported on GPU"
          } else {
            PaimonReflection.optionalList(split, "convertToRawFiles").toSeq
              .flatMap(_.asScala)
              .foreach { rawFile =>
                if (!PaimonReflection.invoke(rawFile, "format").toString
                    .equalsIgnoreCase("parquet")) {
                  errors += "Paimon GPU scans support only Parquet data files"
                }
              }
          }
        }
      }
    }
    errors.result().distinct
  }

  override def readSchema(): StructType = cpuScan.readSchema()
  override def estimateStatistics(): Statistics = statisticsScan.estimateStatistics()
  override def supportedCustomMetrics(): Array[CustomMetric] = cpuScan.supportedCustomMetrics()
  override def reportDriverMetrics(): Array[CustomTaskMetric] = cpuScan.reportDriverMetrics()
  override def description(): String = cpuScan.description()
  override def outputPartitioning(): Partitioning = partitioningScan.outputPartitioning()
  override def outputOrdering(): Array[SortOrder] = orderingScan.outputOrdering()
  override def filterAttributes(): Array[NamedReference] = filterableScan.filterAttributes()
  override def filter(predicates: Array[Predicate]): Unit = filterableScan.filter(predicates)

  override def toBatch: Batch = new GpuPaimonBatch(
    cpuScan.toBatch, table, readSchema(), partitionKeys, rapidsConf, () => metrics)

  override def withInputFile(): GpuScan = this
}

private class GpuPaimonBatch(
    cpuBatch: Batch,
    table: AnyRef,
    readSchema: StructType,
    partitionKeys: Set[String],
    rapidsConf: RapidsConf,
    metrics: () => Map[String, GpuMetric]) extends Batch {

  override def planInputPartitions(): Array[InputPartition] = cpuBatch.planInputPartitions()

  override def createReaderFactory(): PartitionReaderFactory = {
    val spark = SparkSession.active
    val hadoopConf = spark.sparkContext.broadcast(
      new SerializableConfiguration(spark.sparkContext.hadoopConfiguration))
    val dataSchema = StructType(readSchema.filterNot(f => partitionKeys.contains(f.name)))
    val partitionSchema = StructType(readSchema.filter(f => partitionKeys.contains(f.name)))
    val parquetFactory = GpuParquetPartitionReaderFactory(
      spark.sessionState.conf,
      hadoopConf,
      dataSchema,
      dataSchema,
      partitionSchema,
      Array.empty[Filter],
      rapidsConf,
      metrics(),
      Map.empty)
    new GpuPaimonPartitionReaderFactory(
      table, readSchema, dataSchema, partitionSchema, parquetFactory)
  }
}

private case class PaimonTaskMetric(nameValue: String, override val value: Long)
    extends CustomTaskMetric {
  override def name(): String = nameValue
}

private class GpuPaimonPartitionReaderFactory(
    table: AnyRef,
    readSchema: StructType,
    dataSchema: StructType,
    partitionSchema: StructType,
    parquetFactory: GpuParquetPartitionReaderFactory) extends PartitionReaderFactory {

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GPU Paimon reader does not support row reads")

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createColumnarReader(
      partition: InputPartition): PartitionReader[ColumnarBatch] = {
    new GpuPaimonPartitionReader(
      partition, table, readSchema, dataSchema, partitionSchema, parquetFactory)
  }
}

private class GpuPaimonPartitionReader(
    partition: InputPartition,
    table: AnyRef,
    readSchema: StructType,
    dataSchema: StructType,
    partitionSchema: StructType,
    parquetFactory: GpuParquetPartitionReaderFactory) extends PartitionReader[ColumnarBatch] {

  private val outputMapping = readSchema.fields.map { field =>
    dataSchema.fields.indexWhere(_.name == field.name) match {
      case -1 => dataSchema.length + partitionSchema.fields.indexWhere(_.name == field.name)
      case index => index
    }
  }

  private val splits = PaimonReflection.scalaSeq(partition, "splits")
  private val files: Iterator[PartitionedFile] = splits.iterator.flatMap { split =>
    val partitionValues = PaimonReflection.partitionRow(split, table, partitionSchema)
    PaimonReflection.optionalList(split, "convertToRawFiles").toSeq
      .flatMap(_.asScala)
      .iterator
      .map { rawFile =>
        PartitionedFileUtilsShim.newPartitionedFile(
          partitionValues,
          PaimonReflection.invoke(rawFile, "path").toString,
          PaimonReflection.long(rawFile, "offset"),
          PaimonReflection.long(rawFile, "length"))
      }
  }

  private var currentReader: PartitionReader[ColumnarBatch] = _

  private def openNextReader(): Boolean = {
    if (files.hasNext) {
      currentReader = parquetFactory.buildColumnarReader(files.next())
      true
    } else {
      currentReader = null
      false
    }
  }

  override def next(): Boolean = {
    if (currentReader == null && !openNextReader()) {
      false
    } else {
      var hasBatch = false
      var exhausted = false
      while (!hasBatch && !exhausted) {
        if (currentReader.next()) {
          hasBatch = true
        } else {
          currentReader.close()
          exhausted = !openNextReader()
        }
      }
      hasBatch
    }
  }

  override def get(): ColumnarBatch = {
    val input = currentReader.get()
    if (outputMapping.indices.forall(i => outputMapping(i) == i)) {
      input
    } else {
      withResource(input) { batch =>
        val columns = outputMapping.safeMap { index =>
          batch.column(index).asInstanceOf[GpuColumnVector].incRefCount()
        }
        new ColumnarBatch(columns.toArray, batch.numRows())
      }
    }
  }

  override def currentMetricsValues(): Array[CustomTaskMetric] = {
    val dataSplits = splits.filter(
      _.getClass.getName == "org.apache.paimon.table.source.DataSplit")
    val splitSize = dataSplits.flatMap(PaimonReflection.javaList(_, "dataFiles").asScala)
      .map(PaimonReflection.long(_, "fileSize")).sum
    val count = dataSplits.length
    val paimonMetrics = if (count == 0) {
      Array.empty[CustomTaskMetric]
    } else {
      Array(
        PaimonTaskMetric("numSplits", count),
        PaimonTaskMetric("splitSize", splitSize),
        PaimonTaskMetric("avgSplitSize", splitSize / count))
    }
    super.currentMetricsValues() ++ paimonMetrics
  }

  override def close(): Unit = {
    if (currentReader != null) {
      currentReader.close()
      currentReader = null
    }
  }
}
