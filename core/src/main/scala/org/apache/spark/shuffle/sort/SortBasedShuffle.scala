/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.shuffle.sort

import java.io.File
import java.util.Comparator
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

import org.apache.spark.executor.ShuffleWriteMetrics

import scala.collection.mutable.ArrayBuffer

import org.apache.spark._
import org.apache.spark.util.TimeStampedHashMap
import org.apache.spark.util.collection.ExternalAppendOnlyMap
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.serializer.Serializer
import org.apache.spark.shuffle.{BaseShuffleHandle, ShuffleWriter}
import org.apache.spark.storage.{ShuffleBlockId, FileSegment}

class SortBasedShuffle {
  import SortBasedShuffle._

  // missing clean shuffle data.
  private val shuffleMetadataMap = new TimeStampedHashMap[Int, ShuffleMetadata]
  private val nextFileId = new AtomicInteger(0)

  def getBlockLocation(id: ShuffleBlockId): FileSegment = {
    val shuffleMetadata = shuffleMetadataMap(id.shuffleId)
    shuffleMetadata.getFileSegment(id.mapId, id.reduceId).getOrElse(
      throw new IllegalStateException(s"Failed to find shuffle block: $id"))
  }

  def removeShuffle(shuffleId: Int) {
    val cleaned = shuffleMetadataMap(shuffleId)
    cleaned.cleanup()
    shuffleMetadataMap.remove(shuffleId)
  }

  def createShuffleWriter[K, V](
      handle: BaseShuffleHandle[K, V, Any],
      mapId: Int,
      context: TaskContext) = new SortBasedShuffleWriter[K, V](handle, mapId, context)

  protected class SortBasedShuffleWriter[K, V](
      handle: BaseShuffleHandle[K, V, Any],
      mapId: Int,
      context: TaskContext)
    extends ShuffleWriter[K, V] with Logging {

    private val dep = handle.dependency
    private val numOutputSplits = dep.partitioner.numPartitions

    private val blockManager = SparkEnv.get.blockManager
    private val conf = blockManager.conf
    private val ser = Serializer.getSerializer(dep.serializer.getOrElse(null))
    private val bufSize = conf.getInt("spark.shuffle.file.buffer.kb", 100) * 1024

    private val partKeyComparator = new Comparator[(K, _)] {
      // TODO. Use hashcode comparison temporarily, consider keyOrdering
      val keyComparator = new ExternalAppendOnlyMap.KCComparator[K, Any]

      def compare(comp1: (K, _), comp2: (K, _)): Int = {
        val hash1 = dep.partitioner.getPartition(comp1._1)
        val hash2 = dep.partitioner.getPartition(comp2._1)
        if (hash1 != hash2) {
          hash1 - hash2
        } else {
          if (dep.aggregator.isDefined) {
            keyComparator.compare(comp1, comp2)
          } else {
            0
          }
        }
      }
    }

    private val aggregator = if (dep.aggregator.isDefined && dep.mapSideCombine) {
      // Copy to avoid contaminating reduce-side aggregator
      val aggre = dep.aggregator.get
      Aggregator[K, V, Any](aggre.createCombiner, aggre.mergeValue, aggre.mergeCombiners)
        .asInstanceOf[Aggregator[K, Any, Any]]
    } else {
      Aggregator[K, Product2[K, V], ArrayBuffer[Product2[K, V]]](
        v => ArrayBuffer(v),
        (c, v) => c += v,
        (c1, c2) => c1 ++= c2)
        .asInstanceOf[Aggregator[K, Any, Any]]
    }
    aggregator.setComparator(partKeyComparator)

    shuffleMetadataMap.putIfAbsent(dep.shuffleId, ShuffleMetadata(dep.shuffleId, numOutputSplits))

    private val shuffleMetadata = shuffleMetadataMap(dep.shuffleId)
    private var mapStatus: Option[MapStatus] = None

    override def write(records: Iterator[_ <: Product2[K, V]]): Unit = {
      val iter = if (dep.mapSideCombine) {
        aggregator.combineValuesByKey(records, context)
      } else {
        // Additional step for iterator here to keep not breaking the original type of Product2
        aggregator.combineValuesByKey(records.map(r => (r._1, r)), context)
      }

      val shuffleBlockId = ShuffleBlockId(dep.shuffleId, mapId, nextFileId.getAndIncrement())
      val shuffleFile = blockManager.diskBlockManager.getFile(shuffleBlockId)
      var writer = blockManager.getDiskWriter(shuffleBlockId, shuffleFile, ser, bufSize)

      val offsets = new Array[Long](numOutputSplits)
      val lengths = new Array[Long](numOutputSplits)
      var previousBucketId: Int = 0
      var totalBytes = 0L
      var totalTime = 0L

      try {
        for (it <- iter) {
          val bucketId = dep.partitioner.getPartition(it._1)
          if (previousBucketId != bucketId) {
            writer.commit()
            writer.close()
            val fileSegment = writer.fileSegment()
            offsets(previousBucketId) = fileSegment.offset
            lengths(previousBucketId) = fileSegment.length
            totalBytes += fileSegment.length
            totalTime += writer.timeWriting()
            previousBucketId = bucketId

            // Reopen the file for another partition, must recreate, otherwise will cause issue.
            writer = blockManager.getDiskWriter(shuffleBlockId, shuffleFile, ser, bufSize)
          }

          if (dep.mapSideCombine) {
            writer.write(it)
          } else {
            it._2.asInstanceOf[ArrayBuffer[_]].foreach(r => writer.write(r))
          }
        }

        writer.commit()
      } catch {
        case e: Exception =>
          writer.revertPartialWrites()
          throw e
      } finally {
        writer.close()
      }

      val fileSegment = writer.fileSegment()
      offsets(previousBucketId) = fileSegment.offset
      lengths(previousBucketId) = fileSegment.length
      totalBytes += fileSegment.length
      totalTime += writer.timeWriting()

      val shuffleMetrics = new ShuffleWriteMetrics
      shuffleMetrics.shuffleBytesWritten = totalBytes
      shuffleMetrics.shuffleWriteTime = totalTime
      context.taskMetrics.shuffleWriteMetrics = Some(shuffleMetrics)

      // Fill some offsets of zero size partitions
      var i = 1
      while (i < offsets.length) {
        if (offsets(i) == 0) {
          offsets(i) = offsets(i - 1) + lengths(i - 1)
        }
        i += 1
      }

      shuffleMetadata.recordMapOutput(mapId, shuffleFile, offsets)
      mapStatus = Some(new MapStatus(blockManager.blockManagerId,
        lengths.map(MapOutputTracker.compressSize(_))))
    }

    override def stop(success: Boolean): Option[MapStatus] = {
      mapStatus
    }
  }

}

object SortBasedShuffle {
  case class ShuffleMetadata(val shuffleId: Int, val numOutputSplits: Int) {
    // To keep each map output's file and offsets
    private val offsetsByReducer = new ConcurrentHashMap[Int, (File, Array[Long])]()

    def recordMapOutput(mapId: Int, shuffleFile: File, offsets: Array[Long]) {
      offsetsByReducer.put(mapId, (shuffleFile, offsets))
    }

    def getFileSegment(mapId: Int, reduceId: Int): Option[FileSegment] = {
      val (file, offsets) = offsetsByReducer.get(mapId)
      if (offsets != null) {
        val offset = offsets(reduceId)
        val length = if (reduceId + 1 < numOutputSplits) {
          offsets(reduceId + 1) - offset
        } else {
          file.length() - offset
        }
        assert(length >= 0)

        Some(new FileSegment(file, offset, length))
      } else {
        None
      }
    }

    def cleanup() {
      import scala.collection.JavaConversions._
      offsetsByReducer.foreach { case (_, (f, _)) => f.delete() }
    }
  }
}
