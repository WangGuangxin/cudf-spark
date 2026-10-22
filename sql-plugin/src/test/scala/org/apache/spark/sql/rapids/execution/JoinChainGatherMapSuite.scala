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

package org.apache.spark.sql.rapids.execution

import ai.rapids.cudf.{ColumnVector, Cuda, DeviceMemoryBuffer, GatherMap}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import org.scalatest.funsuite.AnyFunSuite

class JoinChainGatherMapSuite extends AnyFunSuite {
  test("transferred indices retain their buffer after the source and gather map close") {
    val values = Array(4, 1, 3, 0)
    val column = withResource(ColumnVector.fromInts(values: _*)) { source =>
      withResource(DeviceMemoryBuffer.allocate(source.getRowCount * 4)) { buffer =>
        withResource(source.getData) { data =>
          buffer.copyFromDeviceBufferAsync(0, data, 0, data.getLength, Cuda.DEFAULT_STREAM)
        }
        buffer.incRefCount()
        withResource(new GatherMap(buffer)) { map =>
          closeOnExcept(GpuJoinChain.columnFromGatherMap(map)) { result =>
            withResource(result.getData) { data =>
              assert(data.getAddress == buffer.getAddress)
            }
            result
          }
        }
      }
    }
    withResource(column) { indices =>
      withResource(indices.copyToHost()) { host =>
        assert(host.getRowCount == values.length)
        values.indices.foreach(i => assert(host.getInt(i) == values(i)))
      }
    }
  }

  test("empty gather maps can transfer ownership") {
    withResource(new GatherMap(DeviceMemoryBuffer.allocate(0))) { map =>
      withResource(GpuJoinChain.columnFromGatherMap(map)) { column =>
        assert(column.getRowCount == 0)
      }
    }
  }
}
