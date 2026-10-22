/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.fileio.hadoop;

import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.fileio.RapidsInputFile;
import com.nvidia.spark.rapids.jni.fileio.SeekableInputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.lang.ref.SoftReference;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Implementation of {@link RapidsInputFile} using the Hadoop file system.
 * <br/>
 * This class provides methods to get the length of the file and to open a seekable input stream
 * for reading the file.
 */
public class HadoopInputFile implements RapidsInputFile {
    private static final String PARQUET_READ_ALLOCATION_SIZE = "parquet.read.allocation.size";
    // Reader threads outlive individual files. Keep at most one default-sized scratch array per
    // thread, instead of allocating it for every file/range read. Larger configured buffers are
    // still supported but are not retained. Soft references let heap pressure reclaim idle scratch.
    private static final ThreadLocal<SoftReference<byte[]>> COPY_BUFFER = new ThreadLocal<>();

    private final Path filePath;
    private final FileSystem fs;
    private final int copyBufferSize;

    public static HadoopInputFile create(Path filePath, Configuration conf) throws IOException {
        Objects.requireNonNull(filePath, "filePath can't be null!");
        Objects.requireNonNull(conf, "Hadoop conf can't be null");
        FileSystem fs = filePath.getFileSystem(conf);
        int copyBufferSize = conf.getInt(PARQUET_READ_ALLOCATION_SIZE,
                RapidsInputFile.DEFAULT_READ_VECTORED_COPY_BUFFER_SIZE);
        return new HadoopInputFile(filePath, fs, copyBufferSize);
    }

    private HadoopInputFile(Path filePath, FileSystem fs, int copyBufferSize) {
        Objects.requireNonNull(filePath, "filePath can't be null!");
        Objects.requireNonNull(fs, "FileSystem can't be null");
        if (copyBufferSize <= 0) {
            throw new IllegalArgumentException(PARQUET_READ_ALLOCATION_SIZE + " must be positive");
        }
        this.filePath = filePath;
        this.fs = fs;
        this.copyBufferSize = copyBufferSize;
    }

    @Override
    public String path() {
        return filePath.toString();
    }

    @Override
    public long getLength() throws IOException {
        return fs.getFileStatus(this.filePath).getLen();
    }

    @Override
    public OptionalLong getLastModificationTime() throws IOException {
        return OptionalLong.of(fs.getFileStatus(this.filePath).getModificationTime());
    }

    @Override
    public SeekableInputStream open() throws IOException {
        return new HadoopInputStream(fs.open(filePath));
    }

    @Override
    public void readVectored(HostMemoryBuffer output, List<RapidsInputFile.CopyRange> copyRanges)
            throws IOException {
        if (copyRanges.isEmpty()) {
            return;
        }
        long maxRangeLength = 0;
        for (RapidsInputFile.CopyRange range : copyRanges) {
            maxRangeLength = Math.max(maxRangeLength, range.getLength());
        }
        int requiredSize = (int) Math.min(copyBufferSize, maxRangeLength);
        SoftReference<byte[]> cached = COPY_BUFFER.get();
        byte[] copyBuffer = cached == null ? null : cached.get();
        // A filesystem may reenter this method on the same thread. Loan the array exclusively
        // until the complete read/copy has finished, including exceptional exits.
        COPY_BUFFER.set(null);
        if (copyBuffer == null || copyBuffer.length < requiredSize ||
                copyBuffer.length > copyBufferSize) {
            copyBuffer = new byte[requiredSize];
            cached = requiredSize <= RapidsInputFile.DEFAULT_READ_VECTORED_COPY_BUFFER_SIZE ?
                    new SoftReference<>(copyBuffer) : null;
        }
        try {
            RapidsInputFile.readVectoredUsingCopyBuffer(this, output, copyRanges, copyBuffer);
        } finally {
            if (copyBuffer.length <= RapidsInputFile.DEFAULT_READ_VECTORED_COPY_BUFFER_SIZE) {
                SoftReference<byte[]> availableRef = COPY_BUFFER.get();
                byte[] available = availableRef == null ? null : availableRef.get();
                if (available == null || available.length < copyBuffer.length) {
                    COPY_BUFFER.set(cached);
                }
            }
        }
    }
}
