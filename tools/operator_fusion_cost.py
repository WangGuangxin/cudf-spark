#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run resident-input cost matrices; requires compiled tests and their runtime classpath."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess


def cases(matrix):
    if matrix == "full":
        for survival, card, strings, batches in itertools.product(
                [0, 1, 10, 50, 100], [16, 131072], [False, True], [1, 8]):
            yield ("filter", survival, card, strings, batches, 32, "cpu", "filter:agg:pipeline")
        for card, strings, batches, parts, path in itertools.product(
                [16, 131072], [False, True], [1, 8], [1, 32, 256], ["cpu", "gpu", "packed"]):
            yield ("shuffle", 100, card, strings, batches, parts, path,
                   "agg:partition:pack:prepare:pipeline")
    elif matrix == "breakdown":
        for survival, (card, strings, batches) in itertools.product(
                [0, 1, 10, 50, 100], [(16, False, 1), (131072, True, 8)]):
            yield ("filter", survival, card, strings, batches, 32, "cpu",
                   "predicate:gather:filter:agg:pipeline")
    else:
        for survival, card, strings, batches in [
                (1, 16, False, 1), (50, 16, False, 1),
                (10, 131072, True, 8), (100, 131072, True, 1)]:
            yield ("filter", survival, card, strings, batches, 32, "cpu",
                   "predicate:gather:pipeline")
        for (card, strings), path in itertools.product(
                [(16, False), (131072, True)], ["cpu", "gpu", "packed"]):
            yield ("shuffle", 100, card, strings, 1, 32, path, "partition:pack:pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classpath-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--matrix", choices=["full", "breakdown", "profile"], default="full")
    parser.add_argument("--allocator", choices=["ASYNC", "NONE"], default="ASYNC")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.matrix + ("-reverse" if args.reverse else "-forward")
    selected = list(cases(args.matrix))
    if args.reverse:
        selected.reverse()
    case_file = args.output_dir / (name + ".csv")
    case_file.write_text("\n".join(",".join(str(v).lower() for v in c) for c in selected) + "\n")
    profile = args.matrix == "profile"
    command = [args.java, "-Xmx4g", "-Xss4m", "-Dspark.ui.enabled=false",
               "-Dcom.nvidia.spark.rapids.runningTests=true", "-Dai.rapids.refcount.debug=false"]
    # Spark 3.5 on JDK 17 needs access to these JDK internals even without a Spark plugin.
    for package in ["java.lang", "java.lang.invoke", "java.lang.reflect", "java.io", "java.net",
                    "java.nio", "java.util", "java.util.concurrent", "java.util.concurrent.atomic",
                    "sun.nio.ch", "sun.nio.cs", "sun.security.action", "sun.util.calendar"]:
        command.append("--add-opens=java.base/" + package + "=ALL-UNNAMED")
    command += ["-cp", args.classpath_file.read_text().strip(),
                "com.nvidia.spark.rapids.OperatorFusionCostBenchmark", str(case_file),
                str(args.iterations), str(args.output_dir / (name + "-rmm.csv")) if profile else "-",
                str(profile).lower(), args.allocator]
    if profile:
        captures = sum(len(c[-1].split(":")) for c in selected)
        command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=repeat:" + str(captures),
                   "--force-overwrite=true", "-o", str(args.output_dir / name)] + command
    (args.output_dir / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
    with (args.output_dir / (name + ".log")).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(f"exit={result.returncode} log={args.output_dir / (name + '.log')}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
