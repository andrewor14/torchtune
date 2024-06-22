# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import statistics
import sys

args = sys.argv
if len(args) != 2:
    print("Usage: python parse_train_data.py [/path/to/log12345.txt]")
    sys.exit(1)

tokens_per_second = []
peak_memory = []
with open(args[1], "r") as f:
    for line in f.readlines():
        split = line.rstrip().split(" ")
        tokens_per_second.append(float(split[5].split(":")[1]))
        peak_memory.append(float(split[-1].split(":")[1]))

print("Max tok/s: ", max(tokens_per_second))
print("Average tok/s: ", sum(tokens_per_second) / len(tokens_per_second))
print("Median tok/s: ", statistics.median(tokens_per_second))
print()
print("Max peak memory: ", max(peak_memory))
print("Average peak memory: ", sum(peak_memory) / len(peak_memory))
print("Median peak memory: ", statistics.median(peak_memory))
