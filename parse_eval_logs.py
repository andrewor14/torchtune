# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Eval log example:

# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:222] Eval completed in 1869.87 seconds.
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] piqa: {'acc,none': 0.7709466811751904, 'acc_stderr,none': 0.009804509865175504, 'acc_norm,none': 0.7769314472252449, 'acc_norm_stderr,none': 0.009713057213018543, 'alias': 'piqa'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] openbookqa: {'acc,none': 0.328, 'acc_stderr,none': 0.021017027165175485, 'acc_norm,none': 0.434, 'acc_norm_stderr,none': 0.022187215803029008, 'alias': 'openbookqa'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] winogrande: {'acc,none': 0.681136543014996, 'acc_stderr,none': 0.013097928420088771, 'alias': 'winogrande'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] arc_easy: {'acc,none': 0.7617845117845118, 'acc_stderr,none': 0.008741163824469184, 'acc_norm,none': 0.7196969696969697, 'acc_norm_stderr,none': 0.009216306864088029, 'alias': 'arc_easy'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] arc_challenge: {'acc,none': 0.4684300341296928, 'acc_stderr,none': 0.014582236460866978, 'acc_norm,none': 0.4735494880546075, 'acc_norm_stderr,none': 0.014590931358120172, 'alias': 'arc_challenge'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] anli_r3: {'acc,none': 0.3675, 'acc_stderr,none': 0.013923529685359278, 'alias': 'anli_r3'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] anli_r2: {'acc,none': 0.394, 'acc_stderr,none': 0.01545972195749338, 'alias': 'anli_r2'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] anli_r1: {'acc,none': 0.391, 'acc_stderr,none': 0.015438826294681797, 'alias': 'anli_r1'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] wikitext: {'word_perplexity,none': 11.769043842838464, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.585747199705589, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6651627944965424, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
# 2024-05-12:08:13:17,367 INFO     [eleuther_eval.py:224] hellaswag: {'acc,none': 0.581557458673571, 'acc_stderr,none': 0.004922953651577674, 'acc_norm,none': 0.7539334793865764, 'acc_norm_stderr,none': 0.0042983749363655985, 'alias': 'hellaswag'}

import json
import os
import re
import sys

args = sys.argv
if len(args) != 2:
    print("Usage: python parse_eval_logs.py [/path/to/eval.log]")
    sys.exit(1)

print("Parsing log ", args[1])
with open(args[1], "r") as f:
    lines = f.readlines()
    start_parsing = False
    result = {}
    for line in lines:
        if "Eval completed in" in line:
            start_parsing = True
            continue
        if not start_parsing:
            continue
        line = re.sub(".*\[eleuther_eval.py.*\] ", "", line)
        eval_dict = re.match(".*(\{.*\})", line).groups()[0]
        eval_dict = json.loads(eval_dict.replace("'", '"'))
        task_name = eval_dict["alias"]
        result[task_name] = {}
        for k, v in eval_dict.items():
            if "stderr" not in k and k != "alias":
                k = k.split(",")[0]
                if "acc" in k:
                    v *= 100
                    v = "%.3f" % v  # + "%"
                else:
                    v = "%.3f" % v
                result[task_name][k] = v

# Print results as csv
line1, line2, line3 = [], [], []
tasks = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
]
if os.getenv("WIKITEXT", "") == "true":
    tasks = ["wikitext"]
# tasks = [
#     "hellaswag",
#     "wikitext",
#     "anli_r1",
#     "anli_r2",
#     "anli_r3",
#     "arc_easy",
#     "arc_challenge",
#     "piqa",
#     "openbookqa",
# ]
for task in tasks:
    if task not in result:
        continue
    for i, (k, v) in enumerate(result[task].items()):
        line1.append(task if i == 0 else "")
        line2.append(k)
        line3.append(v)
print()
print(",".join(line1))
print(",".join(line2))
print(",".join(line3))
