# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import matplotlib.pyplot as plt
import numpy as np


# ==============
#  Llama2 8da4w
# ==============

llama2_8da4w_result = {
    'No quantization': [55.517,74.437,74.116,70.623,43.003,43.686,33.800,43.200,77.203,77.856],
    'QAT (quantized)': [54.292,73.442,72.854,68.771,41.041,42.662,31.800,42.400,76.605,77.203],
    'PTQ': [53.157,71.808,71.970,66.919,42.065,43.003,31.200,41.200,76.007,76.986],
}

llama2_8da4w_wikitext_result = {
    'No quantization': [9.599,1.526,0.610],
    'QAT (quantized)': [10.324,1.547,0.630],
    'PTQ': [11.347,1.575,0.655],
}

# ==============
#  Llama3 8da4w
# ==============

llama3_8da4w_result = {
    'No quantization': [57.857,76.598,80.976,78.746,48.720,53.669,32.800,42.000,79.053,79.489],
    'QAT (quantized)': [57.250,76.389,79.040,76.936,46.928,49.915,32.600,43.200,78.509,78.836],
    'PTQ': [51.743,70.663,75.295,71.044,42.406,45.990,30.000,40.600,76.605,76.442],
}

llama3_8da4w_wikitext_result = {
    'No quantization': [8.905,1.505,0.590],
    'QAT (quantized)': [9.852,1.534,0.617],
    'PTQ': [11.878,1.588,0.668],
}

# ===========
#  Llama3 3w
# ===========

llama3_3w_result = {
    'No quantization': [57.857,76.598,80.976,78.746,48.720,53.669,32.800,42.000,79.053,79.489],
    'QAT (quantized)': [52.450,70.026,73.527,72.138,41.638,45.392,27.800,39.000,76.659,77.149],
    'PTQ': [43.199,59.709,57.029,51.894,32.167,33.191,22.600,33.800,69.750,69.967],
    'QAT (quantized + skip)': [53.884,72.326,75.337,73.274,42.662,47.099,29.600,42.000,77.258,77.693],
    'PTQ (skip)': [47.670,64.897,68.350,65.152,37.201,42.321,25.200,36.800,75.245,74.755],
}

llama3_3w_wikitext_result = {
    'No quantization': [8.905,1.505,0.590],
    'QAT (quantized)': [12.812,1.611,0.688],
    'PTQ': [32.015,1.912,0.935],
    'QAT (quantized + skip)': [11.587,1.581,0.661],
    'PTQ (skip)': [18.576,1.727,0.788],
}

# ===========
#  Llama3 2w
# ===========

llama3_2w_result = {
    'No quantization': [57.857,76.598,80.976,78.746,48.720,53.669,32.800,42.000,79.053,79.489],
    'QAT (quantized)': [25.822,25.553,26.221,25.463,21.075,26.109,13.200,27.200,52.448,50.871],
    'PTQ': [25.632,26.548,24.874,25.884,21.331,25.341,15.400,29.600,54.135,52.067],
    'QAT (quantized + skip)': [41.297,54.461,60.354,54.377,27.474,31.741,20.800,33.800,71.164,71.328],
    'PTQ (skip)': [26.529,28.869,26.599,26.431,22.440,26.365,15.000,27.000,54.244,52.992],
}
llama3_2w_wikitext_result = {
    'No quantization': [8.905,1.505,0.590],
    'QAT (quantized)': [12419.731,5.829,2.543],
    'PTQ': [603335.882,12.050,3.591],
    'QAT (quantized + skip)': [29.936,1.888,0.917],
    'PTQ (skip)': [6765.611,5.203,2.379],
}


def plot_8da4w(result, output_path):
    tasks = (
        "hellaswag\n(acc)",
        "hellaswag\n(acc_norm)",
        "arc_easy\n(acc)",
        "arc_easy\n(acc_norm)",
        "arc_challenge\n(acc)",
        "arc_challenge\n(acc_norm)",
        "openbookqa\n(acc)",
        "openbookqa\n(acc_norm)",
        "piqa\n(acc)",
        "piqa\n(acc_norm)",
    )
    x = np.arange(len(tasks))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots()
    for quant_type, data in result.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, data, width, label=quant_type)
        ax.bar_label(rects, fmt="%.1f", padding=3)
        multiplier += 1
    ax.set_ylabel("Accuracy (%)", fontsize=24)
    ax.set_xticks(x + width, tasks, fontsize=16)
    ax.legend(loc="upper left", ncols=3, fontsize=20)
    ax.set_xlim(-0.4, 9.9)
    ax.set_ylim(0, 110)
    fig.set_size_inches(20, 4)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_8da4w_hellaswag(result, output_path):
    tasks = ("hellaswag\n(acc)", "hellaswag\n(acc_norm)")
    new_result = {}
    for k, v in result.items():
        new_result[k] = v[:2]
    result = new_result
    x = np.arange(len(tasks))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots()
    for quant_type, data in result.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, data, width, label=quant_type)
        ax.bar_label(rects, fmt="%.1f", padding=3)
        multiplier += 1
    ax.set_ylabel('Accuracy (%)', fontsize=24)
    ax.set_xticks(x + width, tasks, fontsize=16)
    ax.legend(loc='upper left', ncols=2, columnspacing=0.75, fontsize=15)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 119)
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_8da4w_wikitext(result, output_path):
    tasks = ("word", "byte", "bits_per_byte")
    x = np.arange(len(tasks))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots()
    for quant_type, data in result.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, data, width, label=quant_type)
        ax.bar_label(rects, fmt="%.3f", padding=3, rotation=60)
        multiplier += 1
    max_value = max(itertools.chain(*result.values()))
    ax.set_ylabel("Perplexity", fontsize=24)
    ax.set_xticks(x + width, tasks, fontsize=16)
    ax.legend(loc="upper right", ncols=1, fontsize=16)
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(0, max_value * 1.3)
    fig.set_size_inches(6, 4)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_2w_3w(result, output_path):
    tasks = (
        "hellaswag\n(acc)",
        "hellaswag\n(acc_norm)",
        "arc_easy\n(acc)",
        "arc_easy\n(acc_norm)",
        "arc_challenge\n(acc)",
        "arc_challenge\n(acc_norm)",
        "openbookqa\n(acc)",
        "openbookqa\n(acc_norm)",
        "piqa\n(acc)",
        "piqa\n(acc_norm)",
    )
    x = np.arange(len(tasks))
    width = 0.15
    multiplier = 0
    fig, ax = plt.subplots()
    for quant_type, data in result.items():
        offset = width * multiplier
        hatch = "//" if "QAT" in quant_type else None
        rects = ax.bar(
            x + offset,
            data,
            width,
            label=quant_type,
            hatch=hatch,
            edgecolor="white",
            linewidth=0,
        )
        ax.bar_label(rects, fmt="%.1f", padding=3, rotation=60)
        multiplier += 1
    ax.set_ylabel("Accuracy (%)", fontsize=24)
    ax.set_xticks(x + width, tasks, fontsize=16)
    ax.legend(loc="upper left", ncols=5, fontsize=16)
    ax.set_xlim(-0.35, 9.9)
    ax.set_ylim(0, 115)
    fig.set_size_inches(20, 4)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_2w_3w_wikitext(result, output_path):
    tasks = ("word", "byte", "bits_per_byte")
    x = np.arange(len(tasks))
    width = 0.15
    multiplier = 0
    fig, ax = plt.subplots()
    for quant_type, data in result.items():
        offset = width * multiplier
        hatch = "//" if "QAT" in quant_type else None
        rects = ax.bar(
            x + offset,
            data,
            width,
            label=quant_type,
            hatch=hatch,
            edgecolor="white",
            linewidth=0,
        )
        ax.bar_label(rects, fmt="%.3f", padding=3, rotation=60)
        multiplier += 1
    max_value = max(itertools.chain(*result.values()))
    ax.set_ylabel("Perplexity", fontsize=24)
    if max_value > 10000:
        ax.set_yscale("log")
    ax.set_xticks(x + width, tasks, fontsize=16)
    ax.legend(loc="upper right", ncols=1, fontsize=14, columnspacing=0.8)
    ax.set_xlim(-0.3, 2.85)
    ax.set_ylim(0, max_value * 1.3)
    fig.set_size_inches(6, 4)
    fig.tight_layout()
    fig.savefig(output_path)


# Plot things!

output_dir = "/tmp/my_plots/"

plot_8da4w(llama2_8da4w_result, output_dir + "llama2_8da4w.png")
plot_8da4w_hellaswag(llama2_8da4w_result, output_dir + "llama2_8da4w_hellaswag.png")
plot_8da4w_wikitext(
    llama2_8da4w_wikitext_result, output_dir + "llama2_8da4w_wikitext.png"
)

plot_8da4w(llama3_8da4w_result, output_dir + "llama3_8da4w.png")
plot_8da4w_hellaswag(llama3_8da4w_result, output_dir + "llama3_8da4w_hellaswag.png")
plot_8da4w_wikitext(
    llama3_8da4w_wikitext_result, output_dir + "llama3_8da4w_wikitext.png"
)

plot_2w_3w(llama3_3w_result, output_dir + "llama3_3w.png")
plot_2w_3w_wikitext(llama3_3w_wikitext_result, output_dir + "llama3_3w_wikitext.png")

plot_2w_3w(llama3_2w_result, output_dir + "llama3_2w.png")
plot_2w_3w_wikitext(llama3_2w_wikitext_result, output_dir + "llama3_2w_wikitext.png")
