import glob
import os
import re
import tabulate

DISCARD_FIRST_N_STEPS = 10
METRICS_FIELDS = [
    "tokens_per_second_per_gpu",
    "peak_memory_active",
    "peak_memory_alloc",
    "peak_memory_reserved",
]
EVAL_FIELDS = [
    "hellaswag_acc",
    "wikitext_word_perplexity",
]

log_dir_base = os.getenv("LOG_DIR_BASE", "/home/andrewor/local/logs/tune")
log_dir_name = os.getenv("LOG_DIR_NAME", "")
log_dir = os.path.join(log_dir_base, log_dir_name)

baseline_name = "full_finetune_distributed_baseline"

experiment_names = [
    "full_finetune_distributed_baseline",
    "full_finetune_distributed_baseline_tp",
    "fp8_quantized_training_noname",
    "fp8_quantized_training_noname_tp",
    "fp8_quantized_training_tensorwise",
    "fp8_quantized_training_tensorwise_tp",
    "fp8_quantized_training_rowwise",
    "fp8_quantized_training_rowwise_with_gw_hp",
]

all_data = {}
for experiment_name in experiment_names:
    experiment_dir = os.path.join(log_dir, experiment_name)
    all_data[experiment_name] = {}

    # extract metrics data
    metrics_file = glob.glob(experiment_dir + "/metrics/log*txt")
    assert len(metrics_file) > 0, "did not find metrics file under %s" % experiment_dir
    metrics_file = metrics_file[0]
    with open(metrics_file, "r") as f:
        lines = f.readlines()[DISCARD_FIRST_N_STEPS:]
        for metric in METRICS_FIELDS:
            values = []
            for l in lines:
                pattern = ".*%s:([\\d.]*) .*" % metric
                values.append(float(re.match(pattern, l).groups()[0]))
            avg_value = sum(values) / len(values)
            all_data[experiment_name][metric] = avg_value

    # extract eval data
    hellaswag_eval_file = os.path.join(experiment_dir, "eval_hellaswag_float.log")
    wikitext_eval_file = os.path.join(experiment_dir, "eval_wikitext_float.log")
    with open(hellaswag_eval_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|acc     \|↑  \|([\\d.]*)\|.*", l)
            if m is not None:
                all_data[experiment_name]["hellaswag_acc"] = float(m.groups()[0])
                break
    with open(wikitext_eval_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|word_perplexity\|↓  \|([\\d.]*)\|.*", l)
            if m is not None:
                all_data[experiment_name]["wikitext_word_perplexity"] = float(m.groups()[0])
                break
    assert "hellaswag_acc" in all_data[experiment_name]
    assert "wikitext_word_perplexity" in all_data[experiment_name]


# print data in a nice format
short_metrics_fields = ["tok/s", "peak_mem_active", "peak_mem_alloc", "peak_mem_reserved"]
table1_headers = ["experiment_name"] + short_metrics_fields
table2_headers = ["experiment_name"] + EVAL_FIELDS
table1_data = []
table2_data = []
for experiment_name, data in all_data.items():
    short_exp_name = experiment_name.replace("fp8_quantized_training", "fp8")
    short_exp_name = short_exp_name.replace("full_finetune_distributed_baseline", "full")
    my_data_for_table1 = [short_exp_name]
    my_data_for_table2 = [short_exp_name]
    for metric in METRICS_FIELDS + EVAL_FIELDS:
        baseline_value = all_data[baseline_name][metric]
        my_value = data[metric]
        if metric in METRICS_FIELDS:
            percent_increase = (my_value - baseline_value) / baseline_value * 100
            sign = "+" if percent_increase >= 0 else ""
            print_value = "%.3f (%s%.3f%%)" % (my_value, sign, percent_increase)
            my_data_for_table1.append(print_value)
        else:
            difference = my_value - baseline_value
            sign = "+" if difference >= 0 else ""
            print_value = "%.3f (%s%.3f)" % (my_value, sign, difference)
            my_data_for_table2.append(print_value)
    table1_data.append(my_data_for_table1)
    table2_data.append(my_data_for_table2)
print(tabulate.tabulate(table1_data, headers=table1_headers))
print()
print(tabulate.tabulate(table2_data, headers=table2_headers))
