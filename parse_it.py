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
    #"hellaswag_acc",
    "wikitext_word_perplexity",
]

log_dir = os.getenv("LOG_DIR", "/home/andrewor/local/logs/tune")

experiment_names = [
#    "Llama3-8B_oasst1_full",
#    "Llama3-8B_oasst1_qat",
#    "Llama3.1-8B_oasst1_full",
#    "Llama3.1-8B_oasst1_qat",
#    "Llama3.2-3B_oasst1_full",
#    "Llama3.2-3B_oasst1_qat",
    "Llama3.2-3B_alpaca_full",
    "Llama3.2-3B_alpaca_qat",
]

def extract_hellaswag_acc(log_file: str) -> float:
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|acc     \|↑  \|([\\d.]*)\|.*", l)
            if m is not None:
                return float(m.groups()[0])
    raise ValueError("Did not find hellaswag accuracy in %s" % log_file)

def extract_wikitext_perplexity(log_file: str) -> float:
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|word_perplexity\|↓  \|([\\d.]*)\|.*", l)
            if m is not None:
                return float(m.groups()[0])
    raise ValueError("Did not find wikitext perplexity in %s" % log_file)

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
    if "hellaswag_acc" in EVAL_FIELDS:
        float_hellaswag_eval_file = os.path.join(experiment_dir, "eval_hellaswag_float.log")
        quantized_hellaswag_eval_file = os.path.join(experiment_dir, "eval_hellaswag_quantized.log")
        all_data[experiment_name]["hellaswag_acc_float"] = extract_hellaswag_acc(float_hellaswag_eval_file)
        all_data[experiment_name]["hellaswag_acc_quantized"] = extract_hellaswag_acc(quantized_hellaswag_eval_file)
    if "wikitext_word_perplexity" in EVAL_FIELDS:
        float_wikitext_eval_file = os.path.join(experiment_dir, "eval_wikitext_float.log")
        quantized_wikitext_eval_file = os.path.join(experiment_dir, "eval_wikitext_quantized.log")
        all_data[experiment_name]["wikitext_word_perplexity_float"] = extract_wikitext_perplexity(float_wikitext_eval_file)
        all_data[experiment_name]["wikitext_word_perplexity_quantized"] = extract_wikitext_perplexity(quantized_wikitext_eval_file)


# print data in a nice format
short_metrics_fields = ["tok/s", "peak_mem_active", "peak_mem_alloc", "peak_mem_reserved"]
table1_headers = ["experiment_name"] + short_metrics_fields
table2_headers = ["experiment_name"] + EVAL_FIELDS
table1_data = []
table2_data = []
for experiment_name, data in all_data.items():
    baseline_name = experiment_name.replace("_qat", "_full")
    short_exp_name = experiment_name.replace("_oasst1", "")
    my_data_for_table1 = [short_exp_name]
    my_data_for_table2 = [short_exp_name]
    for metric in METRICS_FIELDS + EVAL_FIELDS:
        if metric in METRICS_FIELDS:
            my_value = data[metric]
            baseline_value = all_data[baseline_name][metric]
            percent_increase = (my_value - baseline_value) / baseline_value * 100
            sign = "+" if percent_increase >= 0 else ""
            print_value = "%.3f (%s%.3f%%)" % (my_value, sign, percent_increase)
            my_data_for_table1.append(print_value)
        else:
            baseline_float_value = all_data[baseline_name][metric + "_float"]
            baseline_quantized_value = all_data[baseline_name][metric + "_quantized"]
            if experiment_name.endswith("_qat"):
                qat_quantized_value = all_data[experiment_name][metric + "_quantized"]
                recovered = (qat_quantized_value - baseline_quantized_value) / (baseline_float_value - baseline_quantized_value) * 100
                print_value = "%.3f quant, recovered %.3f%%" % (qat_quantized_value, recovered)
            else:
                print_value = "%.3f quant, %.3f float" % (baseline_quantized_value, baseline_float_value)
            my_data_for_table2.append(print_value)
    table1_data.append(my_data_for_table1)
    table2_data.append(my_data_for_table2)
print(tabulate.tabulate(table1_data, headers=table1_headers))
print()
print(tabulate.tabulate(table2_data, headers=table2_headers))
