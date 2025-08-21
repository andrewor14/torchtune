# How to reproduce QAT numbers

## Setup

First, install local torchtune, make sure you have're on this branch:
https://github.com/andrewor14/torchtune/commits/nvfp4/

```
cd torchtune
pip install -e .
```

Then, install torchao nightly or build from source:
```
# nightly (select your own cuda version)
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128

# or build from source
git clone git@github.com:pytorch/ao
cd ao
USE_CUDA=1 USE_CPP=0 python setup.py develop
```

Download HF checkpoints:
```
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <YOUR TOKEN>
tune download meta-llama/Llama-3.2-3B-Instruct --output-dir /tmp/Llama-3.2-3B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <YOUR TOKEN>
```

My commits:
```
torchtune 97bd210b3512aff0f048be81f1f00f539707a5b3 (aug 12)
torchao 5c0d6a3fa6b86f7711165fb1c4a4bdf6bd771944 (aug 18)
torch 1f1900369435933a013df9a7d5e07c75c1cebb5d (aug 18)
```

## Run the experiments

The way the scripts are structured is:
- `run_it.sh` runs finetune -> quantize -> eval
- `super_run_it.sh` calls `run_it.sh` multiple times, once per experiment (e.g. 1 QAT, 1 baseline fine-tuning)
- `parse_it.py` helps format outputs nicely (optional)


Configure log dir path and CUDA_VISIBLE_DEVICES:
```
# edit LOG_DIR
# edit CUDA_VISIBLE_DEVICES
vim super_run_it.sh
```

Go ahead and run the experiments! By default, this fine-tunes Llama-3.2-3B with and without QAT on alpaca for 1 epoch. This is full fine-tuning, i.e. no LoRA.
Quantization scheme is int8 dynamic activation + int4 weights, but this can be configured to NVFP4 through the `QUANTIZER` variable in `super_run_it.sh`.
All eval is done on both original bf16 and quantized int4 models.
```
# Will take maybe 10-15 mins
# Logs will be at the LOG_DIR you configured in the previous step
# The main log of interest will be `eval_wikitext_quantized.log` in experiment directory
./super_run_it.sh

# optional, parse results nicely
LOG_DIR=<your_log_dir> parse_it.py
```

## Example output

```
==> /home/andrewor/local/logs/tune/Llama3.2-3B_alpaca_full/eval_wikitext_quantized.log <==

2025-08-21:08:54:21,467 INFO     [eleuther_eval.py:583] 

| Tasks  |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
|--------|------:|------|------|---------------|---|------:|---|------|
|wikitext|      2|none  |None  |bits_per_byte  |↓  | 0.7209|±  |   N/A|
|        |       |none  |None  |byte_perplexity|↓  | 1.6482|±  |   N/A|
|        |       |none  |None  |word_perplexity|↓  |14.4697|±  |   N/A|



==> /home/andrewor/local/logs/tune/Llama3.2-3B_alpaca_qat/eval_wikitext_quantized.log <==

2025-08-21:09:02:21,286 INFO     [eleuther_eval.py:583] 

| Tasks  |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
|--------|------:|------|------|---------------|---|------:|---|------|
|wikitext|      2|none  |None  |bits_per_byte  |↓  | 0.7030|±  |   N/A|
|        |       |none  |None  |byte_perplexity|↓  | 1.6278|±  |   N/A|
|        |       |none  |None  |word_perplexity|↓  |13.5390|±  |   N/A|
```

Example parsed output:
```
$ LOG_DIR=/home/andrewor/local/logs/tune python parse_it.py 

experiment_name          tok/s                peak_mem_active    peak_mem_alloc     peak_mem_reserved
-----------------------  -------------------  -----------------  -----------------  -------------------
Llama3.2-3B_alpaca_full  7887.617 (+0.000%)   9.922 (+0.000%)    9.922 (+0.000%)    14.394 (+0.000%)
Llama3.2-3B_alpaca_qat   3051.554 (-61.312%)  11.548 (+16.385%)  11.548 (+16.385%)  16.453 (+14.311%)

experiment_name          wikitext_word_perplexity
-----------------------  -------------------------------
Llama3.2-3B_alpaca_full  14.470 quant, 12.254 float
Llama3.2-3B_alpaca_qat   13.539 quant, recovered 42.012%
```
