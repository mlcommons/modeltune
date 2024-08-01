# ML Commons AI Safety Evaluator Training and Evaluation Pipeline

This repository contains code for MLC AIS own pipeline for model training and deployment. 

It leverages SOTA pre-trained and instruction tuned model backbones to train on various datasets open source, synthetic, and MLCs own data (TODO). 

We also integrate with efficient training pipelines for distributed training and Fully Sharded Data Parallel (FSDP).

## Models that we finetune

LLamaGuard 2 - meta-llama/Meta-Llama-Guard-2-8B (HuggingFace weights)[PR open]

https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B

Mistral - mistralai/Mistral-7B-v0.1 (uses Hugging Face weights) [ONGOING]

https://huggingface.co/mistralai/Mistral-7B-v0.1

## Pipelines that we integrate with

1. Meta's Llama Recipes (Hugging Face weights) - https://github.com/meta-llama/llama-recipes/tree/main/src/llama_recipes/utils

2. HuggingFace

## Resources
MLC Auto Cluster 8xH100s


## Installation
1. conda create environment (tested for python==3.10.14)
2. pip install -e mlc_evaluator/model/mlc_llama_guard/meta_llama_recipes/
3. cd mlc_evaluator/
4. export PYTHONPATH=. to your enviroment (preferably add to ~/.bashrc) #TODO  Add setup.py 

## Training Data Creation
The dataset has to be in a specific format. Once you have the dataset in the specific format, you can create the training compatible dataset using the command below.

```bash
python model/mlc_llama_guard/meta_llama_recipes/src/llama_recipes/data/llama_guard/mlc_data/mlc_data_formatter.py  --file_path /home/shaona/modeltune/mlc_evaluator/data/source_datasets/source_dataset.json    --label_column <labels_column_name> --text_column <text_column_name>
```

To try running on a MLC dataset, you will need access to the dataset first to download and run. 

## Launch fine-tuning
Pre-requisite : Obtain access to meta-llama/Meta-Llama-Guard-2-8B  on Hugging Face as its a gated model.

From inside 
`modeltune/mlc_evaluator/` after changing permissions of the script, run: 

```bash
./runner_scripts/finetune_llamguard2_mlc_v0_5_peft_hyperparam_1.sh 
```

## Evaluate the fine-tuned model 
Pre-requisite : Obtain access to the MLC evaluation dataset first. 

From the root directory `modeltune/mlc_evaluator/`, run
```bash
 python inference/infer.py --dataset_name mlc-1320  --variant_type mlc
 ```

## For running tests, run 
From the root directory `modeltune/mlc_evaluator/`, run

```bash
python -m unittest model/mlc_llama_guard/meta_llama_recipes/src/tests/test_mlc_data_formatter.py
``` 


