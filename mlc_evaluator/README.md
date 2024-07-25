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
The dataset has to be in a specific format (TODO for commiting data preprocessing code)
```bash
python model/mlc_llama_guard/meta_llama_recipes/src/llama_recipes/data/llama_guard/aegis/aegis_data_formatter.py  --file_path FILE_PATH --label_column LABEL_COLUMN --text_column TEXT_COLUMN
```
Try on toy example dataset:
```
python model/mlc_llama_guard/meta_llama_recipes/src/llama_recipes/data/llama_guard/aegis/aegis_data_formatter.py  --file_path data/source_datasets/aegis_small_july_0727.json    --label_column labels --text_column text
```

## Launch fine-tuning
Pre-requisite : Obtain access to meta-llama/Meta-Llama-Guard-2-8B  on Hugging Face as its a gated model.


The training dataset created above can be used for training. Additionally, a pre-created toy training set is available in mlc_evaluator/data/source_datasets/ if you directly want to launch training. From inside 
modeltune/mlc_evaluator/ after changing permissions: 
```bash
./runner_scripts/finetune_llamguard2_aegis_peft.sh 
```




