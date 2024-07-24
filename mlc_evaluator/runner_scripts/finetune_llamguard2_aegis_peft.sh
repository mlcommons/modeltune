#!/bin/bash

# Define variables
NNODES=1
NPROC_PER_NODE=8
MODEL_NAME="/home/shaona/.cache/huggingface/hub/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
DATASET="dania_dataset"
OUTPUT_DIR="/home/shaona/trained_models_test"
PATH_TO_FINETUNE="model/mlc_llama_guard/llama_recipes/recipes/quickstart/finetuning"

# For multi GPU parameter efficient finetuning
torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE $PATH_TO_FINETUNE/finetuning.py \
  --enable_fsdp --enable_fsdp \
  --model_name $MODEL_NAME \
  --pure_bf16 \
  --use_fast_kernels \
  --use_peft \
  --peft_method lora \
  --dataset $DATASET \
  --output_dir $OUTPUT_DIR \
  --save_metrics \
  --save_model