# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = (
        "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    )


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class aegis_dataset:
    dataset: str = "aegis_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path = "data/llama_guard_training_data/aegis_small_july_0727_aegis_training_data_20240724-231749.json"
