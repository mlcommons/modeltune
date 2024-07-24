# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class MLCDatasetsFormatterBase(ABC):
    """
    Base class for processing datasets for MLC Evaluators.
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """Path to the dataset."""
        self.dataset_path = dataset_path

    @abstractmethod
    def set_annotation_column_name(self):
        """Extract annotations column name from the data."""

    @abstractmethod
    def set_conversation_column_name(self):
        """Extract the text field from the data."""

    @abstractmethod
    def get_training_examples(self, file_path):
        """Extract training examples from the dataset."""
