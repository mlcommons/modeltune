# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import re
import unittest
from model.mlc_llama_guard.meta_llama_recipes.src.llama_recipes.data.llama_guard.mlc_data.mlc_data_formatter import (
    MLCDataFormatter,
)

EXPECTED_TOKEN_START = "<|begin_of_text|>"
EXPECTED_RESPONSE_CLASSIFICATION_PHRASE = (
    "Check if there is unsafe content in 'agent' messages"
)
EXPECTED_INST_TOKEN = "[/INST]"
SIZE_OF_INST_TOKEN = 7


MLC_TAXONOMY_0_5_TEST = {
    "violent_crimes": "S1",
    "non_violent_crimes": "S2",
    "sex_related_crimes": "S3",
    "child_sexual_exploitation": "S4",
    "indiscriminate_weapons_cbrne": "S5",
    "suicide_and_self_harm": "S6",
    "hate": "S7",
    "ambiguous_hazard": "S8",
}


class TestMLCFormatter(unittest.TestCase):
    """
    MLCFormatterTest Class for testing training data creation routines.
    """

    def setUp(self):
        self.mlc_formatter = MLCDataFormatter(dataset_path=None)

    def test_get_training_examples(self):
        """Test to validate training examples"""

        file_path = "model/mlc_llama_guard/meta_llama_recipes/src/tests/test_mlc_source_data.json"
        self.mlc_formatter.set_annotation_column_name("labels")
        self.mlc_formatter.set_conversation_column_name("text")
        self.training_examples, training_examples_dict = (
            self.mlc_formatter.get_training_examples(file_path=file_path)
        )
        self.assertTrue(isinstance(training_examples_dict, dict))

        violated_category_codes = training_examples_dict[0]["violated_category_codes"]
        label = training_examples_dict[0]["label"]
        for code in violated_category_codes:
            self.assertIn(code, list(MLC_TAXONOMY_0_5_TEST.values()))
            self.assertIn(label, ["safe", "unsafe"])

    def test_get_formatted_examples(self):
        """
        MLCFormatterTest Class for testing training data formatting routines.
        """
        file_path = "model/mlc_llama_guard/meta_llama_recipes/src/tests/test_mlc_training_data.json"
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

            self.assertTrue(isinstance(json_data, dict))
            self.assertEqual(len(json_data), 1)
            for key, text in json_data.items():

                index = text.find(EXPECTED_INST_TOKEN)
                labels = text[index + SIZE_OF_INST_TOKEN : -1]

                index_of_instr_enc = text.find(
                    EXPECTED_INST_TOKEN,
                    text.find(EXPECTED_INST_TOKEN) + SIZE_OF_INST_TOKEN,
                )
                labels_to_test = text[index_of_instr_enc + SIZE_OF_INST_TOKEN : -1]

                re_matches = re.findall(r"unsafe\s*(S\d{1,2})\b", labels_to_test)
                self.assertLessEqual(len(re_matches), 1)

                for match in re_matches:
                    cat = match
                    self.assertIn(cat, {v: k for k, v in MLC_TAXONOMY_0_5_TEST.items()})
