from datasets import load_dataset
from pydantic import BaseModel, Field
import json
import os
from together import Together
from dotenv import load_dotenv
from llama_recipes.inference.prompt_format_utils import build_custom_prompt, create_conversation, LlamaGuardVersion, AgentType, SafetyCategory, LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2

from taxonomy import TaxonomyCategory, MLCTaxonomyCategories
from constants import BEAVER_AMBIGUOUS_CATEGORIES, BEAVER_EASY_CATEGORIES
from llama_guard_utils import build_llama_guard_2_prompt, lg_parse
from safety_labeler import SafetyLabeler, LabelerResponse, LabelerModels, LabelerTaxonomy
import datetime
import argparse

load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def _map_beaver_to_llama_categories(beaver_categories: list[str]):
    mlc_taxonomy = MLCTaxonomyCategories()
    result: set[TaxonomyCategory] = set()
    for category in beaver_categories:
        if category == "animal_abuse":
            result.update([mlc_taxonomy.violent_crimes, mlc_taxonomy.non_violent_crimes, mlc_taxonomy.unknown_hazard])
        elif category == "child_abuse":
            result.update([mlc_taxonomy.child_sexual_exploitation])
        elif category == "controversial_topics,politics": 
            result.update([mlc_taxonomy.hate, mlc_taxonomy.unknown_hazard])
        elif category == "discrimination,stereotype,injustice":
            result.update([mlc_taxonomy.hate])
        elif category == "drug_abuse,weapons,banned_substance":
            result.update([mlc_taxonomy.violent_crimes, mlc_taxonomy.indiscriminate_weapons_cbrne, mlc_taxonomy.unknown_hazard])
        elif category == "financial_crime,property_crime,theft":
            result.update([mlc_taxonomy.violent_crimes, mlc_taxonomy.non_violent_crimes, mlc_taxonomy.unknown_hazard])
        elif category == "hate_speech,offensive_language":
            result.update([mlc_taxonomy.hate])
        elif category == "misinformation_regarding_ethics,laws_and_safety":
            result.update([mlc_taxonomy.specialized_advice])
        elif category == "non_violent_unethical_behavior":
            result.update([mlc_taxonomy.non_violent_crimes])
        elif category == "privacy_violation":
            result.update([mlc_taxonomy.intellectual_property, mlc_taxonomy.non_violent_crimes, mlc_taxonomy.unknown_hazard])
        elif category == "self_harm":
            result.update([mlc_taxonomy.suicide_and_self_harm])
        elif category == "sexually_explicit,adult_content":
            result.update([mlc_taxonomy.sex_related_crimes, mlc_taxonomy.child_sexual_exploitation, mlc_taxonomy.unknown_hazard])
        elif category == "terrorism,organized_crime":
            result.update([mlc_taxonomy.violent_crimes])
        elif category == "violence,aiding_and_abetting,incitement":
            result.update([mlc_taxonomy.violent_crimes])
    
    result = list(map((lambda x: x.to_llama_safety_category()), result))
    return result

def build_beaver_to_llama_prompt(user_prompt: str, beaver_categories: list[str]):
    possible_safety_categories = _map_beaver_to_llama_categories(beaver_categories)
    return build_llama_guard_2_prompt(user_prompt, possible_safety_categories)

def add_pretty_columns(sample, mlc_categories: str):
    original_categories = sample["category"]
    original_detected_categories = {k: v for k, v in original_categories.items() if v is True}
    sample["original_detected_categories"] = ",".join(original_detected_categories.keys())
    mlc_detected_categories = {k: v for k, v in mlc_categories.model_dump().items() if v is True}
    sample["mlc_detected_categories"] = ";".join(mlc_detected_categories.keys())
    return sample

def map_beaver_to_mlc(sample):
    category_dict = sample["category"]
    detected_categories = {k: v for k, v in category_dict.items() if v is True}

    # If no detected categories, return empty MLC categorization
    if len(detected_categories) == 0:
        mlc_tax_categories = MLCTaxonomyCategories()
    # If detected categories are all 1:1 mappings, return the MLC categorization with 1:1 mappings
    elif all(key in BEAVER_EASY_CATEGORIES for key in detected_categories.keys()):
        mlc_tax_categories = MLCTaxonomyCategories()
        for key in detected_categories.keys():
            if key == "child_abuse":
                mlc_tax_categories.child_sexual_exploitation.value = True
            elif key == "discrimination,stereotype,injustice":
                mlc_tax_categories.hate.value = True
            elif key == "hate_speech,offensive_language":
                mlc_tax_categories.hate.value = True
            elif key == "misinformation_regarding_ethics,laws_and_safety":
                mlc_tax_categories.specialized_advice.value = True
            elif key == "non_violent_unethical_behavior":
                mlc_tax_categories.non_violent_crimes.value = True
            elif key == "self_harm":
                mlc_tax_categories.suicide_and_self_harm.value = True
            elif key == "terrorism,organized_crime":
                mlc_tax_categories.violent_crimes.value = True
            elif key == "violence,aiding_and_abetting,incitement":
                mlc_tax_categories.violent_crimes.value = True
    # Otherwise (if there are ambiguous categories), mark as unknown and skip these for now
    # TODO this is where we'd want to engage LG2 to try and subcategorize it for us
    else:
        # Ask LG2 to do the categorization.
        user_prompt = sample["prompt"]
        test_lg_prompt = build_beaver_to_llama_prompt(user_prompt, detected_categories.keys())

        test_response = client.completions.create(
            model="meta-llama/LlamaGuard-2-8b",
            prompt=test_lg_prompt,
        )

        lg_categorization = test_response.choices[0].text

        possible_safety_categories = _map_beaver_to_llama_categories(detected_categories.keys())
        parsed_categories = lg_parse(
            output=lg_categorization,
            categories=[category.name for category in possible_safety_categories],
            category_prefix=LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX
        )
        print(f"Original phrase: {user_prompt}\nOriginal categories: {detected_categories.keys()}\nLlama Guard response: {lg_categorization}\nParsed categories: {parsed_categories}\n")

        mlc_tax_categories = MLCTaxonomyCategories.from_category_list(parsed_categories)

        # Default. Just mark it as unknown
        # mlc_tax_categories = MLCTaxonomyCategories()
        # mlc_tax_categories.unknown_hazard.value = True

    mlc_categories_json = mlc_tax_categories.model_dump_json()
    sample["mlc_category"] = mlc_categories_json

    # Add easy readable columns
    sample = add_pretty_columns(sample, mlc_tax_categories)
    
    return sample 

global_correct = 0
global_incorrect = 0

def validate_beaver_safety_labels(sample):
    global global_correct, global_incorrect
    label_checker = SafetyLabeler(client=client)

    user_message = sample["prompt"]
    is_safe = sample["is_safe"]
    category_dict = sample["category"]
    detected_categories = {k: v for k, v in category_dict.items() if v is True}

    response = label_checker.check_label(user_message, is_safe, detected_categories.keys())

    if response.correct:
        print("Categorization is correct.")
        global_correct += 1
    else:
        print("Categorization is incorrect.")
        print(f"Reasoning: {response.reasoning}")
        global_incorrect += 1

    sample["label_is_correct"] = response.correct
    sample["label_validation_dict"] = response.model_dump_json(indent = 2)

    return sample

def label_beaver(sample, idx, labeler: SafetyLabeler):
    # TODO create a beaver dataset parser for this to avoid repeating code
    user_message = sample["prompt"]

    response: LabelerResponse = labeler.label(message = user_message)

    if response is None:
        # TODO log which lines failed so we can rerun. E.g. add the index of the sample to the function
        print(f"Failed to parse response. Not adding to dataset. Index: {idx}")
        sample["labeled_is_safe"] = False
        sample["labeled_safety_categories"] = []
        sample["label_reasoning"] = ""
        sample["label_succeeded"] = False
    else:
        sample["labeled_is_safe"] = response.is_safe
        sample["labeled_safety_categories"] = response.safety_categories
        sample["label_reasoning"] = response.reasoning
        sample["label_succeeded"] = True

    return sample



def main():
    # parser = argparse.ArgumentParser(description="Remap hazard datasets to different taxonomies.")
    # # TODO allow for multiple datasets. For now it's hardcorded to Beavertails
    # parser.add_argument("--dataset", required=False, type=str, help="Name of the HuggingFace dataset to process")
    # parser.add_argument("--model", required=True, type=str, help="Name of the model to use for labeling", choices=[model.name for model in LabelerModels])
    # parser.add_argument("--taxonomy", required=True, type=str, help="The target taxonomy to use for labeling", choices=list(taxonomy.name for taxonomy in LabelerTaxonomy))
    # args = parser.parse_args()
    # # TODO parse args into a config object and do something with it. For now just code dataset, model, taxonomy manually below

    try:
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Run full dataset
    # working_dataset = dataset
    # Test on a slice of the dataset - here we get every 100th example up to 10,000.
    working_dataset = dataset
    # working_dataset = dataset.filter(lambda example, idx: idx % 100 == 0 and idx < 10000, with_indices=True)
    # working_dataset = dataset.select(range(10))

    print(f"Length sample dataset: ", len(working_dataset))

    # label_checker = LabelChecker(client=client)
    # label_checker.check_label("This is a test message", False, ["violence"])

    print("Starting map operation...")
    labeler = SafetyLabeler(client=client, labeler_model=LabelerModels.LLAMA_3_70B_INSTRUCT, taxonomy=LabelerTaxonomy.MLC_0_5)
    # result = working_dataset.map(lambda sample: label_beaver(sample, labeler))
    result = working_dataset.map(label_beaver, fn_kwargs={"labeler": labeler}, with_indices=True)

    # updated_dataset = working_dataset.map(map_beaver_to_mlc)

    # print("Operation completed")
    # print("Length updated dataset", len(updated_dataset))

    # Save the updated dataset
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_used = labeler.labeler_model.name
    dataset_size = f"{len(working_dataset)}_rows"
    result.to_csv(f"/Users/ryan/projects/mlc/modeltune/beavertails/outputs/{model_used}_{dataset_size}_{current_datetime}.csv")

if __name__ == "__main__":
    main()
