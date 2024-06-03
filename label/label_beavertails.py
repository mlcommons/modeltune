import argparse
import asyncio
import datetime
import os

from datasets import load_dataset
from dotenv import load_dotenv
from together import AsyncTogether, Together

from safety_labeler import (LabelerModels, LabelerResponse, LabelerTaxonomy,
                            SafetyLabeler)

load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
async_client = AsyncTogether(
    api_key=os.getenv("TOGETHER_API_KEY"),
)


def batch_label_beaver(samples, labeler: SafetyLabeler):
    # TODO create a beaver dataset parser for this to avoid repeating code
    user_messages = samples["prompt"]

    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(labeler.async_label(messages=user_messages))

    # Create dict keys to store extra columns
    samples["labeled_is_safe"] = [None] * len(user_messages)
    samples["labeled_safety_categories"] = [None] * len(user_messages)
    samples["label_reasoning"] = [None] * len(user_messages)
    samples["label_succeeded"] = [None] * len(user_messages)

    for index, response in enumerate(responses):
        if response is None:
            # TODO log which lines failed so we can rerun. E.g. add the index of the sample to the function
            print(f"Failed to parse response. Not adding to dataset.")
            samples["labeled_is_safe"][index] = False
            samples["labeled_safety_categories"][index] = []
            samples["label_reasoning"][index] = ""
            samples["label_succeeded"][index] = False
        else:
            samples["labeled_is_safe"][index] = response.is_safe
            samples["labeled_safety_categories"][index] = response.safety_categories
            samples["label_reasoning"][index] = response.reasoning
            samples["label_succeeded"][index] = True

    return samples


def label_beaver(sample, labeler: SafetyLabeler):
    # TODO create a beaver dataset parser for this to avoid repeating code
    user_message = sample["prompt"]

    response: LabelerResponse = labeler.label(message=user_message)

    if response is None:
        # TODO log which lines failed so we can rerun. E.g. add the index of the sample to the function
        print(f"Failed to parse response. Not adding to dataset.")
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
    parser = argparse.ArgumentParser(
        description="Remap hazard datasets to different taxonomies."
    )
    # # TODO allow for multiple datasets. For now it's hardcorded to Beavertails
    # parser.add_argument("--dataset", required=False, type=str, help="Name of the HuggingFace dataset to process")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Name of the model to use for labeling",
        choices=[model.name for model in LabelerModels],
    )
    # parser.add_argument("--taxonomy", required=True, type=str, help="The target taxonomy to use for labeling", choices=list(taxonomy.name for taxonomy in LabelerTaxonomy))
    args = parser.parse_args()
    # # TODO parse args into a config object and do something with it. For now just code dataset, model, taxonomy manually below

    # Load the dataset from hugging face
    try:
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Select the working dataset

    # Run full dataset
    # working_dataset = dataset

    # Test on every X samples
    # working_dataset = dataset.filter(lambda example, idx: idx % 100 == 0 and idx < 10000, with_indices=True)

    # Test dataset on the first 10 examples
    working_dataset = dataset.select(range(11))

    print(f"Length sample dataset: ", len(working_dataset))

    # Set up the Safety Labeler

    # Get the selected labeler model from the argparse parameter
    selected_model = LabelerModels[args.model].value
    print(f"Selected model for safety labler: {selected_model}")

    # Set up the Safety Labeler with the selected model
    labeler = SafetyLabeler(
        client=client,
        async_client=async_client,
        labeler_model=selected_model,
        taxonomy=LabelerTaxonomy.MLC_0_5,
    )

    print("Starting map operation...")
    # Synchronous, 1 at a time. Good for testing
    # result = working_dataset.map(lambda sample: label_beaver(sample, labeler))

    # Process threaded version. Had trouble getting this to work sometimes
    # result = working_dataset.map(
    #     label_beaver, fn_kwargs={"labeler": labeler}, num_proc=12
    # )

    # Async, batched map operation. This works well for large datasets
    result = working_dataset.map(
        batch_label_beaver, fn_kwargs={"labeler": labeler}, batched=True, batch_size=20
    )

    # Save the updated dataset
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_used = labeler.labeler_model.name
    dataset_size = f"{len(working_dataset)}_rows"
    output_location = f"/Users/ryan/projects/mlc/modeltune/label/outputs/{model_used}_{dataset_size}_{current_datetime}.csv"
    result.to_csv(output_location)
    print(f"Saved labeled dataset to: {output_location}")


if __name__ == "__main__":
    main()
