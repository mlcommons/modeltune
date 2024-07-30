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
    """Batch labeler for the BeaverTails dataset. This is used to label the dataset in batches."""
    user_messages = samples["prompt"]

    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(labeler.async_label(messages=user_messages))

    # Create extra columns
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
    """Labeler for the BeaverTails dataset. This is used to label the dataset one at a time."""
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
        description="Remap hazard datasets to MLC taxonomy ."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Name of the model to use for labeling",
        choices=[model.name for model in LabelerModels],
    )
    args = parser.parse_args()

    # Load the dataset from hugging face
    try:
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Select how much of the dataset you want to process
    # working_dataset = dataset.select(range(11))
    working_dataset = dataset
    print(f"Length sample dataset: ", len(working_dataset))

    # Set up the Safety Labeler
    selected_model = LabelerModels[args.model].value
    print(f"Selected model for safety labler: {selected_model}")
    labeler = SafetyLabeler(
        client=client,
        async_client=async_client,
        labeler_model=selected_model,
        taxonomy=LabelerTaxonomy.MLC_0_5,
    )

    # Label the dataset
    print("Starting map operation...")
    # Async, batched map operation. This works well for large datasets
    result = working_dataset.map(
        batch_label_beaver, fn_kwargs={"labeler": labeler}, batched=True, batch_size=20
    )
    # Synchronous, 1 at a time. Good for testing
    # result = working_dataset.map(lambda sample: label_beaver(sample, labeler))

    # Save the updated dataset
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_used = labeler.labeler_model.name
    dataset_size = f"{len(working_dataset)}_rows"
    output_location = f"/Users/ryan/projects/mlc/modeltune/label/outputs/{model_used}_{dataset_size}_{current_datetime}.csv"
    result.to_csv(output_location)
    print(f"Saved labeled dataset to: {output_location}")


if __name__ == "__main__":
    main()
