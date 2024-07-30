import asyncio
import json
import time
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict
from together import AsyncTogether, Together

from constants import (BEAVERTAILS_FULL_TAXONOMY, MLC_TAXONOMY_0_5,
                       TOGETHER_GEMMA_7B_INSTRUCT,
                       TOGETHER_LLAMA_3_70B_INSTRUCT,
                       TOGETHER_MISTRAL_8x22B_INSTRUCT)
from prompts import (LABEL_CHECKER_SYS_PROMPT, LABEL_CHECKER_USER_PROMPT,
                     LABELER_SYS_PROMPT, LABELER_USER_PROMPT)


class LabelCheckerResponse(BaseModel):
    """NOT CURRENTLY BEING USED. Response object when teacher model checks label correctness"""

    correct: bool
    is_safe: bool
    safety_categories: List[str]
    reasoning: str


class LabelerResponse(BaseModel):
    """Response object when teacher model labels safety hazards"""

    is_safe: bool
    safety_categories: List[str]
    reasoning: str


class MessageType(Enum):
    """Enum for message types to be used in prompts. Can be either user chat message or assistant response. Will eventually support multi-turn conversation"""

    USER_PROMPT = "user_prompt"
    ASSISTANT_RESPONSE = "assistant_response"


class LabelerModels(Enum):
    """Enum for teacher models to be used for labeling safety hazards"""

    LLAMA_3_70B_INSTRUCT = TOGETHER_LLAMA_3_70B_INSTRUCT
    MISTRAL_8x22B_INSTRUCT = TOGETHER_MISTRAL_8x22B_INSTRUCT
    GEMMA_7B_INSTRUCT = TOGETHER_GEMMA_7B_INSTRUCT


class LabelerTaxonomy(Enum):
    """Enum for taxonomies to be used for labeling safety hazards"""

    BEAVERTAILS = BEAVERTAILS_FULL_TAXONOMY
    MLC_0_5 = MLC_TAXONOMY_0_5


class SafetyLabeler(BaseModel):
    """
    This class can
    1. Label safety hazards according to a taxonomy using powerful teacher model
    2. Check correctness of hazard labeled datasets (NOT CURRENTLY BEING USED). To be used in case of licensing issues generating direct labels

    Currently, we'll hardcode Llama 3 model for this purpose
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Together
    async_client: AsyncTogether
    labeler_model: LabelerModels
    taxonomy: LabelerTaxonomy

    def label(
        self, message: str, message_type: MessageType = MessageType.USER_PROMPT
    ) -> LabelerResponse:
        """
        Synchronous method: Assuming the message is a user chat message. TODO support assessing assistant response
        """
        # Set up prompts
        system_prompt = LABELER_SYS_PROMPT.format(taxonomy=self.taxonomy.value)
        user_prompt = LABELER_USER_PROMPT.format(
            message_type=message_type.value, message=message
        )

        # Send request and get response
        response = None
        max_retries = 5
        retry_counter = 0
        response = None
        while retry_counter < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.labeler_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except Exception as e:
                print(f"Error in label: {e}")
                retry_counter += 1
                time.sleep(30)

                # Try again unless it's the last retry. In which case throw fatal error
                if retry_counter == max_retries:
                    print("Fatal error: Maximum retries exceeded")
                    raise Exception("Maximum retries exceeded")

        # Try parse response
        try:
            response_json = response.choices[0].message.content
            parsed_response = json.loads(response_json)

            # Parse this into a pydantic model
            parsed_response_model = LabelerResponse(**parsed_response)
            return parsed_response_model

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(
                f"Failed sample: {message}, message_type: {message_type}, response (should be json but is not): {response_json}"
            )
            return None

    async def async_label(
        self, messages: List[str], message_type: MessageType = MessageType.USER_PROMPT
    ) -> List[LabelerResponse]:
        """
        Async method: Assuming the message is a user chat message. TODO support assessing assistant response
        """
        # Set up prompts
        system_prompt = LABELER_SYS_PROMPT.format(taxonomy=self.taxonomy.value)
        user_prompts = [
            LABELER_USER_PROMPT.format(message_type=message_type.value, message=message)
            for message in messages
        ]

        max_retries = 5
        retry_counter = 0
        responses = []
        while retry_counter < max_retries:
            try:
                tasks = [
                    self.async_client.chat.completions.create(
                        model=self.labeler_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    for user_prompt in user_prompts
                ]

                responses = await asyncio.gather(*tasks)
                break
            except Exception as e:
                print(f"Error in async_label: {e}")
                # TODO need better backoff strategy. Currently just give 30 seconds to allow full recovery
                time.sleep(30)
                if retry_counter == max_retries:
                    print("Fatal error: Maximum retries exceeded")
                    raise Exception("Maximum retries exceeded")

        results = []
        for index, response in enumerate(responses):
            try:
                response_json = response.choices[0].message.content
                parsed_response = json.loads(response_json)

                parsed_response_model = LabelerResponse(**parsed_response)
                results.append(parsed_response_model)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                print(
                    f"Failed sample: {messages[index]}, message_type: {message_type}, response (should be json but is not): {response_json}"
                )
                results.append(None)

        return results

    def check_label(
        self, message: str, is_safe: bool, assigned_categories: list[str] = []
    ) -> LabelCheckerResponse:
        """
        NOT CURRENTLY BEING USED. This method is used to check the correctness
        of a labeled dataset in case there are licensing issues using the
        teacher model to label directly
        """
        system_prompt = LABEL_CHECKER_SYS_PROMPT.format(taxonomy=self.taxonomy.value)
        user_prompt = LABEL_CHECKER_USER_PROMPT.format(
            user_message=message, is_safe=is_safe, categories=assigned_categories
        )
        response = self.client.chat.completions.create(
            model=self.labeler_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        try:
            response_json = response.choices[0].message.content
            parsed_response = json.loads(response_json)

            # Parse this into a pydantic model
            parsed_response_model = LabelCheckerResponse(**parsed_response)
            return parsed_response_model

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(
                f"Failed sample: {message}, is_safe: {is_safe}, assigned_categories: {assigned_categories}"
            )
            return None
