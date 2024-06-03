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
    correct: bool
    is_safe: bool
    safety_categories: List[str]
    reasoning: str


class LabelerResponse(BaseModel):
    is_safe: bool
    safety_categories: List[str]
    reasoning: str


class MessageType(Enum):
    USER_PROMPT = "user_prompt"
    ASSISTANT_RESPONSE = "assistant_response"


class LabelerModels(Enum):
    LLAMA_3_70B_INSTRUCT = TOGETHER_LLAMA_3_70B_INSTRUCT
    MISTRAL_8x22B_INSTRUCT = TOGETHER_MISTRAL_8x22B_INSTRUCT
    GEMMA_7B_INSTRUCT = TOGETHER_GEMMA_7B_INSTRUCT


class LabelerTaxonomy(Enum):
    BEAVERTAILS = BEAVERTAILS_FULL_TAXONOMY
    MLC_0_5 = MLC_TAXONOMY_0_5


class SafetyLabeler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Together
    async_client: AsyncTogether
    labeler_model: LabelerModels
    taxonomy: LabelerTaxonomy

    """
    This class can
    1. Label safety hazards according to a taxonomy using powerful teacher model
    2. Check correctness of hazard labeled datasets by using more powerful teacher models
    
    Currently, we'll hardcode Llama 3 model for this purpose
    """

    def label(
        self, message: str, message_type: MessageType = MessageType.USER_PROMPT
    ) -> LabelerResponse:
        """
        Assuming the message is a user chat message

        TODO use MLC taxonomy
        TODO support assessing assistant response
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
        Assuming the message is a user chat message

        TODO use MLC taxonomy
        TODO support assessing assistant response
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
                # Allow full reset
                # TODO need better backoff strategy
                time.sleep(30)

                # Try again unless it's the last retry. In which case throw fatal error
                if retry_counter == max_retries:
                    print("Fatal error: Maximum retries exceeded")
                    raise Exception("Maximum retries exceeded")

        results = []
        for index, response in enumerate(responses):
            # Try parse response
            try:
                response_json = response.choices[0].message.content
                parsed_response = json.loads(response_json)

                # Parse this into a pydantic model
                parsed_response_model = LabelerResponse(**parsed_response)
                results.append(parsed_response_model)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                print(
                    f"Failed sample: {messages[index]}, message_type: {message_type}, response (should be json but is not): {response_json}"
                )
                # TODO probably have to append something here
                results.append(None)

        return results

    def check_label(
        self, message: str, is_safe: bool, assigned_categories: list[str] = []
    ) -> LabelCheckerResponse:
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

        # print(response.choices[0].message.content)

        # try to parse the json response and return it
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
