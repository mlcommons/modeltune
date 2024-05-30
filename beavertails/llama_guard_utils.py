from llama_recipes.inference.prompt_format_utils import (
    build_custom_prompt,
    create_conversation,
    AgentType,
    SafetyCategory,
    LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX,
    PROMPT_TEMPLATE_2,
)


def build_llama_guard_2_prompt(user_prompt: str, categories: list[SafetyCategory]):
    formatted_prompt = build_custom_prompt(
        agent_type=AgentType.USER,
        conversations=create_conversation(
            [
                user_prompt
                # "<agent prompt placeholder>"
            ]
        ),
        categories=categories,
        category_short_name_prefix=LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX,
        prompt_template=PROMPT_TEMPLATE_2,
        with_policy=True,
    )

    return formatted_prompt


def lg_parse(output: str, categories: list[str], category_prefix: str):
    """
    Llama Guard 2 output format:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories.
    - Categories are prefixed with the category_prefix.
    - Category index starts at 1 not 0

    This method should parse the output and return the categories violated
    """
    lines = output.strip().split("\n")
    if lines[0] == "safe":
        return []
    elif lines[0] == "unsafe":
        violated_categories = lines[1].split(",")

        # Remove the prefix, convert to int, and get the actual category name
        violated_categories = [
            int(category.replace(category_prefix, ""))
            for category in violated_categories
        ]
        violated_categories = [categories[index - 1] for index in violated_categories]
        return violated_categories
    else:
        raise ValueError("Invalid Llama Guard 2 output format")
