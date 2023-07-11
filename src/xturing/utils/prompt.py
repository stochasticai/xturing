from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

OpenAICreatePrompt = Union[str, List[str], List[int], List[List[int]]]
OpenAIChatMessage = Dict[
    str, str
]  # A message is a dictionary with "role" and "content" keys
OpenAICreateChatPrompt = List[OpenAIChatMessage]  # A chat log is a list of messages


@dataclass
class Prompt(ABC):
    """
    A `Prompt` encapsulates everything required to present the `raw_prompt` in different formats,
    e.g., a normal unadorned format vs. a chat format.
    """

    @abstractmethod
    def to_openai_create_prompt(self):
        """
        Return the actual data to be passed as the `prompt` field to either `openai.ChatCompletion.create`,
        if the model is a chat model, or `openai.Completion.create` otherwise.
        See the above types to see what each API call is able to handle.
        """


def chat_prompt_to_text(prompt: OpenAICreateChatPrompt) -> str:
    """
    Render a chat prompt as a text prompt. User and assistant messages are separated by newlines
    and prefixed with "User: " and "Assistant: ", respectively, unless there is only one message.
    System messages have no prefix.
    """
    assert is_chat_prompt(prompt), f"Expected a chat prompt, got {prompt}"
    chat_to_prefixes = {
        # roles
        "system": "",
        # names
        "example_user": "User: ",
        "example_assistant": "Assistant: ",
    }

    # For a single message, be it system, user, or assistant, just return the message
    if len(prompt) == 1:
        return prompt[0]["content"]

    text = ""
    for msg in prompt:
        role = msg["name"] if "name" in msg else msg["role"]
        prefix = chat_to_prefixes.get(role, role.capitalize() + ": ")
        content = msg["content"]
        text += f"{prefix}{content}\n"
    text += "Assistant: "
    return text.lstrip()


def text_prompt_to_chat_prompt(prompt: str) -> OpenAICreateChatPrompt:
    assert isinstance(prompt, str), f"Expected a text prompt, got {prompt}"
    return [
        {"role": "system", "content": prompt},
    ]


def is_chat_prompt(prompt: Prompt) -> bool:
    return isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt)
