# misc.py
"""
This file defines miscellanous utilities.
"""
import functools
import importlib
from typing import Any


def t(duration: float) -> str:
    if duration is None:
        return "n/a"
    if duration < 1:
        return f"{(1000*duration):0.3f}ms"
    elif duration < 60:
        return f"{duration:0.3f}s"
    else:
        return f"{duration//60}min{int(duration%60)}s"


def make_object(object_ref: Any, *args: Any, **kwargs: Any) -> Any:
    modname, qualname_separator, qualname = object_ref.partition(":")
    obj = importlib.import_module(modname)
    if qualname_separator:
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
    return functools.partial(obj, *args, **kwargs)


# api_utils.py
"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import logging

import backoff
import openai


def generate_dummy_chat_completion():
    return {
        "id": "dummy-id",
        "object": "chat.completion",
        "created": 12345,
        "model": "dummy-chat",
        "usage": {"prompt_tokens": 56, "completion_tokens": 6, "total_tokens": 62},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a dummy response.",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def generate_dummy_completion():
    return {
        "id": "dummy-id",
        "object": "text_completion",
        "created": 12345,
        "model": "dummy-completion",
        "choices": [
            {
                "text": "This is a dummy response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    if kwargs["model"] == "dummy-completion":
        return generate_dummy_completion()

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    if kwargs["model"] == "dummy-chat":
        return generate_dummy_chat_completion()

    result = openai.ChatCompletion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result
