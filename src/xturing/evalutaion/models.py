# import os

# from dotenv import load_dotenv, find_dotenv

"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
import os
from typing import Callable, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models.base import BaseModel

from .base import ModelSpec
from .prompt import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from .record import record_match, record_sampling

logger = logging.getLogger(__name__)

# # load openai key
# load_dotenv(find_dotenv())
# OPENAI_KEY = os.environ["OPENAI_KEY"]

# HELPER FUNCTIONS


def chat_prompt_to_text(prompt):
    if type(prompt) == str:
        return prompt
    else:
        return " ".join([message["content"] for message in prompt])


def load_model(model_name):
    if not os.path.exists(f"./{model_name}"):
        print(f"LOADING MODEL: {model_name}")
        model = BaseModel.create(model_name)
        model.save(f"./{model_name}")

    return BaseModel.load(f"./{model_name}")


def completion_query(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    **kwargs,
) -> Tuple[dict, Union[OpenAICreatePrompt, OpenAICreateChatPrompt], dict]:
    """
    Query the API for a completion.

    ARGS
    ====
    `model_spec`: `ModelSpec` containing model details to use in the query.
        This should be the dict returned by `registry.get_model()`.
        If `model_spec` is not provided, we use the default model that was
            intialized at the beginning of the run.
    `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
        the approriate `Prompt` class.
    `kwargs`: Other arguments passed to the API.

    RETURNS
    =======
    The result of the API call.
    The prompt that was fed into the API call as a str.
    A dict containing metadata about the query.
    """

    # parse prompt

    # Initialize model
    # TODO: pass kwargs to model!

    # model = AutoModelForCausalLM.from_pretrained(model_spec.name)

    # huggingface_models = ["gpt2"]

    # if model_spec.name in huggingface_models:
    #     model = AutoModelForCausalLM.from_pretrained(model_spec.name)
    # else:
    #     model = BaseModel.load(model_spec.name)
    # tokenizer = AutoTokenizer.from_pretrained(model_spec.name, return_tensors="pt")

    # TODO: is concatenating the contents a good solution to transform chat-style inputs to one string?

    # inputs = tokenizer(actual_prompt, return_tensors="pt").input_ids

    # Run completion
    # outputs = model.generate(
    #     input_ids=inputs, return_dict_in_generate=True, output_scores=True, **kwargs
    # )

    actual_prompt = chat_prompt_to_text(prompt)

    # TODO add config

    model = load_model(model_spec.name)

    text_out = model.generate(texts=[actual_prompt])

    # parse results
    result = {
        "text": text_out,
        "tokens": None,
        "logprobs": None,
    }
    # TODO: change metadata based on model
    metadata = {"model": model_spec.name}

    return result, actual_prompt, metadata


def check_sampled_text(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    expected: Union[str, List[str], Tuple[str]],
    *,
    options: Optional[List[str]] = None,
    separator: Callable[[str], bool] = None,
) -> Optional[str]:
    """
    Generates a completion using the prompt, checks whether the completion is
        one of the expected completions, and then records the result.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `options`: The list of canonical options, defaults to `expected` if None.
        The completion will be converted to one of these options.
    `expected`: The desired completion or the list of desired completions.
    `separator`: A callable which check the character sampled after the option
        to see if it is a valid separator.

    RETURNS
    =======
    The option that was picked, i.e., matched the completion, or None.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    result, actual_prompt, metadata = completion_query(
        prompt=prompt,
        model_spec=model_spec,
    )

    choice = result["text"][0]

    # TODO: check what result is supposed to look like [from OPENAI API]
    sampled = choice.strip() if model_spec.strip_completion else choice

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break

    result = {
        "prompt": actual_prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    result["metadata"] = metadata
    print("result", result)
    record_sampling(**result)
    record_match(match, expected=expected, picked=picked, sampled=sampled)
    return picked


def sample_freeform(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_tokens: int = 512,
    stop: Optional[str] = None,
    n_samples: int = None,
    return_logprobs: bool = False,
    **kwargs,
) -> Union[str, List[str], dict]:
    """
    Samples a freeform response from the specified model, records the sampling,
        and returns the sampled text.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `temperature`: Passed to `openai.Completion.create`.
    `top_p`: Passed to `openai.Completion.create`.
    `max_tokens`: Passed to `openai.Completion.create`.
    `stop`: Passed to `openai.Completion.create`.
    `n_samples`: The number of samples to generate (1 if None).
    `return_logprobs`: If True, returns the tokens and corresponding logprobs
        in addition to the sampled text.
    `kwargs`: See `completion_query`.

    RETURNS
    =======
    If `return_logprobs` is True, returns a dict with the sampled text, tokens,
        and corresponding logprobs. If `n_samples` is None, the outer list is
        removed from all values.
    Otherwise, returns the sampled text, or a list of sampled texts if
        `n_samples` is not None.
    """

    # TODO: add kwargs to completion query (see api.py for reference)
    result, actual_prompt, metadata = completion_query(
        prompt=prompt,
        model_spec=model_spec,
        do_sample=True,
        num_return_sequences=n_samples if n_samples else 1,
        max_new_tokens=max_tokens,
        top_p=top_p,
    )

    if n_samples is None:
        sampled = result["text"][0]
    else:
        sampled = result["text"]

    record_sampling(prompt=actual_prompt, sampled=sampled, metadata=metadata)

    if return_logprobs:
        # assert not model_spec.is_chat, "logprobs only works for non-chat models"
        # assert not kwargs.get("logprobs") is None

        tokens = result["tokens"]
        logprobs = result["logprobs"]
        top_logprobs = logprobs  # TODO: check how to get top logprobs, for now I return all logprobs
        if n_samples is None:
            tokens = tokens[0]
            logprobs = logprobs[0]
            top_logprobs = top_logprobs[0]
        return {
            "text": sampled,
            "tokens": tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

    return sampled
