import json
import random
import re
import string
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

from xturing.model_apis import TextGenerationAPI

random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """
    Encode multiple prompt instructions into a single string.

    Args:
        prompt_instructions: a list of strings representing the prompt instructions.
        classification: a boolean flag indicating whether the prompt is for classification tasks.

    Returns:
        A string representing the encoded prompt.
    """
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        # Remove excess whitespace and trailing colons
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, n):
    """
    Sample n machine instructions from a list of machine instructions.

    Args:
        machine_instructions: a list of strings representing the machine instructions.
        n: an integer representing the number of instructions to sample.

    Returns:
        A list of n machine instructions, sampled from the input list.
    """
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(word, string):
    """
    Find a word in a string, ignoring case and word boundaries.

    Args:
        word: a string representing the word to find.
        string: a string to search for the word.

    Returns:
        A match object if the word is found, None otherwise.
    """
    return re.compile(r"\b({0})\b".format(word), flags=re.IGNORECASE).search(string)


def post_process_gpt3_response(response):
    # If response is None or the finish reason is length, return an empty list
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []

    # Split the text by line number
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []

    # Process each instruction
    for instruction in raw_instructions:
        # Remove excess whitespace and capitalize the first letter
        instruction = re.sub(r"\s+", " ", instruction).strip().capitalize()

        # Skip empty instructions
        if instruction == "":
            continue

        # Filter out instructions that are too short or too long
        if len(instruction.split()) <= 3 or len(instruction.split()) > 150:
            continue

        # Filter out instructions containing unsuitable keywords
        unsuitable_keywords = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
        ]
        if any(keyword in instruction.lower() for keyword in unsuitable_keywords):
            continue

        # Filter out instructions starting with "Write a program"
        if instruction.startswith("Write a program"):
            continue

        # Filter out instructions starting with punctuation or non-ASCII characters
        if instruction[0] in string.punctuation or not instruction[0].isascii():
            continue

        instructions.append(instruction)

    # Return the filtered list of instructions
    return instructions


def load_data_from_jsonl_file(file_path: Path):
    data = [json.loads(l) for l in file_path.open("r")]
    return data


def extract_seed_instructions(seed_tasks, use_clf_seed_tasks_only):
    """Loads the seed tasks from the JSONL file at the given path and returns seed_instructions."""
    if use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    return seed_instructions


def bootstrap_instructions(
    seed_tasks_path: Path,
    output_file: Path,
    num_instructions_to_generate: int,
    use_clf_seed_tasks_only: bool,
    engine: TextGenerationAPI,
    num_prompt_instructions: int,
) -> None:
    """
    Generates machine-generated instructions using OpenAI's GPT-3 and saves them to a file.

    Args:
        seed_tasks_path: The path to a JSONL file containing the seed tasks.
        output_file: The file where the generated instructions should be saved.
        num_instructions_to_generate: The number of machine-generated instructions to generate.
        use_clf_seed_tasks_only: Whether to use only the seed tasks that were classified as valid.
        engine: The name of the OpenAI GPT-3 engine to use.
        num_prompt_instructions: The number of instructions to use as prompts for each GPT-3 request.
        request_batch_size: The number of GPT-3 requests to make in each batch.
        api_key: The API key for the OpenAI API.
        organization: The name of the OpenAI organization to use.
    """

    # Load the seed tasks from the JSONL file
    seed_tasks = load_data_from_jsonl_file(seed_tasks_path)

    # Extract the seed instructions from the seed tasks
    seed_instructions = extract_seed_instructions(seed_tasks, use_clf_seed_tasks_only)

    # Load the existing machine-generated instructions from a file, if it exists
    request_idx = 0
    machine_instructions = []
    if output_file.exists():
        with output_file.open("r") as f:
            for line in f:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(
            f"Loaded {len(machine_instructions)} machine-generated instructions from file."
        )

    # Initialize the Rouge scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Initialize the progress bar
    progress_bar = tqdm(total=num_instructions_to_generate)

    # Update progress bar with number of existing machine instructions
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    # Open output file for writing generated instructions
    with output_file.open("a") as fout:
        # Generate new instructions until desired number of instructions is reached
        while len(machine_instructions) < num_instructions_to_generate:
            # Initialize batch inputs
            batch_inputs = []

            for _ in range(engine.request_batch_size):
                # Sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, n=2
                )
                # Sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions,
                    num_prompt_instructions - len(prompt_instructions)
                    if num_prompt_instructions - len(prompt_instructions) > 0
                    else 1,
                )
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(
                    prompt_instructions, classification=use_clf_seed_tasks_only
                )
                batch_inputs.append(prompt)

            # Use OpenAI GPT3 to generate new instructions
            results = engine.generate_text(
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
            )

            instructions = []
            all_metadata = []

            for result in results:
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(scorer.score, inst),
                        seed_instructions + machine_instructions,
                    )
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]

                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]

                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                fout.write(
                    json.dumps(
                        {
                            "instruction": inst,
                            "most_similar": most_similar_instructions,
                            "avg_similarity_score": float(np.mean(rouge_scores)),
                            "metadata": metadata,
                            "request_idx": request_idx,
                        }
                    )
                    + "\n"
                )

                progress_bar.update(1)

            request_idx += 1
