import json
import random
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

from xturing.model_apis import TextGenerationAPI
from xturing.self_instruct.templates.instance_gen_template import (
    input_first_template_for_gen,
    output_first_template_for_clf,
)

random.seed(42)


def generate_instances(
    input_file: Path,
    classification_file: Path,
    output_file: Path,
    num_instructions: int,
    max_instances_to_generate: int,
    generation_tasks_only: bool,
    classification_tasks_only: bool,
    engine: TextGenerationAPI,
):
    # Load the machine generated instructions from the input file.
    with input_file.open() as fin:
        lines = fin.readlines()
        if num_instructions is not None:
            lines = lines[:num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    # Generate instance inputs for each instruction.
    # Write the generated instances to the output file.
    task_clf_types = {}
    with classification_file.open() as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in [
                "Yes",
                "yes",
                "YES",
            ]

    if classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]

    if generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    existing_requests = {}
    if output_file.exists():
        with output_file.open() as fin:
            for line in tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm(total=len(tasks))
    with output_file.open("w") as fout:
        for batch_idx in range(0, len(tasks), engine.request_batch_size):
            batch = tasks[batch_idx : batch_idx + engine.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k])
                        for k in [
                            "instruction",
                            "raw_instances",
                            "instance_metadata",
                            "instruction_metadata",
                            "most_similar",
                            "avg_similarity_score",
                        ]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = (
                            output_first_template_for_clf
                            + " "
                            + task["instruction"].strip()
                            + "\n"
                        )
                        prompts.append(prompt)
                    else:
                        prompt = (
                            input_first_template_for_gen
                            + " "
                            + task["instruction"].strip()
                            + "\n"
                        )
                        prompts.append(prompt)

                results = engine.generate_text(
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    max_tokens=300
                    if any(task_clf_types[task["instruction"]] for task in batch)
                    else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[
                        f"Example {max_instances_to_generate + 1}",
                        "Task:",
                    ],
                    logprobs=1,
                    n=1,
                    best_of=1,
                )

                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0][
                            "text"
                        ]
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k])
                        for k in [
                            "instruction",
                            "raw_instances",
                            "instance_metadata",
                            "instruction_metadata",
                            "most_similar",
                            "avg_similarity_score",
                        ]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
