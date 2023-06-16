import json
import random
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

from xturing.model_apis import TextGenerationAPI
from xturing.self_instruct.templates.clf_task_template import template_1

random.seed(42)


templates = {"template_1": template_1}


# Creates a function out of the main function that accepts batch_dir, num_instructions, template, engine, request_batch_size, api_key, and organization as arguments.


def identify_if_classification(
    input_file: Path,
    output_file: Path,
    num_instructions: int,
    template: str,
    engine: TextGenerationAPI,
):
    # Load the machine generated instructions
    with input_file.open() as fin:
        lines = fin.readlines()
        if num_instructions is not None:
            lines = lines[:num_instructions]

    # Create the output path

    # Load the existing requests
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

    # Create the progress bar
    progress_bar = tqdm(total=len(lines))

    # Write the output to file
    with output_file.open("w") as fout:
        # Iterate over the lines in batches
        for batch_idx in range(0, len(lines), engine.request_batch_size):
            batch = [
                json.loads(line)
                for line in lines[batch_idx : batch_idx + engine.request_batch_size]
            ]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[template]
                prompts = [
                    prefix
                    + " "
                    + d["instruction"].strip()
                    + "\n"
                    + "Is it classification?"
                    for d in batch
                ]
                results = engine.generate_text(
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                )

                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][
                            0
                        ]["text"]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"],
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            progress_bar.update(len(batch))
