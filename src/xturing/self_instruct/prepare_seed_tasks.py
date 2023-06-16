import json
import os

from tqdm import tqdm

from xturing.model_apis import TextGenerationAPI
from xturing.utils.text_splitter import RecursiveCharacterTextSplitter


def instruction_input_suggest(
    original_text: str,
    engine: TextGenerationAPI,
    chunk_size: int = 8000,
    num_samples_per_chunk=7,
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    texts = text_splitter.split_text(original_text)
    # print(f"Split the document into {len(texts)} parts")

    questions = []
    answers = []
    for text in texts:
        prompt = f"""Given  a document. Suggest {num_samples_per_chunk} questions that could be asked related to the document. Generate a comprehensive and informative answer (but no more than 80 words) for each question.
            Document: {text}
            """

        outputs = engine.get_completion(prompts=[prompt])
        pairs = outputs.split("\n\n")
        for pair in pairs:
            try:
                question, answer = pair.split("\n")
                questions.append(question[3:])
                answers.append(answer)
            except:
                continue
    assert len(questions) == len(answers)
    return questions, answers


def prepare_seed_tasks(
    data_path, seed_path, engine, chunk_size=8000, num_samples_per_chunk=7
):
    instructions = []
    outputs = []
    for file in tqdm(os.listdir(data_path)):
        if file[-4:] != ".txt":
            continue
        with open(os.path.join(data_path, file)) as f:
            text = f.read()

        pairs = instruction_input_suggest(
            text, engine, chunk_size, num_samples_per_chunk
        )
        instructions.extend(pairs[0])
        outputs.extend(pairs[1])
        # break
    with open(seed_path, "a") as f:
        for i, instruction in enumerate(instructions):
            f.write(
                json.dumps(
                    {
                        "id": f"seed_task_{i}",
                        "instruction": instruction,
                        "instances": [{"input": "", "output": outputs[i]}],
                    }
                )
                + "\n"
            )
