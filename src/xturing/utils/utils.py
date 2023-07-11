import contextlib
import io
import logging
import os
import random
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import yaml

SHUFFLE_SEED = 123
_MAX_SAMPLES = None


def read_yamls(config_path):
    conf = {}

    with open(config_path) as f:
        conf.update(yaml.safe_load(f))

    return conf


@contextlib.contextmanager
def no_std_out():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def create_temp_directory(path):
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"{path} directory already exists. Please specify other directory if you don't want to use previous cache."
        )

    return dir_path


def extract_text_from_directory(directory_path):
    try:
        import textract
    except:
        print(
            "To use this functionality we ask you to install the following textract lib and dependencies: \n\n\
  apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig \n\
  pip install textract\n\n\
See more details at https://xturing.stochastic.ai/datasets/generate#from-your-data\n"
        )
        exit()

    directory_path = Path(directory_path)

    # Check if the input is a directory
    if not directory_path.is_dir():
        print("The path {} should be a directory".format(directory_path.resolve()))
        exit()

    print(f"Processing directory {directory_path.resolve()}...")

    # Create a temporary directory to store the txt files
    temp_dir = tempfile.mkdtemp()

    # Walk through the directory tree and process each file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            print(f"Processing file {file}...")
            # Get the file extension and check if it's supported by textract
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)

            # skip files that are supported
            if ext not in {
                ".csv",
                ".doc",
                ".docx",
                ".eml",
                ".epub",
                ".gif",
                ".jpg",
                ".jpeg",
                ".json",
                ".html",
                ".htm",
                ".mp3",
                ".msg",
                ".odt",
                ".ogg",
                ".pdf",
                ".png",
                ".pptx",
                ".rtf",
                ".tiff",
                ".tif",
                ".txt",
                ".wav",
                ".xlsx",
                ".xls",
            }:
                print(f"{ext} extension is not supported")
                continue

            # Extract the text from the file and save it to a txt file
            try:
                text = textract.process(file_path).decode("utf-8")
                txt_filename = name + ".txt"
                txt_filepath = os.path.join(temp_dir, txt_filename)

                # If the txt file already exists, append a random string to its name
                if os.path.exists(txt_filepath):
                    random_suffix = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=8)
                    )
                    txt_filename = name + "_" + random_suffix + ".txt"
                    txt_filepath = os.path.join(temp_dir, txt_filename)

                with open(txt_filepath, "w") as txt_file:
                    txt_file.write(f"{name}\n\n{text}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Finished processing directory, the text files are stored in {temp_dir}.")

    return temp_dir


def _filter_args(arguments: dict):
    to_delete = []
    for key, value in arguments.items():
        if value is None:
            to_delete.append(key)
    for key in to_delete:
        del arguments[key]
    return arguments


def _index_samples(samples: List[Any], logger: logging.Logger):
    """Shuffle `samples` and pair each sample with its index."""
    indices = list(range(len(samples)))
    random.Random(SHUFFLE_SEED).shuffle(indices)
    if _MAX_SAMPLES is not None:
        indices = indices[:_MAX_SAMPLES]
    logger.info(f"Evaluating {len(indices)} samples")
    work_items = [(samples[i], i) for i in indices]
    return work_items
