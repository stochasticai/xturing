from pathlib import Path

import click

from xturing import BaseModel


@click.command(name="chat")
@click.option("-m", "--model_name_or_path")
def chat_command(model_name_or_path: str):
    wrapped_model_path = Path(model_name_or_path)

    click.secho("[*] Loading your model...", fg="blue", bold=True)

    if wrapped_model_path.is_dir():
        model = BaseModel.load(str(wrapped_model_path))
    else:
        try:
            model = BaseModel.create(model_name_or_path)
        except:
            click.secho(
                f"[-] The model_name_or_path you have provided {model_name_or_path} is not valid",
                fg="red",
                bold=True,
            )
            return

    click.secho(
        "[+] Model loaded successfully. Happy chatting!\n\n", fg="green", bold=True
    )

    while True:
        user_input = input("USER > ")
        model_output = model.generate(texts=[user_input])[0]
        print("MODEL > {}".format(model_output))
