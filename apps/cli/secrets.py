from getpass import getpass

import click

from autumn import secrets as secrets_module


@click.group()
def secrets():
    """Reading and writing of secrets"""


@secrets.command("read")
@click.option("--password", type=str, default="")
def read_secrets(password: str):
    """
    Decrypt all secrets into secret files.
    """
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    secrets_module.read(password)


@secrets.command("write")
@click.argument("file_path", type=str)
@click.option("--password", type=str, default="")
def write_secret(file_path: str, password: str):
    """
    Encrypt a secret
    """
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    secrets_module.write(file_path, password)
