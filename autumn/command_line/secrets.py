import os
from getpass import getpass

import click

from autumn.settings import PASSWORD_ENVAR


@click.group()
def secrets():
    """Reading and writing of secrets"""


@secrets.command("read")
def read_secrets():
    """
    Decrypt all secrets into secret files.
    """
    from autumn.tools.utils import secrets as secrets_module

    password = os.environ.get(PASSWORD_ENVAR, "")
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    secrets_module.read(password)


@secrets.command("write")
@click.argument("file_path", type=str)
def write_secret(file_path: str):
    """
    Encrypt a secret
    """
    from autumn.tools.utils import secrets as secrets_module

    password = os.environ.get(PASSWORD_ENVAR, "")
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    secrets_module.write(file_path, password)
