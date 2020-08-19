"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m remote --help

"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import click

import sentry_sdk

from .luigi import luigi

# Setup Sentry error reporting - https://sentry.io/welcome/
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(SENTRY_DSN)


@click.group()
def cli():
    """Remote tasks CLI"""


cli.add_command(luigi)
cli()
