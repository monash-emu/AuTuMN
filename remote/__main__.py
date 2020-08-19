"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m remote --help

"""
import os
import logging
import warnings

import click

# Configure command logging
logging.basicConfig(format="%(asctime)s %(module)s:%(levelname)s: %(message)s", level=logging.INFO)

# Configure logging for the Boto3 library
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)

from .aws import aws
from .buildkite import buildkite


@click.group()
def cli():
    """Remote tasks CLI"""


cli.add_command(aws)
cli.add_command(buildkite)
cli()
