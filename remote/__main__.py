"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m remote --help

"""
import logging

import click

# Configure logging for the Boto3 library
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)

from .aws.cli import aws_cli
from .buildkite.cli import buildkite_cli
from .download.cli import download_cli


@click.group()
def cli():
    """Remote tasks CLI"""


cli.add_command(aws_cli, "aws")
cli.add_command(buildkite_cli, "buildkite")
cli.add_command(download_cli, "download")
cli()
