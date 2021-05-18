"""
Runs remote tasks
"""
import logging

import click

# Configure logging for the Boto3 library
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)


from autumn.remote.aws.cli import aws_cli
from autumn.remote.buildkite.cli import buildkite_cli
from autumn.remote.download.cli import download_cli


@click.group()
def remote():
    """Remote tasks CLI"""


remote.add_command(aws_cli, "aws")
remote.add_command(buildkite_cli, "buildkite")
remote.add_command(download_cli, "download")
