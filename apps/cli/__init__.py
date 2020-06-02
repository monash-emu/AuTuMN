"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import os
import click

import sentry_sdk

from .database import db
from .run import run
from .run_mcmc import run_mcmc
from .calibrate import calibrate

# Setup Sentry error reporting - https://sentry.io/welcome/
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(SENTRY_DSN)


@click.group()
def cli():
    """AuTuMN CLI"""


cli.add_command(run)
cli.add_command(run_mcmc)
cli.add_command(calibrate)
cli.add_command(db)
cli()
