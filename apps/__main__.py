"""
Run the command line interface
"""
import os

# Ensure NumPy only uses 1 thread for matrix multiplication,
# because numpy is stupid and tries to use heaps of threads which is quite wasteful
# and it makes our models run way more slowly.
os.environ["OMP_NUM_THREADS"] = "1"

from apps.cli import cli

cli()