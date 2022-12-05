import os
import shutil

from pathlib import Path


def recreate_dir(dirpath: str):
    """
    Remove (if exists) and then re-create a directory.
    """
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

    os.makedirs(dirpath)


def ls(path: Path, pattern="*"):
    return list(path.glob(pattern))
