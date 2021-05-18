import os
import shutil


def recreate_dir(dirpath: str):
    """
    Remove (if exists) and then re-create a directory.
    """
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

    os.makedirs(dirpath)
