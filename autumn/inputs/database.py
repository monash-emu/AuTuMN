import os

from autumn.tool_kit import Timer
from autumn.db import Database
from autumn import constants

from .mobility.preprocess import preprocess_mobility
from .social_mixing.preprocess import preprocess_social_mixing
from .demography.preprocess import preprocess_demography

_input_db = None
input_db_hash_path = os.path.join(constants.INPUT_DATA_PATH, "inputs-hash.txt")
input_db_path = os.path.join(constants.INPUT_DATA_PATH, "inputs.db")


def get_input_db():
    global _input_db
    if _input_db:
        return _input_db
    else:
        _input_db = build_input_database()
        return _input_db


def build_input_database(force: bool = False, rebuild: bool = False):
    """
    Builds the input database from scratch.
    If force is True, build the database from scratch and ignore any previous hashes.
    If force is False, do not build if it already exists, 
    and crash if the built database hash does not match.

    If rebuild is True, then we force rebuild the database, but we don't write a new hash.

    Returns a Database, representing the input database.
    """
    if os.path.exists(input_db_path) and not (force or rebuild):
        input_db = Database(input_db_path)
    else:
        print("Building a new database.")
        input_db = Database(input_db_path)
        with Timer("Deleting all existing data."):
            input_db.delete_everything()

        with Timer("Ingesting demography data."):
            country_df = preprocess_demography(input_db)

        with Timer("Ingesting social mixing data."):
            preprocess_social_mixing(input_db, country_df)

        with Timer("Ingesting mobility data."):
            preprocess_mobility(input_db, country_df)

    current_db_hash = input_db.get_hash()
    if force:
        # Write the file hash
        write_file_hash(current_db_hash, input_db_hash_path)
    else:
        # Read the file hash and compare
        saved_db_hash = read_file_hash(input_db_hash_path)
        is_hash_mismatch = current_db_hash != saved_db_hash
        if rebuild and is_hash_mismatch:
            msg = "Input database does not match canonical version."
            raise ValueError(msg)
        elif is_hash_mismatch:
            print("Hash mismatch, try rebuilding database...")
            build_input_database(rebuild=True)

    return input_db


def read_file_hash(hash_path: str):
    """
    Read file hash, which is on the last line of the file
    """
    try:
        with open(hash_path, "r") as f:
            file_hash = [l for l in f.readlines() if l][-1].strip()

    except FileNotFoundError:
        msg = "No input database hash found, rebuild the input db with --force"
        raise FileNotFoundError(msg)

    return file_hash


def write_file_hash(file_hash: str, hash_path: str):
    """
    Write file hash to a text file, so it can be read later
    """
    text = [
        "# INPUT DATABASE FILE HASH",
        "# This is a MD5 hash of the canonical input database",
        "# If your input db has a different hash, something is wrong",
        "# The database, and this file, is managed by `build_input_database`",
        "# You can build a new input database using `python -m apps db build --help`",
        "# Ensure you add this file to Git if you make changes to the database",
        file_hash,
    ]
    with open(hash_path, "w") as f:
        f.write("\n".join(text) + "\n")

