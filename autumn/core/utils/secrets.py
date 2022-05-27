"""
Utilities for managing secret data that is kept under source control.
"""
import glob
import hashlib
import json
import logging
import os

import pyAesCrypt

from autumn.settings import PROJECTS_PATH, DATA_PATH

SALT = "ElvHoxZ83kAUgqy9x2KqM"
BUFFER_SIZE = 64 * 1024
HASH_FILE = os.path.join(DATA_PATH, "secret-hashes.json")

logger = logging.getLogger(__name__)


def read(password: str):
    """
    Decrypt all secrets into secret files.
    """
    check_password(password)
    encrypt_dirs = [DATA_PATH, PROJECTS_PATH]
    for encrypt_dir in encrypt_dirs:
        encrypt_glob = os.path.join(encrypt_dir, "**", "*.encrypted.*")
        encrypt_paths = glob.glob(encrypt_glob, recursive=True)
        num_files = len(encrypt_paths)
        logger.info("Decrypting %s files from %s", num_files, encrypt_dir)
        for path in encrypt_paths:
            encrypt_path = path.replace("\\", "/")
            secret_path = encrypt_path.replace(".encrypted.", ".secret.")
            logger.info("\tDecrypting %s", encrypt_path)
            pyAesCrypt.decryptFile(encrypt_path, secret_path, password, BUFFER_SIZE)
            check_hash(secret_path)

    logger.info("Finished decrypting %s files", num_files)


def write(file_path: str, password: str):
    """
    Decrypt all secrets into secret files.
    """
    check_password(password)
    assert (
        ".secret." in file_path
    ), "Can only encrypt files that are marked as secret with *.secret.*"
    set_hash(file_path)
    encrypt_path = file_path.replace(".secret.", ".encrypted.").replace("\\", "/")
    pyAesCrypt.encryptFile(file_path, encrypt_path, password, BUFFER_SIZE)
    check_hash(file_path)


def check_password(password: str):
    password_hash = get_str_hash(password + SALT)
    with open(HASH_FILE, "r") as f:
        hashes = json.load(f)

    assert hashes["password"] == password_hash, f"Wrong password."


def set_hash(file_path: str):
    """
    Check that a secret file is the latest version.
    """
    assert (
        ".secret." in file_path
    ), "Can only set the hash for files that are marked as secret with *.secret.*"
    pass
    with open(HASH_FILE, "r") as f:
        hashes = json.load(f)

    hashes = {k: v for k, v in hashes.items() if os.path.exists(k) or k == "password"}
    fp_key = os.path.relpath(file_path).replace("\\", "/")
    hashes[fp_key] = get_file_hash(file_path)

    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f)


def check_hash(file_path: str):
    """
    Check that a secret file is the latest version.
    """
    assert (
        ".secret." in file_path
    ), "Can only check the hash files that are marked as secret with *.secret.*"
    with open(HASH_FILE, "r") as f:
        hashes = json.load(f)

    file_hash = get_file_hash(file_path)
    fp_key = os.path.relpath(file_path).replace("\\", "/")
    if fp_key in hashes:
        assert (
            hashes[fp_key] == file_hash
        ), f"Secret file {fp_key} is not using the latest data, try re-reading encrypted files."


def get_file_hash(file_path: str):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        return hashlib.sha256(file_bytes).hexdigest()


def get_str_hash(s: str):
    s_bytes = s.encode("utf-8")
    return hashlib.sha256(s_bytes).hexdigest()
