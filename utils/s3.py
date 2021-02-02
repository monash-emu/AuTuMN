import os
import glob
import logging

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ProfileNotFound

import settings

logger = logging.getLogger(__name__)


# AWS S3 upload settings
S3_UPLOAD_EXTRA_ARGS = {"ACL": "public-read"}
S3_UPLOAD_CONFIG = TransferConfig(
    max_concurrency=2,
)
S3_DOWNLOAD_CONFIG = TransferConfig(
    max_concurrency=2,
    num_download_attempts=3,
)


# Get an AWS S3 client
if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
    # Use keys from environment variable
    session = boto3.session.Session(
        region_name=settings.AWS_REGION,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
else:
    try:
        # Try use profile credentials from ~/.aws/credentials
        session = boto3.session.Session(
            region_name=settings.AWS_REGION, profile_name=settings.AWS_PROFILE
        )
    except ProfileNotFound:
        # Try use IAM role credentials
        session = boto3.session.Session(region_name=settings.AWS_REGION)

s3 = session.client("s3")


def download_from_run_s3(run_id: str, src_key: str, quiet: bool, retries=5):
    """
    Download file or folder from the run's AWS S3 directory
    """

    relpath = os.path.relpath(src_key, run_id)
    dest_path = os.path.join(settings.BASE_DIR, relpath)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        os.remove(dest_path)

    download_from_s3(src_key, dest_path, quiet, retries)
    return src_key


def download_from_s3(src_key: str, dest_path: str, quiet: bool, retries=5):
    if quiet:
        logging.disable(logging.INFO)

    retry_count = 0
    while True:
        try:
            download_s3(src_key, dest_path)
            break
        except Exception:
            retry_count += 1
            if retry_count < retries:
                logger.exception(f"Download to {dest_path} failed, trying again.")
            else:
                logger.error(
                    f"Download to {dest_path} failed, tried {retries} times, still failing."
                )
                raise

    if quiet:
        logging.disable(logging.NOTSET)


def upload_to_run_s3(run_id: str, src_path: str, quiet: bool):
    """
    Upload file or folder to the run's AWS S3 directory
    """
    if quiet:
        logging.disable(logging.INFO)

    relpath = os.path.relpath(src_path, settings.BASE_DIR)
    dest_key = os.path.join(run_id, relpath)
    upload_s3(src_path, dest_key)

    if quiet:
        logging.disable(logging.NOTSET)

    return src_path


def list_s3(key_prefix: str, key_suffix: str):
    """Returns the item keys in a path in AWS S3"""
    response = s3.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=key_prefix)
    objs = response["Contents"]
    is_truncated = response["IsTruncated"]
    while is_truncated:
        token = response["NextContinuationToken"]
        response = s3.list_objects_v2(
            Bucket=settings.S3_BUCKET, Prefix=key_prefix, ContinuationToken=token
        )
        objs += response["Contents"]
        is_truncated = response["IsTruncated"]

    return [o["Key"] for o in objs if o["Key"].endswith(key_suffix)]


def download_s3(src_key, dest_path):
    """Downloads a file from AWS S3"""
    logger.info("Downloading from %s to %s", src_key, dest_path)
    s3.download_file(settings.S3_BUCKET, src_key, dest_path, Config=S3_DOWNLOAD_CONFIG)


def upload_s3(src_path, dest_key):
    """Upload a file or folder to S3"""
    if os.path.isfile(src_path):
        upload_file_s3(src_path, dest_key)
    elif os.path.isdir(src_path):
        upload_folder_s3(src_path, dest_key)
    else:
        raise ValueError(f"Path is not a file or folder {src_path}")


def upload_folder_s3(folder_path, dest_folder_key):
    """Upload a folder to S3"""
    nodes = glob.glob(os.path.join(folder_path, "**", "*"), recursive=True)
    files = [f for f in nodes if os.path.isfile(f)]
    rel_files = [os.path.relpath(f, folder_path) for f in files]
    for rel_filepath in rel_files:
        src_path = os.path.join(folder_path, rel_filepath)
        dest_key = os.path.join(dest_folder_key, rel_filepath)
        upload_file_s3(src_path, dest_key)


def upload_file_s3(src_path, dest_key):
    """Upload a file to S3"""
    logger.info("Uploading from %s to %s", src_path, dest_key)
    s3.upload_file(
        src_path,
        settings.S3_BUCKET,
        dest_key,
        ExtraArgs=S3_UPLOAD_EXTRA_ARGS,
        Config=S3_UPLOAD_CONFIG,
    )
