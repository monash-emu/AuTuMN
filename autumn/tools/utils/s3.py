import glob
import logging
import os
from pathlib import PurePath

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ProfileNotFound

from autumn import settings

logger = logging.getLogger(__name__)


# AWS S3 upload settings
S3_UPLOAD_EXTRA_ARGS = {"ACL": "public-read"}
S3_UPLOAD_CONFIG = TransferConfig(
    max_concurrency=2, multipart_threshold=(1024**2) * 100  # 100mb
)
S3_DOWNLOAD_CONFIG = TransferConfig(
    max_concurrency=2,
    num_download_attempts=3,
)


def get_s3_client():
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

    return session.client("s3")


def download_from_run_s3(client, run_id: str, src_key: str, quiet: bool, retries=5):
    """
    Download file or folder from the run's AWS S3 directory
    """

    relpath = os.path.relpath(src_key, run_id)
    dest_path = os.path.join(settings.REMOTE_BASE_DIR, relpath)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        os.remove(dest_path)

    download_from_s3(client, src_key, dest_path, quiet, retries)
    return src_key


def download_from_s3(client, src_key: str, dest_path: str, quiet: bool, retries=5):
    if quiet:
        logging.disable(logging.INFO)

    retry_count = 0
    while True:
        try:
            download_s3(client, src_key, dest_path)
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


def upload_to_run_s3(client, run_id: str, src_path: str, quiet: bool):
    """
    Upload file or folder to the run's AWS S3 directory
    """
    if quiet:
        logging.disable(logging.INFO)

    relpath = os.path.relpath(src_path, settings.REMOTE_BASE_DIR)
    dest_key = os.path.join(run_id, relpath)
    upload_s3(client, src_path, dest_key)

    if quiet:
        logging.disable(logging.NOTSET)

    return src_path


def list_s3(client, key_prefix: str, key_suffix: str = None):
    """Returns the item keys in a path in AWS S3"""

    key_prefix = sanitise_path(key_prefix)

    response = client.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=key_prefix)
    if response["KeyCount"] == 0:
        raise KeyError(f"No S3 results for key_prefix {key_prefix}")
    objs = response["Contents"]
    is_truncated = response["IsTruncated"]
    while is_truncated:
        token = response["NextContinuationToken"]
        response = client.list_objects_v2(
            Bucket=settings.S3_BUCKET, Prefix=key_prefix, ContinuationToken=token
        )
        objs += response["Contents"]
        is_truncated = response["IsTruncated"]

    if key_suffix:
        return [o["Key"] for o in objs if o["Key"].endswith(key_suffix)]
    else:
        return [o["Key"] for o in objs]


def download_s3(client, src_key, dest_path):
    """Downloads a file from AWS S3"""
    logger.info("Downloading from %s to %s", src_key, dest_path)
    client.download_file(
        settings.S3_BUCKET, src_key, dest_path, Config=S3_DOWNLOAD_CONFIG
    )


def upload_s3(client, src_path, dest_key):
    """Upload a file or folder to S3"""
    if os.path.isfile(src_path):
        upload_file_s3(client, src_path, dest_key)
    elif os.path.isdir(src_path):
        upload_folder_s3(client, src_path, dest_key)
    else:
        raise ValueError(f"Path is not a file or folder {src_path}")


def upload_folder_s3(client, folder_path, dest_folder_key):
    """Upload a folder to S3"""
    nodes = glob.glob(os.path.join(folder_path, "**", "*"), recursive=True)
    files = [f for f in nodes if os.path.isfile(f)]
    rel_files = [os.path.relpath(f, folder_path) for f in files]
    for rel_filepath in rel_files:
        src_path = os.path.join(folder_path, rel_filepath)
        dest_key = os.path.join(dest_folder_key, rel_filepath)
        upload_file_s3(client, src_path, dest_key)


def upload_file_s3(client, src_path, dest_key, max_retry=5):
    """Upload a file to S3"""
    dest_key = sanitise_path(dest_key)
    logger.info("Uploading from %s to %s", src_path, dest_key)

    # Enforce mime types for common cases
    extra_args = get_mime_args(src_path)

    retry_count = 0
    while True:
        try:
            client.upload_file(
                src_path,
                settings.S3_BUCKET,
                dest_key,
                ExtraArgs=extra_args,
                Config=S3_UPLOAD_CONFIG,
            )
            break
        except Exception:
            # Make sure we capture any issues here...
            retry_count += 1
            if retry_count < max_retry:
                logger.exception(f"Upload to {dest_key} failed, trying again.")
            else:
                logger.error(
                    f"Upload to {dest_key} failed, tried {retry_count} times, still failing."
                )
                raise


MIME_MAP = {
    "png": "image/png",
    "log": "text/plain",
    "txt": "text/plain",
    "yml": "text/x-yaml",
}


def get_mime_args(src_path):
    extension = src_path.split(".")[-1]
    mime_type = MIME_MAP.get(extension)
    if mime_type:
        extra_args = S3_UPLOAD_EXTRA_ARGS.copy()
        extra_args["ContentType"] = mime_type
    else:
        extra_args = S3_UPLOAD_EXTRA_ARGS
    return extra_args


def sanitise_path(path):
    """
    Return a posix path to ensure all s3 paths have forward slashes
    """
    pp = PurePath(path)
    return pp.as_posix()


def update_mle_from_remote_calibration(model: str, region: str, run_id=None):
    """
    Download MLE parameters from s3 and update the relevant project automatically.

    Args:
        model: Model name (e.g. 'covid_19')
        region: Region name
        run_id: Optional run identifier such as 'covid_19/calabarzon/1626941419/5495a75'. If None, the latest run is used.

    """

    s3_client = get_s3_client()

    if run_id is None:
        run_id = find_latest_run_id(model, region, s3_client)

    msg = f'Could not read run ID {run_id}, use format "{{app}}/{{region}}/{{timestamp}}/{{commit}}" - exiting'
    assert len(run_id.split("/")) == 4, msg

    # Define the destination folder
    project_registry_name = registry._PROJECTS[model][region]
    region_subfolder_names = (
        project_registry_name.split(f"autumn.projects.{model}.")[1]
        .split(".project")[0]
        .split(".")
    )
    destination_dir_path = os.path.join(PROJECTS_PATH, model)
    for region_subfolder_name in region_subfolder_names:
        destination_dir_path = os.path.join(destination_dir_path, region_subfolder_name)
    destination_dir_path = os.path.join(destination_dir_path, "params")
    dest_path = os.path.join(destination_dir_path, "mle-params.yml")

    # Download the mle file
    key_prefix = os.path.join(run_id, "data", "calibration_outputs").replace("\\", "/")
    all_mle_files = list_s3(s3_client, key_prefix, key_suffix="mle-params.yml")

    if len(all_mle_files) == 0:
        print(f"WARNING: No MLE file found for {run_id}")
    else:
        s3_key = all_mle_files[0]
        download_from_s3(s3_client, s3_key, dest_path, quiet=True)
        print(f"Updated MLE file for {region}'s {model} model, using {run_id}")


def find_latest_run_id(model, region, s3_client):

    # Workout latest timestamp
    key_prefix = f"{model}/{region}/"
    all_files = list_s3(s3_client, key_prefix)
    all_timestamps = [int(f.split("/")[2]) for f in all_files]
    latest_timestamp = max(all_timestamps)

    # Identify commit id
    key_prefix = f"{model}/{region}/{latest_timestamp}/"
    all_files = list_s3(s3_client, key_prefix)
    all_commits = [f.split("/")[3] for f in all_files]

    n_commits = len(list(numpy.unique(all_commits)))
    msg = f"There must be exactly one run on the server associated with the latest timestamp, found {n_commits}."
    msg += f"Please enter the run_id manually for model {model}, region {region}."
    assert n_commits == 1, msg

    commit = all_commits[0]

    return f"{model}/{region}/{latest_timestamp}/{commit}"
