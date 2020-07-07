import os
import glob
import logging
from abc import ABC, abstractmethod

import boto3
import luigi
import sentry_sdk
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ProfileNotFound
from luigi.contrib.s3 import S3Target


from . import settings

logger = logging.getLogger(__name__)

# Setup Sentry error reporting - https://sentry.io/welcome/
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(SENTRY_DSN)

# AWS S3 upload settings
S3_UPLOAD_EXTRA_ARGS = {"ACL": "public-read"}
S3_UPLOAD_CONFIG = TransferConfig(
    multipart_threshold=1024 * 25,
    max_concurrency=10,
    multipart_chunksize=1024 * 25,
    use_threads=True,
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


class BaseTask(luigi.Task, ABC):
    """Ensures errors are logged correctly."""

    @abstractmethod
    def safe_run(self):
        pass

    def run(self):
        try:
            self.safe_run()
        except Exception as e:
            logger.exception("Task failed")
            if SENTRY_DSN:
                sentry_sdk.capture_exception()

            raise


class ParallelLoggerTask(luigi.Task, ABC):
    """
    Mixin to allow parallel tasks to log to multiple files in parallel
    """

    run_id = luigi.Parameter()  # Unique run id string

    @abstractmethod
    def get_log_filename(self):
        """Returns the name of the log file for this task"""
        pass

    @abstractmethod
    def safe_run(self):
        pass

    def run(self):
        self.setup_logging()
        try:
            self.safe_run()
        except Exception as e:
            logger.exception("Task failed: %s", self)
            if SENTRY_DSN:
                sentry_sdk.capture_exception()

            raise

    def setup_logging(self):
        """Setup logging for this task, to be called in run()"""
        logfile_path = os.path.join(settings.BASE_DIR, "logs", self.get_log_filename())
        log_format = "%(asctime)s %(module)s:%(levelname)s: %(message)s"
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(logfile_path)
        handler.setFormatter(formatter)
        logger_names = ["tasks", "apps", "autumn", "summer"]
        for logger_name in logger_names:
            _logger = logging.getLogger(logger_name)
            _logger.handlers = []
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

    @staticmethod
    def upload_chain_logs_on_success(task):
        """Ensure log files are uploaded every time the task succeeds"""
        logger.info("Task suceeded, uploading logs: %s", task)
        filename = task.get_log_filename()
        src_path = os.path.join(settings.BASE_DIR, "logs", filename)
        dest_key = os.path.join(task.run_id, "logs", filename.replace(".log", ".success.log"))
        upload_s3(src_path, dest_key)

    @staticmethod
    def upload_chain_logs_on_failure(task, exception):
        """Ensure log files are uploaded even if the task fails"""
        logger.info("Task failed, uploading logs: %s", task)
        filename = task.get_log_filename()
        src_path = os.path.join(settings.BASE_DIR, "logs", filename)
        dest_key = os.path.join(task.run_id, "logs", filename.replace(".log", ".failure.log"))
        upload_s3(src_path, dest_key)


ParallelLoggerTask.event_handler(luigi.Event.SUCCESS)(
    ParallelLoggerTask.upload_chain_logs_on_success
)

ParallelLoggerTask.event_handler(luigi.Event.FAILURE)(
    ParallelLoggerTask.upload_chain_logs_on_failure
)


class BuildLocalDirectoryTask(BaseTask):
    """
    Creates an output directory with the specified name
    """

    dirname = luigi.Parameter()

    def get_dirpath(self):
        return os.path.join(settings.BASE_DIR, self.dirname)

    def output(self):
        return luigi.LocalTarget(self.get_dirpath())

    def safe_run(self):
        os.makedirs(self.get_dirpath(), exist_ok=True)


class UploadFileS3Task(BaseTask, ABC):
    """
    Uploads a file or folder to S3
    """

    run_id = luigi.Parameter()  # Unique run id string

    def output(self):
        return S3Target(self.get_s3_uri())

    def safe_run(self):
        upload_s3(self.get_src_path(), self.get_dest_key())

    @abstractmethod
    def get_src_path(self):
        """Returns the path of the file to upload"""
        pass

    def get_dest_key(self):
        rel_path = os.path.relpath(self.get_src_path(), settings.BASE_DIR)
        return os.path.join(self.run_id, rel_path)

    def get_s3_uri(self):
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", self.get_dest_key())
        return s3_uri


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
