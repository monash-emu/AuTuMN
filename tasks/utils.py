import os

import boto3
import luigi
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ProfileNotFound
from luigi.contrib.s3 import S3Target

from . import settings

S3_UPLOAD_EXTRA_ARGS = {"ACL": "public-read"}
S3_UPLOAD_CONFIG = TransferConfig(
    multipart_threshold=1024 * 25,
    max_concurrency=10,
    multipart_chunksize=1024 * 25,
    use_threads=True,
)

try:
    session = boto3.session.Session(
        region_name=settings.AWS_REGION, profile_name=settings.AWS_PROFILE
    )
except ProfileNotFound:
    session = boto3.session.Session(
        region_name=settings.AWS_REGION,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

s3 = session.client("s3")


class BuildLocalDirectoryTask(luigi.Task):
    """
    Creates an output directory with the specified name
    """

    dirname = luigi.Parameter()
    base_path = luigi.Parameter()

    def get_dirpath(self):
        return os.path.join(self.base_path, self.dirname)

    def output(self):
        return luigi.LocalTarget(self.get_dirpath())

    def run(self):
        os.makedirs(self.get_dirpath(), exist_ok=True)


class UploadFileS3Task(luigi.Task):
    """
    Uploads a file or folder to S3
    """

    src_path = luigi.Parameter()
    dest_key = luigi.Parameter()
    bucket = luigi.Parameter()

    def output(self):
        return S3Target(self.get_s3_uri())

    def run(self):
        if os.path.isfile(self.src_path):
            upload_file_s3(self.src_path, self.dest_key)
        elif os.path.isdir(self.src_path):
            upload_folder_s3(folder_name, dest_folder_key)
        else:
            raise ValueError(f"Path is not a file or folder {self.src_path}")

    def get_s3_uri(self):
        return os.path.join(f"s3://{self.bucket}", self.dest_key)


def upload_file_s3(src_path, dest_key):
    s3.upload_file(
        src_path,
        settings.S3_BUCKET,
        dest_key,
        ExtraArgs=S3_UPLOAD_EXTRA_ARGS,
        Config=S3_UPLOAD_CONFIG,
    )


def upload_folder_s3(folder_name, dest_folder_key):
    # TODO
    pass


def upload_logs_on_failure(task, exception):
    # TODO add "fail" to filename
    # task.run_id?
    # return upload_folder_s3(folder_name, dest_folder_key)
    pass
