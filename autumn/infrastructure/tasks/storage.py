import shutil
from pathlib import Path, PurePosixPath

from autumn.core.runs import ManagedRun
from autumn.core.utils.s3 import upload_s3


class StorageMode:
    S3 = "s3"
    LOCAL = "local"
    MOCK = "mock"


class MockStorage:
    def store(self, src_path):
        pass


class S3Storage:
    def __init__(self, client, run_id, local_base_path, verbose=False):
        self.client = client
        self.run_id = run_id
        self.local_base_path = local_base_path
        self.verbose = verbose

    def store(self, src_path: Path):

        src_path = Path(src_path)
        rel_path = src_path.relative_to(self.local_base_path)

        src_path_str = str(src_path_str)
        dest_key = str(PurePosixPath(self.run_id) / rel_path)

        upload_s3(self.client, src_path_str, dest_key)


class LocalStorage:
    def __init__(self, run_id, local_base_path):
        self.managed_run = ManagedRun(run_id)
        self.source_base = local_base_path
        self.dest_base = self.managed_run.local_path

    def store(self, src_path: Path):
        src_path = Path(src_path)
        rel_path = src_path.relative_to(self.source_base)
        dest_path = self.dest_base / rel_path
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(src_path, dest_path)
