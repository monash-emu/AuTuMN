"""
Wrappers for handling remote run data (S3)
"""

from pathlib import Path

from typing import List

from autumn import settings
from autumn.core.utils import s3

class RemoteRunData:
    def __init__(self, run_id: str, client=None):
        """Remote (S3) wrapper for a given run_id

        Args:
            run_id (str): AuTuMN run_id string
            client (optional): S3 client object (will be created if not supplied)
        """
        self.run_id = run_id
        
        if client is None:
            client = s3.get_s3_client()
        self.client = client
        
        self.local_path_base = Path(settings.DATA_PATH) / 'outputs/runs'
        self.local_path_run = self.local_path_base / run_id

    def list_contents(self, suffix:str =None) -> List[str]:
        """Return a list of all files for this run
        These can be passed directly into the download method

        Args:
            suffix ([str], optional): Filter output by suffix

        Returns:
            [List[str]]: List of files
        """
        return s3.list_s3(self.client, self.run_id, suffix)
            
    def _get_full_metadata(self):
        """Complete S3 metadata for all objects in this run

        Returns:
            [dict]: Metadata
        """
        return self.client.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=self.run_id)
    
    def download(self, remote_path: str):
        """Download a remote file and place it in the corresponding local path

        Args:
            remote_path (str): Full string of remote file path
        """
        # Strip the filename from the end of the path
        split_path = remote_path.split('/')
        filename = split_path[-1]
        dir_only = '/'.join(split_path[:-1])
        
        local_path = self.local_path_base / dir_only
        local_path.mkdir(parents=True, exist_ok=True)
        
        full_local = local_path.joinpath(filename)
        s3.download_s3(self.client, remote_path, str(full_local))
        
    def __repr__(self):
        return f"RemoteRunData: {self.run_id}"
