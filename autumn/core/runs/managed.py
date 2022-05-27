from pathlib import Path

from .remote import RemoteRunData
from .utils import website_addr_for_run
from .powerbi import ManagedPowerBI
from .full import ManagedFullRun
from .calibration.managed import ManagedCalibrationRun

from autumn.coreimport db
from autumn.core.utils.display import get_link

class ManagedRun:
    def __init__(self, run_id, s3_client=None):
        self.run_id = run_id
        self.remote = RemoteRunData(run_id, s3_client)
        self.local_path = self.remote.local_path_run
        self.calibration = ManagedCalibrationRun(self)
        self.powerbi = ManagedPowerBI(self)
        self.full_run = ManagedFullRun(self)
        web_addr = website_addr_for_run(self.run_id)
        self.website_link = get_link(web_addr)
        
    def list_local(self, glob_str='*'):
        return list(self.local_path.rglob(glob_str))
    
    def __repr__(self):
        return f"ManagedRun: {self.run_id}"

