from pathlib import Path

from .remote import RemoteRunData
from .utils import website_addr_for_run
from .powerbi import ManagedPowerBI
from .full import ManagedFullRun
from .calibration.managed import ManagedCalibrationRun

#from autumn.tools import db
from autumn.settings import DATA_PATH
from autumn.tools.utils.display import get_link

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

def get_managed_run(run_id):
    """
    Recommended user function for obtaining a ManagedRun
    """

    return ManagedRun(run_id)

class LocalRunStore:
    def __init__(self, store_base: Path = None):
        """Provides a convenient method for accessing and managing
        locally stored runs via the ManagedRun interface

        Args:
            store_base (Path, optional): Base path of the run store
            Defaults to AuTuMN system run path
        """
        store_base = store_base or Path(DATA_PATH) / 'outputs/runs'
        self.path = store_base

    @property
    def models(self) -> dict[str, Path]:
        models = {p.parts[-1]:p for p in self.path.glob('*')}
        return models

    @property
    def runs(self) -> dict[str, Path]:
        return {'/'.join(p.parts[-4:]):p for p in self.path.glob('*/*/*/*')}

    def get(self, run_id: str):
        return ManagedRun(run_id)

    def create(self, run_id: str, exist_ok: bool = True) -> ManagedRun:
        """Create a directory for run_id and return a ManagedRun
        Will if an existing run exists at this path

        Args:
            run_id: Standard AuTuMN run_id
            exist_ok: Flag for whether an existing run this path is OK

        Returns:
            ManagedRun
        """
        mr = ManagedRun(run_id)
        mr.local_path.mkdir(parents=True, exist_ok=exist_ok)
        (mr.local_path / "logs").mkdir()
        return mr