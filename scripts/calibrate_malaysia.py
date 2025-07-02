
from autumn.infrastructure.tasks import calibrate

runtime = 60.0 * 60 * 2

calibrate.calibrate_task("sm_sir/malaysia/02072025/shortrun", runtime, 8, True, calibrate.StorageMode.LOCAL)
