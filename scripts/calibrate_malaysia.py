
from autumn.infrastructure.tasks import calibrate

runtime = 60.0 * 60 * 14

calibrate.calibrate_task("sm_sir/malaysia/123456/newpriors3", 60000.0, 8, True, calibrate.StorageMode.LOCAL)
