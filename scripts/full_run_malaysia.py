
from autumn.infrastructure.tasks import full

run_id = "sm_sir/malaysia/02072025/shortrun"
burn_in = 200
sample_size = 200

full.full_model_run_task(run_id, burn_in, sample_size, quiet=False, store=full.StorageMode.LOCAL)
