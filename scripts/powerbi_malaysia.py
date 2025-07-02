
from autumn.infrastructure.tasks import powerbi

run_id = "sm_sir/malaysia/02072025/shortrun"

powerbi.powerbi_task(run_id, "mle", quiet=False, store=powerbi.StorageMode.LOCAL)
