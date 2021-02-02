from apps import covid_19, tuberculosis, tuberculosis_strains
from utils.runs import read_run_id

APP_MAP = {
    "covid_19": covid_19,
    "tuberculosis": tuberculosis,
    "tuberculosis_strains": tuberculosis_strains,
}


def get_app_region(run_id: str):
    app_name, region_name, _, _ = read_run_id(run_id)
    app_module = APP_MAP[app_name]
    return app_module.app.get_region(region_name)
