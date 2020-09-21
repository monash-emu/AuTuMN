import time


def read_run_id(run_id: str):
    """Read data from run id"""
    parts = run_id.split("/")
    if len(parts) < 2:
        # It's an old style path
        # central-visayas-1600644750-9fdd80c
        parts = run_id.split("-")
        git_commit = parts[-1]
        timestamp = parts[-2]
        region_name = "-".join(parts[:-2])
        app_name = "covid_19"
    else:
        # It's an new style path
        # covid_19/central-visayas/1600644750/9fdd80c
        app_name = parts[0]
        region_name = parts[1]
        timestamp = parts[2]
        git_commit = parts[3]

    return app_name, region_name, timestamp, git_commit


def build_run_id(app_name: str, region_name: str, git_commit: str):
    timestamp = str(int(time.time()))
    return "/".join([app_name, region_name, timestamp, git_commit])
