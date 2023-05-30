#!/usr/bin/env python
"""
Reads data from AWS S3 and saves in a JSON that can be consumed by a website builder.
"""
import json
import os
import sys

import boto3
from botocore.exceptions import ProfileNotFound

# Do some super sick path hacks to get script to work from command line.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(BASE_DIR)


# from autumn.coreutils.runs import read_run_id
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


BUCKET = "autumn-data"
AWS_PROFILE = "autumn"
AWS_REGION = "ap-southeast-2"
BUCKET_URL = f"https://{BUCKET}.s3-ap-southeast-2.amazonaws.com/"

try:
    session = boto3.session.Session(region_name=AWS_REGION, profile_name=AWS_PROFILE)
except ProfileNotFound:
    session = boto3.session.Session(
        region_name=AWS_REGION,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

client = session.client("s3")


def fetch_all_objects():
    response = client.list_objects_v2(Bucket=BUCKET)
    objs = response["Contents"]
    is_truncated = response["IsTruncated"]
    while is_truncated:
        token = response["NextContinuationToken"]
        response = client.list_objects_v2(Bucket=BUCKET, ContinuationToken=token)
        objs += response["Contents"]
        is_truncated = response["IsTruncated"]

    return objs


def is_website_asset(key):
    return key.endswith(".html")


print("Fetching object list from AWS S3")
objs = fetch_all_objects()
keys = [o["Key"] for o in objs]
print("Found", len(keys), "objects.")
APPS = [
    "covid_19",
    "tuberculosis",
    "sm_sir",
    "hierarchical_sir",
    "sm_covid2",
    "sm_jax",
    "tb_dynamics",
]
apps = {}
reports = {
    "dhhs": {"title": "DHHS", "description": "Weekly report for DHHS", "files": []},
    "ensemble": {
        "title": "Monash Ensemble",
        "description": "Weekly forceast for the Commonwealth ensemble model, collated by Rob Hyndman",
        "files": [],
    },
}


print("Creating data structure...")
for k in keys:
    if is_website_asset(k):
        continue

    is_report = False
    for report in reports.keys():
        if k.startswith(report):
            # Report specific data, not model runs.
            path_parts = k.split("/")
            name = path_parts[-1]
            file = {"filename": name, "url": os.path.join(BUCKET_URL, k)}
            reports[report]["files"].append(file)
            is_report = True
            break

    if is_report:
        continue

    if any([k.startswith(app) for app in APPS]):
        # New storage structure.
        # app/region/timestamp/commit/path
        path_parts = k.split("/")
        run_id = "/".join(path_parts[0:4])
        path = "/".join(path_parts[4:])
    else:
        # Old storage structure.
        # modelname-timestamp-commit/path
        path_parts = k.split("/")
        run_id, path = path_parts[0], "/".join(path_parts[1:])

    try:
        app_name, region_name, timestamp, commit = read_run_id(run_id)
    except Exception as e:
        print(f"Failed to parse run_id {run_id} for key {k}")
        continue

    if not app_name in apps:
        apps[app_name] = {}

    if not region_name in apps[app_name]:
        apps[app_name][region_name] = {}

    uuid = f"{timestamp}-{commit}"
    if not uuid in apps[app_name][region_name]:
        apps[app_name][region_name][uuid] = {
            "id": run_id,
            "app": app_name,
            "region": region_name,
            "timestamp": timestamp,
            "commit": commit,
            "files": [],
        }

    file = {"path": path, "url": os.path.join(BUCKET_URL, run_id, path)}
    apps[app_name][region_name][uuid]["files"].append(file)


data = {
    "apps": apps,
    "reports": reports,
}
output_path = "website.json"
print("Writing website data to", output_path)
with open(output_path, "w") as f:
    json.dump(data, f)
