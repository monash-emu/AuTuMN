#!/usr/bin/env python
"""
Reads data from AWS S3 and saves in a JSON that can be consumed by a website builder.
"""
import os
import json
import sys

import boto3
from botocore.exceptions import ProfileNotFound

# Do some super sick path hacks to get script to work from command line.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(BASE_DIR)

from remote.aws.utils import read_run_id

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
dhhs_files = []
apps = {}
print("Creating data structure...")
for k in keys:
    if is_website_asset(k):
        continue

    if k.startswith("dhhs"):
        # DHHS specific data, not model runs.
        path_parts = k.split("/")
        name = path_parts[-1]
        file = {"filename": name, "url": os.path.join(BUCKET_URL, k)}
        dhhs_files.append(file)
        continue

    if k.startswith("covid_19") or k.startswith("tuberculosis"):
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

    app_name, region_name, timestamp, commit = read_run_id(run_id)
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
    "dhhs": dhhs_files,
}
output_path = "website.json"
print("Writing website data to", output_path)
with open(output_path, "w") as f:
    json.dump(data, f)
