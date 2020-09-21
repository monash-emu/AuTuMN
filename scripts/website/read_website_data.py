#!/usr/bin/env python
"""
Reads data from AWS S3 and saves in a JSON that can be consumed by a website builder.
"""
import os
import json

import boto3
from botocore.exceptions import ProfileNotFound

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
model_names = set()
dhhs_files = []
runs = {}
print("Creating data structure...")
for k in keys:
    if is_website_asset(k):
        continue

    path_parts = k.split("/")
    run_id, path = path_parts[0], "/".join(path_parts[1:])

    if k.startswith("dhhs"):
        name = path_parts[-1]
        file = {"filename": name, "url": os.path.join(BUCKET_URL, k)}
        dhhs_files.append(file)

    try:
        app_name, region_name, model, timestamp, commit = read_run_id(run_id)
    except Exception:
        continue

    assert False, "Matt needs to fix this."

    model_names.add(model)
    if not model in runs:
        runs[model] = {}

    if not run_id in runs[model]:
        runs[model][run_id] = {
            "id": run_id,
            "app": app_name,
            "region": region_name,
            "timestamp": timestamp,
            "commit": commit,
            "files": [],
        }

    file = {"path": path, "url": os.path.join(BUCKET_URL, run_id, path)}
    runs[model][run_id]["files"].append(file)


data = {
    "models": list(model_names),
    "runs": runs,
    "dhhs": dhhs_files,
}
output_path = "website.json"
print("Writing website data to", output_path)
with open(output_path, "w") as f:
    json.dump(data, f)
