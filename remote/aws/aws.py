import os
import subprocess
from datetime import datetime
from dateutil.tz import tzutc
import logging

import boto3
from botocore.exceptions import ProfileNotFound
import timeago
from tabulate import tabulate

from . import settings

logger = logging.getLogger(__name__)

try:
    session = boto3.session.Session(
        region_name=settings.AWS_REGION, profile_name=settings.AWS_PROFILE
    )
except ProfileNotFound:
    session = boto3.session.Session(
        region_name=settings.AWS_REGION,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


client = session.client("ec2")

DESCRIBE_KEYS = ["InstanceId", "InstanceType", "LaunchTime", "State"]


def get_instance_type(min_cores: int, min_ram: int):
    cores_set = set()
    ram_set = set()
    for instance_type, specs in settings.EC2_INSTANCE_SPECS.items():
        if specs["cores"] > min_cores:
            cores_set.add(instance_type)
        if specs["ram"] > min_ram:
            ram_set.add(instance_type)

    specs_set = cores_set.intersection(ram_set)
    min_price = 10000
    chosen_instance = None
    for instance_type in specs_set:
        price = settings.EC2_INSTANCE_SPECS[instance_type]["price"]
        if price < min_price:
            chosen_instance = instance_type
            min_price = price

    assert chosen_instance, "Could not find an instance to match specs"
    return chosen_instance


def download_s3(s3_key, dest):
    cmd = f"aws --profile {settings.AWS_PROFILE} s3 cp --recursive {s3_key} {dest}"
    subprocess.run(args=[cmd], shell=True, check=True)


class NoInstanceAvailable(Exception):
    pass


def run_job(job_id: str, instance_type=None):
    if not instance_type:
        instance_type = settings.EC2InstanceType.m5_8xlarge

    run_instance(job_id, instance_type)


def stop_job(job_id: str):
    logger.info(f"Stopping EC2 instances running job {job_id}... ")
    instance_ids = [i["InstanceId"] for i in describe_instances() if i["name"] == job_id]
    client.terminate_instances(InstanceIds=instance_ids)
    logger.info("Stop request sent.")


def cleanup_volumes():
    """
    Delete orphaned volumes so we don't pay for them
    """
    volumes = client.describe_volumes()
    volume_ids = [v["VolumeId"] for v in volumes["Volumes"] if v["State"] == "available"]
    for v_id in volume_ids:
        logger.info(f"Deleting orphaned volume {v_id}")
        client.delete_volume(VolumeId=v_id)

    if not volume_ids:
        logger.info("No volumes to delete.")


def cleanup_instances():
    """
    Delete old EC2 instances so we don't pay for them
    """
    instances = describe_instances()
    stop_instance_ids = []
    for i in instances:
        if i["name"] == "buildkite":
            # Don't kill buildkite server
            continue

        launched_time = i["LaunchTime"]
        uptime_delta = datetime.utcnow() - launched_time.replace(tzinfo=None)
        hours = uptime_delta.total_seconds() // 3600
        if hours > settings.EC2_MAX_HOURS:
            launch_time_str = launched_time.isoformat()
            logger.info(
                f"Stopping instance {i['name']} with id {i['InstanceId']} with {hours}h uptime since {launch_time_str}"
            )
            stop_instance_ids.append(i["InstanceId"])

    if stop_instance_ids:
        logger.info("Stopping instance ids %s", stop_instance_ids)
        client.terminate_instances(InstanceIds=stop_instance_ids)
    else:
        logger.info("No instances to stop.")


def run_instance(job_id: str, instance_type: str):
    logger.info(f"Creating EC2 instance {instance_type} for job {job_id}... ")
    client.run_instances(
        MaxCount=1,
        MinCount=1,
        ImageId=settings.EC2_AMI,
        InstanceType=instance_type,
        SecurityGroupIds=[settings.EC2_SECURITY_GROUP],
        IamInstanceProfile={"Name": settings.EC2_IAM_INSTANCE_PROFILE},
        KeyName=settings.EC2_KEYFILE.split(".")[0],
        InstanceInitiatedShutdownBehavior="terminate",
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": settings.EC2_SPOT_MAX_PRICE,
                "SpotInstanceType": "one-time",
            },
        },
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": job_id}]}
        ],
    )
    logger.info("Create request sent.")


def find_instance(name):
    instances = describe_instances()
    for instance in instances:
        if instance["name"] == name:
            return instance


def start_instance(instance):
    name = instance["name"]
    logger.info(f"Starting EC2 instance {name}")
    response = client.start_instances(InstanceIds=[instance["InstanceId"]])


def stop_instance(instance):
    name = instance["name"]
    logger.info(f"Stopping EC2 instance {name}")
    response = client.stop_instances(InstanceIds=[instance["InstanceId"]])
    logger.info(response)


def is_running(instance):
    status = instance["State"]["Name"]
    return status == "running"


def print_status(instances):
    now = datetime.utcnow().replace(tzinfo=tzutc())
    logger.info("EC2 instance statuses\n")
    table_data = [
        [
            i["name"],
            i["InstanceType"],
            i["State"]["Name"],
            i["ip"],
            timeago.format(i["LaunchTime"], now),
        ]
        for i in instances
    ]
    table_str = tabulate(table_data, headers=["Name", "Type", "Status", "IP", "Launched"])
    print(table_str, "\n")


def describe_instances():
    response = client.describe_instances()
    aws_instances = []
    for reservation in response["Reservations"]:
        for aws_instance in reservation["Instances"]:
            aws_instances.append(aws_instance)

    instances = []
    for aws_instance in aws_instances:
        if aws_instance["State"]["Name"] == "terminated":
            continue

        name = ""
        for tag in aws_instance.get("Tags", []):
            if tag["Key"] == "Name":
                name = tag["Value"]

        instance = {}
        instance["name"] = name
        instances.append(instance)
        for k, v in aws_instance.items():
            if k in DESCRIBE_KEYS:
                instance[k] = v

        # Read IP address
        try:
            network_interface = aws_instance["NetworkInterfaces"][0]
            instance["ip"] = network_interface["Association"]["PublicIp"]
        except (KeyError, IndexError):
            instance["ip"] = None

    return instances
