from typing import Optional, List, Union

from dataclasses import dataclass, asdict
from enum import Enum
from time import time

import boto3
from cloudpickle import instance
import numpy as np

# Keep everything we 'borrow' from autumn in this block
# Should eventually be replaced for a free-living and independent springboard
from autumn.infrastructure.remote.aws import aws as autumn_aws
from autumn.settings import aws as aws_settings


class EC2InstanceType(str, Enum):
    """Instance type of requested EC2 machine
    See:
    https://aws.amazon.com/ec2/instance-types/
    """

    GENERAL = "general"
    COMPUTE = "compute"
    MEMORY = "memory"
    TINY = "tiny"


@dataclass
class EC2MachineSpec:
    """Requirements for requested EC2 instances
    category should be one of EC2InstanceType
    """

    min_cores: int
    min_ram: int
    category: str


class InstanceStateError(Exception):
    pass


def wait_instance(instance_id: str, timeout: Optional[float] = None) -> dict:
    """Waits for a given instance to be in a state other than 'pending'
    Used just after launch to wait for an instance to be ready

    Args:
        instance_id: The AWS InstanceID
        timeout: Maximum time to wait, in seconds.  Defaults to no timeout

    Returns:
        The instance dictionary of the running instance (rinst)

    Raises:
        InstanceStateError: The instance entered a state other than running
    """

    if timeout is None:
        timeout = np.inf

    start = time()
    elapsed = 0.0
    state = "pending"
    while state == "pending" and (elapsed < timeout):
        try:
            rinst = autumn_aws.find_instance_by_id(instance_id)
            state = rinst["State"]["Name"]
        except:
            pass
        elapsed = time() - start
    if state == "running":
        return rinst
    else:
        raise InstanceStateError("Instance failed to launch with state", state)


def start_ec2_instance(mspec: EC2MachineSpec, name: str, ami: Optional[str] = None) -> dict:
    """Request and launch an EC2 instance for the given machine specifications

    Args:
        mspec: EC2MachineSpec detailing requirements
        name: The instance name
        ami: The AMI identifier string; defaults to AuTuMN's springboard AMI

    Returns:
        The instance dictionary of the running instance (rinst)
    """
    instance_type = autumn_aws.get_instance_type(**asdict(mspec))

    # +++: Borrow default from AuTuMN; move to springboard rcparams?
    ami = ami or aws_settings.EC2_AMI["springboardtest"]

    inst_req = autumn_aws.run_instance(name, instance_type, False, ami_name=ami)
    iid = inst_req["Instances"][0]["InstanceId"]

    # Will raise exception if instance fails to start
    rinst = wait_instance(iid, 60.0)
    return rinst


def start_ec2_multi_instance(
    mspec: EC2MachineSpec, name: str, n_instances: int, run_group: str, ami: Optional[str] = None
) -> list:
    """Request and launch an EC2 instance for the given machine specifications

    Args:
        mspec: EC2MachineSpec detailing requirements
        name: The instance name
        ami: The AMI identifier string; defaults to AuTuMN's springboard AMI

    Returns:
        The instance dictionary of the running instance (rinst)
    """
    instance_type = autumn_aws.get_instance_type(**asdict(mspec))

    # +++: Borrow default from AuTuMN; move to springboard rcparams?
    ami = ami or aws_settings.EC2_AMI["springboardtest"]

    inst_req = autumn_aws.run_multiple_instances(
        name, instance_type, n_instances, run_group=run_group, ami_name=ami
    )
    iid = inst_req["Instances"][0]["InstanceId"]

    req_instances = inst_req["Instances"]
    instances = []

    for rinst in req_instances:
        iid = rinst["InstanceId"]
        instances.append(wait_instance(iid, 60.0))

    return instances


def set_cpu_termination_alarm(
    instance_id: str, time_minutes: int = 15, min_cpu=5.0, region=aws_settings.AWS_REGION
):
    # Create alarm

    cloudwatch = boto3.client("cloudwatch")

    alarm = cloudwatch.put_metric_alarm(
        AlarmName="CPU_Utilization",
        ComparisonOperator="LessThanThreshold",
        EvaluationPeriods=2,
        MetricName="CPUUtilization",
        Namespace="AWS/EC2",
        Period=time_minutes * 60,
        Statistic="Average",
        Threshold=min_cpu,
        ActionsEnabled=True,
        AlarmDescription=f"Alarm when worker CPU less than {min_cpu}%",
        AlarmActions=[f"arn:aws:automate:{region}:ec2:terminate"],
        Dimensions=[
            {"Name": "InstanceId", "Value": instance_id},
        ],
    )
    return alarm


def get_instances_by_tag(name, value, verbose=False) -> List[dict]:
    import boto3

    ec2client = boto3.client("ec2")
    custom_filter = [{"Name": f"tag:{name}", "Values": [value]}]

    response = ec2client.describe_instances(Filters=custom_filter)
    instances = [[i for i in r["Instances"]] for r in response["Reservations"]]
    instances = [i for r in instances for i in r]
    if verbose:
        return instances
    else:
        concise_instances = [
            {
                "InstanceId": inst["InstanceId"],
                "InstanceType": inst["InstanceType"],
                "LaunchTime": inst["LaunchTime"],
                "State": inst["State"],
                "ip": inst["PublicIpAddress"],
                "tags": {tag["Key"]: tag["Value"] for tag in inst["Tags"]},
            }
            for inst in instances
        ]
        for inst in concise_instances:
            inst["name"] = inst["tags"]["Name"]
        return concise_instances


def set_tag(instance_ids: Union[str, List[str]], key: str, value):
    import boto3

    ec2client = boto3.client("ec2")

    if isinstance(instance_ids, str):
        instance_ids = [instance_ids]

    resp = ec2client.create_tags(Resources=instance_ids, Tags=[{"Key": key, "Value": value}])
    return resp
