import secrets


class EC2InstanceType:
    m5_8xlarge = "m5.8xlarge"


EC2_INSTANCE_TYPE_LIMIT = 2
EC2_AMI = "ami-0c11789bddea34b53"
EC2_SPOT_MAX_PRICE = "0.9"
EC2_SECURITY_GROUP = "sg-3d0ecf44"
EC2_LAUNCH_PREFERENCE = [EC2InstanceType.m5_8xlarge]
AWS_REGION = "ap-southeast-2"
BUILDKITE_ACCESS_TOKEN = getattr(secrets, "BUILDKITE_ACCESS_TOKEN", "")
