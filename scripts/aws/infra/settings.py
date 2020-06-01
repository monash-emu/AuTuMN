import secrets


class EC2InstanceType:
    m5_8xlarge = "m5.8xlarge"  # ~60c / h, 32 cores, 128GB RAM
    m5_12xlarge = "m5.12xlarge"  # ~90c / h, 48 cores, 192GB RAM
    m5_16xlarge = "m5.16xlarge"  # ~120c / h, 64 cores, 256GB RAM


EC2_INSTANCE_TYPE_LIMIT = 2
EC2_AMI = "ami-0c11789bddea34b53"
EC2_SPOT_MAX_PRICE = "0.9"
EC2_SECURITY_GROUP = "sg-3d0ecf44"
EC2_LAUNCH_PREFERENCE = [EC2InstanceType.m5_8xlarge]
AWS_REGION = "ap-southeast-2"
