import secrets


class EC2InstanceType:
    r5_2xlarge = "r5.2xlarge"
    m5_2xlarge = "m5.2xlarge"
    m5_4xlarge = "m5.4xlarge"
    m5_8xlarge = "m5.8xlarge"
    c5_9xlarge = "c5.9xlarge"
    c5_12xlarge = "c5.12xlarge"
    m5_12xlarge = "m5.12xlarge"
    m5_16xlarge = "m5.16xlarge"


# RAM in GB, price in approx cents per hour.
EC2_INSTANCE_SPECS = {
    EC2InstanceType.r5_2xlarge: {"cores": 8, "ram": 64, "price": 15},
    EC2InstanceType.m5_2xlarge: {"cores": 8, "ram": 32, "price": 15},
    EC2InstanceType.m5_4xlarge: {"cores": 16, "ram": 64, "price": 30},
    EC2InstanceType.m5_8xlarge: {"cores": 32, "ram": 128, "price": 60},
    EC2InstanceType.c5_9xlarge: {"cores": 36, "ram": 72, "price": 61},
    EC2InstanceType.c5_12xlarge: {"cores": 48, "ram": 96, "price": 80},
    EC2InstanceType.m5_12xlarge: {"cores": 48, "ram": 192, "price": 90},
    EC2InstanceType.m5_16xlarge: {"cores": 64, "ram": 256, "price": 120},
}


EC2_INSTANCE_TYPE_LIMIT = 2
EC2_AMI = "ami-060a95ac585c93df7"
EC2_SPOT_MAX_PRICE = "0.9"
EC2_SECURITY_GROUP = "sg-3d0ecf44"
EC2_LAUNCH_PREFERENCE = [EC2InstanceType.m5_8xlarge]
AWS_REGION = "ap-southeast-2"
