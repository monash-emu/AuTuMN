class EC2InstanceState:
    pending = "pending"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    terminated = "terminated"
    shutting_down = "shutting-down"
    rebooting = "rebooting"

    LIVE_STATES = [pending, running, rebooting]
    DEAD_STATES = [stopping, stopped, terminated, shutting_down]

    @staticmethod
    def is_dead(state):
        return state in EC2InstanceState.DEAD_STATES

    @staticmethod
    def is_live(state):
        return state in EC2InstanceState.LIVE_STATES


class EC2InstanceType:
    r5_2xlarge = "r5.2xlarge"
    m5_2xlarge = "m5.2xlarge"
    r5_4xlarge = "r5.4xlarge"
    r5_8xlarge = "r5.8xlarge"
    r5d_8xlarge = "r5d.8xlarge"
    r5a_8xlarge = "r5a.8xlarge"
    r5a_16xlarge = "r5a.16xlarge"
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
    EC2InstanceType.r5_4xlarge: {"cores": 16, "ram": 128, "price": 30},
    EC2InstanceType.r5_8xlarge: {"cores": 32, "ram": 256, "price": 60},
    EC2InstanceType.r5d_8xlarge: {"cores": 32, "ram": 256, "price": 61},
    EC2InstanceType.r5a_8xlarge: {"cores": 32, "ram": 256, "price": 61},
    EC2InstanceType.r5a_16xlarge: {"cores": 32, "ram": 256, "price": 61},
    EC2InstanceType.m5_4xlarge: {"cores": 16, "ram": 64, "price": 30},
    EC2InstanceType.m5_8xlarge: {"cores": 32, "ram": 128, "price": 60},
    EC2InstanceType.c5_9xlarge: {"cores": 36, "ram": 72, "price": 61},
    EC2InstanceType.c5_12xlarge: {"cores": 48, "ram": 96, "price": 80},
    EC2InstanceType.m5_12xlarge: {"cores": 48, "ram": 192, "price": 90},
    EC2InstanceType.m5_16xlarge: {"cores": 64, "ram": 256, "price": 120},
}


AWS_PROFILE = "autumn"
AWS_REGION = "ap-southeast-2"
EC2_SPOT_MAX_PRICE = "1.2"
EC2_AMI = "ami-09c1853274b19da33"
EC2_SECURITY_GROUP = "sg-0b2fe230ac8853538"
EC2_IAM_INSTANCE_PROFILE = "worker-profile"
S3_BUCKET = "autumn-data"
EC2_MAX_HOURS = 24
