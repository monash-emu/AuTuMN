from enum import Enum


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


class EC2Instance:
    def __init__(self, cores: int, ram: int):
        self.cores = cores
        self.ram = ram

    def __repr__(self):
        return f"{self.cores} cores, {self.ram} Gb RAM"


class EC2InstanceCategory:
    GENERAL = "general"
    MEMORY = "memory"
    COMPUTE = "compute"
    TINY = "tiny"


EC2_INSTANCE_SPECS = {
    EC2InstanceCategory.TINY: {
        "t3.micro": EC2Instance(1, 1),
        "t3.small": EC2Instance(1, 2),
        "t3.medium": EC2Instance(1, 4),
    },
    EC2InstanceCategory.GENERAL: {
        "m6i.large": EC2Instance(1, 4),
        "m6i.xlarge": EC2Instance(2, 8),
        "m6i.2xlarge": EC2Instance(4, 16),
        "m6i.4xlarge": EC2Instance(8, 32),
        "m6i.8xlarge": EC2Instance(16, 64),
        "m6i.12xlarge": EC2Instance(24, 96),
        "m6i.16xlarge": EC2Instance(32, 128),
        "m6i.24xlarge": EC2Instance(48, 192),
        "m6i.32xlarge": EC2Instance(64, 256),
    },
    EC2InstanceCategory.MEMORY: {
        "r6i.large": EC2Instance(1, 16),
        "r6i.xlarge": EC2Instance(2, 32),
        "r6i.2xlarge": EC2Instance(4, 64),
        "r6i.4xlarge": EC2Instance(8, 128),
        "r6i.8xlarge": EC2Instance(16, 256),
        "r6i.12xlarge": EC2Instance(24, 384),
        "r6i.16xlarge": EC2Instance(32, 512),
        "r6i.24xlarge": EC2Instance(48, 768),
        "r6i.32xlarge": EC2Instance(64, 1024),
    },
    EC2InstanceCategory.COMPUTE: {
        "c6i.large": EC2Instance(1, 4),
        "c6i.xlarge": EC2Instance(2, 8),
        "c6i.2xlarge": EC2Instance(4, 16),
        "c6i.4xlarge": EC2Instance(8, 32),
        "c6i.8xlarge": EC2Instance(16, 64),
        "c6i.12xlarge": EC2Instance(24, 96),
        "c6i.16xlarge": EC2Instance(32, 128),
        "c6i.24xlarge": EC2Instance(48, 192),
        "c6i.32xlarge": EC2Instance(64, 256),
    },
}


AWS_PROFILE = "autumn"
AWS_REGION = "ap-southeast-2"
EC2_SPOT_MAX_PRICE = "1.2"
EC2_AMI = {
    "36venv": "ami-0d27c531f813ff1cf",
    "310conda": "ami-0dbd9adffb9f07a62",
    "springboard310": "ami-02ee8f059b67898ee",
}
EC2_SECURITY_GROUP = "sg-0b2fe230ac8853538"
EC2_IAM_INSTANCE_PROFILE = "worker-profile"
S3_BUCKET = "autumn-data"
EC2_MAX_HOURS = 24
