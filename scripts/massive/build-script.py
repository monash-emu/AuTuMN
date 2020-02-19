"""
Uses config.yml to build a Slurm batch script
"""
import os
import sys

import yaml

job_name = sys.argv[1]
job_dir = sys.argv[2]

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

batch_kwargs = {
    "job_name": job_name,
    "job_dir": job_dir,
    "array_end": config["num_jobs"] - 1,
    "num_tasks": config["num_tasks_per_job"],
    "cores_per_task": config["cores_per_task"],
    "mem_per_cpu": config["mem_per_cpu"],
    "runtime": config["runtime"],
    "notification_email": config["notification_email"],
    "job_command": config["job_command"],
}

with open("batch.template.sh") as f:
    script_template = f.read()

script_text = script_template.format(**batch_kwargs)

with open("batch.sh", "w") as f:
    f.write(script_text)
