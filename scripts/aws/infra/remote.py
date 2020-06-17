import os
import subprocess

from . import settings

SSH_ARGS = f"-o StrictHostKeyChecking=no -i ~/.ssh/{settings.EC2_KEYFILE}"
AWS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip}", shell=True)


def ssh_run_job(instance: dict, script_name: str, script_args):
    """
    Copy and run a script on a given instance.
    TODO: Pipe stdout to logs as well as console.
    """
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    run_script_path = os.path.join(AWS_DIR, "tasks", script_name)
    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Could not find {run_script_path}")

    subprocess.call(
        f"scp {SSH_ARGS} {run_script_path} ubuntu@{ip}:/home/ubuntu/{script_name}", shell=True,
    )
    args = " ".join([str(a) for a in script_args])
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/{script_name} {args}'", shell=True)
