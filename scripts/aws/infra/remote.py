import os
import subprocess

SSH_ARGS = "-o StrictHostKeyChecking=no -i ~/.ssh/wizard.pem"


INFRA_DIR = os.path.dirname(os.path.abspath(__file__))


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip}", shell=True)


def ssh_run_job(instance: dict, script_name: str, script_args):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    run_script_path = os.path.join(INFRA_DIR, script_name)
    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Could not find {run_script_path}")

    subprocess.call(
        f"scp {SSH_ARGS} {run_script_path} ubuntu@{ip}:/home/ubuntu/{script_name}",
        shell=True,
    )
    args = " ".join([str(a) for a in script_args])
    subprocess.call(
        f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/{script_name} {args}'", shell=True
    )
