import os
import subprocess

SSH_ARGS = "-o StrictHostKeyChecking=no -i ~/.ssh/wizard.pem"


INFRA_DIR = os.path.dirname(os.path.abspath(__file__))


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip}", shell=True)


def ssh_run_job(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    run_script_path = os.path.join(INFRA_DIR, "run_aws.sh")
    subprocess.call(
        f"scp {SSH_ARGS} {run_script_path} ubuntu@{ip}:/home/ubuntu/run.sh", shell=True
    )
    subprocess.call(f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/run.sh'", shell=True)
