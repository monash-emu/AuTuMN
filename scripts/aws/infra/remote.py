import os
import subprocess

from . import settings

AWS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSH_OPTIONS = {
    "StrictHostKeyChecking": "no",
    # https://superuser.com/questions/522094/how-do-i-resolve-a-ssh-connection-closed-by-remote-host-due-to-inactivity
    "TCPKeepAlive": "yes",
    "ServerAliveInterval": "30",
}
SSH_OPT_STR = " ".join([f"-o {k}={v}" for k, v in SSH_OPTIONS.items()])
SSH_KEY_STR = f"-i ~/.ssh/{settings.EC2_KEYFILE}"
SSH_ARGS = f"{SSH_OPT_STR} {SSH_KEY_STR}"


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    print(f"Starting SSH session with instance {name}.")
    cmd_str = f"ssh {SSH_ARGS} ubuntu@{ip}"
    print("Entering ssh session with:", cmd_str)
    subprocess.call(cmd_str, shell=True)


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

    # Copy script to server
    cmd_str = f"scp {SSH_ARGS} {run_script_path} ubuntu@{ip}:/home/ubuntu/{script_name}"
    print("Uploading script with:", cmd_str)
    subprocess.call(cmd_str, shell=True)

    # Run script
    args = " ".join([str(a) for a in script_args])
    cmd_str = f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/{script_name} {args}'"
    print("Running ssh session with:", cmd_str)
    subprocess.call(cmd_str, shell=True)
