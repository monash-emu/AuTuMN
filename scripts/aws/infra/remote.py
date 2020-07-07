import os
import sys
import subprocess
import logging
import time

from fabric import Connection

from . import settings

logger = logging.getLogger(__name__)


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
CODE_PATH = "/home/ubuntu/code"


def get_conn(instance):
    ip = instance["ip"]
    key_filepath = os.path.expanduser(f"~/.ssh/{settings.EC2_KEYFILE}")
    return Connection(host=ip, user="ubuntu", connect_kwargs={"key_filename": key_filepath},)


def fabric_test(instance):
    with get_conn(instance) as conn:
        update_repo(conn)
        install_requirements(conn)
        run_id = get_run_id(conn, "test")
        set_run_id(conn, run_id)
        # import pdb

        # pdb.set_trace()
        # pass


def set_run_id(conn: Connection, run_id: str):
    """Set git to use the commit for a given run ID"""
    logger.info("Setting up repo using a run id %s", run_id)
    commit = run_id.split("-")[-1]
    with conn.cd(CODE_PATH):
        conn.run("git fetch --quiet", echo=True)
        conn.run(f"git checkout --quiet {commit}", echo=True)

    logger.info("Done updating repo.")


def get_run_id(conn: Connection, job_name: str):
    """Get the run ID for a given job name name"""
    logger.info("Building run id.")
    with conn.cd(CODE_PATH):
        git_branch = conn.run("git rev-parse --abbrev-ref HEAD", hide="out").stdout.strip()
        git_commit = conn.run("git rev-parse HEAD", hide="out").stdout.strip()

    timestamp = int(time.time())
    run_id = f"{job_name}-{timestamp}-{git_branch}-{git_commit}"
    logger.info("Using run id %s", run_id)
    return run_id


def update_repo(conn: Connection, branch: str = "master"):
    """Update remote Git repo to use the latest code"""
    logger.info("Updating git repository to run the latest code.")
    conn.sudo(f"chown -R ubuntu:ubuntu {CODE_PATH}", echo=True)
    with conn.cd(CODE_PATH):
        conn.run(f"git checkout --quiet {branch}", echo=True)
        conn.run("git pull --quiet", echo=True)
    logger.info("Done updating repo.")


def install_requirements(conn: Connection):
    """Install Python requirements on remote server"""
    logger.info("Ensuring latest Python requirements are installed.")
    with conn.cd(CODE_PATH):
        conn.run("./env/bin/pip install --quiet -r requirements.txt", echo=True)
    logger.info("Finished installing requirements.")


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    logger.info(f"Starting SSH session with instance {name}.")
    cmd_str = f"ssh {SSH_ARGS} ubuntu@{ip}"
    logger.info("Entering ssh session with: %s", cmd_str)
    subprocess.call(cmd_str, shell=True)


def ssh_run_job(instance: dict, script_name: str, script_args):
    """
    Copy and run a script on a given instance.
    """
    ip = instance["ip"]
    name = instance["name"]
    logger.info(f"Starting SSH session with instance {name}.")
    run_script_path = os.path.join(AWS_DIR, "tasks", script_name)
    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Could not find {run_script_path}")

    # Copy script to server
    cmd_str = f"scp {SSH_ARGS} {run_script_path} ubuntu@{ip}:/home/ubuntu/{script_name}"
    logger.info("Uploading script with: %s", cmd_str)
    retcode = subprocess.call(cmd_str, shell=True)
    if not retcode == 0:
        logger.error("scp file upload failed with return code %s", retcode)
        sys.exit(retcode)

    # Run script
    args = " ".join([str(a) for a in script_args])
    cmd_str = f"ssh {SSH_ARGS} ubuntu@{ip} 'bash ~/{script_name} {args}'"
    logger.info("Running ssh session with: %s", cmd_str)
    retcode = subprocess.call(cmd_str, shell=True)
    if not retcode == 0:
        logger.error("ssh session failed with return code %s", retcode)
        sys.exit(retcode)
