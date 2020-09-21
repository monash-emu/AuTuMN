import os
import sys
import subprocess
import logging

from fabric import Connection

from . import settings
from .utils import build_run_id, read_run_id

logger = logging.getLogger(__name__)


CODE_PATH = "/home/ubuntu/code"


def run_dhhs(instance, commit: str, branch: str):
    """Run DHHS processing on the remote server"""
    msg = "Running DHHS processing for commit %s on AWS instance %s"
    logger.info(msg, commit, instance["InstanceId"])
    with get_connection(instance) as conn:
        update_repo(conn, branch=branch)
        install_requirements(conn)
        read_secrets(conn)
        pipeline_name = "dhhs"
        pipeline_args = {
            "commit": commit,
            "workers": 60,
        }
        run_luigi_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("DHHS processing completed for commit %s", commit)


def run_powerbi(instance, run_id: str, branch: str):
    """Run PowerBI processing on the remote server"""
    run_id = run_id.lower()
    msg = "Running PowerBI processing for run %s on AWS instance %s"
    logger.info(msg, run_id, instance["InstanceId"])
    with get_connection(instance) as conn:
        update_repo(conn, branch=branch)
        install_requirements(conn)
        read_secrets(conn)
        pipeline_name = "powerbi"
        pipeline_args = {
            "run": run_id,
            "workers": 7,
        }
        run_luigi_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("PowerBI processing completed for %s", run_id)


def run_full_model(instance, run_id: str, burn_in: int, use_latest_code: bool, branch: str):
    """Run full model job on the remote server"""
    run_id = run_id.lower()
    msg = "Running full models for run %s burn-in %s on AWS instance %s"
    logger.info(msg, run_id, burn_in, instance["InstanceId"])
    with get_connection(instance) as conn:
        if use_latest_code:
            update_repo(conn, branch=branch)
        else:
            set_run_id(conn, run_id)

        install_requirements(conn)
        read_secrets(conn)
        pipeline_name = "full"
        pipeline_args = {
            "run": run_id,
            "burn": burn_in,
            "workers": 7,
        }
        run_luigi_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("Full model runs completed for %s", run_id)


def run_calibration(
    instance, app_name: str, region_name: str, num_chains: int, runtime: int, branch: str,
):
    """Run calibration job on the remote server"""
    msg = "Running calibration %s %s with %s chains for %s seconds on AWS instance %s."
    logger.info(msg, app_name, region_name, num_chains, runtime, instance["InstanceId"])
    run_id = None
    with get_connection(instance) as conn:
        update_repo(conn, branch=branch)
        install_requirements(conn)
        read_secrets(conn)
        run_id = get_run_id(conn, app_name, region_name)
        pipeline_name = "calibrate"
        pipeline_args = {
            "run": run_id,
            "chains": num_chains,
            "runtime": runtime,
            "workers": num_chains,
        }
        run_luigi_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("Calibration completed for %s", run_id)

    return run_id


def run_luigi_pipeline(conn: Connection, pipeline_name: str, pipeline_args: dict):
    """Run a Luigi pipeline on the remote machine"""
    logger.info("Running Luigi pipleine %s", pipeline_name)
    pipeline_args_str = " ".join([f"--{k} {v}" for k, v in pipeline_args.items()])
    cmd_str = f"./env/bin/python -m tasks {pipeline_name} {pipeline_args_str}"
    with conn.cd(CODE_PATH):
        conn.run(cmd_str, echo=True)

    logger.info("Finished running Luigi pipleine %s", pipeline_name)


def set_run_id(conn: Connection, run_id: str):
    """Set git to use the commit for a given run ID"""
    logger.info("Setting up repo using a run id %s", run_id)
    conn.sudo(f"chown -R ubuntu:ubuntu {CODE_PATH}", echo=True)
    _, _, _, commit = read_run_id(run_id)
    with conn.cd(CODE_PATH):
        conn.run("git fetch --quiet", echo=True)
        conn.run(f"git checkout --quiet {commit}", echo=True)

    logger.info("Done updating repo.")


def get_run_id(conn: Connection, app_name: str, region_name: str):
    """Get the run ID for a given job name name"""
    logger.info("Building run id.")
    with conn.cd(CODE_PATH):
        git_commit = conn.run("git rev-parse HEAD", hide="out").stdout.strip()

    git_commit = git_commit[:7]
    run_id = build_run_id(app_name, region_name, git_commit)
    logger.info("Using run id %s", run_id)
    return run_id


def update_repo(conn: Connection, branch: str = "master"):
    """Update remote Git repo to use the latest code"""
    logger.info("Updating git repository to run the latest code.")
    conn.sudo(f"chown -R ubuntu:ubuntu {CODE_PATH}", echo=True)
    with conn.cd(CODE_PATH):
        conn.run("git fetch --quiet", echo=True)
        conn.run(f"git checkout --quiet {branch}", echo=True)
        conn.run("git pull --quiet", echo=True)
    logger.info("Done updating repo.")


def read_secrets(conn: Connection):
    """Read any encrypted files"""
    logger.info("Decrypting Autumn secrets.")
    with conn.cd(CODE_PATH):
        conn.run("./env/bin/python -m autumn secrets read", echo=True)


def install_requirements(conn: Connection):
    """Install Python requirements on remote server"""
    logger.info("Ensuring latest Python requirements are installed.")
    with conn.cd(CODE_PATH):
        conn.run("./env/bin/pip install --quiet -r requirements.txt", echo=True)
    logger.info("Finished installing requirements.")


def get_connection(instance):
    ip = instance["ip"]
    key_filepath = os.path.expanduser(f"~/.ssh/{settings.EC2_KEYFILE}")
    return Connection(host=ip, user="ubuntu", connect_kwargs={"key_filename": key_filepath},)


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
    logger.info(f"Starting SSH session with instance {name}.")
    cmd_str = f"ssh {SSH_ARGS} ubuntu@{ip}"
    logger.info("Entering ssh session with: %s", cmd_str)
    subprocess.call(cmd_str, shell=True)
