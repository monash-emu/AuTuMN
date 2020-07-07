import os
import sys
import subprocess
import logging
import time

from fabric import Connection

from . import settings

logger = logging.getLogger(__name__)


CODE_PATH = "/home/ubuntu/code"


def run_powerbi(instance, run_id: str):
    """Run PowerBI processing on the remote server"""
    run_id = "test"  # FIXME: Debug
    logger.info("Running PowerBI processing for run %s", run_id)
    with get_connection(instance) as conn:
        update_repo(conn, branch="luigi-redux")
        install_requirements(conn)
        start_luigi_scheduler(conn, instance)
        # with conn.cd(CODE_PATH):
        #     cmd_str = (
        #         "./env/bin/python -m luigi"
        #         " --module tasks"
        #         " RunCalibrate"
        #         f" --run-id {run_id}"
        #         f" --num-chains {num_chains}"
        #         f" --workers {num_chains}"
        #         f" --CalibrationChainTask-model-name {model_name}"
        #         f" --CalibrationChainTask-runtime {runtime}"
        #         " --logging-conf-file tasks/luigi-logging.ini"
        #     )
        #     conn.run(cmd_str, echo=True, pty=True)
        #     logger.info("Calibration chains finished.")
        #     # Upload Luigi log files
        #     logger.info("Uploading Luigi log files.")
        #     src = "/home/ubuntu/code/data/outputs/luigid/luigi-server.log"
        #     dest = f"{run_id}/logs/powerbi/luigi-worker.log"
        #     copy_s3(conn, src, dest)
        #     src = "/home/ubuntu/code/data/outputs/remote/luigi-worker.log"
        #     dest = f"{run_id}/logs/powerbi/luigi-server.log"
        #     copy_s3(conn, src, dest)

        logger.info("PowerBI processing completed for %s", run_id)


def run_full_model(instance, run_id: str, burn_in: int, use_latest_code: bool):
    """Run full model job on the remote server"""
    run_id = "test"  # FIXME: Debug
    logger.info("Running full models for run %s with %s burn-in.", run_id, burn_in)
    with get_connection(instance) as conn:
        if use_latest_code:
            update_repo(conn, branch="luigi-redux")
        else:
            set_run_id(conn, run_id)

        install_requirements(conn)
        start_luigi_scheduler(conn, instance)
        # with conn.cd(CODE_PATH):
        #     cmd_str = (
        #         "./env/bin/python -m luigi"
        #         " --module tasks"
        #         " RunCalibrate"
        #         f" --run-id {run_id}"
        #         f" --num-chains {num_chains}"
        #         f" --workers {num_chains}"
        #         f" --CalibrationChainTask-model-name {model_name}"
        #         f" --CalibrationChainTask-runtime {runtime}"
        #         " --logging-conf-file tasks/luigi-logging.ini"
        #     )
        #     conn.run(cmd_str, echo=True, pty=True)
        #     logger.info("Calibration chains finished.")
        #     # Upload Luigi log files
        #     logger.info("Uploading Luigi log files.")
        #     src = "/home/ubuntu/code/data/outputs/luigid/luigi-server.log"
        #     dest = f"{run_id}/logs/full/luigi-worker.log"
        #     copy_s3(conn, src, dest)
        #     src = "/home/ubuntu/code/data/outputs/remote/luigi-worker.log"
        #     dest = f"{run_id}/logs/full/luigi-server.log"
        #     copy_s3(conn, src, dest)

        logger.info("Full model runs completed for %s", run_id)


def run_calibration(
    instance, model_name: str, num_chains: int, runtime: int, branch: str,
):
    """Run calibration job on the remote server"""
    with get_connection(instance) as conn:
        update_repo(conn, branch=branch)
        install_requirements(conn)
        start_luigi_scheduler(conn, instance)
        run_id = get_run_id(conn, model_name)
        run_id = "test"  # FIXME: Debug
        logger.info(
            "Running calibration %s with %s chains for %s seconds.", model_name, num_chains, runtime
        )
        with conn.cd(CODE_PATH):
            cmd_str = (
                "./env/bin/python -m luigi"
                " --module tasks"
                " RunCalibrate"
                f" --run-id {run_id}"
                f" --num-chains {num_chains}"
                f" --workers {num_chains}"
                f" --CalibrationChainTask-model-name {model_name}"
                f" --CalibrationChainTask-runtime {runtime}"
                " --logging-conf-file tasks/luigi-logging.ini"
            )
            conn.run(cmd_str, echo=True, pty=True)
            logger.info("Calibration chains finished.")
            # Upload Luigi log files
            logger.info("Uploading Luigi log files.")
            src = "/home/ubuntu/code/data/outputs/luigid/luigi-server.log"
            dest = f"{run_id}/logs/calibrate/luigi-worker.log"
            copy_s3(conn, src, dest)
            src = "/home/ubuntu/code/data/outputs/remote/luigi-worker.log"
            dest = f"{run_id}/logs/calibrate/luigi-server.log"
            copy_s3(conn, src, dest)

        logger.info("Calibration completed for %s", run_id)


def copy_s3(conn: Connection, src_path: str, dest_key: str):
    conn.run(f"./env/bin/aws s3 cp {src_path} s3://{settings.S3_BUCKET}/{dest_key}", echo=True)


def start_luigi_scheduler(conn: Connection, instance):
    """Start the Luigi scheduling server"""
    ip = instance["ip"]
    url = f"http://{ip}:8082/static/visualiser/index.html"
    logger.info("Starting Luigi scheduling server")
    log_dir = "/home/ubuntu/code/data/outputs/luigid"
    conn.run(f"mkdir -p {log_dir}")
    cmd_str = (
        "/home/ubuntu/code/env/bin/luigid"
        " --background"
        f" --logdir {log_dir}"
        " --address 0.0.0.0"
        " --port 8082"
    )
    conn.sudo(cmd_str, echo=True)
    logger.info("Started Luigi scheduling server")
    logger.info("Luigi server available at %s", url)


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
        conn.run("git fetch --quiet", echo=True)
        conn.run(f"git checkout --quiet {branch}", echo=True)
        conn.run("git pull --quiet", echo=True)
    logger.info("Done updating repo.")


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
