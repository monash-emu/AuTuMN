import logging
import os
import subprocess

from fabric import Connection

from autumn.tools.utils.runs import build_run_id, read_run_id

logger = logging.getLogger(__name__)


CODE_PATH = "/home/ubuntu/code"


def cleanup_builds(instance):
    with get_connection(instance) as conn:
        conn.run("sudo rm -rf /var/lib/buildkite-agent/builds/", echo=True)


def run_powerbi(instance, run_id: str, urunid: str, commit: str):
    """Run PowerBI processing on the remote server"""
    run_id = run_id.lower()
    msg = "Running PowerBI processing for run %s on AWS instance %s"
    logger.info(msg, run_id, instance["InstanceId"])
    with get_connection(instance) as conn:
        print_hostname(conn)
        if commit != "use_original_commit":
            set_repo_to_commit(conn, commit=commit)
        else:
            set_run_id(conn, run_id)
        install_requirements(conn)
        read_secrets(conn)
        pipeline_name = "powerbi"
        pipeline_args = {"run": run_id, "urunid": urunid}
        run_task_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("PowerBI processing completed for %s", run_id)


def run_full_model(
    instance, run_id: str, burn_in: int, sample: int, commit: str
):
    """Run full model job on the remote server"""
    run_id = run_id.lower()
    msg = "Running full models for run %s burn-in %s on AWS instance %s"
    logger.info(msg, run_id, burn_in, instance["InstanceId"])
    with get_connection(instance) as conn:
        print_hostname(conn)
        if commit != "use_original_commit":
            set_repo_to_commit(conn, commit=commit)
        else:
            set_run_id(conn, run_id)

        install_requirements(conn)
        read_secrets(conn)
        pipeline_name = "full"
        pipeline_args = {
            "run": run_id,
            "burn": burn_in,
            "sample": sample,
        }
        run_task_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("Full model runs completed for %s", run_id)


def resume_calibration(
    instance,
    baserun: str,
    num_chains: int,
    runtime: int,
):
    """Resume calibration job on the remote server"""
    msg = "Resuming calibration with %s chains for %s seconds on AWS instance %s."
    logger.info(msg, num_chains, runtime, instance["InstanceId"])
    run_id = None

    app_name, region_name, _, orig_commit = baserun.split('/')

    with get_connection(instance) as conn:
        print_hostname(conn)
        set_run_id(conn, baserun)
        install_requirements(conn)
        read_secrets(conn)
        run_id = get_run_id(conn, app_name, region_name)
        pipeline_name = "resume_calibration"
        pipeline_args = {
            "run": run_id,
            "baserun": baserun,
            "chains": num_chains,
            "runtime": runtime,
        }
        run_task_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("Calibration resume completed for %s", run_id)

    return run_id

def run_calibration(
    instance,
    app_name: str,
    region_name: str,
    num_chains: int,
    runtime: int,
    commit: str,
):
    """Run calibration job on the remote server"""
    msg = "Running calibration %s %s with %s chains for %s seconds on AWS instance %s."
    logger.info(msg, app_name, region_name, num_chains, runtime, instance["InstanceId"])
    run_id = None
    with get_connection(instance) as conn:
        print_hostname(conn)
        set_repo_to_commit(conn, commit=commit)
        install_requirements(conn)
        read_secrets(conn)
        run_id = get_run_id(conn, app_name, region_name)
        pipeline_name = "calibrate"
        pipeline_args = {
            "run": run_id,
            "chains": num_chains,
            "runtime": runtime,
        }
        run_task_pipeline(conn, pipeline_name, pipeline_args)
        logger.info("Calibration completed for %s", run_id)

    return run_id


def run_task_pipeline(conn: Connection, pipeline_name: str, pipeline_args: dict):
    """Run a task pipeline on the remote machine"""
    logger.info("Running task pipeline %s", pipeline_name)
    pipeline_args_str = " ".join([f"--{k} {v}" for k, v in pipeline_args.items()])
    cmd_str = f"./env/bin/python -m autumn tasks {pipeline_name} {pipeline_args_str}"
    with conn.cd(CODE_PATH):
        conn.run(cmd_str, echo=True)

    logger.info("Finished running task pipeline %s", pipeline_name)


LOGS_URL = "https://monashemu.grafana.net/explore?orgId=1&left=%5B%22now-1h%22,%22now%22,%22grafanacloud-monashemu-logs%22,%7B%22expr%22:%22%7Bjob%3D%5C%22app%5C%22%7D%20%7C%3D%20%5C%22${HOSTNAME}%5C%22%22%7D%5D"
METRICS_URL = "https://monashemu.grafana.net/d/SkUUUHyMk/nodes?orgId=1&refresh=30s&var-datasource=grafanacloud-monashemu-prom&var-instance=${HOSTNAME}:12345"


def print_hostname(conn: Connection):
    conn.run('echo "Running on host $HOSTNAME"', echo=True)
    conn.run(f'echo "METRICS: {METRICS_URL}"', echo=True)
    conn.run(f'echo "LOGS: {LOGS_URL}"', echo=True)


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

def set_repo_to_commit(conn: Connection, commit: str):
    """Update remote Git repo to use the specified commit"""
    logger.info(f"Updating git repository to use commit {commit}")
    conn.sudo(f"chown -R ubuntu:ubuntu {CODE_PATH}", echo=True)
    with conn.cd(CODE_PATH):
        conn.run("git fetch --quiet", echo=True)
        conn.run(f"git checkout --quiet {commit}", echo=True)
        # Do a pull here so we can use branch names as commits and reliably update them
        conn.run(f"git pull --quiet", echo=True)
        conn.run(f"git checkout --quiet {commit}", echo=True)
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
    key_filepath = try_get_ssh_key_path(instance["name"])
    return Connection(
        host=ip,
        user="ubuntu",
        connect_kwargs={"key_filename": key_filepath},
    )


SSH_OPTIONS = {
    "StrictHostKeyChecking": "no",
    # https://superuser.com/questions/522094/how-do-i-resolve-a-ssh-connection-closed-by-remote-host-due-to-inactivity
    "TCPKeepAlive": "yes",
    "ServerAliveInterval": "30",
}
SSH_OPT_STR = " ".join([f"-o {k}={v}" for k, v in SSH_OPTIONS.items()])
SSH_KEYS_TO_TRY = ["buildkite", "id_rsa"]


def ssh_interactive(instance):
    ip = instance["ip"]
    name = instance["name"]
    logger.info(f"Starting SSH session with instance {name}.")
    ssh_key_path = try_get_ssh_key_path(name)
    cmd_str = f"ssh {SSH_OPT_STR} -i {ssh_key_path} ubuntu@{ip}"
    logger.info("Entering ssh session with: %s", cmd_str)
    subprocess.call(cmd_str, shell=True)


def try_get_ssh_key_path(name=None):
    keypath = None
    keys_to_try = (
        ["autumn.pem"]
        if name and (name.startswith("buildkite") or name == "website")
        else SSH_KEYS_TO_TRY
    )
    for keyname in keys_to_try:
        keypath = os.path.expanduser(f"~/.ssh/{keyname}")
        if os.path.exists(keypath):
            break

    if not keypath:
        raise FileNotFoundError(
            f"Could not find SSH key at {keypath} or for alternate names {keys_to_try}."
        )

    return keypath
