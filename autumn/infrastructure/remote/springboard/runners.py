from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import PurePosixPath, Path
from enum import Enum
import json
import sys
import inspect

import logging
import logging.config
import socket
import os

import s3fs
import boto3
import cloudpickle

from autumn.settings import aws as aws_settings
from autumn.infrastructure.remote.aws import aws
from autumn.infrastructure.remote.aws import cli
from autumn.infrastructure.tasks.storage import S3Storage
from autumn.core.utils.s3 import get_s3_client
from autumn.core.utils.parallel import gather_exc_plus

from .clients import SSHRunner
from .scripting import process_script, process_dumpbin


@dataclass
class EC2MachineSpec:
    min_cores: int
    min_ram: int
    category: str


class TaskStatus(str, Enum):
    INIT = "INIT"  # Task has been created on S3, but nothing else has happened
    LAUNCHING = "LAUNCHING"  # Any machine provisioning and other setup is happening here
    RUNNING = "RUNNING"  # Task is actively running
    # Final possible states are SUCCESS or FAILED
    # Anything else is still in progress/hung
    SUCCESS = "SUCCESS"  # Everything went to according to plan
    FAILED = "FAILURE"  # Something did not go according to plan...


class S3TaskManager:
    """Wrapper for a remote task syncronized via S3
    The STATUS file on S3 is the 'ground truth' status for any given job
    """

    def __init__(
        self,
        project_path: str,
        fs: s3fs.S3FileSystem = None,
        bucket: PurePosixPath = PurePosixPath(aws_settings.S3_BUCKET),
    ):
        fs = fs or s3fs.S3FileSystem()
        self.fs = fs
        self.bucket = bucket
        self.project_path = project_path
        self._full_rpath = self.bucket / self.project_path
        self._remote_taskmeta = self._full_rpath / ".taskmeta"
        self._status_file = self._full_rpath / ".taskmeta" / "STATUS"

    def exists(self):
        """Check if this project exists on S3"""
        return self.fs.exists(self._status_file)

    def _setup_task(self):
        if self.fs.exists(self._status_file):
            cur_status = self.get_status()
            raise FileExistsError(f"Existing task found with status {cur_status}", self._full_rpath)

    def _read_taskdata(self, key):
        with self.fs.open(self._remote_taskmeta / key, "r") as f:
            return f.read()

    def _write_taskdata(self, key, value):
        with self.fs.open(self._remote_taskmeta / key, "w") as f:
            f.write(value)

    def get_status(self):
        return self._read_taskdata("STATUS").strip("\n")

    def set_status(self, status):
        if isinstance(status, TaskStatus):
            status = status.value
        self._write_taskdata("STATUS", status)

    def set_instance(self, rinst):
        instance_json = json.dumps(rinst, default=str)
        self._write_taskdata("instance.json", instance_json)

    def get_instance(self):
        return json.loads(self._read_taskdata("instance.json"))


def gen_run_name(description: str) -> str:
    """Generate a run name consisting of a timestamp and user supplied description
    This is used as the final segment of a run_path

    Args:
        desc: Short descriptive name; don't use special characters!

    Returns:
        Run name (timestamp + desc)
    """
    t = datetime.now()
    tstr = t.strftime("%Y-%m-%dT%H%M")
    return f"{tstr}-{description}"


def get_autumn_project_run_path(project, region, run_desc) -> str:
    """Generate a full run_path of the format
       projects/{project}/{region}/run_name
       This is the canonical "run_id" that can be used by
       launch_synced_autumn_task, ManagedRun etc

    Args:
        project: Model/application name
        region: Region name
        run_desc: Short description of run - no special characters

    Returns:
        The complete identifying run_path
    """
    run_name = gen_run_name(run_desc)
    return f"projects/{project}/{region}/{run_name}"


class TaskSpec:
    def __init__(self, run_func, func_kwargs: dict = None):
        self.run_func = run_func
        self.func_kwargs = func_kwargs or {}


class TaskBridge:
    def __init__(self, local_base: Path, run_path: str):
        """A simple wrapper that exposes logging and storage to
        remote tasks.  An object of this type will always be
        available as the first argument in wrapped tasks.

        The primary entry points for user code (wrapped tasks) are
        TaskBridge.out_path: The (task-machine local) path whose data will be
                             synced at the end of the run
        TaskBridge.logger:   A python logging.logger whose output is stored with the run

        Args:
            local_base: The 'working directory' which is local to the machine
                        where the task is running
            run_path: Full identifying path to the remote run
        """
        self.out_path = local_base / "output"
        set_rtask_logging_config(local_base / "log")
        self.logger = logging.getLogger()
        self._log_f = local_base / "log/task.log"

        s3_client = get_s3_client()
        self._storage = S3Storage(s3_client, run_path, local_base)

    def sync_logs(self):
        """Can be called by the task during a run to ensure remote log files are up to
        date with current run state
        """
        sys.stdout.flush()
        sys.stderr.flush()
        self._storage.store(self._log_f)


def test_stub_task(bridge: TaskBridge, **kwargs):
    """A simple validation stub, which serves as a reference for writing wrapped tasks

    Args:
        bridge: _description_
    """
    logger = bridge.logger
    logger.info("Test stub entry")
    bridge.sync_logs()

    with open(bridge.out_path / "kwdump.json", "w") as f:
        f.write(json.dumps(kwargs))

    logger.info("Test stub exit")


def launch_synced_autumn_task(
    task_spec, mspec: EC2MachineSpec, run_path, branch="master", verbose=False
):
    # create an S3 task
    # check if it exists already then fail if so

    s3task = S3TaskManager(run_path)
    s3task.setup_task()

    # get a machine
    # confirm that it's running, then set an alarm

    instance_type = aws.get_instance_type(**asdict(mspec))

    instance_name = run_path.split("/")[-1]
    inst_req = aws.run_instance(
        instance_name, instance_type, False, ami_name=aws_settings.EC2_AMI["springboard310"]
    )
    iid = inst_req["Instances"][0]["InstanceId"]

    print(iid)
    state = wait_instance(iid)
    print(state)

    rinst = aws.find_instance_by_id(iid)

    # Store the instance data as part of the run
    s3task.set_instance(rinst)

    # Set a cloudwatch alarm to auto-terminate hanging jobs
    set_cpu_termination_alarm(iid)

    # init the S3 task (call this locally, but have the task init on s3)
    s3task.set_status("LAUNCHING")

    # Grab an SSH runner
    runner = cli.runner.get_runner(rinst)
    # Give it a dummy command to make sure it's initialised
    # retry a few times to give it a chance...
    for retry in range(5):
        try:
            runner.conn.run("")
            break
        except:
            pass

    # conda environment activation preamble
    conda_preamble = (
        'eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate autumn310;'
    )

    cd_autumn = "cd code/autumn;"
    print(f"Checking out autumn {branch}")

    git_res = runner.conn.run(
        f"cd code/autumn; git fetch --quiet; git checkout --quiet {branch}; git pull --quiet"
    )
    if verbose:
        print(git_res)

    print(f"Installing requirements")
    pip_res = runner.conn.run(
        f"{conda_preamble} {cd_autumn} pip install -r requirements/requirements310.txt"
    )
    if verbose:
        print(pip_res)

    # put the cpkl on the remote machine
    # launch the job via ssh with reference to cpkl
    # remote job will update the S3 Status
    # you can check this via the local S3ManagedTask
    # or try to interact with the active EC2 instance via SSH....

    ftp = runner.conn.sftp()

    with open("task.cpkl", "wb") as taskcpkl_f:
        cloudpickle.dump(task_spec, taskcpkl_f)

    ftp.put("task.cpkl", "task.cpkl")

    stdin, stdout, stderr = runner.conn.client.exec_command(
        f"{conda_preamble} python -m autumn tasks springboard --run {run_path} --shutdown"
    )

    return s3task, runner, (stdin, stdout, stderr)


def set_rtask_logging_config(log_path: Path, verbose=False):
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.host = socket.gethostname()
        return record

    logging.setLogRecordFactory(record_factory)

    log_format = "%(asctime)s %(host)s %(levelname)s %(message)s"

    log_path.mkdir(exist_ok=True)

    logfile = log_path / "task.log"
    root_logger = {"level": "INFO", "handlers": ["file"]}
    handlers = {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": logfile,
            "formatter": "app",
            "encoding": "utf-8",
        }
    }
    if verbose:
        root_logger["handlers"].append("stream")
        handlers["stream"] = {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "app",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "root": root_logger,
            "handlers": handlers,
            "formatters": {
                "app": {
                    "format": log_format,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
        }
    )


def autumn_task_entry(run_path):
    local_base = Path().resolve() / "taskdata"
    local_base.mkdir(exist_ok=True)
    (local_base / "output").mkdir(exist_ok=True)
    (local_base / "log").mkdir(exist_ok=True)
    bridge = TaskBridge(local_base, run_path)
    task_manager = S3TaskManager(run_path)

    task_manager.set_status(TaskStatus.RUNNING)
    bridge.logger.info("Running AuTuMN task")

    success = False

    try:
        with open("task.cpkl", "rb") as taskcpkl_f:
            task_spec = cloudpickle.load(taskcpkl_f)

        task_spec.run_func(bridge, **task_spec.func_kwargs)
        bridge.logger.info("Task completed")
        task_manager.set_status(TaskStatus.SUCCESS)
        success = True

    except:
        bridge.logger.error("Task failed - see crash.log")
        gather_exc_plus(local_base / "log" / "crash.log")
        task_manager.set_status(TaskStatus.FAILED)
        success = False
    finally:
        bridge.sync_logs()
        bridge._storage.store(local_base / "output")
        bridge._storage.store(local_base / "log")
        logging.shutdown()

    if success:
        return 0
    else:
        return 255


class SpringboardTaskRunner:
    def __init__(self, rinst, run_path):
        self.sshr = SSHRunner(rinst["ip"])
        self.s3 = S3TaskManager(run_path)
        self.run_path = run_path
        self.cres = None

    def _script_callback(self, script_path):
        self.sshr.ftp.put(script_path, "taskscript.sh")
        return self.sshr.run("chmod +x taskscript.sh")

    def _taskpkl_callback(self, pkl_path):
        self.sshr.ftp.put(pkl_path, "task.cpkl")

    def run_script(self, script: str, task_spec=None):
        chmres = process_script(script, self._script_callback)
        # If we failed to chmod the remote script, something is wrong...
        assert chmres.exit_status == 0

        if task_spec is not None:
            process_dumpbin(task_spec, cloudpickle.dump, self._taskpkl_callback)

        self.cres = cres = self.sshr.run("./taskscript.sh 2>&1 | tee iodump")
        return cres

    def get_log(self, logtype="task"):
        return (
            self.s3.fs.open(f"autumn-data/{self.run_path}/log/{logtype}.log", "rb").read().decode()
        )

    def get_iodump(self):
        return (
            self.s3.fs.open(f"autumn-data/{self.run_path}/.taskmeta/iodump", "rb").read().decode()
        )
