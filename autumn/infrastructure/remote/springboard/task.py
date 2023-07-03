from pathlib import PurePosixPath, Path
from enum import Enum
import json
import sys
from time import time, sleep

import logging
import logging.config
import socket

import s3fs
import cloudpickle

# autumn imports
# Most of these should get refactored away eventually...
from autumn.settings import aws as aws_settings
from autumn.infrastructure.tasks.storage import S3Storage
from autumn.core.utils.s3 import get_s3_client
from autumn.core.utils.parallel import gather_exc_plus

# Multi-library SSH wrapper
from .clients import SSHRunner

# Callback wrappers for script/cpkl task management
from .scripting import process_script, process_dumpbin


class TaskStatus(str, Enum):
    INIT = "INIT"  # Task has been created on S3, but nothing else has happened
    LAUNCHING = "LAUNCHING"  # Any machine provisioning and other setup is happening here
    RUNNING = "RUNNING"  # Task is actively running
    # Final possible states are SUCCESS or FAILURE
    # Anything else is still in progress/hung
    SUCCESS = "SUCCESS"  # Everything went to according to plan
    FAILURE = "FAILURE"  # Something did not go according to plan...


class TaskSpec:
    def __init__(self, run_func, func_kwargs: dict = None):
        """Used to specify wrapped tasks for springboard runners
        Args:
            run_func: Any function taking a TaskBridge as its first argument
            func_kwargs: Optional additional kwargs to run_func
        """
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

    def store_all(self):
        self._storage.store(self.local_base / "output")
        self._storage.store(self.local_base / "log")


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


def autumn_task_entry(run_path: str) -> int:
    """Used to run a cpickled TaskSpec
    This handles S3 persistance/logging/cleanup via
    the TaskBridge interface

    Args:
        run_path: The (S3) run_path of the current task

    Returns:
        Exit code - can be returned to bash callers
    """
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
        task_manager.set_status(TaskStatus.FAILURE)
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
    """This is primary local 'user facing' interface for running remote Springboard tasks
    Typically a task will be created by some other method and a SpringboardTaskRunner returned

    """

    def __init__(self, rinst, run_path, shutdown=True):
        self.sshr = SSHRunner(rinst["ip"])
        self.s3 = S3TaskManager(run_path)
        self.run_path = run_path
        self.cres = None
        self.instance = rinst
        self.shutdown = shutdown
        self._logger = logging.getLogger("springboard")

    def _script_callback(self, script_path):
        self.sshr.ftp.put(script_path, "taskscript.sh")
        return self.sshr.run("chmod +x taskscript.sh")

    def _taskpkl_callback(self, pkl_path):
        # FIXME: This is throwing all kinds of loopy botocore path errors for some users...
        # Let's just leave it for now and try a non-s3fs method later (go back to the autumn helpers?)
        # self.s3.fs.put(str(pkl_path.resolve()), str(self.s3._full_rpath / ".taskmeta/task.cpkl"))
        self.sshr.ftp.put(pkl_path, "task.cpkl")

    def run_script(self, script: str, task_spec=None):
        try:
            chmres = process_script(script, self._script_callback)
            # If we failed to chmod the remote script, something is wrong...
            assert chmres.exit_status == 0

            if task_spec is not None:
                process_dumpbin(task_spec, cloudpickle.dump, self._taskpkl_callback)
        except Exception as e:
            if self.shutdown:
                self._logger.error("Processing remote script failed, shutting down remote machine")
                self.sshr.run("sudo shutdown now")
            raise e

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

    def tail(self, n=10) -> str:
        """Return the output of the linux 'tail' command on the remote machine; ie
        display the last {n} lines of stdio of a running task

        Will fail (raise exception) if the remote machine has terminated

        Args:
            n: Number of lines to

        Returns:
            A string capturing the stdout of the tail command
        """
        return self.sshr.run(f"tail -n {n} iodump").stdout

    def top(self, sort_key="+%CPU") -> str:
        """Return the output of the linux 'top' command on the remote machine,
        sorted by sort_key; defaults to (descending) CPU usage, to sort by (descending) memory, use
        top(sort_key="+%MEM")

        Args:
            sort_key: Key consisting of + or - for descending or ascending, followed by the
                      name of the column

        Returns:
            A string capturing the stdout of the top command
        """
        return self.sshr.run(f"top -b -n 1 -o {sort_key}").stdout

    def wait(self, freq: int = 5, maxtime: int = 60 * 60) -> str:
        """Wait for a task to complete
        Will return the status str of the task on completion,
        or raise a TimeoutError if maxtime exceeded

        Args:
            freq: Polling frequency in seconds (defaults to 5 seconds)
            maxtime: Maximum wait time in seconds (defaults to 1 hour)

        Raises:
            TimeoutError: Maximum wait time exceeded - does not indicate task failure
                          Contains current task status as its only argument

        Returns:
            Status string (one of TaskStatus.SUCCESS or TaskStatus.FAILURE)
        """

        tot_time = 0.0

        cur_status = ""

        while tot_time < maxtime:
            start = time()
            cur_status = self.s3.get_status()
            if cur_status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
                return cur_status
            sleep(freq)
            tot_time += time() - start

        raise TimeoutError(cur_status)


class S3TaskManager:
    """Wrapper for a remote task syncronized via S3
    The STATUS file on S3 is the 'ground truth' status for any given job
    This wrapper only handles sync with S3, and is usually exposed via
    SpringboardTaskRunner which also manages SSH connections for a complete bridge
    """

    def __init__(
        self,
        project_path: str,
        fs: s3fs.S3FileSystem = None,
        bucket: PurePosixPath = PurePosixPath(aws_settings.S3_BUCKET),
    ):
        # s3fs seems to have intermittent trouble accessing files created remotely
        # use_listings_cache might (should?) help...
        fs = fs or s3fs.S3FileSystem(use_listings_cache=False)
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

    def _write_taskdata(self, key, value, mode="w"):
        with self.fs.open(self._remote_taskmeta / key, mode) as f:
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

    def put(self, lpath, rpath, absolute_rpath=False):
        self.fs.put(lpath, self._full_rpath / rpath)
