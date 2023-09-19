from typing import Optional

from pathlib import PurePosixPath, Path
from enum import Enum
import json
import sys
from time import time, sleep
from functools import wraps

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
from autumn.core.runs.remote import RemoteRunData

# Multi-library SSH wrapper
from .clients import CommandResult, SSHRunner

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
    def __init__(self, run_func, func_kwargs: Optional[dict] = None):
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

    print("Exiting autumn task runner")

    if success:
        sys.exit(0)
    else:
        sys.exit(255)


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
        self._shutdown = shutdown
        self._logger = logging.getLogger("springboard")

    def _script_callback(self, script_path) -> CommandResult:
        self.sshr.ftp.put(script_path, "taskscript.sh")
        return self.sshr.run("chmod +x taskscript.sh")

    def _taskpkl_callback(self, pkl_path):
        # FIXME: This is throwing all kinds of loopy botocore path errors for some users...
        # Let's just leave it for now and try a non-s3fs method later (go back to the autumn helpers?)
        # self.s3.fs.put(str(pkl_path.resolve()), str(self.s3._full_rpath / ".taskmeta/task.cpkl"))
        self.sshr.ftp.put(pkl_path, "task.cpkl")
        try:
            with open(pkl_path, "rb") as pkl_f:
                self.s3._write_taskdata("task.cpkl", pkl_f.read(), "wb")
        except:
            self._logger.warning("Could not store cpkl on S3, continuing")

    def run_script(self, script: str, task_spec=None):
        try:
            chmres = process_script(script, self._script_callback)
            # If we failed to chmod the remote script, something is wrong...
            assert chmres is not None
            assert chmres.exit_status == 0

            if task_spec is not None:
                process_dumpbin(task_spec, cloudpickle.dump, self._taskpkl_callback)
        except Exception as e:
            if self._shutdown:
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

    def terminate(self):
        """
        Immediately terminate a remote run - no logs or data will be persisted
        """
        return self.sshr.run("sudo shutdown now")

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

    def get_managed_task(self):
        return ManagedTask(self.run_path)


class S3TaskManager:
    """Wrapper for a remote task syncronized via S3
    The STATUS file on S3 is the 'ground truth' status for any given job
    This wrapper only handles sync with S3, and is usually exposed via
    SpringboardTaskRunner which also manages SSH connections for a complete bridge
    """

    def __init__(
        self,
        project_path: str,
        fs: Optional[s3fs.S3FileSystem] = None,
        bucket: PurePosixPath = PurePosixPath(aws_settings.S3_BUCKET),
    ):
        # s3fs seems to have intermittent trouble accessing files created remotely
        # use_listings_cache might (should?) help...
        fs = fs or s3fs.S3FileSystem(use_listings_cache=False)
        self.fs = fs
        self.bucket = bucket
        self.project_path = project_path
        self.remote_path = self.bucket / self.project_path
        self._remote_taskmeta = self.remote_path / ".taskmeta"
        self._status_file = self.remote_path / ".taskmeta" / "STATUS"

    def exists(self):
        """Check if this project exists on S3"""
        return self.fs.exists(self._status_file)

    def _setup_task(self):
        if self.fs.exists(self._status_file):
            cur_status = self.get_status()
            raise FileExistsError(f"Existing task found with status {cur_status}", self.remote_path)

    def read_taskdata(self, key, mode="r"):
        with self.fs.open(self._remote_taskmeta / key, mode) as f:
            return f.read()

    def _write_taskdata(self, key, value, mode="w"):
        with self.fs.open(self._remote_taskmeta / key, mode) as f:
            f.write(value)

    def get_status(self):
        return self.read_taskdata("STATUS").strip("\n")

    def set_status(self, status):
        if isinstance(status, TaskStatus):
            status = status.value
        self._write_taskdata("STATUS", status)

    def set_instance(self, rinst):
        instance_json = json.dumps(rinst, default=str)
        self._write_taskdata("instance.json", instance_json)

    def get_instance(self):
        return json.loads(self.read_taskdata("instance.json"))

    def get_log(self, logtype="task"):
        return self.fs.open(self.remote_path / "log" / f"{logtype}.log", "rb").read().decode()

    def get_iodump(self):
        return self.fs.open(self._remote_taskmeta / "iodump", "rb").read().decode()

    def put(self, lpath, rpath, absolute_rpath=False):
        if not absolute_rpath:
            rpath = self.remote_path / rpath
        self.fs.put(lpath, rpath)


class ManagedTask(S3TaskManager):
    def __init__(
        self,
        project_path: str,
        fs: Optional[s3fs.S3FileSystem] = None,
        bucket: PurePosixPath = PurePosixPath(aws_settings.S3_BUCKET),
    ):
        """

        Args:
            run_path (str): Description
            fs (s3fs.S3FileSystem, optional): Description
            bucket (PurePosixPath, optional): Description
        """

        super().__init__(project_path, fs, bucket)
        self.remote = RemoteTaskStore(project_path, fs, bucket)
        from autumn.settings import DATA_PATH

        local_path_base = Path(DATA_PATH) / "managed" / str(bucket) / project_path

        self.local = LocalStore(local_path_base)
        self._remotedata = RemoteRunData(project_path, local_path_base=local_path_base)

    def download(self, remote_path, recursive=False):
        full_remote = self.remote._ensure_full_path(remote_path)
        rel_path = full_remote.relative_to(self.remote_path)

        full_local = self.local.path / rel_path
        if full_local.exists() and full_local.is_dir():
            full_local = self.local.path

        return self.fs.get(str(full_remote), str(full_local), recursive=recursive)

    def download_all(self):
        return self.download(None, recursive=True)

    def get_runner(self):
        return SpringboardTaskRunner(self.get_instance(), self.project_path)


class LocalStore:
    def __init__(self, base_path):
        self.path = base_path

    def open(self, file, mode="r"):
        file = self._ensure_full_path(file)
        return open(file, mode)

    def _using_root(self, path=None):
        if path is None:
            return False
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, Path):
            if path.parts[0] == self.path.parts[0]:
                return True
        return False

    def _ensure_full_path(self, path=None):
        if path is None:
            return self.path
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, Path):
            if path.parts[0] == self.path.parts[0]:
                return path
            else:
                return self.path / path
        else:
            raise TypeError("Path must be str or Path", path)

    def ls(self, path=None, full=False, recursive=False, **kwargs):
        using_root = self._using_root(path)

        path = self._ensure_full_path(path)
        if recursive:
            results = path.rglob("*", **kwargs)
        else:
            results = path.glob("*", **kwargs)

        if using_root:
            ref_path = path
        else:
            ref_path = self.path

        if not full:
            results = [str(Path(res).relative_to(ref_path)) for res in results]
        else:
            results = [res for res in results]

        return results

    def __truediv__(self, divisor):
        return self.path / divisor


class RemoteTaskStore:
    def __init__(
        self, base_path: Optional[str] = None, fs=None, bucket=PurePosixPath("autumn-data")
    ):
        self.fs = fs or s3fs.S3FileSystem(use_listings_cache=False)
        self.bucket = bucket
        self.cwd = bucket
        if base_path is not None:
            self._set_cwd(self.bucket / base_path)

        self.glob = self._wrap_ensure_path(self.fs.glob, True)
        self.read_text = self._wrap_ensure_path(self.fs.read_text)

    def _set_cwd(self, path):
        if self.fs.exists(path):
            self.cwd = path
        else:
            raise FileNotFoundError(path)

    def _validate_path(self, path=None, as_str=False):
        path = self._ensure_full_path(path)
        if as_str:
            return str(path)
        else:
            return path

    def _ensure_full_path(self, path=None):
        if path is None:
            return self.cwd
        if isinstance(path, str):
            path = PurePosixPath(path)
        if isinstance(path, PurePosixPath):
            if path.parts[0] == str(self.bucket):
                return path
            else:
                return self.cwd / path
        else:
            raise TypeError("Path must be str or PurePosixPath", path)

    def _using_root(self, path=None):
        if path is None:
            return False
        if isinstance(path, str):
            path = PurePosixPath(path)
        if isinstance(path, PurePosixPath):
            if path.parts[0] == str(self.bucket):
                return True
        return False

    def _wrap_ensure_path(self, func, as_str=False):
        @wraps(func)
        def wrapper(path=None, *args, **kwargs):
            path = self._validate_path(path, as_str)
            return func(path, *args, **kwargs)

        return wrapper

    def cd(self, path):
        path = self._ensure_full_path(path)
        self._set_cwd(path)

    def ls(self, path=None, full=False, recursive=False, **kwargs):
        using_root = self._using_root(path)

        path = self._ensure_full_path(path)
        if recursive:
            results = self.fs.glob(str(path / "**"), **kwargs)
        else:
            results = self.fs.ls(path, **kwargs)

        if using_root:
            ref_path = path
        else:
            ref_path = self.cwd

        if not full:
            results = [str(PurePosixPath(res).relative_to(ref_path)) for res in results]

        return results

    def get_managed_task(self, path=None):
        path = self._ensure_full_path(path)
        run_path = "/".join(path.parts[1:])
        mt = ManagedTask(run_path, self.fs, self.bucket)
        if mt.exists():
            return mt
        else:
            raise FileNotFoundError("No task exists", run_path)
