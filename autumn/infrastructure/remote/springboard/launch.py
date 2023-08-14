from typing import Dict

from datetime import datetime

# For dumping task metadata
import inspect
import yaml

# Import a few type definitions for clarity in declarations
from .task import TaskSpec, TaskStatus, SpringboardTaskRunner
from .aws import EC2MachineSpec

# Import the rest of the springboard machinery
from . import task
from . import aws
from . import scripting


def launch_synced_autumn_task(
    task_spec: TaskSpec,
    mspec: EC2MachineSpec,
    run_path: str,
    branch: str = "master",
    job_id: str = None,
    set_alarm: bool = True,
    extra_commands: list = None,
    auto_shutdown_time: int = 4 * 60,  # Time in minutes
) -> SpringboardTaskRunner:
    """Launch a task on an EC2 instance

    Args:
        task_spec: The task to run
        mspec: The specifications of an EC2 instance to launch
        run_path: The run_path (or 'run_id') of the launched job
        branch: The branch of autumn to pull, or pass None if not using autumn
        job_id: The EC2 instance descriptor; defaults to using run_path
        set_alarm: Set a cloudwatch alarm to shutdown on inactivity. Defaults to True.
        extra_commands: List of strings to add to bash script (will be run before the python task)
        auto_shutdown_time: Time in minutes to automatically terminate this instance (defaults to 240; ie 4 hours)

    Raises:
        Exception: run_path already exists

    Returns:
        The active task runner
    """
    s3t = task.S3TaskManager(run_path)
    if s3t.exists():
        raise Exception("Task already exists", run_path)

    s3t.set_status(TaskStatus.LAUNCHING)

    if job_id is None:
        job_id = run_path

    rinst = aws.start_ec2_instance(mspec, job_id)
    s3t.set_instance(rinst)

    if set_alarm:
        aws.set_cpu_termination_alarm(rinst["InstanceId"])

    srunner = task.SpringboardTaskRunner(rinst, run_path)

    sdown_res = srunner.sshr.run(f"sudo shutdown -P +{auto_shutdown_time}")

    script = scripting.gen_autumn_run_bash(run_path, branch, extra_commands=extra_commands)

    s3t._write_taskdata("taskscript.sh", script)

    task_spec_meta = {
        "run_func": {
            "name": task_spec.run_func.__name__,
            "module": task_spec.run_func.__module__,
            "source": inspect.getsource(task_spec.run_func),
        },
        "func_kwargs": task_spec.func_kwargs,
    }
    s3t._write_taskdata("task_spec.yml", yaml.dump(task_spec_meta))

    cres = srunner.run_script(script, task_spec)

    return srunner


def launch_synced_multiple_autumn_task(
    task_dict, mspec, branch="master", job_id=None
) -> Dict[str, SpringboardTaskRunner]:
    for run_path in task_dict.keys():
        s3t = task.S3TaskManager(run_path)
        if s3t.exists():
            raise Exception("Task already exists", run_path)
        s3t.set_status(TaskStatus.LAUNCHING)

    if job_id is None:
        job_id = gen_run_name("autumntask")

    instances = aws.start_ec2_multi_instance(mspec, job_id, len(task_dict))

    runners = {}

    for rinst, (run_path, task_spec) in zip(instances, task_dict.items()):
        s3t = task.S3TaskManager(run_path)
        s3t.set_instance(rinst)
        aws.set_cpu_termination_alarm(rinst["InstanceId"])
        srunner = task.SpringboardTaskRunner(rinst, run_path)

        script = scripting.gen_autumn_run_bash(run_path, branch)

        s3t._write_taskdata("taskscript.sh", script)

        task_spec_meta = {
            "run_func": {
                "name": task_spec.run_func.__name__,
                "module": task_spec.run_func.__module__,
                "source": inspect.getsource(task_spec.run_func),
            },
            "func_kwargs": task_spec.func_kwargs,
        }
        s3t._write_taskdata("task_spec.yml", yaml.dump(task_spec_meta))

        cres = srunner.run_script(script, task_spec)

        runners[run_path] = srunner

    return runners


def launch_managed_instance(
    mspec: EC2MachineSpec,
    run_path: str,
    job_id=None,
    set_alarm=True,
) -> SpringboardTaskRunner:
    s3t = task.S3TaskManager(run_path)
    if s3t.exists():
        raise Exception("Task already exists", run_path)

    s3t.set_status(TaskStatus.LAUNCHING)

    if job_id is None:
        job_id = run_path

    rinst = aws.start_ec2_instance(mspec, job_id)
    s3t.set_instance(rinst)

    if set_alarm:
        aws.set_cpu_termination_alarm(rinst["InstanceId"])

    srunner = task.SpringboardTaskRunner(rinst, run_path)

    return srunner


def gen_run_name(description: str) -> str:
    """Generate a run name consisting of a timestamp and user supplied description
    This is used to generate an identifying segment of run_path, but should not
    be used as a run_path on its own

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
