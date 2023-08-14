import textwrap
from tempfile import NamedTemporaryFile
from pathlib import Path, PurePosixPath
from typing import List, Callable


def gen_autumn_run_bash(
    run_path: str,
    branch: str,
    shutdown: bool = True,
    bucket: PurePosixPath = PurePosixPath("autumn-data"),
    extra_commands: list = None,
) -> List[str]:
    define_dump_io = f"""
    write_ios_s3 () {{
       aws s3 cp $BASE_PATH/iodump s3://{bucket}/{run_path}/.taskmeta/
    }}\n
    """

    define_dump_io = textwrap.dedent(define_dump_io)

    if extra_commands is None:
        extra_commands = []

    set_base_path = "export BASE_PATH=$PWD"
    cd_base_path = "cd $BASE_PATH"
    cd_autumn = "cd code/autumn;"
    conda_preamble = (
        'eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate autumn310;'
    )

    def emit_status(status):
        return f"echo {status} | aws s3 cp - s3://{bucket}/{run_path}/.taskmeta/STATUS\n"

    def test_to_exit(fail_str):
        shutdown_str = "sudo shutdown now" if shutdown else ""

        testbad = f"""
        if [ $? -ne 0 ]; then
            echo {fail_str} failed, cleaning up
            {emit_status("FAILURE")}
            write_ios_s3
            {shutdown_str}
        fi
        """
        return textwrap.dedent(testbad)

    if branch is None:
        git_script = []
    else:
        git_script = [
            "git fetch",
            test_to_exit("git fetch"),
            f"git checkout {branch}",
            test_to_exit("git checkout"),
            "git pull",
            test_to_exit("git pull"),
            "pip install -r requirements/requirements310.txt",
            test_to_exit("pip install"),
        ]

    script = [
        "#!/bin/bash\n",
        define_dump_io,
        set_base_path,
        cd_autumn,
        conda_preamble,
        # "echo BASHENTRY > STATUS\n",
        # f"aws s3 cp STATUS s3://autumn-data/{run_path}/STATUS\n",
        *git_script,
        cd_base_path,
        *extra_commands,
        f"echo Launching python task on {run_path}",
        f"python -m autumn tasks springboard --run {run_path}",
        test_to_exit("run python task"),
        "echo Python task complete",
        # "echo SUCCESS > STATUS\n",
        # f"aws s3 cp STATUS s3://autumn-data/{run_path}/STATUS\n",
        "write_ios_s3",
    ]

    if shutdown:
        script.append("sudo shutdown now")

    script = [enforce_newline(line) for line in script]
    script = "".join(script)

    return script


def gen_autumn_nbserver(
    run_path: str,
    branch: str,
    shutdown: bool = True,
    bucket: PurePosixPath = PurePosixPath("autumn-data"),
) -> List[str]:
    define_dump_io = f"""
    write_ios_s3 () {{
       aws s3 cp $BASE_PATH/iodump s3://{bucket}/{run_path}/.taskmeta/
    }}\n
    """

    define_dump_io = textwrap.dedent(define_dump_io)

    set_base_path = "export BASE_PATH=$PWD"
    cd_base_path = "cd $BASE_PATH"
    cd_autumn = "cd code/autumn;"
    conda_preamble = (
        'eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate autumn310;'
    )

    def emit_status(status):
        return f"echo {status} | aws s3 cp - s3://{bucket}/{run_path}/.taskmeta/STATUS\n"

    def test_to_exit(fail_str):
        shutdown_str = "sudo shutdown now" if shutdown else ""

        testbad = f"""
        if [ $? -ne 0 ]; then
            echo {fail_str} failed, cleaning up
            {emit_status("FAILURE")}
            write_ios_s3
            {shutdown_str}
        fi
        """
        return textwrap.dedent(testbad)

    if branch is None:
        git_script = []
    else:
        git_script = [
            "git fetch",
            test_to_exit("git fetch"),
            f"git checkout {branch}",
            test_to_exit("git checkout"),
            "git pull",
            test_to_exit("git pull"),
        ]

    script = [
        "#!/bin/bash\n",
        define_dump_io,
        set_base_path,
        cd_autumn,
        conda_preamble,
        # "echo BASHENTRY > STATUS\n",
        # f"aws s3 cp STATUS s3://autumn-data/{run_path}/STATUS\n",
        *git_script,
        # "echo GITCOMPLETE > STATUS\n",
        # f"aws s3 cp STATUS s3://autumn-data/{run_path}/STATUS\n",
        "pip install -r requirements/requirements310.txt",
        test_to_exit("pip install"),
        "pip install notebook",
        test_to_exit("pip install notebook server"),
        cd_base_path,
        f"Starting notebook server for run_id {run_path}",
        f"jupyter nbclassic",
        test_to_exit("launch notebook server"),
        "echo Notebook server exited",
        # "echo SUCCESS > STATUS\n",
        # f"aws s3 cp STATUS s3://autumn-data/{run_path}/STATUS\n",
        "write_ios_s3",
    ]

    if shutdown:
        script.append("sudo shutdown now")

    script = [enforce_newline(line) for line in script]
    script = "".join(script)

    return script


def process_script(script: str, callback: Callable[[Path], None] = None):
    scriptfile = NamedTemporaryFile(mode="w", newline="\n", delete=False)
    scriptfile.write(script)
    scriptfile.close()
    script_path = Path(scriptfile.name)
    if callback is not None:
        callback_res = callback(script_path)
    else:
        callback_res = None
    # ftp.put(scriptfile.name, "testjob.sh")
    # fs.put_file(scriptfile.name, bucket / run_path / "runtask.sh")
    script_path.unlink()
    return callback_res


def process_dumpbin(dumpobj, dump_func, callback: Callable[[Path], None] = None):
    dumpfile = NamedTemporaryFile(mode="wb", delete=False)
    dump_func(dumpobj, dumpfile)
    dumpfile.close()
    dumpf_path = Path(dumpfile.name)
    if callback is not None:
        callback_res = callback(dumpf_path)
    else:
        callback_res = None
    # ftp.put(scriptfile.name, "testjob.sh")
    # fs.put_file(scriptfile.name, bucket / run_path / "runtask.sh")
    dumpf_path.unlink()
    return callback_res


def enforce_newline(line_str) -> str:
    out_str = line_str if line_str.endswith("\n") else line_str + "\n"
    return out_str
