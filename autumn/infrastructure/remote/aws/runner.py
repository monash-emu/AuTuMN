import logging
import os
import subprocess

from fabric import Connection

from autumn.core.utils.runs import build_run_id, read_run_id

logger = logging.getLogger(__name__)


def get_runner(instance):
    """Consider this a bit of a hack - will match the type of runner based on ImageId and
    the EC2_AMI table in autumn.settings.aws
    +++FIXME Not working (yet), just return hardcoded Conda310 runner for now
    """

    return CondaRunner(instance)


class SSHRunner:
    _python_preamble = None
    _python_bin = None
    _pip_bin = None
    _requirements = None

    def __init__(self, instance):
        self.instance = instance
        self.conn = get_connection(instance)
        self.code_path = "/home/ubuntu/code"

    def print_hostname(self):
        self.conn.run('echo "Running on host $HOSTNAME"', echo=True)

    def set_repo_to_commit(self, commit: str):
        """Update remote Git repo to use the specified commit"""
        logger.info(f"Updating git repository to use commit {commit}")

        if commit.startswith("branch:"):
            commit = commit.split(":")[-1]
            is_branch = True
        else:
            is_branch = False

        self.conn.sudo(f"chown -R ubuntu:ubuntu {self.code_path}", echo=True)
        with self.conn.cd(self.code_path):
            self.conn.run("git fetch --quiet", echo=True)
            self.conn.run(f"git checkout --quiet {commit}", echo=True)
            if is_branch:
                self.conn.run("git pull --quiet", echo=True)

        logger.info("Done updating repo.")

    def update_repo(self, branch: str = "master"):
        """Update remote Git repo to use the latest code"""
        logger.info("Updating git repository to run the latest code.")
        self.conn.sudo(f"chown -R ubuntu:ubuntu {self.code_path}", echo=True)
        with self.conn.cd(self.code_path):
            self.conn.run("git fetch --quiet", echo=True)
            self.conn.run(f"git checkout --quiet {branch}", echo=True)
            self.conn.run("git pull --quiet", echo=True)
        logger.info("Done updating repo.")

    def set_run_id(self, run_id: str):
        """Set git to use the commit for a given run ID"""
        logger.info("Setting up repo using run id %s", run_id)
        self.conn.sudo(f"chown -R ubuntu:ubuntu {self.code_path}", echo=True)
        _, _, _, commit = read_run_id(run_id)
        with self.conn.cd(self.code_path):
            self.conn.run("git fetch --quiet", echo=True)
            self.conn.run(f"git checkout --quiet {commit}", echo=True)

        logger.info("Done updating repo.")

    def get_run_id(self, app_name: str, region_name: str):
        """Get the run ID for a given job name name"""
        logger.info("Building run id.")
        with self.conn.cd(self.code_path):
            git_commit = self.conn.run("git rev-parse HEAD", hide="out").stdout.strip()

        git_commit = git_commit[:7]
        run_id = build_run_id(app_name, region_name, git_commit)
        logger.info("Using run id %s", run_id)
        return run_id

    def install_requirements(self):
        """Install Python requirements on remote server"""
        logger.info("Ensuring latest Python requirements are installed.")
        with self.conn.cd(self.code_path):
            self.pip(f"install --quiet -r {self._requirements}")
        logger.info("Finished installing requirements.")

    def read_secrets(self):
        """Read any encrypted files"""
        logger.info("Decrypting Autumn secrets.")
        with self.conn.cd(self.code_path):
            self.python("-m autumn secrets read")

    def run_task_pipeline(self, pipeline_name: str, pipeline_args: dict):
        """Run a task pipeline on the remote machine"""
        logger.info("Running task pipeline %s", pipeline_name)
        pipeline_args_str = " ".join([f"--{k} {v}" for k, v in pipeline_args.items()])
        cmd_str = f"-m autumn tasks {pipeline_name} {pipeline_args_str}"
        with self.conn.cd(self.code_path):
            self.python(cmd_str)

        logger.info("Finished running task pipeline %s", pipeline_name)

    def pip(self, command):
        full_str = f"{self._python_preamble} {self._pip_bin} {command}"
        self.conn.run(full_str, echo=True)

    def python(self, command):
        full_str = f"{self._python_preamble} {self._python_bin} {command}"
        self.conn.run(full_str, echo=True)


class VEnvRunner(SSHRunner):
    """
    Run on the original AuTuMN preactivated environment AMI
    Matches pre-refactor behaviour
    """

    def __init__(self, instance):
        super().__init__(instance)
        self._python_preamble = ""
        self._python_bin = "./env/bin/python"
        self._pip_bin = "./env/bin/pip"
        self._requirements = "requirements.txt"


class CondaRunner(SSHRunner):
    """
    Run on a new style Conda AMI with the specified environment
    """

    def __init__(self, instance, conda_env="autumn310"):
        super().__init__(instance)
        self.conda_env = conda_env
        self._python_preamble = f'eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate {self.conda_env};'
        self._python_bin = "python"
        self._pip_bin = "pip"
        self._requirements = "requirements/requirements310.txt"


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
SSH_KEYS_TO_TRY = ["buildkite", "id_rsa", "springboard_rsa"]


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
