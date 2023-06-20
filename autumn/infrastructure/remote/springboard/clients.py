"""Wrappers for various SSH client libraries;
we use paramiko for the initial connection (mostly for ease of syncing FTP),
then pssh to run the actual task (due to the non-blocking reads)
"""

from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass
from time import sleep

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from pssh.output import HostOutput

import paramiko

from logging import getLogger

logger = getLogger("springboard")


@dataclass
class CommandResult:
    complete: bool
    exit_status: int
    stdout: str
    stderr: str
    host_output: HostOutput

    def refresh(self):
        return read_pssh_output(self.host_output, self)


def get_rsa_key(key_path: str = "springboard_rsa"):
    with open(Path.home() / ".ssh" / key_path) as f:
        return paramiko.RSAKey.from_private_key(f)


def get_paramiko_client(ip_addr: str, retries=5):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client_exception = Exception("SSH connection error")

    for r in range(retries):
        try:
            client.connect(ip_addr, username="ubuntu", pkey=get_rsa_key())
            return client
        except Exception as e:
            logger.warning(f"SSH connection still waiting, retrying {r+1}/{retries}")
            client_exception = e
            sleep(5)

    raise client_exception


def run_cmd_paramiko(client: paramiko.SSHClient, cmd: str) -> CommandResult:
    _, stdout, stderr = client.exec_command(cmd)
    stdout_res = stdout.read().decode()
    stderr_res = stderr.read().decode()
    return CommandResult(True, stdout.channel.recv_exit_status(), stdout_res, stderr_res)


def get_pssh_client(ip_addr: str) -> SSHClient:
    pssh_client = SSHClient(ip_addr, user="ubuntu", pkey="~/.ssh/springboard_rsa")
    return pssh_client


def read_pssh_output(host_output, existing: CommandResult = None) -> CommandResult:
    if existing is not None:
        complete = existing.complete
        if complete:
            return existing

        stderr = existing.stderr
        stdout = existing.stdout
    else:
        complete = False
        stdout = ""
        stderr = ""

    try:
        for line in host_output.stdout:
            stdout += line + "\n"
        for line in host_output.stderr:
            stderr += line + "\n"
        complete = True
    except Timeout:
        pass

    return CommandResult(complete, host_output.exit_code, stdout, stderr, host_output)


class SSHRunner:
    """Entry point for other springboard client code;
    this manages and abstracts the underlying paramiko/pssh connections
    """

    def __init__(self, ip_addr: str):
        self.paramiko = get_paramiko_client(ip_addr)
        self.pssh = get_pssh_client(ip_addr)
        self.ftp = self.paramiko.open_sftp()

    def run(self, cmd, read_timeout=1.0):
        host_out = self.pssh.run_command(cmd, read_timeout=read_timeout)
        return read_pssh_output(host_out)
