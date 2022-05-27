import git

from autumn.settings.folders import BASE_PATH
from autumn.core.utils import run_command


def get_latest_commit(branch: str) -> str:
    """Return the SHA of the most recent commit for the given branch
    Note this returns the remote commit (ie what appears on github), so will
    ignore any unpushed local commits

    Args:
        branch (str): The branch to get the commit from

    Returns:
        str: SHA hash of the commit
    """
    repo = git.repo.Repo(BASE_PATH)
    remote_refs = repo.remote().refs
    return remote_refs[branch].commit.hexsha


def get_git_hash() -> str:
    """
    Return the current commit hash, or an empty string.

    """
    return run_command("git rev-parse HEAD").strip()


def get_git_branch() -> str:
    """
    Return the current git branch, or an empty string.

    """
    return run_command("git rev-parse --abbrev-ref HEAD").strip()


def get_git_modified() -> bool:
    """
    Return True if there are (tracked and uncommited) modifications.

    """

    status = run_command("git status --porcelain").split("\n")
    return any([s.startswith(" M") for s in status])
