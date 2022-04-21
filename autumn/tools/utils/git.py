import git

from autumn.settings.folders import BASE_PATH


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
