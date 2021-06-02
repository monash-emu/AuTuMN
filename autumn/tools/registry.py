"""
A global registry of projects.

This lives separately from tools.projects so that importing it has no imported dependencies.
"""
_PROJECTS = {}


def register_project(model_name: str, project_name: str, import_path: str):
    if model_name not in _PROJECTS:
        _PROJECTS[model_name] = {}
    if project_name not in _PROJECTS[model_name]:
        _PROJECTS[model_name][project_name] = import_path
    else:
        raise ValueError(f"Project {project_name} using model {model_name} already exists.")


def get_registered_model_names():
    return sorted(list(set(_PROJECTS.keys())))


def get_registered_project_names(model_name=None):
    projects = set()
    if model_name:
        model_names = [model_name]
    else:
        model_names = get_registered_model_names()

    for model_name in model_names:
        for project_name in _PROJECTS[model_name].keys():
            projects.add(project_name)

    return sorted(list(projects))
