from typing import List, Dict
from pathlib import Path

from autumn.core.project.project import Project


def get_params_folder(
    model: str,
    country: str,
    region: str,
) -> Path:
    """
    Find the directory to where we want to keep the files for the parameters,
    including add any paths that weren't already present.
    
    Args:
        model: Name of the model type
        country: The country from which the region comes
        region: The region considered
    
    """
    
    base_dir = Path().absolute().parent.parent.parent  # Will need to change this
    projects_dir = base_dir / "docs" / "tex_descriptions" / "projects"
    
    model_dir = projects_dir / model
    model_dir.mkdir(exist_ok=True)
    
    country_dir = model_dir / country
    country_dir.mkdir(exist_ok=True)
    
    app_dir = country_dir / region
    app_dir.mkdir(exist_ok=True)

    return app_dir / "auto_params.tex"


def write_param_table_rows(
    file_name: Path,
    project: Project,
    params_to_write: List[str],
    rationales: Dict[str, str]
):
    """
    Write parameter values to a TeX file in a format that can be incorporated
    into a standard TeX table.
    
    Args:
        params_to_write: The names of the parameters to be written
        
    """
    
    with open(file_name, "w") as tex_file:
        for i_param, param in enumerate(params_to_write):        
            param_name = param.replace("_", " ")
            value = project.param_set.baseline[param]
            rationale = rationales[param] if param in rationales else "pending"

            # Note that for some TeX-related reason, we can't put the \\ on the last line
            line_end = "" if i_param == len(params_to_write) - 1 else " \\\\ \n\hline"

            table_line = f"\n{param_name} & {value} & {rationale}{line_end}"
            tex_file.write(table_line)
