from typing import List, Union
from pathlib import Path
from functools import reduce
import operator

from numpy import isin
from autumn.models.sm_sir.constants import PARAMETER_NAMES, PARAMETER_EXPLANATIONS, PARAMETER_UNITS

from autumn.core.project.project import Project

"""
Should possibly go to ./autumn/core/project/params.py
"""


def get_param_from_nest_string(
    parameters: dict, 
    param_request: str,
) -> Union[int, float, str]:
    """
    Get the value of a parameter from a parameters dictionary, using a single string
    defining the parameter name, with "." characters to separate the tiers of the
    keys in the nested parameter dictionary.
    
    
    Args:
        parameters: The full parameter set to look int
        param_request: The single request submitted by the user
    Return:
        The value of the parameter being requested
    
    """
    
    param_value = reduce(operator.getitem, param_request.split("."), parameters.to_dict())
    msg = "Haven't indexed into single parameter"
    assert not isinstance(param_value, dict), msg
    return param_value


"""
The following functions should possibly replace ./autumn/core/utils/tex_tools/py
But will need to adjust the folder structure from that location
"""


def get_params_folder(
    model: str,
    country: str,
    region: str,
    file_name: str,
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

    return app_dir / f"{file_name}.tex"


def get_param_name(param: str) -> str:
    """
    Simple function to essentially return the appropriate value of the PARAMETER_NAMES dictionary.

    Args:
        param: Name of the parameter of interest, with hierachical keys joined with "."

    Returns:
        The parameter name in an appropriate format to go into a table
    """

    name = PARAMETER_NAMES[param] if param in PARAMETER_NAMES else param.replace("_", " ")
    return name[:1].upper() + name[1:]


def get_param_explanation(param: str) -> str:
    """
    Simple function to essentially return the appropriate value of the PARAMETER_EXPLANATIONS dictionary.

    Args:
        param: Name of the parameter of interest, with hierachical keys joined with "."

    Returns:
        The parameter explanation in an appropriate format to go into a table
    """

    explanation = PARAMETER_EXPLANATIONS[param] if param in PARAMETER_EXPLANATIONS else "assumed"
    return explanation[:1].upper() + explanation[1:]


def format_value_for_tex(project, param_name, value: Union[float, int, str]) -> str:
    """
    Get the parameter value itself in the format needed for writing to a TeX table.
    Only adjusts float values, leaves both integers and strings unaffected.

    Args:
        value: The parameter's value

    Returns:
        The string version of the parameter value ready to write 
    """
    prior_names = [prior["param_name"] for prior in project.calibration.all_priors]
    if param_name in prior_names:
        return "calibrated"
    elif isinstance(value, float):
        return float('%.3g' % value)
    else:
        return value


def write_param_table_rows(
    file_name: Path,
    project: Project,
    params_to_write: List[str],
):
    """
    Write parameter values to a TeX file in a format that can be incorporated
    into a standard TeX table.
    
    Args:
        params_to_write: The names of the requested parameters to be written
        
    """
    
    base_params = project.param_set.baseline

    with open(file_name, "w") as tex_file:
        for i_param, param in enumerate(params_to_write):        
            param_name = get_param_name(param)
            value = format_value_for_tex(project, param, get_param_from_nest_string(base_params, param))
            unit = "" if param not in PARAMETER_UNITS or value == "calibrated" else PARAMETER_UNITS[param]
            explanation = get_param_explanation(param)

            # Note that for some TeX-related reason, we can't put the \\ on the last line
            line_end = "" if i_param == len(params_to_write) - 1 else " \\\\ \n\hline"

            table_line = f"\n{param_name} & {value} {unit} & {explanation}{line_end}"
            tex_file.write(table_line)
