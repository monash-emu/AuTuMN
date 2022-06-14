from typing import List, Union, Dict
from pathlib import Path
from functools import reduce
import operator
from importlib import import_module

from autumn.core.project.project import Project
from autumn.settings.folders import BASE_PATH


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
    
    projects_dir = Path(BASE_PATH) / "docs" / "tex" / "tex_descriptions" / "projects"
    app_dir = projects_dir / model / country / region    
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir / f"{file_name}.tex"


def get_param_name(
    parameter_definition: Dict[str, str],
    param: str
) -> str:
    """
    Simple function to essentially return the appropriate value of the PARAMETER_NAMES dictionary.

    Args:
        param: Name of the parameter of interest, with hierachical keys joined with "."

    Returns:
        The parameter name in an appropriate format to go into a table
    """

    name = parameter_definition[param] if param in parameter_definition else param.replace("_", " ")
    return name[:1].upper() + name[1:]


def get_param_explanation(
    parameter_evidence: Dict[str, str], 
    param: str
) -> str:
    """
    Simple function to essentially return the appropriate value of the PARAMETER_EXPLANATIONS dictionary.

    Args:
        param: Name of the parameter of interest, with hierachical keys joined with "."

    Returns:
        The parameter explanation in an appropriate format to go into a table
    """

    explanation = parameter_evidence[param] if param in parameter_evidence else "assumed"
    return explanation[:1].upper() + explanation[1:]


def format_value_for_tex(value: Union[float, int, str]) -> str:
    """
    Get the parameter value itself in the format needed for writing to a TeX table.
    Only adjusts float values, leaves both integers and strings unaffected.

    Args:
        value: The parameter's value

    Returns:
        The string version of the parameter value ready to write 
    """
    if isinstance(value, float):
        return float('%.3g' % value)
    elif isinstance(value, str):
        return value[:1].upper() + value[1:]
    else:
        return value


def format_prior_values(
    distribution: str, 
    parameters: List[float]
) -> str:
    """
    Get the TeX-ready string to represent the values of the prior distribution.
    This function needs to be extended considerably, only one example in here for now.

    Args:
        distribution: Name of the distribution, which will determine the parameter format
        parameters: The values of the distribution parameters

    Returns:
        String in TeX format        
    """
    if distribution == "uniform":
        return f"Range: [{parameters[0]}, {parameters[1]}]"


def get_line_end(
    is_last_line: bool,
) -> str:
    """
    Get the characters needed for the end of a row of a TeX table being created by one
    of the functions below.
    Note that for some TeX-related reason, we can't put the \\ on the last line.

    Args:
        is_last_line: Whether this is the last line of the 

    Returns:
        The characters needed for the end of the line of the table
    """
    return "" if is_last_line else " \\\\ \n\hline"


def write_param_table_rows(
    file_name: Path,
    project: Project,
    params_to_write: List[str],
    ignore_priors: bool=True,
):
    """
    Write parameter values to a TeX file in a format that can be incorporated
    into a standard TeX table.
    
    Args:
        file_name: Path to the file to be written
        project: The AuTuMN project object being interrogated
        params_to_write: The names of the requested parameters to be written
        ignore_priors: Whether to ignore parameters that are project priors
    """
    base_params = project.param_set.baseline

    # Get the dictionaries to pull the text from
    model_constants = import_module(f"autumn.models.{project.model_name}.param_format")

    with open(file_name, "w") as tex_file:
        for i_param, param in enumerate(params_to_write):

            # Get the ingredients
            param_name = get_param_name(model_constants.PARAMETER_DEFINITION, param)
            unit = model_constants.PARAMETER_UNITS[param] if param in model_constants.PARAMETER_UNITS else ""

            # Ignore if the parameter is a calibration prior
            if param in (prior["param_name"] for prior in project.calibration.all_priors) and ignore_priors:
                value = "Calibrated"
                unit = ""
            else:
                value = format_value_for_tex(get_param_from_nest_string(base_params, param))
            explanation = get_param_explanation(model_constants.PARAMETER_EVIDENCE, param)

            line_end = get_line_end(i_param == len(params_to_write) - 1)

            # Format for TeX
            table_line = f"\n{param_name} & {value} {unit} & {explanation}{line_end}"

            # Write
            tex_file.write(table_line)


def write_prior_table_rows(
    file_name: Path,
    project: Project,
):
    """
    Write prior values to a TeX file in a format that can be incorporated
    into a standard TeX table.
    
    Args:
        params_to_write: The names of the requested parameters to be written
    """

    # Get the dictionaries to pull the text from
    model_constants = import_module(f"autumn.models.{project.model_name}.param_format")

    with open(file_name, "w") as tex_file:
        for i_prior, prior in enumerate(project.calibration.all_priors):

            # Get the ingredients
            param_name = get_param_name(model_constants.PARAMETER_DEFINITION, prior["param_name"])
            distribution_type = format_value_for_tex(prior["distribution"])
            prior_parameters = format_prior_values(prior["distribution"], prior["distri_params"])
            line_end = get_line_end(i_prior == len(project.calibration.all_priors) - 1)

            # Format for TeX
            table_line = f"\n{param_name} & {distribution_type} & {prior_parameters}{line_end}"

            # Write
            tex_file.write(table_line)
