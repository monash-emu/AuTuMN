import pylatex as pl
from pylatex.utils import NoEscape, bold

from summer2 import CompartmentalModel


def get_fixed_param_value_text(
    param: str,
    parameters: dict,
    param_units: dict,
    prior_names: list,
    decimal_places=2,
    calibrated_string="Calibrated, see priors table",
) -> str:
    """
    Get the value of a parameter being used in the model for the parameters table,
    except indicate that it is calibrated if it's one of the calibration parameters.
    
    Args:
        param: Parameter name
        parameters: All parameters expected by the model
        param_units: The units for the parameter being considered
        prior_names: The names of the parameters used in calibration
        decimal_places: How many places to round the value to
        calibrated_string: The text to use if the parameter is calibrated
    Return:
        Description of the parameter value
    """
    return calibrated_string if param in prior_names else f"{round(parameters[param], decimal_places)} {param_units[param]}"


def get_prior_dist_type(
    prior,
) -> str:
    """
    Clunky way to extract the type of distribution used for a prior.
    
    Args:
        The prior object
    Return:
        Description of the distribution
    """
    dist_type = str(prior.__class__).replace(">", "").replace("'", "").split(".")[-1].replace("Prior", "")
    return f"{dist_type} distribution"


def get_prior_dist_param_str(
    prior,
) -> str:
    """
    Extract the parameters to the distribution used for a prior.
    
    Args:
        prior: The prior object
    Return:
        The parameters to the prior's distribution joined together
    """
    return " ".join([f"{param}: {prior.distri_params[param]}" for param in prior.distri_params])


def get_prior_dist_support(
    prior,
) -> str:
    """
    Extract the bounds to the distribution used for a prior.
    
    Args:
        prior: The prior object
    Return:        
        The bounds to the prior's distribution joined together
    """
    return " to ".join([str(i) for i in prior.bounds()])


def add_param_table_to_doc(
    model: CompartmentalModel,
    doc: pl.document.Document, 
    params: list, 
    param_descriptions: dict, 
    units: dict, 
    evidence: dict, 
    priors: list,
):
    """
    Include a table for the non-calibrated parameters in a PyLaTeX document.
    
    Args:
        doc: The document to modify
        params: The parameters the model is expecting
        descriptions: The longer parameter names
        units: The Units of each parameter's value
        evidence: Description of the evidence for each parameter
        priors: The priors being used in calibration
    """
    doc.append("Parameter interpretation, with value (for parameters not included in calibration algorithm) and summary of evidence.\n")
    param_headers = ["Name", "Value", "Evidence"]
    with doc.create(pl.Tabular("p{2.7cm} " * 2 + "p{5.8cm}")) as parameters_table:
        parameters_table.add_hline()
        parameters_table.add_row([bold(i) for i in param_headers])
        for param in model.get_input_parameters():
            param_value_text = get_fixed_param_value_text(param, params, units, priors)
            parameters_table.add_hline()
            param_table_row = (param_descriptions[param], param_value_text, NoEscape(evidence[param]))
            parameters_table.add_row(param_table_row)
        parameters_table.add_hline()


def add_calib_table_to_doc(
    doc: pl.document.Document, 
    priors: list, 
    descriptions: dict,
):
    """
    Include a table for the calibrated parameters in a PyLaTeX document.

    Args:
        doc: The document to modify
        priors: The priors being used in calibration
        descriptions: The longer parameter names
    """
    doc.append("Input parameters varied through calibration with uncertainty distribution parameters and support.\n")
    calib_headers = ["Name", "Distribution", "Distribution parameters", "Support"]
    with doc.create(pl.Tabular("p{2.7cm} " * 4)) as calibration_table:
        calibration_table.add_hline()
        calibration_table.add_row([bold(i) for i in calib_headers])
        for prior in priors:
            prior_desc = descriptions[prior.name]
            dist_type = get_prior_dist_type(prior)
            dist_params = get_prior_dist_param_str(prior)
            dist_range = get_prior_dist_support(prior)
            calibration_table.add_hline()
            calib_table_row = (prior_desc, dist_type, dist_params, dist_range)
            calibration_table.add_row(calib_table_row)
        calibration_table.add_hline()
