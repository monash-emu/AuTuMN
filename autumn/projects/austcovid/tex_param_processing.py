import pylatex as pl
from pylatex.utils import NoEscape, bold
import arviz as az
import pandas as pd

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


def add_calib_metric_table_to_doc(
    doc: pl.document.Document,
    calib_summary: pd.DataFrame, 
    descriptions: dict,
):
    """
    Include a table of the summary outputs from the calibration algorithm in the document.

    Args:
        doc: The document to modify
        calib_summary: Grid of outputs from arviz's summary function
        descriptions: The longer parameter names
    """

    headers = ["Para-meter", "Mean (SD)", "3-97% high-density interval", "MCSE mean (SD)", "ESS bulk", "ESS tail", "R_hat"]
    with doc.create(pl.Tabular("p{1.3cm} " * 7)) as calib_metrics_table:
        calib_metrics_table.add_hline()
        calib_metrics_table.add_row([bold(i) for i in headers])
        for param in calib_summary.index:
            calib_metrics_table.add_hline()
            summary_row = calib_summary.loc[param]
            name = descriptions[param]
            mean_sd = f"{summary_row['mean']} ({summary_row['sd']})"
            hdi = f"{summary_row['hdi_3%']} to {summary_row['hdi_97%']}"
            mcse = f"{summary_row['mcse_mean']} ({summary_row['mcse_sd']})"
            calib_metrics_table.add_row([name, mean_sd, hdi, mcse] + [str(metric) for metric in summary_row[6:]])
        calib_metrics_table.add_hline()


def add_parameter_progression_fig_to_doc(
    outputs: az.data.inference_data.InferenceData,
    doc: pl.document.Document,
    priors: list,
    descriptions: dict,
):
    """
    Include a figure of the parameter posteriors and trace in document.

    Args:
        outputs: Results of inference algorithm in arviz format
        doc: The document to modify
        priors: The priors being used in calibration
        descriptions: The longer parameter names
    """
    axes = az.plot_trace(outputs, figsize=(15, 10))
    for i_prior, prior in enumerate(priors):
        column_names = ["posterior", "trace"]
        for col in range(2):
            ax = axes[i_prior][col]
            ax.set_title(f"{descriptions[prior]}, {column_names[col]}", fontsize=20)
            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
    with doc.create(pl.Figure()) as plot:
        plot.add_plot(width=NoEscape(r"1\textwidth"))
        plot.add_caption("Parameter posteriors and progression.")
