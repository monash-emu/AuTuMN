import pylatex as pl
from pylatex.utils import NoEscape, bold
from pylatex.section import Section
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

from summer2 import CompartmentalModel
from autumn.projects.austcovid.model_features import DocumentedProcess, FigElement
from estival.calibration.mcmc.adaptive import AdaptiveChain

BASE_PATH = Path(__file__).parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


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


class DocumentedCalibration(DocumentedProcess):
    def __init__(
        self, 
        priors, 
        targets, 
        iterations, 
        burn_in, 
        descriptions, 
        units, 
        evidence, 
        doc=None,
        add_documentation=False, 
    ):
        self.doc_sections = {}
        self.iterations = iterations
        self.burn_in = burn_in
        self.priors = priors
        self.prior_names = [priors[i_prior].name for i_prior in range(len(priors))]
        self.targets = targets
        self.descriptions = descriptions
        self.units = units
        self.evidence = evidence
        self.doc = doc
        self.add_documentation = add_documentation
        
    def get_analysis(self, model, params, start, end):
        uncertainty_analysis = AdaptiveChain(
            model, params, self.priors, self.targets, params,
            build_model_kwargs={"start_date": start, "end_date": end, "doc": None},
        )
        uncertainty_analysis.run(max_iter=self.iterations)
        self.uncertainty_outputs = uncertainty_analysis.to_arviz(self.burn_in)
    
    def graph_param_progression(self):

        if self.add_documentation:
            axes = az.plot_trace(self.uncertainty_outputs, figsize=(16, 12))
            for i_prior, prior_name in enumerate(self.prior_names):
                column_names = ["posterior", "trace"]
                for col in range(2):
                    ax = axes[i_prior][col]
                    ax.set_title(f"{self.descriptions[prior_name]}, {column_names[col]}", fontsize=20)
                    ax.xaxis.set_tick_params(labelsize=15)
                    ax.yaxis.set_tick_params(labelsize=15)
            location = "progression.jpg"
            plt.savefig(SUPPLEMENT_PATH / location)
            caption = "Parameter posteriors and progression."
            self.add_element_to_doc("Calibration", FigElement(location, caption=caption))

    def add_calib_table_to_doc(self):

        with self.doc.create(Section("Calibration algorithm")):
            self.doc.append("Input parameters varied through calibration with uncertainty distribution parameters and support.\n")
            calib_headers = ["Name", "Distribution", "Distribution parameters", "Support"]
            with self.doc.create(pl.Tabular("p{2.7cm} " * 4)) as calibration_table:
                calibration_table.add_hline()
                calibration_table.add_row([bold(i) for i in calib_headers])
                for prior in self.priors:
                    prior_desc = self.descriptions[prior.name]
                    dist_type = get_prior_dist_type(prior)
                    dist_params = get_prior_dist_param_str(prior)
                    dist_range = get_prior_dist_support(prior)
                    calibration_table.add_hline()
                    calib_table_row = (prior_desc, dist_type, dist_params, dist_range)
                    calibration_table.add_row(calib_table_row)
                calibration_table.add_hline()
            
    def table_param_results(self):
        with self.doc.create(Section("Calibration metrics")):
            calib_summary = az.summary(self.uncertainty_outputs)
            headers = ["Para-meter", "Mean (SD)", "3-97% high-density interval", "MCSE mean (SD)", "ESS bulk", "ESS tail", "R_hat"]
            with self.doc.create(pl.Tabular("p{1.3cm} " * 7)) as calib_metrics_table:
                calib_metrics_table.add_hline()
                calib_metrics_table.add_row([bold(i) for i in headers])
                for param in calib_summary.index:
                    calib_metrics_table.add_hline()
                    summary_row = calib_summary.loc[param]
                    name = self.descriptions[param]
                    mean_sd = f"{summary_row['mean']} ({summary_row['sd']})"
                    hdi = f"{summary_row['hdi_3%']} to {summary_row['hdi_97%']}"
                    mcse = f"{summary_row['mcse_mean']} ({summary_row['mcse_sd']})"
                    calib_metrics_table.add_row([name, mean_sd, hdi, mcse] + [str(metric) for metric in summary_row[6:]])
                calib_metrics_table.add_hline()
            
    def add_param_table_to_doc(self,
        model: CompartmentalModel,
        params: list, 
    ):
        with self.doc.create(Section("Calibration metrics")):
            self.doc.append("Parameter interpretation, with value (for parameters not included in calibration algorithm) and summary of evidence.\n")
            param_headers = ["Name", "Value", "Evidence"]
            with self.doc.create(pl.Tabular("p{2.7cm} " * 2 + "p{5.8cm}")) as parameters_table:
                parameters_table.add_hline()
                parameters_table.add_row([bold(i) for i in param_headers])
                for param in model.get_input_parameters():
                    param_value_text = get_fixed_param_value_text(param, params, self.units, self.prior_names)
                    parameters_table.add_hline()
                    param_table_row = (self.descriptions[param], param_value_text, NoEscape(self.evidence[param]))
                    parameters_table.add_row(param_table_row)
                parameters_table.add_hline()
