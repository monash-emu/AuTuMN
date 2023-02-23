import pandas as pd
import pylatex as pl
from pylatex.utils import NoEscape
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from random import sample
import copy
from datetime import datetime

from autumn.projects.austcovid.doc_utils import DocumentedProcess, FigElement, TextElement, TableElement
from autumn.projects.austcovid.model import build_aust_model
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
        priors: list, 
        targets: list, 
        iterations: int, 
        burn_in: int, 
        model_func: callable,
        parameters: dict,
        descriptions: dict, 
        units: dict, 
        evidence: dict, 
        start: datetime,
        end: datetime,
        doc: pl.Document=None,
    ):
        """
        Supports calibration of a summer model,
        with documentation to TeX document as the processes proceed.
        Most of this should be general enough to use for any summer calibration.

        Args:
            priors: The prior objects
            targets: The targets to fit to
            iterations: The number of iterations to run
            burn_in: The number of iterations to discard as burn-in
            model_func: The function to build the model
            parameters: The base parameter requests before updating through calibration
            descriptions: Strings to describe the parameters properly
            units: Strings for the units of each parameter
            evidence: Strings with a more detailed description of the evidence for each parameter
            start: Starting date for simulation
            end: Finish date for simulation
            doc: The TeX document to populate
        """
        super().__init__(doc, True)
        self.iterations = iterations
        self.burn_in = burn_in
        self.model_func = model_func
        self.priors = priors
        self.prior_names = [priors[i_prior].name for i_prior in range(len(priors))]
        self.targets = targets
        self.params = parameters
        self.descriptions = descriptions
        self.units = units
        self.evidence = evidence
        self.start = start
        self.end = end
        self.model = build_aust_model(start, end, None, add_documentation=False)
        
    def get_analysis(self):
        """
        Run the uncertainty analysis and get outputs in arviz format.
        """
        uncertainty_analysis = AdaptiveChain(
            self.model_func, self.params, self.priors, self.targets, self.params,
            build_model_kwargs={"start_date": self.start, "end_date": self.end, "doc": None},
        )
        uncertainty_analysis.run(max_iter=self.iterations)
        self.uncertainty_outputs = uncertainty_analysis.to_arviz(self.burn_in)
    
    def graph_param_progression(self):
        """
        Plot progression of parameters over model iterations with posterior density plots.
        """
        axes = az.plot_trace(self.uncertainty_outputs, figsize=(16, 12))
        for i_prior, prior_name in enumerate(self.prior_names):
            for i_col, column in enumerate(["posterior", "trace"]):
                ax = axes[i_prior][i_col]
                ax.set_title(f"{self.descriptions[prior_name]}, {column}", fontsize=20)
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_tick_params(labelsize=15)
        location = "progression.jpg"
        plt.savefig(SUPPLEMENT_PATH / location)
        caption = "Parameter posteriors and progression."
        self.add_element_to_doc("Calibration", FigElement(location, caption=caption))

    def add_calib_table_to_doc(self):
        """
        Report calibration input choices in table.
        """

        text = "Input parameters varied through calibration with uncertainty distribution parameters and support. \n"
        self.add_element_to_doc("Calibration", TextElement(text))

        headers = ["Name", "Distribution", "Distribution parameters", "Support"]
        col_widths = "p{2.7cm} " * 4
        rows = []
        for prior in self.priors:
            prior_desc = self.descriptions[prior.name]
            dist_type = get_prior_dist_type(prior)
            dist_params = get_prior_dist_param_str(prior)
            dist_range = get_prior_dist_support(prior)
            rows.append([prior_desc, dist_type, dist_params, dist_range])
        self.add_element_to_doc("Calibration", TableElement(col_widths, headers, rows))

    def table_param_results(self):
        """
        Report results of calibration analysis.
        """

        calib_summary = az.summary(self.uncertainty_outputs)
        headers = ["Para-meter", "Mean (SD)", "3-97% high-density interval", "MCSE mean (SD)", "ESS bulk", "ESS tail", "R_hat"]
        rows = []
        for param in calib_summary.index:
            summary_row = calib_summary.loc[param]
            name = self.descriptions[param]
            mean_sd = f"{summary_row['mean']} ({summary_row['sd']})"
            hdi = f"{summary_row['hdi_3%']} to {summary_row['hdi_97%']}"
            mcse = f"{summary_row['mcse_mean']} ({summary_row['mcse_sd']})"
            rows.append([name, mean_sd, hdi, mcse] + [str(metric) for metric in summary_row[6:]])
        self.add_element_to_doc("Calibration", TableElement("p{1.3cm} " * 7, headers, rows))
            
    def add_param_table_to_doc(self):
        """
        Describe all the parameters used in the model, regardless of whether 
        """
        
        text = "Parameter interpretation, with value (for parameters not included in calibration algorithm) and summary of evidence. \n"
        self.add_element_to_doc("Parameterisation", TextElement(text))

        headers = ["Name", "Value", "Evidence"]
        col_widths = "p{2.7cm} " * 2 + "p{5.8cm}"
        rows = []
        for param in self.model.get_input_parameters():
            param_value_text = get_fixed_param_value_text(param, self.params, self.units, self.prior_names)
            rows.append([self.descriptions[param], param_value_text, NoEscape(self.evidence[param])])
        self.add_element_to_doc("Calibration", TableElement(col_widths, headers, rows))

    def get_sample_outputs(
            self, 
            n_samples: int, 
        ):
        """
        Get a selection of the model runs obtained during calibration in the notebook.

        Args:
            n_samples: Number of samples to choose

        Returns:
            The outputs from consecutive runs
        """

        # How many parameter samples to run through again (suppress warnings if 100+)
        samples = sorted(sample(range(self.burn_in, self.iterations - 200), n_samples))

        # Sample parameters from accepted runs
        sample_params = pd.DataFrame(
            {p.name: self.uncertainty_outputs.posterior[p.name][0, samples].to_numpy() for p in self.priors},
            index=samples,
        )

        # Get model outputs for sampled parameters
        sample_outputs = pd.DataFrame(
            index=self.model.get_derived_outputs_df().index, 
            columns=samples,
        )
        params = copy.deepcopy(self.params)
        for i_param_set in samples:
            params.update(sample_params.loc[i_param_set, :].to_dict())
            self.model.run(parameters=params)
            sample_outputs[i_param_set] = self.model.get_derived_outputs_df()["notifications"]
        
        return sample_outputs
