import apps.covid_19.mixing_optimisation.outputs.plots.calibration.figure_code as fc
import matplotlib as mpl


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")

    # Get calibration data
    calibration_outputs = fc.get_all_countries_calibration_outputs()

    """
    parameter-related
    """
    param_values = fc.get_parameter_values(calibration_outputs)
    fc.make_posterior_ranges_figure(param_values)
    fc.get_all_posterior_detection_percentiles(
        param_values
    )  # Rerun this line to recalculate percentiles
    fc.plot_posterior_detection()

    param_values_best_chain = fc.get_parameter_values(calibration_outputs, best_chain_only=True)
    fc.plot_parameter_traces(param_values_best_chain, max_n_iter=2500)

    """
    output-related
    """
    fc.make_calibration_fits_figure(
        calibration_outputs
    )  # uncertainty figure for all calibration targets
    fc.make_calibration_fits_figure(
        calibration_outputs, seroprevalence=True
    )  # for seroprevalence only
    fc.make_all_sero_by_age_fits_figures(calibration_outputs)


if __name__ == "__main__":
    main()
