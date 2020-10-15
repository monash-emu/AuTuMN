import apps.covid_19.mixing_optimisation.outputs.plots.calibration.figure_code as fc

calibration_outputs = fc.get_all_countries_calibration_outputs()


fc.make_posterior_ranges_figure(calibration_outputs)
