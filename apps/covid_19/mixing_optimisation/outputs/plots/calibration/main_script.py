import apps.covid_19.mixing_optimisation.outputs.plots.calibration.figure_code as fc

import matplotlib as mpl

# Reset pyplot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.pyplot.style.use('ggplot')

# Get calibration data
calibration_outputs = fc.get_all_countries_calibration_outputs()
# param_values = fc.get_parameter_values(calibration_outputs)

# fc.make_posterior_ranges_figure(param_values)

# fc.get_all_posterior_detection_percentiles(param_values)   # Rerun this line to recalculate percentiles
# fc.plot_posterior_detection()


fc.make_calibration_fits_figure(calibration_outputs)
