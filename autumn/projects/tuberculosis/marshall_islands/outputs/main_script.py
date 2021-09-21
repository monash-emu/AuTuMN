import autumn.projects.tuberculosis.marshall_islands.outputs.calibration as cal
import autumn.projects.tuberculosis.marshall_islands.outputs.counterfactual as ctf
import autumn.projects.tuberculosis.marshall_islands.outputs.diabetes as dia
import autumn.projects.tuberculosis.marshall_islands.outputs.elimination as elm
import autumn.projects.tuberculosis.marshall_islands.outputs.get_output_numbers as gon
import autumn.projects.tuberculosis.marshall_islands.outputs.posteriors as pos
import autumn.projects.tuberculosis.marshall_islands.outputs.priors as pri
from autumn.projects.tuberculosis.marshall_islands.outputs.utils import get_format


def make_all_rmi_plots():
    get_format()

    # Print outputs as numbers
    gon.main()

    # calibration outputs
    cal.main()

    # prior table
    # pri.main() # FIXME

    # posterior table
    # pos.main()  # FIXME

    # counterfactual outputs
    # ctf.main() # FIXME

    # elimination outputs
    # elm.main()  # FIXME

    # diabetes plot
    # dia.main()  # FIXME



make_all_rmi_plots()
