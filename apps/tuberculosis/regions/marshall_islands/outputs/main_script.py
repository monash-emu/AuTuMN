from apps.tuberculosis.regions.marshall_islands.outputs.utils import get_format
import apps.tuberculosis.regions.marshall_islands.outputs.calibration as cal
import apps.tuberculosis.regions.marshall_islands.outputs.posteriors as pos
import apps.tuberculosis.regions.marshall_islands.outputs.counterfactual as ctf
import apps.tuberculosis.regions.marshall_islands.outputs.elimination as elm
import apps.tuberculosis.regions.marshall_islands.outputs.diabetes as dia


get_format()

# calibration outputs
cal.main()

# posterior table
pos.main()

# counterfactual outputs
ctf.main()

# elimination outputs
elm.main()

# diabetes plot
dia.main()
