from apps.tuberculosis.regions.marshall_islands.outputs.utils import get_format
import apps.tuberculosis.regions.marshall_islands.outputs.counterfactual as ctf
import apps.tuberculosis.regions.marshall_islands.outputs.elimination as elm



get_format()

# counterfactual outputs
ctf.main()

# elimination outputs
elm.main()
