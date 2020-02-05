from autumn.calibration import *

par_priors = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [1.0, 20.0]},
    {"param_name": "rr_transmission_ger", "distribution": "uniform", "distri_params": [1.0, 5.0]},
    {"param_name": "rr_transmission_urban", "distribution": "uniform", "distri_params": [1.0, 5.0]},
    {
        "param_name": "rr_transmission_province",
        "distribution": "uniform",
        "distri_params": [0.5, 5.0],
    },
    {"param_name": "latency_adjustment", "distribution": "uniform", "distri_params": [1.0, 3.0]},
    {
        "param_name": "dr_amplification_prop_among_nonsuccess",
        "distribution": "uniform",
        "distri_params": [0.05, 0.20],
    },
]

target_outputs = [
    {"output_key": "prevXinfectiousXamongXage_15Xage_60", "years": [2015.0], "values": [560.0]},
    {
        "output_key": "prevXinfectiousXamongXage_15Xage_60Xhousing_ger",
        "years": [2015.0],
        "values": [613.0],
    },
    {
        "output_key": "prevXinfectiousXamongXage_15Xage_60Xlocation_urban",
        "years": [2015.0],
        "values": [586.0],
    },
    {
        "output_key": "prevXinfectiousXamongXage_15Xage_60Xlocation_province",
        "years": [2015.0],
        "values": [513.0],
    },
    {"output_key": "prevXlatentXamongXage_5", "years": [2016.0], "values": [960.0]},
    {
        "output_key": "prevXinfectiousXstrain_mdrXamongXinfectious",
        "years": [2015.0],
        "values": [500],
    },
]

multipliers = {
    "prevXlatentXamongXage_5": 1.0e4,
    "prevXinfectiousXstrain_mdrXamongXinfectious": 1.0e4,
}

for i, output in enumerate(target_outputs):
    if (
        output["output_key"][0:15] == "prevXinfectious"
        and output["output_key"] != "prevXinfectiousXstrain_mdrXamongXinfectious"
    ):
        multipliers[output["output_key"]] = 1.0e5

calib = Calibration(build_mongolia_model, par_priors, target_outputs, multipliers)
calib.run_mode = "lsm"

# _________________________________________________________________
# for Guillaume:
bounds = [par_priors[i]["distri_params"] for i in range(len(par_priors))]
print("Here are the bounds to be used around the different parameters:")
print(bounds)


def objective(parameters):
    return calib.loglikelihood(parameters)


cost = objective([5.0, 2.0, 2.0, 1.0, 2.0, 0.10])
