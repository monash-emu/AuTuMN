from multiprocess.pool import Pool

from autumn.calibration import *

par_priors = [
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.0001, 0.01]},
    {
        "param_name": "beta_decay_rate",
        "distribution": "uniform",
        "distri_params": [0.01, 0.1]},
    {
        "param_name": "minimum_tv_beta_multiplier",
        "distribution": "uniform",
        "distri_params": [0.1, 0.4]},
    {
        "param_name": "rr_transmission_ebeye",
        "distribution": "uniform",
        "distri_params": [1.0, 2.5]},
    {
        "param_name": "rr_transmission_otherislands",
        "distribution": "uniform",
        "distri_params": [0.5, 1.5]},
    {
        "param_name": "cdr_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0]},
    {
        "param_name": "case_detection_ebeye_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0]},
    {
        "param_name": "case_detection_otherislands_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 1.0],
    },
    {
        "param_name": "over_reporting_prevalence_proportion",
        "distribution": "uniform",
        "distri_params": [0.0, 0.5],
    },
]

target_outputs = [
    {
        "output_key": "prevXinfectiousXamongXlocation_ebeye",
        "years": [2017.0],
        "values": [755.0],
        "cis": [(620.0, 894.0)]},
    {
        "output_key": "reported_majuro_prevalence",
        "years": [2018.0],
        "values": [1578.0],
        "cis": [(620.0, 894.0)]},
    {
        "output_key": "prevXlatentXamongXlocation_majuro",
        "years": [2017.0],
        "values": [28.5],
        "cis": [(902.0, 1018.0)]},
    {
        "output_key": "notificationsXlocation_majuro",
        "years": [2016.0],
        "values": [119.0]},
    {
        "output_key": "notificationsXlocation_ebeye",
        "years": [2016.0],
        "values": [53.0]},
    {
        "output_key": "notificationsXlocation_otherislands",
        "years": [2016.0],
        "values": [6.0],
        "cis": [(5.0, 17.0)]}
]

multipliers = {
    "prevXinfectiousXamong": 1.0e5,
    "prevXlatentXamongXlocation_majuro": 1.0e4,
    "prevXinfectiousXstrain_mdrXamongXinfectious": 1.0e4,
}


if __name__ == "__main__":
    n_cpus = 24
    _iterable = []
    for i in range(n_cpus):
        _iterable.append(Calibration(build_rmi_model, par_priors, target_outputs, multipliers, i))

    def run_a_single_chain(_calib):
        _calib.run_fitting_algorithm(
            run_mode="autumn_mcmc",
            n_iterations=100000,
            n_burned=0,
            n_chains=1,
            available_time=3600.0 * 24.0 * 7.0,
        )
        return

    p = Pool(processes=n_cpus)
    p.map(func=run_a_single_chain, iterable=_iterable)
