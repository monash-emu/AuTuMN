def test_import_runner_tools():
    from autumn.projects.sm_covid2.common_school import runner_tools as rt

def test_get_bcm():
    from autumn.projects.sm_covid2.common_school import runner_tools as rt
    bcm = rt.get_bcm_object("AUS")

def test_run_default_params():
    from autumn.projects.sm_covid2.common_school import runner_tools as rt
    bcm = rt.get_bcm_object("AUS")
    defp = bcm.model.builder.get_default_parameters()
    upd_p = {
        "infection_deaths_dispersion_param": 225.0,
    }
    bcm.model.run(defp | upd_p)