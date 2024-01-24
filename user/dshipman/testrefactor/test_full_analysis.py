def test_run_full_analysis():
    from autumn.projects.sm_covid2.common_school import runner_tools as rt

    #+++
    # We get some nasty threading errors (on Windows, at least) if we don't
    # do this.  Check with Romain whether we would ever expect to use
    # the runner_tools in an interactive context (Agg is file-only)
    import matplotlib as mpl
    mpl.use("Agg")

    rt.run_full_analysis("AUS", run_config=rt.TEST_RUN_CONFIG)

def test_run_full_analysis_arg():
    from autumn.projects.sm_covid2.common_school import runner_tools as rt

    #+++
    # We get some nasty threading errors (on Windows, at least) if we don't
    # do this.  Check with Romain whether we would ever expect to use
    # the runner_tools in an interactive context (Agg is file-only)
    import matplotlib as mpl
    mpl.use("Agg")

    rt.run_full_analysis("ARG", run_config=rt.TEST_RUN_CONFIG)    