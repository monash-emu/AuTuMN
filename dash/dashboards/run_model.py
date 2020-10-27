import streamlit as st

from apps import covid_19, tuberculosis, sir_example


def run_dashboard():
    st.header("Run a model")

    apps = {
        "COVID-19": covid_19.app,
        "Tuberculosis": tuberculosis.app,
        "Example": sir_example.app,
    }
    app_keys = list(apps.keys())
    app_key = st.selectbox("Select an app", app_keys)
    app = apps[app_key]
    regions = app.region_names
    region = st.selectbox("Select a region", regions)
    app_region = app.get_region(region)
    should_run_scenarios = st.checkbox("Run scenarios", value=True)

    if st.button("Run the model"):
        with st.spinner(f"Running the {app_key} model for {region}..."):
            app_region.run_model(run_scenarios=should_run_scenarios)

        st.success(f"Finished running the {app_key} model for {region}.")
