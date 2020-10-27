import streamlit as st

from apps import covid_19, tuberculosis, sir_example


def run_dashboard():
    st.header("Calibrate a model")

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
    time_seconds = st.slider(
        "Calibration time (seconds)", min_value=30, max_value=3600, value=30, step=30
    )
    if st.button("Calibrate the model"):
        with st.spinner(f"Calibratning the {app_key} model for {region}..."):
            app_region.calibrate_model(max_seconds=time_seconds, run_id=1, num_chains=1)

        st.success(f"Finished calibrating the {app_key} model for {region}.")
