import streamlit as st
import pandas as pd
from autumn import plots
from autumn.region import Region
from autumn.plots.plotter import StreamlitPlotter
from apps import covid_19


BASE_DATE = pd.datetime(2019, 12, 31)


def run_dashboard():
    st.header("Visualise DHHS data")
    url = st.text_input("Enter the URL of the DHHS CSV", value="", type="default")
    if not url.endswith(".csv"):
        st.warning(f"Not a CSV file: {url}")
        return

    df = get_url_df(url)

    regions = df["region"].unique().tolist()
    region = st.selectbox("Select region", regions)
    region_mask = df["region"] == region
    region_df = df[region_mask].drop(columns=["region"])

    region_name = Region.to_name(region)
    if not region_name == "victoria":
        app_region = covid_19.app.get_region(region_name)
        targets = app_region.targets
    else:
        targets = {}
        for region in Region.VICTORIA_SUBREGIONS:
            app_region = covid_19.app.get_region(region)
            region_targets = app_region.targets
            for key, target in region_targets.items():
                if key in targets:
                    targets[key]["values"] = [
                        a + b for a, b in zip(targets[key]["values"], target["values"])
                    ]
                else:
                    targets[key] = target

    plotter = StreamlitPlotter(targets)

    outputs = region_df["type"].unique().tolist()
    output = st.selectbox("Select output", outputs)

    dates = pd.to_datetime(region_df["time"], infer_datetime_format=True)
    region_df["time"] = (dates - BASE_DATE).dt.days
    region_df["scenario"] = 0
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter,
        uncertainty_df=region_df,
        output_name=output,
        scenario_idxs=[0],
        targets=targets,
    )


@st.cache
def get_url_df(url):
    return pd.read_csv(url)
