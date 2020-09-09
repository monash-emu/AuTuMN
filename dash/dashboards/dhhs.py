import re
from importlib import import_module

import streamlit as st
import pandas as pd
from autumn import db, plots
from autumn.constants import Region
from autumn.plots.plotter import StreamlitPlotter
from apps import covid_19


BASE_DATE = pd.datetime(2019, 12, 31)


URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def run_dashboard():
    st.header("Visualise DHHS data")

    url = st.text_input(
        "Enter the URL of the DHHS CSV", value="", max_chars=None, key=None, type="default"
    )
    is_valid_url = re.match(URL_REGEX, url) is not None
    if not is_valid_url or not url.endswith(".csv"):
        if url:
            st.warning(f"Invalid URL {url}")
        return

    df = get_url_df(url)

    regions = df["region"].unique().tolist()
    region = st.selectbox("Select region", regions)
    region_mask = df["region"] == region
    region_df = df[region_mask].drop(columns=["region"])

    region_name = Region.to_name(region)
    app_region = covid_19.app.get_region(region_name)
    targets = app_region.targets
    plotter = StreamlitPlotter(targets)

    outputs = region_df["type"].unique().tolist()
    output = st.selectbox("Select output", outputs)
    output_mask = region_df["type"] == output
    output_df = region_df[output_mask].drop(columns=["type"])
    target = {"output_key": output, "times": [], "values": []}
    for t in targets.values():
        if t["output_key"] == output:
            target = t
            break

    dates = pd.to_datetime(output_df["time"], infer_datetime_format=True)
    times = (dates - BASE_DATE).dt.days.unique().tolist()
    quantiles = {}
    for q in target["quantiles"]:
        mask = output_df["quantile"] == q
        quantiles[q] = output_df[mask]["value"].tolist()

    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter, output, 0, quantiles, times, targets
    )


@st.cache
def get_url_df(url):
    return pd.read_csv(url)
