import streamlit as st
import base64
import numpy as np


def create_downloadable_csv(
    data_frame_to_download,
    filename,
    include_row=True,
    text="click here to download CSV containing the following data",
):
    """
    Create a link for a downloadable CSV file available in the streamlit interface.
    """
    csv_bytes = data_frame_to_download.to_csv(index=include_row).encode()
    b64_str = base64.b64encode(csv_bytes).decode()
    html_str = f'<a download="{filename}.csv" href="data:file/csv;name={filename}.csv;base64,{b64_str}">{text}</a>'
    st.markdown(html_str, unsafe_allow_html=True)


def round_sig_fig(number: float, significant_figures: int):
    """
    Returns the submitted number rounded to the requested number of significant figures.
    """
    decimal_places = significant_figures - (int(np.floor(np.log10(abs(number)))) + 1)
    return round(number, decimal_places)


def create_standard_plotting_sidebar():
    title_font_size = st.sidebar.slider("Title font size", 1, 15, 8)
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    capitalise_first_letter = st.sidebar.checkbox("Title start capital")
    return title_font_size, label_font_size, dpi_request, capitalise_first_letter


def create_xrange_selector(x_min, x_max):
    x_min = st.sidebar.slider("Plot start time", x_min, x_max, x_min)
    x_max = st.sidebar.slider("Plot end time", x_min, x_max, x_max)
    return x_min, x_max
