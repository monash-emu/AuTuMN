import streamlit as st
import os

from .base_plotter import BasePlotter
from matplotlib import pyplot


class StreamlitPlotter(BasePlotter):
    """
    Plots stuff in Streamlit.
    """

    def __init__(self, targets: dict):
        self.translation_dict = {t["output_key"]: t["title"] for t in targets.values()}

    def save_figure(self, fig, filename: str, subdir=None, title_text=None, dpi_request=300):
        if title_text:
            pretty_title = self.get_plot_title(title_text).replace("X", " ")
            md = f"<p style='text-align: center;padding-left: 80px'>{pretty_title}</p>"
            st.markdown(md, unsafe_allow_html=True)

        st.pyplot(fig, dpi=dpi_request, bbox_inches="tight")

        dir_name = "vic_figures"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        for extension in ["png", "pdf"]:
            dir_extension_name = os.path.join(dir_name, extension)
            if not os.path.isdir(dir_extension_name):
                os.makedirs(dir_extension_name)
            path = os.path.join(dir_extension_name, f"{filename}.{extension}")
            pyplot.savefig(path)
