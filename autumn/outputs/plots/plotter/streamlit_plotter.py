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


class StreamlitSavingPlotter(StreamlitPlotter):
    def save_figure(self, fig, filename: str, subdir=None, title_text=None, dpi_request=300, output_dirname="vic_figures"):
        if title_text:
            pretty_title = self.get_plot_title(title_text).replace("X", " ")
            md = f"<p style='text-align: center;padding-left: 80px'>{pretty_title}</p>"
            st.markdown(md, unsafe_allow_html=True)

        st.pyplot(fig, dpi=dpi_request, bbox_inches="tight")

        if not os.path.isdir(output_dirname):
            os.makedirs(output_dirname)

        for extension in ["png", "pdf"]:
            dir_extension_name = os.path.join(output_dirname, extension)
            if not os.path.isdir(dir_extension_name):
                os.makedirs(dir_extension_name)
            path = os.path.join(dir_extension_name, f"{filename}.{extension}")
            pyplot.savefig(path)
