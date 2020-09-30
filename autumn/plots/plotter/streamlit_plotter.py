import streamlit as st

from .base_plotter import BasePlotter


class StreamlitPlotter(BasePlotter):
    """
    Plots stuff in Streamlit.
    """

    def __init__(self, targets: dict):
        self.translation_dict = {t["output_key"]: t["title"] for t in targets.values()}

    def save_figure(self, fig, filename: str, subdir=None, title_text=None, dpi=300):
        if title_text:
            pretty_title = self.get_plot_title(title_text).replace("X", " ")
            md = f"<p style='text-align: center;padding-left: 80px'>{pretty_title}</p>"
            st.markdown(md, unsafe_allow_html=True)

        st.pyplot(fig, dpi=dpi, bbox_inches="tight")
