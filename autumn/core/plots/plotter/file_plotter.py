import os

from matplotlib import pyplot

from .base_plotter import BasePlotter, add_title_to_plot


class FilePlotter(BasePlotter):
    """
    Plots stuff to a PNG file.
    """

    def __init__(self, out_dir: str, targets: dict):
        self.translation_dict = {t["output_key"]: t["title"] for t in targets.values()}
        self.out_dir = out_dir

    def save_figure(self, fig, filename: str, subdir=None, title_text=None, dpi_request=300):
        """
        Args:
            fig: The figure to add the title to
            filename: The end of the string for the file name
            subdir
            n_plots: number of panels
            title_text: Text for the title of the figure
        """
        if title_text:
            pretty_title = self.get_plot_title(title_text)
            add_title_to_plot(fig, 1, pretty_title)

        if subdir:
            subdir_path = os.path.join(self.out_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            filename = os.path.join(subdir_path, f"{filename}.png")
        else:
            filename = os.path.join(self.out_dir, f"{filename}.png")

        fig.savefig(filename, dpi=dpi_request, bbox_inches="tight")
        pyplot.close(fig)
