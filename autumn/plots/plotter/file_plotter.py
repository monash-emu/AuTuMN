from matplotlib import pyplot

from .base_plotter import BasePlotter


class FilePlotter(BasePlotter):
    """
    Plots stuff to a PNG file.
    """

    def __init__(self, out_dir: str, translation_dict: dict):
        self.translation_dict = translation_dict
        self.out_dir = out_dir

    def save_figure(self, fig, filename: str, subdir=None, title_text=None):
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

        fig.savefig(filename, dpi=300, bbox_inches="tight")
        pyplot.close(fig)
