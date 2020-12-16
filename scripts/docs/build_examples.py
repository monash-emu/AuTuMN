import os, sys
import inspect
from shutil import rmtree
from os.path import abspath, dirname, join, exists
from typing import Dict

import numpy as np
from matplotlib import pyplot
from black import format_file_contents, Mode, NothingChanged

# Do some super sick path hacks to get script to work from command line.
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
EXAMPLES_DIR = join(BASE_DIR, "docs", "examples")

sys.path.append(BASE_DIR)

from summer2.examples import EXAMPLES

if exists(EXAMPLES_DIR):
    rmtree(EXAMPLES_DIR)

os.makedirs(EXAMPLES_DIR)

index_rst = """
Examples
==============

Words words words

.. toctree::
   :maxdepth: 1

"""
markdown = """
# {name}

{docstring}

```python
{source}

# Build the model, run it, and plot the results.
model = build_model()
plot_outputs(model)
```

"""


def get_plotter(slug):
    plot_files = []

    def plot_timeseries(title: str, times: np.ndarray, values: Dict[str, np.ndarray]):
        pyplot.style.use("ggplot")
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel("times")
        legend = []
        for plot_name, plot_vals in values.items():
            ax.plot(times, plot_vals)
            legend.append(plot_name)

        ax.legend(legend)
        title_slug = title.lower().replace(" ", "-")
        filename = f"{slug}-{title_slug}.png"
        filepath = join(EXAMPLES_DIR, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        pyplot.close(fig)
        plot_files.append(filename)

    return plot_timeseries, plot_files


format_mode = Mode(line_length=90)

for name, module in EXAMPLES.items():
    print(f"Building documentation for example {name}")
    # Add to table of contents
    slug = name.lower().replace(" ", "-")
    index_rst += f"   {slug}\n"

    # Format source code to fit in margins
    docstring = inspect.getdoc(module)
    source = inspect.getsource(module)
    try:
        source = format_file_contents(source, fast=True, mode=format_mode)
    except NothingChanged:
        pass

    source = '"""'.join(source.split('"""')[2:])

    # Generate plots
    plot_timeseries, plot_files = get_plotter(slug)
    module.plot_timeseries = plot_timeseries
    model = module.build_model()
    module.plot_outputs(model)

    md = markdown.format(name=name, source=source, docstring=docstring)
    for file in plot_files:
        md += f"![]({file})\n"

    with open(join(EXAMPLES_DIR, f"{slug}.md"), "w") as f:
        f.write(md)

    print(f"Finished building for example {name}")


with open(join(EXAMPLES_DIR, "index.rst"), "w") as f:
    f.write(index_rst)
