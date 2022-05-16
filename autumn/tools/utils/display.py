"""
Utilities for displaying things in (primarily) Jupyter Notebooks

Note that importing this module will register custom display hooks
in IPython

"""

from pprint import PrettyPrinter
from IPython.display import HTML


def pretty_print(obj, indent=2):
    """Custom pretty printer
    Extend classes by adding a '__pretty__' method
    Args:
        obj ([Any]): The object to pretty print
        indent (int, optional): Indentation level. Defaults to 2.
    """
    printer = PrettyPrinter(indent=indent)
    if hasattr(obj, "__pretty__"):
        print(obj.__pretty__(printer))
    else:
        printer.pprint(obj)


def get_link(url: str) -> HTML:
    """
    Return a clickable link (in Jupyter) for any valid URL string
    """
    return HTML(f"<a href={url}>{url}</a>")


def _register_display_hooks():
    try:
        ip = get_ipython()
    except:
        return

    def generator_formatter(generator, pp, cycle):
        """Print generators as if they were lists
        Mostly used for printing glob output of pathlib
        """
        for i, item in enumerate(generator):
            pp.text(repr(item))
            pp.breakable()
            # Prevent iterating forever...
            if i > 128:
                pp.text("...")
                pp.breakable()
                pp.text("Too many items to display")
                return

    plain_formatter = ip.display_formatter.formatters["text/plain"]
    plain_formatter.for_type_by_name("builtins", "generator", generator_formatter)


# Disable this if there are problems... should be relatively non-intrusive
_register_display_hooks()
