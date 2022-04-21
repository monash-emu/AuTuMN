"""
Utilities for pretty printing purposes.  May expand to a generic 'display' module at some point (Jupyter notebook extensions etc)
"""

from pprint import PrettyPrinter


def pretty_print(obj, indent=2):
    printer = PrettyPrinter(indent=indent)
    if hasattr(obj, "__pretty__"):
        print(obj.__pretty__(printer))
    else:
        printer.pprint(obj)
