import datetime as dt
import pandas as pd

class EpochConverter:
    """Convert to and from an epoch (reference datetime) and floating point offsets
    """
    
    def __init__(self, epoch: dt.datetime, units: dt.timedelta):
        self.epoch = epoch
        self.units = units

    def dtf(self, d):
        """Return floating point offsets from a datetimelike object or collection

        Args:
            d (datetimelike): [description]

        Returns:
            offset (float or index): Results offsets
        """
        return (pd.to_datetime(d) - self.epoch) / self.units
    
    def ftd(self, f):
        """Return a datetime (or pandas Float64Index thereof) from given floating point offset(s) 

        Args:
            f: Floating point offset(s)

        Returns:
            datetimelike or collection: The resultant datetime(s)
        """
        if hasattr(f,'__len__'):
            return self.epoch + pd.Float64Index(f) * self.units
        else:
            return self.epoch + f * self.units
