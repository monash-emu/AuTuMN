import os
import json
from typing import List, Union

import numpy as np
from autumn.tools.utils.secrets import check_hash
from autumn.tools.utils.utils import apply_moving_average


class TimeSeries:
    """
    A timeseries of interesting data.
    """

    def __init__(self, name: str, times: List[float], values: List[float]):
        self.name = name
        self.times = times
        self.values = values

    def __len__(self):
        t, v = len(self.times), len(self.values)
        assert t == v, "Times and values have different length"
        return t

    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            idxs = range(*idx.indices(len(self.times)))
        else:
            idxs = [idx]

        return TimeSeries(
            self.name,
            times=[self.times[i] for i in idxs],
            values=[self.values[i] for i in idxs],
        )

    def downsample(self, step: int):
        return TimeSeries(
            self.name,
            times=self.times[::step],
            values=self.values[::step],
        )

    def round_values(self):
        return TimeSeries(
            self.name,
            times=[*self.times],
            values=[round(v) for v in self.values],
        )

    def moving_average(self, window: int):
        return TimeSeries(
            self.name,
            times=[*self.times],
            values=apply_moving_average(self.values, window),
        )

    def truncate_times(self, start: float, end: float):
        """
        Returns a new TimeSeries with the times all greater than the start time,
        and all lesser than the end time
        """
        ts = self.truncate_start_time(start)
        return ts.truncate_end_time(end)

    def truncate_start_time(self, time: float):
        """
        Returns a new TimeSeries with the times all greater than the given time.
        """
        if (not self.times) or time is None or time < min(self.times):
            return self
        else:
            start_idx = next(x[0] for x in enumerate(self.times) if x[1] > time)
            return TimeSeries(
                self.name,
                times=self.times[start_idx:],
                values=self.values[start_idx:],
            )

    def truncate_end_time(self, time: float):
        """
        Returns a new TimeSeries with the times all lesser than the given time.
        """
        if (not self.times) or time is None or time > max(self.times):
            return self
        else:
            end_idx = next(x[0] for x in enumerate(self.times) if x[1] > time)
            return TimeSeries(
                self.name,
                times=self.times[:end_idx],
                values=self.values[:end_idx],
            )


class TimeSeriesSet:
    """
    A collection of timeseries data.
    """

    def __init__(self, data: dict):
        self._timeseries_lookup = {k: (v["times"], v["values"]) for k, v in data.items()}

    def __getitem__(self, key) -> TimeSeries:
        return self.get(key)

    def get(self, key, name: str = None) -> TimeSeries:
        times, values = self._timeseries_lookup[key]
        name = name or key
        return TimeSeries(name, times=[*times], values=[*values])

    @staticmethod
    def from_file(path: str):
        assert path.endswith(".json"), "Timeseries can only load JSON files (.json)"
        is_secret = ".secret." in path
        if is_secret:
            msg = MISSING_MESSAGE_SECRET.format(path=path)
            assert os.path.exists(path), msg
            check_hash(path)
        else:
            msg = MISSING_MESSAGE.format(path=path)
            assert os.path.exists(path), msg

        # Load the JSON
        with open(path, "r") as f:
            data = json.load(f)

        return TimeSeriesSet(data)


MISSING_MESSAGE = """

    A timeseries file is missing at {path}

"""


MISSING_MESSAGE_SECRET = """

    A timeseries file is missing at {path}

    This is probably happening because you need to decrypt some data.
    Have you tried decrypting your data? Try running this from the command line:

        python -m autumn secrets read

"""