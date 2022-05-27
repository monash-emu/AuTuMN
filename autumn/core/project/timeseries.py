import os
import json
import pandas as pd

from autumn.coretils.secrets import check_hash


def load_timeseries(path: str):
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

    out_dict = {}
    
    for k, v in data.items():
        out_dict[k] = pd.Series(data=v['values'], index=v['times'], name=v['output_key'])

    return out_dict


MISSING_MESSAGE = """

    A timeseries file is missing at {path}

"""


MISSING_MESSAGE_SECRET = """

    A timeseries file is missing at {path}

    This is probably happening because you need to decrypt some data.
    Have you tried decrypting your data? Try running this from the command line:

        python -m autumn secrets read

"""