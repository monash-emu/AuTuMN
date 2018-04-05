from __future__ import print_function
import os
import json
import copy
from autumn import gui_params

default_params = gui_params.get_autumn_params()['params']

def get_diff_params(ref_params, test_params):
    test_keys = test_params.keys()
    ref_keys = ref_params.keys()
    diff_params = {}
    for key in set(test_keys).intersection(ref_keys):
        if test_params[key]['value'] != ref_params[key]['value']:
            diff_params[key] = {'value': test_params[key]['value']}
    return diff_params

def restore_diff_params(ref_params, diff_params):
    restored_params = copy.deepcopy(ref_params)
    for key in diff_params:
        print(
            "restored",
            key + ":",
            json.dumps(ref_params[key]['value']),
            '->',
            json.dumps(diff_params[key]['value']))
        restored_params[key]['value'] = diff_params[key]['value']
    return restored_params


def get_country_defaults(country):
    filename = '../../projects/test_%s/params.json' % country
    with open(filename) as f:
        country_params = json.load(f)
    diff_params = get_diff_params(default_params, country_params)
    return diff_params

countries = ['fiji', 'bulgaria']

country_defaults = {}
if os.path.isfile('server/country_defaults.json'):
    with open('server/country_defaults.json') as f:
        country_defaults = json.load(f)

for country in countries:
    country_defaults[country] = get_country_defaults(country)
    print("### Country", country)
    restored_params = restore_diff_params(
        default_params, country_defaults[country])

with open('server/country_defaults.json', 'w') as f:
    json.dump(country_defaults, f, indent=2)

