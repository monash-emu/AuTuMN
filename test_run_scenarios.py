

import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_and_process_data

# Start timer
start_realtime = datetime.datetime.now()

# Decide on country
country = u'Fiji'

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Load data
keys_of_sheets_to_read = [
    'bcg', 'birth_rate', 'life_expectancy', 'attributes', 'parameters', 'miscellaneous', 'time_variants', 'tb',
    'notifications', 'outcomes']
data = read_and_process_data(True, keys_of_sheets_to_read, country)

# Note that it takes about one hour to run all of the possible model structures,
# so perhaps don't do that - and longer if running multiple scenarios
for scenario in data['attributes']['scenarios_to_run']:
    n_organs = data['attributes']['n_organs'][0]
    n_strains =  data['attributes']['n_strains'][0]
    n_comorbidities = data['attributes']['n_comorbidities'][0]
    is_quality = data['attributes']['is_lowquality'][0]
    is_amplification = data['attributes']['is_amplification'][0]
    is_misassignment = data['attributes']['is_misassignment'][0]
    if (is_misassignment and not is_amplification) \
            or (n_strains <= 1 and (is_amplification or is_misassignment)):
        pass
    else:
        name = 'model%d' % n_organs
        base = os.path.join(out_dir, name)

        model = autumn.model.ConsolidatedModel(
            n_organs,
            n_strains,
            n_comorbidities,
            is_quality,  # Low quality care
            is_amplification,  # Amplification
            is_misassignment,  # Misassignment by strain
            scenario,  # Scenario to run
            data)
        print(str(n_organs) + " organ(s),   " +
              str(n_strains) + " strain(s),   " +
              str(n_comorbidities) + " comorbidity(ies),   " +
              "Low quality? " + str(is_quality) + ",   " +
              "Amplification? " + str(is_amplification) + ",   " +
              "Misassignment? " + str(is_misassignment) + ".")

        for key, value in data['parameters'].items():
            model.set_parameter(key, value)
        for key, value in data['miscellaneous'].items():
            model.set_parameter(key, value)

        model.integrate()

        # Only make a flow-diagram if the model isn't overly complex
        if n_organs + n_strains + n_comorbidities <= 5:
            model.make_graph(base + '.workflow')

        autumn.plotting.plot_outputs(
            model, ["incidence", "mortality", "prevalence"],
            data['attributes']['start_time'], base + '.rate_outputs.png')
        autumn.plotting.plot_outputs_against_gtb(
            model, "incidence",
            data['attributes']['recent_time'], base + '.rate_outputs_gtb.png',
            data)
        autumn.plotting.plot_all_outputs_against_gtb(
            model, ["incidence", "mortality", "prevalence", "notifications"],
            data['attributes']['recent_time'], base + '.all_rate_outputs_gtb' + str(scenario) + '.png',
            data, country)


pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))

