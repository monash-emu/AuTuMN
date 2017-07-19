<template>
  <div style=" padding: 1em">
    <md-layout md-column>

      <div style="text-align: center;">
        <br>
        <div class="md-display-2">
          Model Parameters
        </div>
        <br>
        <br>
      </div>

      <md-layout md-row>

        <md-whiteframe style="width: 50%; height: calc(100vh - 230px); overflow: auto">
          <md-tabs>
            <md-tab
                v-for="(paramGroup, i) in paramGroups"
                :key="i"
                :id="paramGroup.name"
                :md-label="paramGroup.name">
              <div>
                <div style="width: 18em;">
                  <md-layout
                      md-column
                      v-for="(key, i) in paramGroups[i].keys"
                      :key="i">

                    <div v-if="params[key].type == 'boolean'">
                      <md-checkbox
                          type="checkbox"
                          tabindex="0"
                          :id="key"
                          v-model="params[key].value">
                        {{ params[key].label }}
                      </md-checkbox>
                    </div>

                    <div v-else-if="params[key].type == 'drop_down'">
                      <md-input-container>
                        <label>{{ params[key].label }}</label>
                        <md-select
                            v-model="params[key].value">
                          <md-option
                              v-for="(option, i) in params[key].options"
                              v-bind:value="option"
                              :key="i">
                            {{option}}
                          </md-option>
                        </md-select>
                      </md-input-container>
                    </div>

                    <div v-else-if="params[key].type == 'number'">
                      <md-input-container>
                        <label>{{ params[key].label }}</label>
                        <md-input
                            type="number"
                            step="any"
                            v-model="params[key].value">
                        </md-input>

                      </md-input-container>
                    </div>

                  </md-layout>
                </div>
              </div>
            </md-tab>
          </md-tabs>
        </md-whiteframe>

        <md-whiteframe style="padding: 1em; width: 50%; height: calc(100vh - 230px); overflow: auto">
          <md-layout
              md-row
              md-align="center"
              md-vertical-align="center">
            <md-button
                class="md-raised"
                :disabled="isRunning"
                @click="run()">
              Run simulation
            </md-button>
          </md-layout>
          <md-layout>
            <ul>
              <li v-for="line in consoleLines">
                {{ line }}
              </li>
            </ul>
          </md-layout>
          <md-layout style="padding-left: 2em;">
            <md-spinner
                :md-size="30"
                md-indeterminate
                v-if="isRunning">
            </md-spinner>
          </md-layout>
        </md-whiteframe>
      </md-layout>
    </md-layout>
  </div>
</template>

<!-- Add 'scoped' attribute to limit CSS to this component only -->
<style scoped>
</style>

<script>
  import rpc from '../modules/rpc'
  import _ from 'lodash'

  const booleanKeys = [
    'output_flow_diagram',
    'output_compartment_populations',
    'output_riskgroup_fractions',
    'output_age_fractions',
    'output_by_subgroups',
    'output_fractions',
    'output_scaleups',
    'output_gtb_plots',
    'output_plot_economics',
    'output_plot_riskgroup_checks',
    'output_param_plots',
    'output_popsize_plot',
    'output_likelihood_plot',
    'output_uncertainty',
    'adaptive_uncertainty',
    'output_spreadsheets',
    'output_documents',
    'output_by_scenario',
    'output_horizontally',
    'output_age_calculations',
    'riskgroup_diabetes',
    'riskgroup_hiv',
    'riskgroup_prison',
    'riskgroup_indigenous',
    'riskgroup_urbanpoor',
    'riskgroup_ruralpoor',
    'is_lowquality',
    'is_amplification',
    'is_misassignment',
    'is_vary_detection_by_organ',
    'is_timevariant_organs',
    'is_timevariant_contactrate',
    'is_vary_force_infection_by_riskgroup',
    'is_treatment_history'
  ]

  _.each(_.range(1, 16), i => {
    booleanKeys.push(`scenario_${i}`)
  })

  const booleanName = {
    'output_uncertainty': 'Run uncertainty',
    'adaptive_uncertainty': 'Adaptive search',
    'output_spreadsheets': 'Write to spreadsheets',
    'output_documents': 'Write to documents',
    'output_by_scenario': 'Output by scenario',
    'output_horizontally': 'Write horizontally',
    'output_gtb_plots': 'Plot outcomes',
    'output_compartment_populations': 'Plot compartment sizes',
    'output_by_subgroups': 'Plot outcomes by sub-groups',
    'output_age_fractions': 'Plot proportions by age',
    'output_riskgroup_fractions': 'Plot proportions by risk group',
    'output_flow_diagram': 'Draw flow diagram',
    'output_fractions': 'Plot compartment fractions',
    'output_scaleups': 'Plot scale-up functions',
    'output_plot_economics': 'Plot economics graphs',
    'output_plot_riskgroup_checks': 'Plot risk group checks',
    'output_age_calculations': 'Plot age calculation weightings',
    'output_param_plots': 'Plot parameter progression',
    'output_popsize_plot': 'Plot "popsizes" for cost-coverage curves',
    'output_likelihood_plot': 'Plot log likelihoods over runs',
    'riskgroup_diabetes': 'Type II diabetes',
    'riskgroup_hiv': 'HIV',
    'riskgroup_prison': 'Prison',
    'riskgroup_urbanpoor': 'Urban poor',
    'riskgroup_ruralpoor': 'Rural poor',
    'riskgroup_indigenous': 'Indigenous',
    'is_lowquality': 'Low quality care',
    'is_amplification': 'Resistance amplification',
    'is_timevariant_organs': 'Time-variant organ status',
    'is_misassignment': 'Strain mis-assignment',
    'is_vary_detection_by_organ': 'Vary case detection by organ status',
    'n_organs': 'Number of organ strata',
    'n_strains': 'Number of strains',
    'is_timevariant_contactrate': 'Time-variant contact rate',
    'is_vary_force_infection_by_riskgroup': 'Heterogeneous mixing'
  }

  let params = {}

  let paramGroups = [
    { keys: [], name: 'Model running' },
    { keys: [], name: 'Model Stratifications' },
    { keys: [], name: 'Elaborations' },
    { keys: [], name: 'Scenarios to run' },
    { keys: [], name: 'Uncertainty' },
    { keys: [], name: 'Plotting' },
    { keys: [], name: 'MS Office outputs' }
  ]

  for (let key of booleanKeys) {
    params[key] = {
      value: false,
      type: "boolean",
      label: key in booleanName ? booleanName[key] : ''
    }
  }
  const defaultBooleanKeys = ['adaptive_uncertainty', 'is_amplification',
    'is_misassignment', 'is_vary_detection_by_organ', 'output_gtb_plots']
  for (let k of defaultBooleanKeys) {
    params[k].value = true
  }

  for (let key of booleanKeys) {
    const name = booleanName[key]
    if (_.includes(name, 'Plot') || _.includes(name, 'Draw')) {
      paramGroups[5].keys.push(key)
    } else if (_.includes(name, 'uncertainty') || _.includes(key, 'uncertainty')) {
      paramGroups[4].keys.push(key)
    } else if (_.includes(key, 'is_')) {
      paramGroups[2].keys.push(key)
    } else if (_.includes(key, 'riskgroup_') || _.includes(key, 'n_')) {
      paramGroups[1].keys.push(key)
    } else if (_.includes(key, 'scenario_')) {
      paramGroups[3].keys.push(key)
    } else {
      paramGroups[6].keys.push(key)
    }
  }

  // Model running options
  params.country = {
    type: 'drop_down',
    options: [
      'Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia',
      'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh',
      'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil',
      'Bulgaria', 'Burundi', 'Cameroon', 'Chad', 'Chile', 'Croatia',
      'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon',
      'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Philippines', 'Romania']
  }
  params.country.value = params.country.options[4]

  params.integration_method = {
    type: 'drop_down',
    options: ['Runge Kutta', 'Explicit']
  }
  params.integration_method.value = params.integration_method.options[1]

  params.fitting_method = {
    type: 'drop_down',
    options: ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
  }
  params.fitting_method.value = params.fitting_method.options[4]

  params.default_smoothness = {
    type: 'number',
    value: 1.0
  }
  params.time_step = {
    type: 'number',
    value: 5.
  }

  const runningKeys = ['country', 'integration_method', 'fitting_method',
    'default_smoothness', 'time_step']
  for (let k of runningKeys) {
    paramGroups[0].keys.push(k)
  }

  // Model stratifications options
  params.n_organs = {
    type: 'drop_down',
    options: ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
  }
  params.n_organs.value = params.n_organs.options[0]
  params.n_strains = {
    type: 'drop_down',
    options: ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
  }
  params.n_strains.value = params.n_strains.options[1]

  for (let k of ['n_organs', 'n_strains']) {
    paramGroups[1].keys.push(k)
  }

  // Uncertainty options
  params.uncertainty_runs = {
    type: 'number',
    value: 10.,
    label: 'Number of uncertainty runs'
  }
  params.burn_in_runs = {
    type: 'number',
    value: 0,
    label: 'Number of burn-in runs'
  }
  params.search_width = {
    type: 'number',
    value: .08,
    label: 'Relative search width'
  }
  params.pickle_uncertainty = {
    type: 'drop_down',
    options: ['No saving or loading', 'Load', 'Save'],
  }
  params.pickle_uncertainty.value = params.pickle_uncertainty.options[0]

  const uncertaintyKeys = ['uncertainty_runs', 'burn_in_runs',
    'search_width', 'pickle_uncertainty']
  for (let k of uncertaintyKeys) {
    paramGroups[4].keys.push(k)
  }

  for (let [key, value] of _.entries(params)) {
    if (!value.label) {
      value.label = key
    }
  }
  console.log('params', params)

  function find_scenario_string_from_number (scenario) {
    if (scenario === null) {
      return 'baseline'
    } else {
      return `scenario_${scenario}`
    }
  }

  export default {
    name: 'experiments',
    data() {
      return {
        paramGroups,
        params,
        isRunning: false,
        consoleLines: []
      }
    },
    created () {
      this.checkRun()
    },
    methods: {
      checkRun () {
        rpc
          .rpcRun(
            'public_check_autumn_run')
          .then((res) => {
            console.log('>> Home.checkRun', res.data.is_running, res.data.console)
            if (res.data.console) {
              this.$data.consoleLines = res.data.console
            }
            if (res.data.is_running) {
              this.$data.isRunning = true
              setTimeout(() => { this.checkRun() }, 2000)
            } else {
              this.$data.isRunning = false
            }
          })
      },
      run() {
        let organ_stratification_keys = {
          'Pos / Neg / Extra': 3,
          'Pos / Neg': 2,
          'Unstratified': 0
        }
        let strain_stratification_keys = {
          'Single strain': 0.286,
          'DS / MDR': 2,
          'DS / MDR / XDR': 3
        }

        let params = this.$data.params
        let guiOutputs = {
          scenarios_to_run: [null],
          scenario_names_to_run: ['baseline']
        }
        for (let [key, value] of _.entries(params)) {
          if (_.includes(key, 'scenario_')) {
            if (value.type == 'boolean') {
              if (value.value) {
                console.log('scenario')
                let i = parseInt(key.substr(9, 2))
                guiOutputs.scenarios_to_run.push(i)
                guiOutputs.scenario_names_to_run.push(find_scenario_string_from_number(i))
              }
            }
          } else if (key == 'fitting_method') {
            guiOutputs[key] = parseInt(_.last(value.value))
          } else if (key == 'n_organs') {
            guiOutputs[key] = organ_stratification_keys[value.value]
          } else if (key == 'n_strains') {
            guiOutputs[key] = strain_stratification_keys[value.value]
          } else {
            guiOutputs[key] = value.value
          }
        }

        console.log('>> Home.run', guiOutputs)
        this.$data.isRunning = true
        rpc
          .rpcRun(
            'public_run_autumn', guiOutputs)
          .then((res) => {
            console.log('>> Home.run res', res)
          })
          .catch((res) => {
            this.$data.isRunning = false
          })

        this.$data.consoleLines = []
        this.checkRun()
      }
    }
  }

</script>

