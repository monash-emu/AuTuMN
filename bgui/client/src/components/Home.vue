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
                v-for="(group, i) in groups"
                :key="i"
                :id="group"
                :md-label="group">
              <div>
                <div style="width: 18em;">
                  <md-layout
                      md-column
                      v-for="(key, i) in inputSets[i]"
                      :key="i">

                    <div v-if="key in booleans">
                      <md-checkbox
                          type="checkbox"
                          tabindex="0"
                          v-bind:id="key"
                          v-bind:value="getName(key)"
                          v-model="booleans[key]">
                        {{ getName(key) }}
                      </md-checkbox>
                    </div>

                    <div v-else-if="key in drop_downs">
                      <md-input-container>
                        <label>{{ getName(key) }}</label>
                        <md-select
                            v-model="raw_outputs[key]">
                          <md-option
                              v-for="(option, i) in drop_downs[key]"
                              v-bind:value="option"
                              :key="i">
                            {{option}}
                          </md-option>
                        </md-select>
                      </md-input-container>
                    </div>

                    <div v-else>
                      <md-input-container>
                        <label>{{ getName(key) }}</label>
                        <md-input
                            type="number"
                            step="any"
                            v-model="raw_outputs[key]">
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
  import axios from 'axios'
  import auth from '../modules/auth'
  import config from '../config'
  import util from '../modules/util'
  import rpc from '../modules/rpc'
  import _ from 'lodash'

  const keys = [
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

  let groups = [
    'Model running',
    'Model Stratifications',
    'Elaborations',
    'Scenarios to run',
    'Uncertainty',
    'Plotting',
    'MS Office outputs',
  ]

  _.each(_.range(1, 16), i => {
    keys.push(`scenario_${i}`)
  })

  let inputSets = _.map(_.range(7), i => [])
  _.each(keys, key => {
    const name = booleanName[key]
    if (_.includes(name, 'Plot') || _.includes(name, 'Draw')) {
      inputSets[5].push(key)
    } else if (_.includes(name, 'uncertainty') || _.includes(key, 'uncertainty')) {
      inputSets[4].push(key)
    } else if (_.includes(key, 'is_')) {
      inputSets[2].push(key)
    } else if (_.includes(key, 'riskgroup_') || _.includes(key, 'n_')) {
      inputSets[1].push(key)
    } else if (_.includes(key, 'scenario_')) {
      inputSets[3].push(key)
    } else {
      inputSets[6].push(key)
    }
  })

  let booleans = {}
  _.each(keys, key => {
    booleans[key] = false
  })
  booleans.adaptive_uncertainty = true
  booleans.is_amplification = true
  booleans.is_misassignment = true
  booleans.is_vary_detection_by_organ = true
  booleans.output_gtb_plots = true

  let raw_outputs = {}
  let drop_downs = {}

  // Model running options
  drop_downs.country = [
    'Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia',
    'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh',
    'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil',
    'Bulgaria', 'Burundi', 'Cameroon', 'Chad', 'Chile', 'Croatia',
    'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon',
    'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Philippines', 'Romania']
  raw_outputs.country = drop_downs.country[4]
  drop_downs.integration_method = ['Runge Kutta', 'Explicit']
  raw_outputs.integration_method = drop_downs.integration_method[1]
  drop_downs.fitting_method = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
  raw_outputs.fitting_method = drop_downs.fitting_method[4]
  raw_outputs.default_smoothness = 1.
  raw_outputs.time_step = 5.
  const runningKeys = ['country', 'integration_method', 'fitting_method',
    'default_smoothness', 'time_step']
  _.each(runningKeys, k => inputSets[0].push(k))

  // Model stratifications options
  drop_downs.n_organs = ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
  raw_outputs.n_organs = drop_downs.n_organs[0]
  drop_downs.n_strains = ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
  raw_outputs.n_strains = drop_downs.n_strains[1]
  _.each(['n_organs', 'n_strains'], k => inputSets[1].push(k))

  // Uncertainty options
  const uncertainty_numeric_list = {
    uncertainty_runs: ['Number of uncertainty runs', 10],
    burn_in_runs: ['Number of burn-in runs', 0],
    search_width: ['Relative search width', .08]
  }
  drop_downs.pickle_uncertainty = ['No saving or loading', 'Load', 'Save']
  raw_outputs.pickle_uncertainty = 'No saving or loading'
  _.each(uncertainty_numeric_list, (v, k) => {
    raw_outputs[k] = v[1]
    inputSets[4].push(k)
  })
  inputSets[4].push('pickle_uncertainty')

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
        inputSets,
        booleans,
        groups,
        drop_downs,
        raw_outputs,
        selectedGroup: groups[0],
        isRunning: false,
        consoleLines: []
      }
    },
    created () {
      this.checkRun()
    },
    methods: {
      getName(key) {
        if (key in booleanName) {
          return booleanName[key]
        } else {
          return key
        }
      },
      selectGroup (group) {
        this.$data.selectedGroup = group
      },
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
        console.log('> Home.run start')
        let booleans = this.$data.booleans
        let raw_outputs = this.$data.raw_outputs

        let guiOutputs = {
          scenarios_to_run: [null],
          scenario_names_to_run: ['baseline']
        }
        for (let key of keys) {
          if (_.includes(key, 'scenario_')) {
            if (booleans[key]) {
              let i = parseInt(key.substr(9, 2))
              guiOutputs.scenarios_to_run.push(i)
              guiOutputs.scenario_names_to_run.push(find_scenario_string_from_number(i))
            }
          } else {
            guiOutputs[key] = booleans[key]
          }
        }

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

        _.each(raw_outputs, (value, option) => {
          if (option == 'fitting_method') {
            guiOutputs[option] = parseInt(_.last(value))
          } else if (option == 'n_organs') {
            guiOutputs[option] = organ_stratification_keys[value]
          } else if (option == 'n_strains') {
            guiOutputs[option] = strain_stratification_keys[value]
          } else {
            guiOutputs[option] = value
          }
        })

        console.log('>> Home.run', guiOutputs)

        rpc
          .rpcRun(
            'public_run_autumn', guiOutputs)
          .then((res) => {
            console.log('>> Home.run res', res)
          })

        this.$data.consoleLines = []
        this.checkRun()
      }
    }
  }

</script>

