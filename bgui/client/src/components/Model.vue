<template>
  <div style="">
    <md-layout md-column>

      <md-layout md-row>

        <md-layout
          md-row
          style="
            width: 50%;
            height: calc(100vh - 48px);
            overflow: auto">

          <md-whiteframe
            style="
              width: 230px;
              padding-top: 30px ">

            <h2
              class="md-heading"
              style="padding-left: 15px">
              Parameter sets
            </h2>

            <md-list>
              <md-list-item
                  v-for="(thisParamGroup, i) in paramGroups"
                  :key="i"
                  :id="thisParamGroup.name"
                  @click="selectParamGroup(i)">
                {{thisParamGroup.name}}
              </md-list-item>
            </md-list>
          </md-whiteframe>

          <md-whiteframe
              style="width: 220px">

            <md-layout
                v-if="paramGroup"
                md-column
                style="padding: 30px 15px">

              <h2 class="md-heading">
                {{paramGroup.name}}
              </h2>

              <md-layout
                  md-column
                  v-for="(key, i) in paramGroup.keys"
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
                        v-model="params[key].value"
                        @change="selectDropDown(key)"
                        >
                      <md-option
                          v-for="(option, i) in params[key].options"
                          v-bind:value="option"
                          :key="i">
                        {{option}}
                      </md-option>
                    </md-select>
                  </md-input-container>
                </div>

                <div v-else-if="
                    (params[key].type === 'number') ||
                    (params[key].type === 'double') ||
                    (params[key].type === 'integer')">
                  <md-input-container>
                    <label>{{ params[key].label }}</label>
                    <md-input
                        type="number"
                        step="any"
                        v-model="params[key].value">
                    </md-input>

                  </md-input-container>
                </div>

                <div v-else-if="params[key].type == 'slider'">
                  <label>{{ params[key].label }}</label>
                  <div style="height: 2.5em"></div>
                  <vue-slider
                      :max="params[key].max"
                      :interval="params[key].interval"
                      v-model="params[key].value">
                  </vue-slider>
                </div>

                <div v-else-if="params[key].type == 'breakpoints'">
                  <label>{{ params[key].label }}</label>
                  <div style="height: 2.5em"></div>
                  <md-layout
                      md-row
                      style="width: 200px"
                      v-for="(breakpoint, i) of params[key].value"
                      :key="i">
                    <vue-slider
                        style="width: 130px"
                        :max="100"
                        :interval="1"
                        v-model="params[key].value[i]"
                        @drag-end="breakpointCallback(params, key)">
                    </vue-slider>
                    <md-button
                        class="md-icon-button md-raised"
                        @click="deleteBreakpoint(params, key, i)">
                      <md-icon>clear</md-icon>
                    </md-button>
                  </md-layout>
                  <md-button
                      class="md-icon-button md-raised"
                      @click="addBreakpoint(params, key)">
                    <md-icon>add</md-icon>
                  </md-button>
                </div>

              </md-layout>
            </md-layout>
          </md-whiteframe>

          <md-layout md-flex>
            <div style="width: 100%; padding: 30px 15px">
              <md-layout
                  md-column
                  md-align="start"
                  md-vertical-align="start">

                <h2 class="md-heading">
                  Run model
                </h2>

                <md-input-container
                    style="
                        width: 200px;">
                  <label>Existing Projects</label>
                  <md-select
                      v-model="project"
                      @change="changeProject">
                    <md-option
                        v-for="(p, i) in projects"
                        v-bind:value="p"
                        :key="i">
                      {{p}}
                    </md-option>
                  </md-select>
                </md-input-container>

                <div style="width: 100%">
                  <md-layout
                      md-row
                      md-vertical-align="center">

                    <md-button
                        md-flex=true
                        class="md-raised"
                        :disabled="isRunning"
                        @click="run()">
                      Run
                    </md-button>

                    <md-spinner
                        :md-size="30"
                        md-indeterminate
                        v-if="isRunning">
                    </md-spinner>
                  </md-layout>
                </div>

                <h2 class="md-heading">
                  Console Output
                </h2>

                <md-layout
                    style="
                      width: 100%;
                      background-color: #EEE;">
                  <div
                      id="console-output"
                      style="
                        width: 100%;
                        height: 350px;
                        overflow-y: scroll;
                        font-family: Courier, monospace;
                        font-size: 0.9em">

                    <div
                        style="
                          margin: 0 8px;
                          word-wrap: break-word;"
                        v-for="(line, i) in consoleLines"
                        :key="i">
                      {{ line }}
                    </div>

                  </div>
                </md-layout>

                <h2
                  v-show="isGraph"
                  class="md-heading">
                  Progress in Uncertainty Runs
                </h2>
                <md-layout >
                  <div id="temp-chart-0">
                  </div>
                  <div id="temp-chart-1">
                  </div>
                  <div id="temp-chart-2">
                  </div>
                  <div id="temp-chart-3">
                  </div>
                  <div id="temp-chart-4">
                  </div>
                  <div id="temp-chart-5">
                  </div>
                </md-layout>

                <h2
                  class="md-heading"
                  style="margin-top: 1.5em;">
                  Model Results
                </h2>

                <vue-slider
                    v-if="filenames.length > 0"
                    style="width: 100%"
                    :max="100"
                    :min="10"
                    :interval="1"
                    v-model="imageWidth"
                    @callback="changeWidth(imageWidth)">
                </vue-slider>

                <md-layout style="width: 100%">
                  <md-card
                      :style="imageStyle"
                      v-for="(filename, i) in filenames"
                      :key="i">
                    <md-card-media>
                      <img
                          style="width: 100%"
                          :src="filename"/>
                    </md-card-media>
                  </md-card>
                </md-layout>
              </md-layout>

            </div>
          </md-layout>
        </md-layout>

      </md-layout>
    </md-layout>
  </div>
</template>

<!-- Add 'scoped' attribute to limit CSS to this component only -->
<style scoped>
</style>

<script>
import rpc from '../modules/rpc'
import util from '../modules/util'
import vueSlider from 'vue-slider-component'
import Vue from 'vue'
import VueScrollTo from 'vue-scrollto'

import _ from 'lodash'
import config from '../config'

import ChartContainer from '../modules/chartContainer'

Vue.use(VueScrollTo)

export default {
  name: 'experiments',
  components: {vueSlider},
  data () {
    return {
      params: {},
      paramGroups: [],
      paramGroup: null,
      iParamGroup: -1,
      isRunning: false,
      consoleLines: [],
      filenames: [],
      project: null,
      projects: [],
      imageWidth: 50,
      imageStyle: 'width: 50%',
      isGraph: false
    }
  },
  async created () {
    let res = await rpc.rpcRun('public_get_autumn_params')
    if (res.result) {
      console.log('> Model.created', res.result)
      this.paramGroups = res.result.paramGroups
      this.defaultParams = _.cloneDeep(res.result.params)
      this.params = _.cloneDeep(res.result.params)
      this.paramGroup = this.paramGroups[0]
      this.projects = res.result.projects
      this.countryDefaults = res.result.countryDefaults
    }
    this.charts = {}
    this.isGraph = false
    // res = await rpc.rpcRun('public_get_example_graph_data')
    // this.updateGraph(res.result.data)
    this.checkRun()
  },
  methods: {
    async checkRun () {
      let res = await rpc.rpcRun('public_check_autumn_run')
      if (res.result) {
        this.consoleLines = res.result.console
        if (this.$el.querySelector) {
          let container = this.$el.querySelector('#console-output')
          container.scrollTop = container.scrollHeight
        }
        if (_.keys(res.result.graph_data).length > 0) {
          this.isGraph = true
          this.updateGraph(res.result.graph_data)
        }
      }
      if (res.result.is_running) {
        this.isRunning = true
        setTimeout(this.checkRun, 2000)
      } else {
        this.isRunning = false
      }
    },
    updateGraph (data) {
      console.log(`> Model.updateGraph`)
      let paramKeys = _.keys(data.all_parameters_tried)
      for (let iParam of _.range(paramKeys.length)) {
        let paramKey = paramKeys[iParam]

        if (!(paramKey in this.charts)) {
          let chart = new ChartContainer(`#temp-chart-${iParam}`)
          chart.setTitle(paramKey)
          chart.setYLabel('')
          chart.setXLabel('accepted runs')
          this.charts[paramKey] = chart
        }

        let chart = this.charts[paramKey]

        let rejectedSets = data.rejection_dict[paramKey]
        let rejectedSetIndices = _.keys(rejectedSets)

        let iDataset = 0
        let values = data.all_parameters_tried[paramKey]
        let yValues = _.filter(values, (v, i) => data.whether_accepted_list[i])
        let xValues = _.range(1, yValues.length + 1)
        if (iDataset >= chart.getDatasets().length) {
          chart.addDataset(paramKey, xValues, yValues, '#037584')
        } else {
          chart.updateDataset(iDataset, xValues, yValues)
        }

        for (let iRejectedSet of rejectedSetIndices) {
          iDataset += 1
          let yValues = rejectedSets[iRejectedSet]
          let xValues = util.makeArray(yValues.length, parseFloat(iRejectedSet) + 1)
          let name = paramKey + iRejectedSet
          if (iDataset >= chart.getDatasets().length) {
            chart.addDataset(name, xValues, yValues, '#FC4A1A')
          } else {
            chart.updateDataset(iDataset, xValues, yValues)
          }
        }
      }
    },
    deleteBreakpoint (params, key, i) {
      this.params[key].value.splice(i, 1)
    },
    addBreakpoint (params, key) {
      this.params[key].value.push(_.max(this.params[key].value))
    },
    async breakpointCallback (params, key) {
      await util.delay(100)
      this.params[key].value = _.sortBy(
        this.params[key].value, v => _.parseInt(v))
    },
    selectParamGroup (i) {
      this.paramGroup = this.paramGroups[i]
    },
    async run () {
      let params = _.cloneDeep(this.params)
      for (let param of _.values(params)) {
        if (param.type === 'breakpoints') {
          param.value = _.sortedUniq(param.value)
        } else if ((param.type === 'number') || (param.type === 'double')) {
          param.value = parseFloat(param.value)
        } else if (param.type === 'integer') {
          param.value = _.parseInt(param.value)
        }
      }
      this.filenames = []
      this.isRunning = true
      this.isGraph = false
      this.project = ''
      this.consoleLines = []
      for (let key of _.keys(this.charts)) {
        let chart = this.charts[key]
        chart.destroy()
        delete this.charts[key]
      }

      setTimeout(this.checkRun, 2000)

      let res = await rpc.rpcRun('public_run_autumn', params)
      if (!res.result || !res.result.success) {
        this.consoleLines.push('Error: model crashed')
        this.isRunning = false
      } else {
        this.project = res.result.project
        if (!_.includes(this.projects, this.project)) {
          this.projects.push(this.project)
        }
        this.filenames = _.map(
          res.result.filenames,
          f => `${config.apiUrl}/file/${f}`)
        console.log('>> Model.run filenames', this.filenames)
      }
    },
    async changeProject (project) {
      let res = await rpc.rpcRun('public_get_project_images', project)
      if (res.result) {
        console.log('> Model.changeProject', project, res.result)
        this.filenames = _.map(
          res.result.filenames, f => `${config.apiUrl}/file/${f}`)
        this.consoleLines = res.result.consoleLines
        if (this.$el.querySelector) {
          let container = this.$el.querySelector('#console-output')
          container.scrollTop = container.scrollHeight
        }
        if (res.result.params) {
          this.params = res.result.params
        }
        console.log('> Model.changeProject', this.filenames)
      }
    },
    changeWidth () {
      this.imageStyle = `width: ${this.imageWidth}%`
      console.log('> Model.changeWidth', this.imageStyle)
    },
    selectDropDown (key) {
      if (key === 'country') {
        let country = this.params[key].value.toLowerCase()
        if (country !== this.country) {
          if (country in this.countryDefaults) {
            console.log('Model.selectDropdown', country)
            let diffParams = this.countryDefaults[country]
            _.assign(this.params, this.defaultParams)
            for (let key of _.keys(diffParams)) {
              console.log(
                '> Model.select',
                country,
                key,
                JSON.stringify(this.params[key].value),
                '->',
                JSON.stringify(diffParams[key].value))
              this.params[key].value = diffParams[key].value
            }
          }
        }
      }
    }

  }
}
</script>
