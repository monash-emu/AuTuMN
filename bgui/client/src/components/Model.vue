<template>
  <md-layout md-row>

    <md-layout
      md-row
      style="
        height: calc(100vh - 48px);
        overflow: hidden">

      <div
        style="
          width: 230px;
          height: calc(100vh - 48px);
          overflow: auto;
          padding: 30px 0 30px 0;
          border-right: 1px solid rgb(221, 221, 221);">

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
            {{ thisParamGroup.name }}
          </md-list-item>
        </md-list>
      </div>

      <div
        style="
          height: calc(100vh - 48px);
          overflow: auto;
          width: 220px;
          padding: 30px 15px;
          border-right: 1px solid rgb(221, 221, 221);">

        <md-layout
          v-if="paramGroup"
          md-column>

          <h2 class="md-heading">
            {{ paramGroup.name }}
          </h2>

          <md-layout
            v-for="(key, i) in paramGroup.keys"
            :key="i"
            md-column>

            <div v-if="params[key].type === 'boolean'">
              <md-checkbox
                :id="key"
                v-model="params[key].value"
                type="checkbox"
                tabindex="0">
                {{ params[key].label }}
              </md-checkbox>
            </div>

            <div v-else-if="params[key].type === 'drop_down'">
              <md-input-container>
                <label>{{ params[key].label }}</label>
                <md-select
                  v-model="params[key].value"
                  @change="selectDropDown(key)"
                >
                  <md-option
                    v-for="(option, i) in params[key].options"
                    :value="option"
                    :key="i">
                    {{ option }}
                  </md-option>
                </md-select>
              </md-input-container>
            </div>

            <div
              v-else-if="(params[key].type === 'number') ||
                (params[key].type === 'double') ||
            (params[key].type === 'integer')">
              <md-input-container>
                <label>{{ params[key].label }}</label>
                <md-input
                  v-model="params[key].value"
                  type="number"
                  step="any"/>

              </md-input-container>
            </div>

            <div v-else-if="params[key].type === 'slider'">
              <label>{{ params[key].label }}</label>
              <div style="height: 2.5em"/>
              <vue-slider
                :max="params[key].max"
                :interval="params[key].interval"
                v-model="params[key].value"/>
            </div>

            <div v-else-if="params[key].type === 'breakpoints'">
              <label>{{ params[key].label }}</label>
              <div style="height: 2.5em"/>
              <md-layout
                v-for="(breakpoint, i) of params[key].value"
                :key="i"
                md-row
                style="width: 200px">
                <vue-slider
                  :max="100"
                  :interval="1"
                  v-model="params[key].value[i]"
                  style="width: 130px"
                  @drag-end="breakpointCallback(params, key)"/>
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
      </div>

      <md-layout md-flex>
        <div
          style="
            height: calc(100vh - 48px);
            overflow: auto;
            padding: 30px 15px">

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
                  :value="p"
                  :key="i">
                  {{ p }}
                </md-option>
              </md-select>
            </md-input-container>

            <div style="width: 100%">
              <md-layout
                md-row
                md-vertical-align="center">

                <md-button
                  :disabled="isRunning"
                  md-flex="true"
                  class="md-raised"
                  @click="run()">
                  Run
                </md-button>

                <md-spinner
                  v-if="isRunning"
                  :md-size="30"
                  md-indeterminate/>
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
                  v-for="(line, i) in consoleLines"
                  :key="i"
                  style="
                      margin: 0 8px;
                      word-wrap: break-word;">
                  {{ line }}
                </div>

              </div>
            </md-layout>

            <h2
              v-show="isGraph"
              class="md-heading">
              Progress in Uncertainty Runs
            </h2>
            <md-layout>
              <div id="temp-chart-0"/>
              <div id="temp-chart-1"/>
              <div id="temp-chart-2"/>
              <div id="temp-chart-3"/>
              <div id="temp-chart-4"/>
              <div id="temp-chart-5"/>
            </md-layout>

            <h2
              class="md-heading"
              style="margin-top: 1.5em;">
              Model Results
            </h2>

            <vue-slider
              v-if="filenames.length > 0"
              :max="100"
              :min="10"
              :interval="1"
              v-model="imageWidth"
              style="width: 100%"
              @callback="changeWidth(imageWidth)"/>

            <md-layout style="width: 100%">
              <md-card
                v-for="(filename, i) in filenames"
                :style="imageStyle"
                :key="i">
                <md-card-media>
                  <img
                    :src="filename"
                    style="width: 100%">
                </md-card-media>
              </md-card>
            </md-layout>
          </md-layout>

        </div>
      </md-layout>
    </md-layout>

  </md-layout>
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
  name: 'Experiments',
  components: { vueSlider },
  data() {
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
  async created() {
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
    async checkRun() {
      let res = await rpc.rpcRun('public_check_autumn_run')

      if (res.result) {
        console.log('Model.checkRun', res.result)
        this.consoleLines = res.result.console_lines
        if (this.$el.querySelector) {
          await util.delay(100)
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
      } else if (res.result.is_completed) {
        this.project = res.result.project
        if (!_.includes(this.projects, this.project)) {
          this.projects.push(this.project)
        }
        this.changeProject(res.result.project)
        this.isRunning = false
      } else {
        this.isRunning = false
      }
    },
    updateGraph(data) {
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
          let xValues = util.makeArray(
            yValues.length,
            parseFloat(iRejectedSet) + 1
          )
          let name = paramKey + iRejectedSet
          if (iDataset >= chart.getDatasets().length) {
            chart.addDataset(name, xValues, yValues, '#FC4A1A')
          } else {
            chart.updateDataset(iDataset, xValues, yValues)
          }
        }
      }
    },
    deleteBreakpoint(params, key, i) {
      this.params[key].value.splice(i, 1)
    },
    addBreakpoint(params, key) {
      this.params[key].value.push(_.max(this.params[key].value))
    },
    async breakpointCallback(params, key) {
      await util.delay(100)
      this.params[key].value = _.sortBy(this.params[key].value, v =>
        _.parseInt(v)
      )
    },
    selectParamGroup(i) {
      this.paramGroup = this.paramGroups[i]
    },
    async run() {
      let params = _.cloneDeep(this.params)
      for (let param of _.values(params)) {
        if (param.type === 'breakpoints') {
          param.value = _.sortedUniq(param.value)
        } else if (param.type === 'number' || param.type === 'double') {
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
      }
    },
    async changeProject(project) {
      let res = await rpc.rpcRun('public_get_project_images', project)
      if (res.result) {
        console.log('> Model.changeProject', project, res.result)
        this.filenames = _.map(
          res.result.filenames,
          f => `${config.apiUrl}/file/${f}`
        )
        this.consoleLines = res.result.consoleLines
        if (this.$el.querySelector) {
          await util.delay(100)
          let container = this.$el.querySelector('#console-output')
          container.scrollTop = container.scrollHeight
        }
        if (res.result.params) {
          this.params = res.result.params
        }
        console.log('> Model.changeProject', this.filenames)
      }
    },
    changeWidth() {
      this.imageStyle = `width: ${this.imageWidth}%`
      console.log('> Model.changeWidth', this.imageStyle)
    },
    selectDropDown(key) {
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
                JSON.stringify(diffParams[key].value)
              )
              this.params[key].value = diffParams[key].value
            }
          }
        }
      }
    }
  }
}
</script>
