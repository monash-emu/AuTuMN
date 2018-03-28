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
              style="width: 180px; padding-top: 30px ">

            <h2 class="md-heading"
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

          <md-whiteframe style="width: 220px">

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
                      <md-icon>delete</md-icon>
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

                <div style="width: 100%">
                  <md-layout md-row>
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
                      height: 350px;
                      overflow-y: scroll;
                      font-family: Courier, fixed;
                      font-size: 0.9em">

                    <div
                        style="margin: 0 8px"
                        v-for="(line, i) in consoleLines"
                        :key="i">
                      {{ line }}
                    </div>

                  </div>
                </md-layout>

                <md-whiteframe
                  style="
                    width: 100%">
                  <md-card
                      v-for="(filename, i) in filenames"
                      :key="i">
                    <img
                        style="width:500px"
                        :src="filename">
                  </md-card>
                </md-whiteframe>
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

Vue.use(VueScrollTo)

export default {
  name: 'experiments',
  components: {vueSlider},
  data () {
    return {
      paramGroups: [],
      params: {},
      isRunning: false,
      consoleLines: [],
      filenames: [],
      paramGroup: null,
      iParamGroup: -1
    }
  },
  async created () {
    this.checkRun()
    let res = await rpc.rpcRun('public_get_autumn_params')
    if (res.result) {
      this.paramGroups = res.result.paramGroups
      this.params = res.result.params
      this.paramGroup = this.paramGroups[0]
    }
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
      }
      if (res.result.is_running) {
        this.isRunning = true
        setTimeout(() => { this.checkRun() }, 2000)
      } else {
        this.isRunning = false
      }
    },
    deleteBreakpoint (params, key, i) {
      this.params[key].value.splice(i, 1)
    },
    addBreakpoint (params, key) {
      this.params[key].value.push(_.max(this.params[key].value))
    },
    breakpointCallback (params, key) {
      // this.params[key].value.sort()
    },
    selectParamGroup (i) {
      this.paramGroup = this.paramGroups[i]
    },
    async run () {
      let params = this.$data.params
      for (let param of _.values(this.params)) {
        if (param.type === 'breakpoints') {
          param.value = _.sortedUniq(param.value)
          console.log(util.jstr(param))
        }
      }
      this.isRunning = true
      this.consoleLines = []
      let res = await rpc.rpcRun('public_run_autumn', params)
      console.log('>> Home.run res', res)
      if (!res.result) {
        this.consoleLines.push('Error: model crashed')
        this.isRunning = false
      } else {
        console.log('what')
        this.filenames = _.map(
          res.result.filenames, f => `${config.apiUrl}/file/${f}`)
        console.log(this.filenames)
        this.checkRun()
      }
      setTimeout(() => this.checkRun(), 1000)
    }
  }
}
</script>
