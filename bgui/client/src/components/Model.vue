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
            <div style="padding: 30px 15px">
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
                    md-flex="100"
                    style="background-color: #EEE;">
                  <div
                      id="console-output"
                      style="
                      height: 350px;
                      overflow-y: scroll;
                      font-family: Courier, fixed;
                      font-size: 0.9em">

                    <div style="margin: 0 8px" v-for="line in consoleLines">
                      {{ line }}
                    </div>

                  </div>
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
        paramGroup: null,
        iParamGroup: -1,
      }
    },
    created () {
      this.checkRun()
      rpc
        .rpcRun('public_get_autumn_params')
        .then(res => {
          this.paramGroups = res.data.paramGroups
          this.params = res.data.params
          this.paramGroup = this.paramGroups[0]
          console.log(util.jstr(this.params))
        })
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
              if (this.$el.querySelector) {
                let container = this.$el.querySelector('#console-output')
                container.scrollTop = container.scrollHeight
              }
            }
            if (res.data.is_running) {
              this.$data.isRunning = true
              setTimeout(() => {
                this.checkRun()
              }, 2000)
            } else {
              this.$data.isRunning = false
            }
          })
      },
      deleteBreakpoint(params, key, i) {
        this.params[key].value.splice(i, 1)
      },
      addBreakpoint(params, key) {
        this.params[key].value.push(_.max(this.params[key].value))
      },
      breakpointCallback(params, key) {
        // this.params[key].value.sort()
      },
      selectParamGroup (i) {
        this.paramGroup = this.paramGroups[i]
      },
      run () {
        let params = this.$data.params
        for (let param of _.values(this.params)) {
          if (param.type === "breakpoints") {
            param.value = _.sortedUniq(param.value)
            console.log(util.jstr(param))
          }
        }
        this.$data.isRunning = true
        this.$data.consoleLines = []
        rpc
          .rpcRun(
            'public_run_autumn', params)
          .then((res) => {
            console.log('>> Home.run res', res)
            if (!res.data.success) {
              this.$data.consoleLines.push('Error: model crashed')
            } else {
              this.checkRun()
            }
          })
          .catch((res) => {
            this.$data.isRunning = false
          })
        setTimeout(() => this.checkRun(), 1000)
      }
    }
  }

</script>

