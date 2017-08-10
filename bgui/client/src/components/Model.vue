<template>
  <div style=" padding: 1em">
    <md-layout md-column>

      <div style="text-align: center;">
        <br>
        <div class="md-display-1">
          TB Epidemiological Model
        </div>
        <br>
        <br>
      </div>

      <md-layout md-row>

        <md-whiteframe
            style="
            width: 50%;
            height: calc(100vh - 230px);
            overflow: auto">
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

                    <div v-else-if="params[key].type == 'slider'">
                      <label>{{ params[key].label }}</label>
                      <div style="height: 2.5em"></div>
                      <vue-slider
                          :max="params[key].max"
                          :interval="params[key].interval"
                          v-model="params[key].value">
                      </vue-slider>
                    </div>

                  </md-layout>
                </div>
              </div>
            </md-tab>
          </md-tabs>
        </md-whiteframe>

        <md-whiteframe
            style="
            padding: 1em;
            width: 50%;
            height: calc(100vh - 230px);
            overflow: auto">
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
  import vueSlider from 'vue-slider-component'

  export default {
    name: 'experiments',
    components: {vueSlider},
    data () {
      return {
        paramGroups: [],
        params: {},
        isRunning: false,
        consoleLines: []
      }
    },
    created () {
      this.checkRun()
      rpc
        .rpcRun('public_get_autumn_params')
        .then(res => {
          this.paramGroups = res.data.paramGroups
          this.params = res.data.params
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
            }
            if (res.data.is_running) {
              this.$data.isRunning = true
              setTimeout(() => { this.checkRun() }, 2000)
            } else {
              this.$data.isRunning = false
            }
          })
      },
      run () {
        let params = this.$data.params
        this.$data.isRunning = true
        rpc
          .rpcRun(
            'public_run_autumn', params)
          .then((res) => {
            console.log('>> Home.run res', res)
            if (!res.data.success) {
              this.$data.consoleLines.push('Error: model crashed')
            }
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

