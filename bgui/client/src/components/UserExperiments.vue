<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12 col-md-12 col-lg-12">
        <h1>
          Your experiments
        </h1>
      </div>
      <div class="col-sm-12 col-md-12 col-lg-12">
        <table class="left-margin">
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th>View</th>
              <th>Delete</th>
            </tr>
          </thead>
          <tr v-for="experiment in experiments">
            <td>
              {{experiment.name}}
            </td>
            <td>{{experiment.description}}</td>
            <td>
              <router-link 
                  class="button"
                  v-bind:to="getExperimentRoute(experiment.id)">
                view
              </router-link>
            </td>
            <td>
              <button @click="deleteExperiment(experiment)">X</button>
            </td>
          </tr>
          <tr>
            <td colspan="6">
              <router-link 
                  class="button"
                  to='/create-experiment'>
                Create New Experiment
              </router-link>
            </td>
          </tr>
        </table>
      </div>
    </div>
  </div>
</template>

<!-- Add 'scoped' attribute to limit CSS to this component only -->
<style scoped>
.left-margin {
  margin-left: 10px
}
</style>

<script>
import axios from 'axios'
import auth from '../modules/auth'
import config from '../config'
import util from '../modules/util'
import rpc from '../modules/rpc'
import _ from 'lodash'


export default {
  name: 'experiments',
  data() {
    return {
      experiments: []
    }
  },
  mounted() {
    rpc.rpcRun('getExperiments', auth.user.id)
      .then((res) => {
        this.$data.experiments = res.data.experiments
      })
  },
  methods: {
    deleteExperiment(experiment) {
      rpc.rpcRun('deleteExperiment', experiment.id)
        .then((res) => {
          util.removeItem(
            this.$data.experiments, experiment, 'id')
        })
    },
    getExperimentRoute(experimentId) {
      return 'experiment/' + experimentId
    }
  }
}

</script>

