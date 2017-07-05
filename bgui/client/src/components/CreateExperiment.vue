<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12 col-md-12 col-lg-12">
        <h1> Create Experiment </h1>
        <form v-on:submit.prevent="submit">
          <label>Name</label>
          <input 
              type="text"
              name="uploadFiles"
              v-model="name">
              </input>
          <input 
              type="file" 
              id="file-input"
              multiple
              @change="filesChange($event)">
              </input>
          <label for="file-input" class="button">
            Upload files
          </label>          
          {{fileStr}}
          <br>
          <button type="submit">Submit</button>
        </form>
      </div>
    </div>
  </div>
</template>

<!-- Add 'scoped' attribute to limit CSS to this component only -->
<style scoped>
</style>

<script>
import axios from 'axios'
import _ from 'lodash'

import config from '../config'
import auth from '../modules/auth'
import util from '../modules/util'
import rpc from '../modules/rpc'

export default {
  name: 'createExperiment',
  data() {
    return {
      target: null,
      name: '',
      files: '',
      fileStr: ''
    }
  },
  methods: {
    filesChange ($event) {
      this.$data.target = $event.target
      this.$data.fileStr = `${event.target.files.length} files`
    },
    submit ($event) {
      rpc.rpcUpload(
        'uploadImages', this.$data.target, this.$data.name, auth.user.id)
        .then(res => {
          console.log('>> CreateExperiment.submit', res.data)
          let experimentId = res.data.experimentId
          this.$router.push('/experiment/' + experimentId)
        })
    }
  }
}
</script>

