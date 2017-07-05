<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12 col-md-12 col-lg-12">
        <h1> Experiment - {{experiment.name}} </h1>
        <div class="row">
          <h3>Participants</h3>
          <button @click="makeInvite">
            Invite participant
          </button>
          <button @click="downloadResults">
            Download Results
          </button>
        </div>
        <div class="row">
          <table class="left-margin">
            <tr>
              <th>Invite</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Best Image</th>
              <th>Created</th>
              <th>Updated</th>
              <th>X</th>
            </tr>
            <tr v-for="participant in experiment.participants">
              <td>
                <router-link 
                    class="button" 
                    v-bind:to="getInviteRoute(participant)">
                  invite
                </router-link>
              </td>
              <td>
                <span v-if="participant.user">
                  {{participant.user.age}}
                </span>
              </td>
              <td>
                <span v-if="participant.user">
                  {{participant.user.gender}}
                </span>
              </td>
              <td>
                <span v-if="participant.state.ranks.length">
                  <a 
                      class="button"
                      v-bind:href="participant.state.ranks[0]">
                    Image
                  </a>
                </span>
              </td>
              <td>
                <span v-if="participant.createdAt">
                  {{participant.createdAt}}
                </span>
              </td>
              <td>
                <span v-if="participant.updatedAt">
                  {{participant.updatedAt}}
                </span>
              </td>
              <td>
              <a class="button" 
                  @click="deleteInvite(participant)">X</a>
              </td>
            </tr>
          </table>
        </div>

        <div class="row">
          <h3>Images</h3>
        </div>

        <div class="row">
          <img 
              class="small card"
              v-for="url in imageUrls"
              v-bind:src="url">
        </div>

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
import config from '../config'
import auth from '../modules/auth'
import util from '../modules/util'
import rpc from '../modules/rpc'

export default {
  name: 'experiment',
  data() {
    return {
      experiment: {}
    }
  },
  computed: {
    imageUrls: function () {
      let images = this.$data.experiment.Images
      return _.map( images, i => config.apiUrl + i.url)
    },
  },
  mounted () {
    let experimentId = this.$route.params.experimentId
    rpc
      .rpcRun('getExperiment', experimentId)
      .then((res) => {
        let experiment = res.data.experiment
        let participants = experiment.participants
        _.each(participants, participant => {
          let state = participant.state
          if ('ranks' in state) {
            state.ranks = _.map(state.ranks, r => config.apiUrl + r)
          }
        })
        console.log('>> Experiment.mounted', experiment)
        this.$data.experiment = experiment
      })
  },
  methods: {
    downloadResults() {
      console.log('>> Experiment.downloadResults')
      let experiment = this.$data.experiment
      let participants = experiment.participants
      let payload = _.map(participants, participant => {
        return {
          user: participant.user,
          ranks: participant.state.ranks,
          time: {
            start: participant.createdAt,
            end: participant.updatedAt
          }
        }
      })
      util.downloadObject('results.json', payload)
    },
    getInviteRoute(participant) {
      return `/participant/${participant.participateId}`
    },
    deleteInvite(participant) {
      rpc
        .rpcRun('deleteParticipant', participant.participateId)
        .then((res) => {
          console.log('>> Experiment.deleteInvite', res.data)
          let participants = this.$data.experiment.participants
          util.removeItem(participants, participant, 'participateId')
        })
    },
    makeInvite () {
      let experimentId = this.$route.params.experimentId
      let participants = this.$data.experiment.participants
      rpc
        .rpcRun('inviteParticipant', experimentId, 'test@test.com')
        .then((res) => {
          console.log('>> Experiment.makeInvite', res.data)
          participants.push(res.data.participant)
        })
    }
  }
}

</script>

