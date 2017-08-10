<template>
  <div style="padding: 1em">
    <h2 class="md-display-2">
      {{ title }}
    </h2>
    <form v-on:submit.prevent="submit">
      <md-input-container>
        <label>User name</label>
        <md-input
            type='text'
            v-model='name'
            placeholder='User name'>
        </md-input>
      </md-input-container>
      <md-input-container>
        <label>E-mail address</label>
        <md-input
            type='text'
            v-model='email'
            placeholder='E-mail address'>
        </md-input>
      </md-input-container>
      <md-input-container>
        <label>New Password</label>
        <md-input
            type='password'
            v-model='password'
            placeholder='New Password'>
        </md-input>
      </md-input-container>
      <md-input-container>
        <label>New Password</label>
        <md-input
            type='password'
            v-model='passwordv'
            placeholder='Confirm Password'>
        </md-input>
      </md-input-container>
      <md-button type="submit" class="md-raised md-primary">
        Update
      </md-button>
      <div v-if="errors.length" class="card error">
        <ul>
          <li v-for="err in errors">
            {{ err }}
          </li>
        </ul>
      </div>
    </form>
  </div>
</template>

<script>
  import axios from 'axios'
  import Router from 'vue-router'
  import auth from '../modules/auth'
  import _ from 'lodash'
  import util from '../modules/util'

  export default {
    name: 'EditUser',
    data () {
      let payload = _.assign({}, auth.user)
      _.assign(payload, {
        title: 'Edit Your Details',
        password: '',
        passwordv: '',
        errors: []
      })
      console.log('>> EditUser.data', payload)
      return payload
    },
    methods: {
      submit (e) {
        let payload = {}
        const keys = ['name', 'email', 'password', 'passwordv']
        for (let key of keys) {
          if (this.$data[key]) {
            payload[key] = this.$data[key]
          }
        }
        console.log('>> EditUser.submit', util.jstr(payload))
        auth
          .update(payload)
          .then((res) => {
            if (res.data.success) {
              console.log('>> Register.submit success: login')
              return auth
                .login({
                  email: payload.email,
                  password: payload.password
                })
            } else {
              console.log('>> Register.submit fail', res.data)
              this.$data.errors = res.data.errors
            }
          })
      }
    }
  }
</script>
