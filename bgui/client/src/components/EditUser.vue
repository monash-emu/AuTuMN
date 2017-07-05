<template>
  <div class="container">
    <div class="row">
      <div 
          class="
            col-sm-6 col-sm-offset-3 
            col-md-6 col-md-3-offset 
            col-lg-4 col-lg-offset-4"
          style="
            padding-top: 4em">
        <form class="login-screen"
              v-on:submit.prevent="submit">
          <h2>{{ title }}</h2>
          <input 
              type='text'
              v-model='firstName'
              placeholder='First name'>
          </input>
          <br>
          <input 
              type='text'
              v-model='lastName'
              placeholder='Last name'>
          </input>
          <br>
          <input 
              type='text'
              v-model='email'
              placeholder='E-mail address'>
          </input>
          <br>
          <input 
              type='password'
              v-model='password'
              placeholder='New Password'>
          </input>
          <br>
          <input 
              type='password'
              v-model='passwordv'
              placeholder='New Password Confirm'>
          </input>
          <br>
          <button>
            Update
          </button>
          <div v-if="errors.length" class="card error">
            <ul>
              <li v-for="err in errors">
                {{ err }}
              </li>
            </ul>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import Router from 'vue-router'
import auth from '../modules/auth'
import _ from 'lodash'

export default {
  name: 'EditUser',
  data() {
    let payload = _.assign({}, auth.user)
    _.assign(payload, {
      title: 'Edit User Details',
      password: '',
      passwordv: '',
      errors: []
    })
    return payload
  },
  methods: {
    submit(e) {
      let credentials = {}
      let keys = ['id', 'firstName', 'lastName', 'email', 'password', 'passwordv']
      _.each(keys, key => {
        if (this.$data[key]) {
          credentials[key] = this.$data[key]
        }
      })
      console.log('>> EditUser.submit', credentials)
      auth
        .update(credentials)
        .then((res) => {
          if (res.data.success) {
            console.log('>> Register.submit success: login')
            return auth.login({
              username: credentials.email,
              password: credentials.password
            })
          } else {
            console.log('>> Register.submit fail', res.data)
            this.$data.errors = res.data.errors
            return { data: { success: false } }
          }
        })
        .then((res) => {
          if (res.data.success) {
            this.$router.push('/experiments')
          }
        })
    }
  }
}
</script>
