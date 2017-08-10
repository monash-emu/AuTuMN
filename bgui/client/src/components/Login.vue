<template>
  <div style="padding: 1em">

    <h2 class="md-display-2">
      {{ title }}
    </h2>

    <form novalidate class="login-screen"
          v-on:submit.prevent="submit">

      <md-input-container>
        <label>E-mail address</label>
        <md-input
            type='text'
            v-model='email'
            placeholder='E-mail address'>
        </md-input>
      </md-input-container>

      <md-input-container>
        <label>Password</label>
        <md-input
            type='password'
            v-model='password'
            placeholder='Password'>
        </md-input>
      </md-input-container>

      <md-button type="submit" class="md-raised md-primary">login</md-button>

      <br>
      <br>
      <br>
      New to Versus? &nbsp;
      <router-link to="/register">Register</router-link>

      </md-layout>


      <div v-if="msg" class="card error">
        {{ msg }}
      </div>

    </form>

  </div>
</template>

<script>
  import axios from 'axios'
  import Router from 'vue-router'
  import auth from '../modules/auth'

  export default {
    name: 'Login',
    data() {
      return {
        title: 'Login to AuTuMN',
        email: '',
        password: '',
        msg: ''
      }
    },
    methods: {
      submit(e) {
        let payload = {
          email: this.$data.email,
          password: this.$data.password
        }
        console.log('>> Login.submit', payload)
        auth
          .login(payload)
          .then((res) => {
            if (res.data.success) {
              this.$router.push('/')
            } else {
              this.$data.msg = res.data.msg
            }
          })
      }
    }
  }
</script>
