<template>
  <md-layout md-align="center">
    <md-whiteframe style="margin-top: 4em; padding: 3em">
      <md-layout 
        md-flex="50" 
        md-align="center" 
        md-column>

        <h2 class="md-display-2">
          Login to {{ title }}
        </h2>

        <form 
          novalidate 
          class="login-screen"
          @submit.prevent="submit">

          <md-input-container>
            <label>E-mail address</label>
            <md-input
              v-model="email"
              type="text"
              placeholder="E-mail address"/>
          </md-input-container>

          <md-input-container>
            <label>Password</label>
            <md-input
              v-model="rawPassword"
              type="password"
              placeholder="Password"/>
          </md-input-container>

          <md-button 
            type="submit" 
            class="md-raised md-primary">login</md-button>

          <div 
            v-if="error" 
            style="color: red">
            {{ error }}
          </div>

          <div style="margin-top: 3em">
            New to {{ title }}? &nbsp;
            <router-link to="/register">Register</router-link>
          </div>

        </form>

      </md-layout>
    </md-whiteframe>
  </md-layout>
</template>

<script>
import auth from '../modules/auth'
import config from '../config'

export default {
  name: 'Login',
  data() {
    return {
      title: config.title,
      email: '',
      rawPassword: '',
      error: ''
    }
  },
  methods: {
    async submit() {
      let payload = {
        email: this.$data.email,
        rawPassword: this.$data.rawPassword
      }
      console.log('> Login.submit', payload)
      let response = await auth.login(payload)
      console.log('> Login.submit response', response)

      if (response.result) {
        this.$router.push('/')
      } else {
        this.error = response.error.message
      }
    }
  }
}
</script>
