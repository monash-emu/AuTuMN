<template>
  <header class="sticky">

    <router-link to="/" class="logo">
      AuTuMN
    </router-link>

    <span v-if="user.authenticated">

      <router-link to='/experiments' class='button'>
        Experiments
      </router-link>

      <span class="user">
        {{user.email}}
      </span>

      <router-link to='/edit-user' class='button'>
        Edit
      </router-link>
      <button @click="logout">Logout</button>

    </span>

    <span v-else>
      <router-link to='/login' class='button'>
        Login
      </router-link>
    </span>

  </header>
</template>

<style scoped>
</style>

<script>

import auth from '../modules/auth'
import axios from 'axios'
import config from '../config'

export default {
  name: 'navbar',
  data() {
    return {
      user: auth.user
    }
  },
  methods: {
    logout () {
      auth
        .logout()
        .then(() => {
          this.$router.push('/login')
        })
    }
  }
}
</script>

