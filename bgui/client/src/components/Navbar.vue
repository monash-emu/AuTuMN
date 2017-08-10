<template>
  <md-toolbar class="md-dense">

    <h2 class="md-title" style="padding-left: 1em; flex: 1" >AuTuMN</h2>

    <router-link to='/' class="md-button">
      Model
    </router-link>

    <span v-if="user.authenticated">

      <md-menu >

        <md-button md-menu-trigger>
            {{user.name}}
        </md-button>

        <md-menu-content>
          <md-menu-item @click="editUser">Edit User
          </md-menu-item>
          <md-menu-item @click="logout">Logout</md-menu-item>
        </md-menu-content>
      </md-menu>

    </span>

    <router-link v-else to='/login' tag='md-button'>
      Login
    </router-link>

  </md-toolbar>
</template>

<style scoped>
</style>

<script>

  import axios from 'axios'
  import config from '../config'
  import auth from '../modules/auth'
  import router from '../router'

  export default {
    name: 'navbar',
    data () {
      return {
        user: auth.user
      }
    },
    methods: {
      editUser () {
        router.push('/edit-user')
      },
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

