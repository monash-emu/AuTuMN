<template>
  <md-toolbar class="md-dense">

    <h2
        class="md-title"
        style="cursor: pointer; flex: 1"
        @click="home()">
      AuTuMN - TB Epi Modelling
    </h2>

    <md-menu v-if="user.authenticated">

      <md-button md-menu-trigger>
        {{user.name}}
      </md-button>

      <md-menu-content>
        <md-menu-item @click="editUser">Edit User</md-menu-item>
        <md-menu-item @click="logout">Logout</md-menu-item>
      </md-menu-content>

    </md-menu>

    <router-link
        v-else to='/login'
        tag='md-button'>
      Login
    </router-link>

  </md-toolbar>
</template>

<script>

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
      home () {
        router.push('/')
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

