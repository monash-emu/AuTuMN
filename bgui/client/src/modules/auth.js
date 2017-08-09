import axios from 'axios'
import _ from 'lodash'
import config from '../config'
import rpc from '../modules/rpc'
import SHA224 from '../modules/sha224/sha224'

axios.defaults.withCredentials = true

function hashPassword (password) {
  return SHA224(password).toString()
}

let user = {
  authenticated: false
}

export default {

  // User object will let us check authentication status
  user: user,

  // Send a request to the login URL and save the returned JWT
  login (newUser) {
    let payload = _.cloneDeep(newUser)
    payload.password = hashPassword(payload.password)
    payload.username = payload.name
    delete payload.passwordv
    return rpc
      .rpcRun('public_login_user', payload)
      .then(res => {
        console.log('>> auth.login res', res)
        if (res.data.success) {
          localStorage.setItem('user', JSON.stringify(payload))
          let returnUser = res.data.user
          user.authenticated = true
          _.assign(user, returnUser)
          console.log('>> auth.login user', user)
        }
        return res
      })
  },

  register (newUser) {
    let payload = _.cloneDeep(newUser)
    payload.password = hashPassword(payload.password)
    payload.username = payload.name
    delete payload.passwordv
    return rpc.rpcRun('public_create_user', payload)
  },

  update (user) {
    console.log('>> auth.update', user)
    return axios.post(config.api + '/update', user)
  },

  restoreLastUser () {
    return new Promise((resolve, reject) => {
      let lastUser = JSON.parse(localStorage.getItem('user'))
      console.log('>> auth.restoreLastUser', lastUser)
      this
        .login(lastUser)
        .then(resolve)
    })
  },

  // To log out, we just need to remove the token
  logout () {
    localStorage.removeItem('user')
    user.authenticated = false
    return rpc.rpcRun('public_logout_user')
  }

  // // if using JWT
  // getAuthHeader () {
  //   return {
  //     'Authorization': 'Bearer ' + localStorage.getItem('access_token')
  //   }
  // }
}
