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
        if (res.data.success) {
          let returnUser = res.data.user
          user.authenticated = true
          _.assign(user, returnUser)
          user.password = payload.password
          console.log('>> auth.login success user', JSON.stringify(user))
          localStorage.setItem('user', JSON.stringify(user))
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

  update (updateUser) {
    console.log('>> auth.update', updateUser)
    updateUser.id = user.id
    // return axios.post(config.api + '/update', user)
    return rpc.rpcRun('login_update_user', updateUser)
  },

  restoreLastUser () {
    return new Promise((resolve, reject) => {
      let lastUser = JSON.parse(localStorage.getItem('user'))
      console.log('>> auth.restoreLastUser', lastUser)
      if (lastUser) {
        this
          .login(lastUser)
          .then(resolve)
      } else {
        resolve()
      }
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
