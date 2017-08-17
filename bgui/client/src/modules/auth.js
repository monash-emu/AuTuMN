import _ from 'lodash'
import rpc from '../modules/rpc'
import SHA224 from '../modules/sha224/sha224'
import util from '../modules/util'

function hashPassword (password) {
  return SHA224(password).toString()
}

let user = {
  authenticated: false
}

export default {

  user: user,

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
          console.log('>> auth.login success user', util.jstr(user))
          localStorage.setItem('user', util.jstr(user))
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
          .catch(resolve)
      } else {
        resolve()
      }
    })
  },

  logout () {
    localStorage.removeItem('user')
    user.authenticated = false
    return rpc.rpcRun('public_logout_user')
  }

}
