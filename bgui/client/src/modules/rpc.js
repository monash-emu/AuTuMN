import axios from 'axios'
import _ from 'lodash'
import config from '../config'
import auth from '../modules/auth'
/**
 * rpc module provides a clean rpc interface for JSON-based
 * api with the server
 */

// really important for using with passport.js
// https://stackoverflow.com/questions/40941118/axios-wont-send-cookie-ajax-xhrfields-does-just-fine

axios.defaults.withCredentials = true

export default {

  rpcRun (name, ...args) {
    console.log('>> rpc.rpcRun', name, args)
    return axios.post(`${config.apiUrl}/api/rpc-run`, {name, args})
  },

  rpcUpload (name, files, ...args) {
    let formData = new FormData()
    formData.append('name', name)
    formData.append('args', JSON.stringify(args))
    _.each(files, f => {
      formData.append('uploadFiles', f, f.name)
    })
    console.log('>> rpc.rpcUpoad', name, args, _.map(files, 'name'))
    return axios.post(`${config.apiUrl}/api/rpc-upload`, formData)
  }

}
