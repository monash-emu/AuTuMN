import axios from 'axios'
import _ from 'lodash'
import config from '../config'

// really important for using with passport.js 
// https://stackoverflow.com/questions/40941118/axios-wont-send-cookie-ajax-xhrfields-does-just-fine
axios.defaults.withCredentials = true

export default {
  rpcRun (...args) {
    const n = args.length
    const payload = {
      name: args[0],
      args: _.takeRight(args, n - 1)
    }
    console.log('>> rpc.rpcRun', args)
    return axios.post(`${config.api}/rpc-run`, payload)
  },

  rpcUpload (name, inputEventTarget, ...args) {
    let files = inputEventTarget.files
    let formData = new FormData()
    formData.append('name', name)
    formData.append('args', JSON.stringify(args))
    _.each(files, file => {
      formData.append('uploadFiles', file, file.name)
    })
    console.log('>> rpcUpoad ', name, args, _.map(files, 'name'))
    return axios.post(`${config.api}/rpc-upload`, formData)
  }

}
