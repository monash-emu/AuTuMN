// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.

import Vue from 'vue'
import VueMaterial from 'vue-material'

import App from './App'
import auth from './modules/auth'

import router from './router.js'

Vue.config.productionTip = false
Vue.use(VueMaterial)

auth.restoreLastUser()
  .then(res => {
    console.log('> Got here')
    new Vue({
      el: '#app',
      router,
      template: '<App/>',
      components: {App}
    })
  })

