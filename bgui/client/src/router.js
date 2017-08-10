import Vue from 'vue'
import Router from 'vue-router'
Vue.use(Router)

import Model from './components/Model'
import Login from './components/Login'
import Register from './components/Register'
import EditUser from './components/EditUser'

import auth from './modules/auth'

let router = new Router({
  routes: [
    {
      path: '/',
      name: 'model',
      component: Model
    },
    {
      path: '/login',
      name: 'login',
      component: Login
    },
    {
      path: '/register',
      name: 'register',
      component: Register
    },
    {
      path: '/edit-user',
      name: 'editUser',
      component: EditUser
    },
  ]
})

export default router