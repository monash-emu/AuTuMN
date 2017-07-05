import _ from 'lodash'
import Vue from 'vue'
import Router from 'vue-router'
import Home from '@/components/Home'
import UserExperiments from '@/components/UserExperiments'
import Experiment from '@/components/Experiment'
import CreateExperiment from '@/components/CreateExperiment'
import Login from '@/components/Login'
import Register from '@/components/Register'
import EditUser from '@/components/EditUser'
import auth from '../modules/auth'

Vue.use(Router)

let router = new Router({
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home
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
    {
      path: '/experiments',
      name: 'expriments',
      component: UserExperiments
    },
    {
      path: '/create-experiment',
      name: 'createExpriment',
      component: CreateExperiment
    },
    {
      path: '/experiment/:experimentId',
      name: 'expriment',
      component: Experiment
    }
  ]
})

export default router
