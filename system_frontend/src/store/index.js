import { createStore } from 'vuex'

export default createStore({
  state: {
    // 存储token
    Authorization: sessionStorage.getItem('Authorization') ? sessionStorage.getItem('Authorization') : '',
    userName:"",
    userEmail:"",
    userDateJoined:"",
    userRoles:"",
    userLastLogin:"",
  },

  mutations: {
    // 修改token，并将token存入sessionStorage
    changeLogin (state, user) {
      state.Authorization = user.Authorization;
      sessionStorage.setItem('Authorization', user.Authorization);
    },
    editUserName(state, x) {
      state.userName = x
    },
    editUserEmail(state, x) {
      state.userEmail = x
    },
    editUserDateJoined(state, x) {
      state.userDateJoined = x
    },
    editUserRoles(state, x) {
      state.userRoles = x
    },
    editUserLastLogin(state, x) {
      state.userLastLogin = x
    },
  }
})
