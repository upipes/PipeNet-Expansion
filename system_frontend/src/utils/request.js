import axios from 'axios'
import router from "../router";
import {ElMessage} from 'element-plus';

// 请求拦截器 请求头中加入token
axios.interceptors.request.use(config => {
    // 如果存在token，请求携带这个token
    const token = sessionStorage.getItem('tokenStr')
    if (token) {
        // 如果token有值  就在请求头信息里添加一个token值
        config.headers['Authorization'] = 'Bearer ' + token
    }
    return config
}, error => {
    console.log(error)
})

//响应拦截器
axios.interceptors.response.use(success => {
    // if (success.status === 200) {
    //   ElMessage.success({ message: 'OK!' })
    // } else if (success.status === 201) {
    //   ElMessage.success({ message: 'Created!' })
    // }
    return success
},error=>{
    if(error.response.code===504||error.response.code===404){
        ElMessage.error({message: '服务器忙!'});
    }else if(error.message.code===403){
        ElMessage.error({message: '权限不足，请联系管理员!'});
    }else if(error.message.code===401){
        ElMessage.error({message: '尚未登录，请登录!'});
        sessionStorage.removeItem('Authorization');
        //跳转到登陆页面
        router.replace('/login');
    }else{
        ElMessage.error({message: '错误!'});
    }
})

const base = `http://127.0.0.1:8000/api`
// 传送json格式的post请求
export const postRequest = (url, params) => {
    return axios.post(`${base}${url}`, params)
}

// 传送json格式的get请求
export const getRequest = (url, params) => {
    return axios.get(`${base}${url}`, params)
}

// 传送json格式的put请求
export const putRequest = (url, params) => {
    return axios.put(`${base}${url}`, params)
}

// 传送json格式的delete请求
export const deleteRequest = (url, params) => {
    return axios.delete(`${base}${url}`, params)
}
