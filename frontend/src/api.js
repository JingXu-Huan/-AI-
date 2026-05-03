import axios from 'axios';
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});
export const getAllRepairs = () => apiClient.get('/repairs');
export const getRepairsByStatus = (status) => apiClient.get('/repairs/status/' + status);
export const getRepairById = (id) => apiClient.get('/repairs/' + id);
export const getDescFromAI = (data) => apiClient.post('/getDesc', data);
export const analyze = (data) => apiClient.post('/analyze', data);
export const writeToDB = (data, imageFile) => {
  const formData = new FormData();
  formData.append('data', new Blob([JSON.stringify(data)], { type: 'application/json' }));
  if (imageFile) {
    formData.append('file', imageFile);
  }
  // 不用 apiClient，直接用 axios 发，不带默认的 JSON header
  return axios.post('/api/writeToDB', formData);
};
export const updateRepairStatus = (id, status) => apiClient.put('/repairs/' + id + '/status?status=' + status);
export const deleteRepair = (id) => apiClient.delete('/repairs/' + id);
export const getRepairImgUrls = (id) => apiClient.get('/repairs/' + id + '/imgUrls');

// 登录注册 (后端是 /user 不是 /api/user)
export const login = (data) => axios.post('/user/login', data);
export const register = (data) => axios.post('/user/register', data);

// 无人机任务 API (后端路径已经是 /fly/*，不用加 /api 前缀)
export const getAllDrones = () => axios.get('/fly/all');
export const getDroneStatus = (droneNo) => axios.get('/fly/status?droneNo=' + droneNo);
export const getTaskList = (droneNo, status) => {
  const params = new URLSearchParams();
  if (droneNo) params.append('droneNo', droneNo);
  if (status) params.append('status', status);
  return axios.get('/fly/task/list?' + params.toString());
};
export const getTaskQueue = () => axios.get('/fly/queue');
export const addTask = (data) => axios.put('/fly/addTask', data);
export const cancelTask = (id) => axios.delete('/fly/task/' + id);
export const retryTask = (id) => axios.put('/fly/task/' + id + '/status?status=queued');
export const addDrone = (data) => axios.put('/fly/drone', data);
export const getTasksByDrone = (droneNo) => axios.get('/fly/task/list?droneNo=' + droneNo);
