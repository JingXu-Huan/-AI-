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
