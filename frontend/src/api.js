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
export const writeToDB = (data) => apiClient.post('/writeToDB', data);
export const updateRepairStatus = (id, status) => apiClient.put('/repairs/' + id + '/status?status=' + status);
export const deleteRepair = (id) => apiClient.delete('/repairs/' + id);
