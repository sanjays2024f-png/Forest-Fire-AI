import axios from 'axios';

const baseURL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

const api = axios.create({
  baseURL,
  timeout: 60000,
});

export const getStatistics = () => api.get('/api/statistics');
export const getModelPerformance = () => api.get('/api/model-performance');
export const getAlerts = () => api.get('/api/alerts');
export const getHistory = (limit = 50) => api.get('/history', { params: { limit } });
export const getHistoryItem = (id) => api.get(`/history/${id}`);
export const deleteHistoryItem = (id) => api.delete(`/history/${id}`);
export const getModelInfo = () => api.get('/model-info');
export const getTemporalStatus = () => api.get('/temporal-status');
export const resetWebcam = () => api.post('/webcam-reset');
export const getSatelliteTraining = () => api.get('/api/satellite/training-report');
export const getSatelliteWarning = (latitude, longitude) => api.get('/api/satellite/nearby', {
  params: { latitude, longitude },
});
// Required workflow: upload request contains only `file`.
export const uploadImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/upload-image', formData);
};
export const analyzeImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/analyze-image', formData);
};

export default api;
