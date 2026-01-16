import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const getStatus = async () => {
  const response = await api.get('/api/status')
  return response.data
}

export const uploadPDF = async (file, clearExisting = false) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('clear_existing', clearExisting)
  
  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const askQuestion = async (question) => {
  const response = await api.post('/api/ask', { question })
  return response.data
}

export const clearDocuments = async () => {
  const response = await api.post('/api/clear')
  return response.data
}

export const getChatHistory = async () => {
  const response = await api.get('/api/chat-history')
  return response.data
}

export const clearChatHistory = async () => {
  const response = await api.post('/api/chat-history/clear')
  return response.data
}

export default api
