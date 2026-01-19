import { useState, useEffect, useRef } from 'react'
import ChatInterface from './components/ChatInterface'
import Sidebar from './components/Sidebar'
import { getStatus, uploadPDF, askQuestion, clearDocuments, getChatHistory, clearChatHistory } from './services/api'

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [status, setStatus] = useState({ document_count: 0, status: 'no_documents', document_name: null })
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    loadStatus()
    loadChatHistory()
  }, [])

  const loadChatHistory = async () => {
    try {
      const data = await getChatHistory()
      if (data.messages && data.messages.length > 0) {
        setMessages(data.messages)
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }

  const loadStatus = async () => {
    try {
      const statusData = await getStatus()
      setStatus(statusData)
    } catch (error) {
      console.error('Failed to load status:', error)
    }
  }

  const handleUpload = async (file, clearExisting) => {
    setIsUploading(true)
    try {
      const result = await uploadPDF(file, clearExisting)
      await loadStatus()
      setMessages([]) // Clear chat when new document is uploaded
      return { success: true, message: result.message }
    } catch (error) {
      return { success: false, message: error.message || 'Upload failed' }
    } finally {
      setIsUploading(false)
    }
  }

  const handleAsk = async (question) => {
    if (!question.trim()) return

    // Add user message
    const userMessage = { role: 'user', content: question }
    setMessages(prev => [...prev, userMessage])

    setIsLoading(true)
    try {
      const response = await askQuestion(question)
      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        source_documents: response.source_documents || []
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `❌ Error: ${error.message || 'Failed to get response'}`,
        source_documents: []
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = async () => {
    try {
      await clearDocuments()
      await loadStatus()
      setMessages([])
    } catch (error) {
      console.error('Failed to clear documents:', error)
    }
  }

  const handleClearChat = async () => {
    try {
      await clearChatHistory()
      setMessages([])
    } catch (error) {
      console.error('Failed to clear chat history:', error)
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onUpload={handleUpload}
        onClear={handleClear}
        onClearChat={handleClearChat}
        status={status}
        isLoading={isUploading}
      />
      <div className="flex-1 flex flex-col">
        <ChatInterface
          messages={messages}
          onAsk={handleAsk}
          isLoading={isLoading}
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
        />
      </div>
    </div>
  )
}

export default App
