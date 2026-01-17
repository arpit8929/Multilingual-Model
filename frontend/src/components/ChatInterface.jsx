import { useState, useRef, useEffect } from 'react'
import { Send, Menu } from 'lucide-react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'

function ChatInterface({ messages, onAsk, isLoading, onMenuClick }) {
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = (question) => {
    if (!isLoading && question.trim()) {
      onAsk(question)
    }
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-white shadow-sm z-10">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuClick}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors z-20 relative"
            aria-label="Menu"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-xl font-semibold text-gray-800">
              Multilingual QnA Assistant
            </h1>
            <p className="text-sm text-gray-500">
              Ask questions in English, Hindi, or Hinglish
            </p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <p className="text-lg mb-2">👋 Welcome!</p>
              <p>Upload a PDF and start asking questions</p>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  )
}

export default ChatInterface
