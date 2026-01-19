import { useRef, useEffect } from 'react'
import { Menu } from 'lucide-react'
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
    <div
      className="flex flex-col h-full bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300"
      style={{ marginLeft: 'var(--sidebar-offset)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-sm z-10 relative transition-colors duration-300">
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation()
              onMenuClick()
            }}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors relative z-30"
            aria-label="Menu"
            type="button"
          >
            <Menu className="w-5 h-5 text-gray-700 dark:text-gray-200" />
          </button>

          <div>
            <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
              Multilingual QnA Assistant
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Ask questions in English, Hindi, or Hinglish
            </p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500 dark:text-gray-400">
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
      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4 transition-colors duration-300">
        <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  )
}

export default ChatInterface
