import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function Message({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg p-4 ${
          isUser
            ? 'bg-primary text-white'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        <div className={`prose prose-sm max-w-none ${isUser ? 'prose-invert' : ''}`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p className={`mb-2 last:mb-0 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</p>,
              ul: ({ children }) => <ul className={`list-disc list-inside mb-2 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</ul>,
              ol: ({ children }) => <ol className={`list-decimal list-inside mb-2 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</ol>,
              li: ({ children }) => <li className={`mb-1 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</li>,
              strong: ({ children }) => <strong className={`font-semibold ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</strong>,
              code: ({ children }) => (
                <code className={`px-1 py-0.5 rounded text-sm ${isUser ? 'bg-white/20 text-white' : 'bg-gray-200 text-gray-800'}`}>{children}</code>
              ),
              h1: ({ children }) => <h1 className={`text-xl font-bold mb-2 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</h1>,
              h2: ({ children }) => <h2 className={`text-lg font-bold mb-2 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</h2>,
              h3: ({ children }) => <h3 className={`text-base font-bold mb-2 ${isUser ? 'text-white' : 'text-gray-800'}`}>{children}</h3>,
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Source Documents */}
        {!isUser && message.source_documents && message.source_documents.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-300">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
            >
              <FileText className="w-4 h-4" />
              <span>View Source Documents ({message.source_documents.length})</span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showSources && (
              <div className="mt-2 space-y-2">
                {message.source_documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded p-2 text-xs border border-gray-200"
                  >
                    <div className="font-semibold text-gray-700 mb-1">
                      Source {idx + 1}: {doc.source.split('/').pop()} | Page {doc.page}
                    </div>
                    <div className="text-gray-600 line-clamp-3">
                      {doc.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Message
