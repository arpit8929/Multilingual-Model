import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'

function ChatInput({ onSubmit, isLoading }) {
  const [input, setInput] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    if (textareaRef.current) {
      const el = textareaRef.current
  
      el.style.height = 'auto'
  
      const maxHeight = 160 // ≈ 5–6 lines (you can tweak)
      const newHeight = Math.min(el.scrollHeight, maxHeight)
  
      el.style.height = `${newHeight}px`
      el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
    }
  }, [input])
  

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSubmit(input)
      setInput('')
      if (textareaRef.current) textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      {/* Gradient glow wrapper */}
      <div className="p-[1.5px] rounded-2xl bg-gradient-to-r from-primary/60 via-blue-500/40 to-purple-500/60">
        
        {/* Actual input container */}
        <div className="
          flex items-center gap-3
          bg-white dark:bg-gray-900
          rounded-2xl
          px-4 py-3
          shadow-xl
          transition-all duration-300
        ">
          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something smart..."
            rows={1}
            disabled={isLoading}
            className="
              flex-1 bg-transparent resize-none outline-none
              text-gray-900 dark:text-gray-100
              placeholder-gray-400 dark:placeholder-gray-500
              text-base leading-relaxed
              min-h-[24px]
              max-h-[160px]
              overflow-y-auto
              scrollbar-thin
            "
          />

          {/* Send Button */}
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="
              p-3 rounded-xl
              bg-gradient-to-br from-primary to-emerald-500
              text-white
              shadow-lg
              hover:scale-105
              active:scale-95
              transition-all duration-200
              disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </form>
  )
}

export default ChatInput
