import { useState, useEffect } from 'react'
import {
  Upload,
  FileText,
  Trash2,
  CheckCircle,
  AlertCircle,
  Sun,
  Moon
} from 'lucide-react'

const SIDEBAR_WIDTH = '20rem'

function Sidebar({ isOpen = true, onUpload, onClear, onClearChat, status = { document_count: 0 }, isLoading }) {
  const [file, setFile] = useState(null)
  const [clearExisting, setClearExisting] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [theme, setTheme] = useState('light')

  // Sidebar shift
  useEffect(() => {
    document.documentElement.style.setProperty(
      '--sidebar-offset',
      isOpen ? SIDEBAR_WIDTH : '0px'
    )
  }, [isOpen])

  // Theme init (local + system)
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const final = saved || (systemDark ? 'dark' : 'light')

    setTheme(final)
    document.documentElement.classList.toggle('dark', final === 'dark')
  }, [])

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light'
    setTheme(next)
    localStorage.setItem('theme', next)
    document.documentElement.classList.toggle('dark', next === 'dark')
  }

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setUploadStatus(null)
    }
  }
  
  const handleUpload = async () => {
    if (!file) {
      setUploadStatus({ success: false, message: 'Please select a PDF file' })
      return
    }
  
    setUploadStatus({ success: null, message: 'Uploading...' })
  
    const result = await onUpload(file, clearExisting)
    setUploadStatus(result)
  }
  
  return (
    <div
      className={`fixed inset-y-0 left-0 w-80 
      bg-white/90 dark:bg-gray-900/90 backdrop-blur-md
      border-r border-gray-200 dark:border-gray-800
      z-50 transform transition-all duration-300
      ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      flex flex-col`}
    >
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-800">
        <div>
          <h2 className="text-lg font-bold text-gray-800 dark:text-white">Multilingual AI</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400">Document Q&A</p>
        </div>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800"
        >
          {theme === 'dark'
            ? <Sun className="w-5 h-5 text-yellow-400" />
            : <Moon className="w-5 h-5 text-gray-700 dark:text-gray-300" />}
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 p-4 space-y-6 overflow-y-auto">

        {/* Status */}
        <div>
          <h3 className="font-semibold text-gray-700 dark:text-gray-200 mb-2 flex gap-2">
            <FileText size={18} /> Status
          </h3>

          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            {status.document_count > 0 ? (
              <div className="text-green-600 flex gap-2 items-center">
                <CheckCircle size={18} />
                {status.document_count} chunks ingested
              </div>
            ) : (
              <div className="text-amber-600 flex gap-2 items-center">
                <AlertCircle size={18} /> No documents loaded
              </div>
            )}
          </div>
        </div>

        {/* Upload */}
        <div>
          <h3 className="font-semibold text-gray-700 dark:text-gray-200 mb-2 flex gap-2">
            <Upload size={18} /> Upload PDF
          </h3>

          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="w-full text-sm text-gray-600 dark:text-gray-300 
    file:bg-primary file:text-white file:border-0 file:px-4 file:py-2 file:rounded-lg"
          />


          <label className="flex gap-2 mt-3 text-gray-700 dark:text-gray-300">
            <input
              type="checkbox"
              checked={clearExisting}
              onChange={(e) => setClearExisting(e.target.checked)}
            />

            Clear existing documents
          </label>

          <button
            onClick={handleUpload}
            className="w-full mt-3 bg-primary text-white py-2 rounded-lg hover:bg-primary/90"
          >
            Upload & Ingest
          </button>

        </div>

        {/* Clear */}
        <div className="space-y-2">
          <h3 className="font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
            <Trash2 className="w-5 h-5" />
            Clear
          </h3>

          {/* Clear Documents */}
          <button
            onClick={onClear}
            disabled={isLoading || status.document_count === 0}
            className="w-full py-2 px-4 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            Clear All Documents
          </button>

          {/* ✅ Restore Clear Chat */}
          {onClearChat && (
            <button
              onClick={onClearChat}
              disabled={isLoading}
              className="w-full py-2 px-4 bg-gray-600 dark:bg-gray-700 text-white rounded-lg hover:bg-gray-700 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Clear Chat History
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default Sidebar
