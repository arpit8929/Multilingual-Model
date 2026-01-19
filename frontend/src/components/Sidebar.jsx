import { useState, useEffect } from 'react'
import {
  Upload,
  FileText,
  Trash2,
  CheckCircle,
  AlertCircle,
  Pin,
  PinOff
} from 'lucide-react'

const SIDEBAR_WIDTH = '20rem' // 320px

function Sidebar({ isOpen, onClose, onUpload, onClear, onClearChat, status, isLoading }) {
  const [file, setFile] = useState(null)
  const [clearExisting, setClearExisting] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [isPinned, setIsPinned] = useState(false)

  const shouldBeOpen = isOpen || isPinned

  // Shift chat window when sidebar is open (ChatGPT-style)
  useEffect(() => {
    const root = document.documentElement
    root.style.setProperty(
      '--sidebar-offset',
      shouldBeOpen ? SIDEBAR_WIDTH : '0px'
    )

    return () => {
      root.style.setProperty('--sidebar-offset', '0px')
    }
  }, [shouldBeOpen])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setUploadStatus(null)
    } else {
      setUploadStatus({ success: false, message: 'Please select a PDF file' })
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus({ success: false, message: 'Please select a file' })
      return
    }

    setUploadStatus({ success: null, message: 'Uploading...' })
    const result = await onUpload(file, clearExisting)
    setUploadStatus(result)

    if (result.success) {
      setFile(null)
      setClearExisting(false)
      const fileInput = document.getElementById('pdf-upload')
      if (fileInput) fileInput.value = ''
    }
  }

  // Sidebar closes only if NOT pinned
  const handleClose = () => {
    if (!isPinned) {
      onClose()
    }
  }

  return (
    <>
      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 w-80 bg-white border-r border-gray-200 z-50 transform transition-transform duration-300 ease-in-out ${
          shouldBeOpen ? 'translate-x-0' : '-translate-x-full'
        } flex flex-col`}
      >
        {/* Header */}

        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-700 rounded-xl flex items-center justify-center shadow-lg">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white"></div>
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-800">Multilingual AI</h2>
              <p className="text-xs text-gray-500">Document Q&A</p>
            </div>
          </div>

          {/* Pin Button (Only control inside sidebar now) */}
          <button
            onClick={() => setIsPinned(!isPinned)}
            title={isPinned ? 'Unpin Sidebar' : 'Pin Sidebar'}
            className={`p-2 rounded-lg transition-colors ${
              isPinned
                ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                : 'hover:bg-gray-100'
            }`}
          >
            {isPinned ? <PinOff className="w-5 h-5" /> : <Pin className="w-5 h-5" />}
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Status */}
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Status
            </h3>
            <div className="p-3 bg-gray-50 rounded-lg">
              {status.document_count > 0 ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />

                  <span className="truncate">
                    {status.document_name ||
                      `${status.document_count} chunks ingested`}
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-amber-600">
                  <AlertCircle className="w-5 h-5" />
                  <span>No documents loaded</span>
                </div>
              )}
            </div>
          </div>

          {/* Upload */}
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload PDF
            </h3>

            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select PDF File
                </label>
                <input
                  id="pdf-upload"
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-white hover:file:bg-primary/90"
                  disabled={isLoading}
                />
                {file && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {file.name}
                  </p>
                )}
              </div>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={clearExisting}
                  onChange={(e) => setClearExisting(e.target.checked)}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                  disabled={isLoading}
                />
                <span className="text-sm text-gray-700">
                  Clear existing documents before upload
                </span>
              </label>

              <button
                onClick={handleUpload}
                disabled={!file || isLoading}
                className="w-full py-2 px-4 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Upload & Ingest
                  </>
                )}
              </button>

              {uploadStatus && (
                <div
                  className={`p-3 rounded-lg text-sm ${
                    uploadStatus.success === true
                      ? 'bg-green-50 text-green-700'
                      : uploadStatus.success === false
                      ? 'bg-red-50 text-red-700'
                      : 'bg-blue-50 text-blue-700'
                  }`}
                >
                  {uploadStatus.message}
                </div>
              )}
            </div>
          </div>

          {/* Clear */}
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2">
              <Trash2 className="w-5 h-5" />
              Clear
            </h3>

            <div className="space-y-2">
              <button
                onClick={onClear}
                disabled={isLoading || status.document_count === 0}
                className="w-full py-2 px-4 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Clear All Documents
              </button>

              {onClearChat && (
                <button
                  onClick={onClearChat}
                  disabled={isLoading}
                  className="w-full py-2 px-4 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear Chat History
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Sidebar
