import { useState } from 'react'
import { X, Upload, FileText, Trash2, CheckCircle, AlertCircle } from 'lucide-react'

function Sidebar({ isOpen, onClose, onUpload, onClear, onClearChat, status, isLoading }) {
  const [file, setFile] = useState(null)
  const [clearExisting, setClearExisting] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)

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
      // Reset file input
      const fileInput = document.getElementById('pdf-upload')
      if (fileInput) fileInput.value = ''
    }
  }

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed lg:static inset-y-0 left-0 w-80 bg-white border-r border-gray-200 z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        } flex flex-col`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Settings</h2>
          <button
            onClick={onClose}
            className="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
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
                  <span>{status.document_count} chunks ingested</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-amber-600">
                  <AlertCircle className="w-5 h-5" />
                  <span>No documents loaded</span>
                </div>
              )}
            </div>
          </div>

          {/* Upload PDF */}
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

          {/* Clear Documents */}
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
