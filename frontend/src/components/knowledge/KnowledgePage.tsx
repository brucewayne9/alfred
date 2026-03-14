import { useEffect, useState } from 'react'
import { Trash2, Search, Loader2 } from 'lucide-react'
import { useKnowledgeStore } from '../../stores/knowledgeStore'
import { DropZone } from './DropZone'

type Tab = 'upload' | 'documents' | 'search'

export function KnowledgePage() {
  const [tab, setTab] = useState<Tab>('upload')
  const {
    status, documents, entities, queryResult,
    isLoading, isUploading, isQuerying, error,
    loadStatus, loadDocuments, loadEntities,
    uploadText, uploadFile, deleteDocument, query, searchEntities,
    clearError,
  } = useKnowledgeStore()

  useEffect(() => {
    loadStatus()
    loadDocuments()
    loadEntities()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Status bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-alfred-border">
        <span
          className={`w-2 h-2 rounded-full ${status?.connected ? 'bg-green-500' : 'bg-red-500'}`}
        />
        <span className="text-xs text-alfred-muted">
          {status?.connected
            ? `Connected — ${Object.values(status.document_counts || {}).reduce((a, b) => a + b, 0)} documents`
            : 'Disconnected'}
        </span>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center justify-between">
          <span className="text-xs text-red-400">{error}</span>
          <button onClick={clearError} className="text-xs text-red-400 hover:text-red-300">dismiss</button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 px-4 pt-3">
        {(['upload', 'documents', 'search'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              tab === t
                ? 'bg-alfred-accent text-white'
                : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {tab === 'upload' && <UploadTab isUploading={isUploading} uploadFile={uploadFile} uploadText={uploadText} />}
        {tab === 'documents' && <DocumentsTab documents={documents} isLoading={isLoading} deleteDocument={deleteDocument} />}
        {tab === 'search' && (
          <SearchTab
            query={query} queryResult={queryResult} isQuerying={isQuerying}
            entities={entities} searchEntities={searchEntities}
          />
        )}
      </div>
    </div>
  )
}

/* ── Upload Tab ─────────────────────────────────────────── */

function UploadTab({
  isUploading, uploadFile, uploadText,
}: {
  isUploading: boolean
  uploadFile: (f: File) => Promise<void>
  uploadText: (text: string, desc: string) => Promise<void>
}) {
  const [text, setText] = useState('')
  const [desc, setDesc] = useState('')

  const handleTextSubmit = async () => {
    if (text.length < 10) return
    await uploadText(text, desc)
    setText('')
    setDesc('')
  }

  return (
    <div className="grid md:grid-cols-2 gap-4">
      <div>
        <h3 className="text-sm font-medium text-white mb-2">Upload File</h3>
        <DropZone onFile={uploadFile} isUploading={isUploading} />
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium text-white">Paste Text</h3>
        <input
          value={desc}
          onChange={e => setDesc(e.target.value)}
          placeholder="Description (optional)"
          className="px-3 py-2 bg-alfred-surface border border-alfred-border rounded-lg text-sm text-white placeholder:text-alfred-muted/50 outline-none focus:border-alfred-accent/50"
        />
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste text content here (min 10 chars)..."
          rows={6}
          className="flex-1 px-3 py-2 bg-alfred-surface border border-alfred-border rounded-lg text-sm text-white placeholder:text-alfred-muted/50 outline-none focus:border-alfred-accent/50 resize-none"
        />
        <button
          onClick={handleTextSubmit}
          disabled={text.length < 10 || isUploading}
          className="self-end px-4 py-2 bg-alfred-accent text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-alfred-accent/80 transition-colors flex items-center gap-2"
        >
          {isUploading && <Loader2 size={14} className="animate-spin" />}
          Upload Text
        </button>
      </div>
    </div>
  )
}

/* ── Documents Tab ──────────────────────────────────────── */

function DocumentsTab({
  documents, isLoading, deleteDocument,
}: {
  documents: { id: string; file_path?: string | null; content_summary?: string; created_at?: string; [key: string]: unknown }[]
  isLoading: boolean
  deleteDocument: (id: string) => Promise<void>
}) {
  const [confirmId, setConfirmId] = useState<string | null>(null)

  const handleDelete = (id: string) => {
    if (confirmId === id) {
      deleteDocument(id)
      setConfirmId(null)
    } else {
      setConfirmId(id)
    }
  }

  if (isLoading) {
    return <div className="flex justify-center py-12"><Loader2 className="animate-spin text-alfred-muted" /></div>
  }

  if (!documents.length) {
    return <p className="text-sm text-alfred-muted text-center py-12">No documents yet. Upload something!</p>
  }

  return (
    <div className="space-y-2">
      {documents.map(doc => (
        <div key={doc.id} className="flex items-center justify-between px-4 py-3 bg-alfred-surface rounded-lg border border-alfred-border">
          <div className="min-w-0 flex-1">
            <p className="text-sm text-white truncate">{doc.file_path || doc.id}</p>
            {doc.content_summary && (
              <p className="text-xs text-alfred-muted truncate mt-0.5">{doc.content_summary.slice(0, 100)}</p>
            )}
            {doc.created_at && (
              <p className="text-xs text-alfred-muted">{new Date(doc.created_at).toLocaleDateString()}</p>
            )}
          </div>
          <button
            onClick={() => handleDelete(doc.id)}
            className={`ml-3 p-2 rounded-lg transition-colors ${
              confirmId === doc.id
                ? 'bg-red-500/20 text-red-400'
                : 'text-alfred-muted hover:text-red-400 hover:bg-alfred-hover'
            }`}
            title={confirmId === doc.id ? 'Click again to confirm' : 'Delete'}
          >
            <Trash2 size={16} />
          </button>
        </div>
      ))}
    </div>
  )
}

/* ── Search Tab ─────────────────────────────────────────── */

function SearchTab({
  query: doQuery, queryResult, isQuerying, entities, searchEntities,
}: {
  query: (q: string, mode: string) => Promise<void>
  queryResult: { response?: string; [key: string]: unknown } | null
  isQuerying: boolean
  entities: { label: string; [key: string]: unknown }[]
  searchEntities: (label: string) => Promise<void>
}) {
  const [q, setQ] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [entitySearch, setEntitySearch] = useState('')

  const handleSearch = () => {
    if (q.trim()) doQuery(q, mode)
  }

  return (
    <div className="space-y-4">
      {/* Query input */}
      <div className="flex gap-2">
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
          placeholder="Search the knowledge base..."
          className="flex-1 px-3 py-2 bg-alfred-surface border border-alfred-border rounded-lg text-sm text-white placeholder:text-alfred-muted/50 outline-none focus:border-alfred-accent/50"
        />
        <select
          value={mode}
          onChange={e => setMode(e.target.value)}
          className="px-3 py-2 bg-alfred-surface border border-alfred-border rounded-lg text-sm text-white outline-none"
        >
          <option value="hybrid">Hybrid</option>
          <option value="local">Local</option>
          <option value="global">Global</option>
          <option value="naive">Naive</option>
        </select>
        <button
          onClick={handleSearch}
          disabled={!q.trim() || isQuerying}
          className="px-4 py-2 bg-alfred-accent text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-alfred-accent/80 transition-colors flex items-center gap-2"
        >
          {isQuerying ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
          Search
        </button>
      </div>

      {/* Results */}
      {queryResult && (
        <div className="p-4 bg-alfred-surface border border-alfred-border rounded-lg">
          <h4 className="text-xs font-medium text-alfred-muted uppercase tracking-wider mb-2">Results</h4>
          <p className="text-sm text-white whitespace-pre-wrap">{queryResult.response || JSON.stringify(queryResult, null, 2)}</p>
        </div>
      )}

      {/* Entity chips */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-xs font-medium text-alfred-muted uppercase tracking-wider">Entities</h4>
          <input
            value={entitySearch}
            onChange={e => { setEntitySearch(e.target.value); if (e.target.value.length >= 2) searchEntities(e.target.value) }}
            placeholder="Filter entities..."
            className="px-2 py-1 bg-alfred-surface border border-alfred-border rounded text-xs text-white placeholder:text-alfred-muted/50 outline-none w-40"
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {entities.map((ent, i) => (
            <span
              key={i}
              className="px-2.5 py-1 bg-alfred-hover text-alfred-muted rounded-full text-xs hover:text-white hover:bg-alfred-accent/20 transition-colors cursor-default"
              title={(ent.description || ent.type || '') as string}
            >
              {ent.label}
            </span>
          ))}
          {!entities.length && (
            <span className="text-xs text-alfred-muted/60">No entities found</span>
          )}
        </div>
      </div>
    </div>
  )
}
