import { useState, useCallback } from 'react'
import { ChevronDown, ChevronRight, FolderPlus, Trash2, Check, X } from 'lucide-react'
import { useSidebarStore } from '../../stores/sidebarStore'
import { useChatStore } from '../../stores/chatStore'
import { conversationsApi } from '../../api/conversations'

const FOLDER_COLORS = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ef4444', '#eab308', '#06b6d4', '#ec4899']

export function ProjectList() {
  const projects = useSidebarStore(s => s.projects)
  const { createProject, deleteProject } = useSidebarStore()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [projectConvs, setProjectConvs] = useState<Record<string, Array<{ id: string; title: string }>>>({})
  const loadConversation = useChatStore(s => s.loadConversation)
  const closeSidebar = useSidebarStore(s => s.close)

  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState(FOLDER_COLORS[0])
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  const toggleProject = async (id: string) => {
    const next = !expanded[id]
    setExpanded(s => ({ ...s, [id]: next }))
    if (next && !projectConvs[id]) {
      try {
        const convs = await conversationsApi.list(20, 0, id)
        setProjectConvs(s => ({ ...s, [id]: convs }))
      } catch { /* ignore */ }
    }
  }

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return
    await createProject(newName.trim(), newColor)
    setNewName('')
    setNewColor(FOLDER_COLORS[0])
    setShowCreate(false)
  }, [newName, newColor, createProject])

  const handleDelete = useCallback(async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (confirmDeleteId !== id) {
      setConfirmDeleteId(id)
      setTimeout(() => setConfirmDeleteId(null), 3000)
      return
    }
    await deleteProject(id)
    setConfirmDeleteId(null)
  }, [confirmDeleteId, deleteProject])

  return (
    <div className="py-2 border-t border-alfred-border">
      <div className="flex items-center justify-between px-4 py-1.5">
        <h3 className="text-[11px] font-medium text-alfred-muted uppercase tracking-wider">
          Folders
        </h3>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="p-0.5 rounded text-alfred-muted hover:text-alfred-accent transition-colors"
          title="New folder"
        >
          {showCreate ? <X size={14} /> : <FolderPlus size={14} />}
        </button>
      </div>

      {showCreate && (
        <div className="px-4 pb-2 space-y-2">
          <input
            value={newName}
            onChange={e => setNewName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleCreate()}
            placeholder="Folder name..."
            autoFocus
            className="w-full px-2.5 py-1.5 bg-alfred-bg border border-alfred-border rounded-lg text-sm text-white placeholder-alfred-muted outline-none focus:border-alfred-accent/50 transition-colors"
          />
          <div className="flex items-center gap-1.5">
            {FOLDER_COLORS.map(color => (
              <button
                key={color}
                onClick={() => setNewColor(color)}
                className={`w-5 h-5 rounded-full transition-transform ${newColor === color ? 'scale-125 ring-2 ring-white/30' : 'hover:scale-110'}`}
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
          <button
            onClick={handleCreate}
            disabled={!newName.trim()}
            className="w-full py-1.5 bg-alfred-accent/10 text-alfred-accent rounded-lg text-sm font-medium hover:bg-alfred-accent/20 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-1.5"
          >
            <Check size={14} /> Create Folder
          </button>
        </div>
      )}

      {projects.map(project => (
        <div key={project.id}>
          <div className="group flex items-center">
            <button
              onClick={() => toggleProject(project.id)}
              className="flex-1 flex items-center gap-2 px-4 py-2 hover:bg-alfred-hover/50 transition-colors"
            >
              {expanded[project.id] ? <ChevronDown size={14} className="text-alfred-muted" /> : <ChevronRight size={14} className="text-alfred-muted" />}
              <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: project.color }} />
              <span className="text-sm text-alfred-text truncate">{project.name}</span>
            </button>
            <button
              onClick={(e) => handleDelete(e, project.id)}
              className={`p-1 mr-2 rounded opacity-0 group-hover:opacity-100 transition-all ${
                confirmDeleteId === project.id
                  ? 'text-red-400 bg-red-400/10 opacity-100'
                  : 'text-alfred-muted hover:text-red-400 hover:bg-alfred-hover'
              }`}
              title={confirmDeleteId === project.id ? 'Click again to delete' : 'Delete folder'}
            >
              <Trash2 size={12} />
            </button>
          </div>
          {expanded[project.id] && (
            <div className="pl-8">
              {(projectConvs[project.id] || []).map(conv => (
                <button
                  key={conv.id}
                  onClick={() => { loadConversation(conv.id); closeSidebar() }}
                  className="w-full text-left px-3 py-1.5 text-sm text-alfred-muted hover:text-white hover:bg-alfred-hover/50 truncate transition-colors"
                >
                  {conv.title || 'Untitled'}
                </button>
              ))}
              {(projectConvs[project.id] || []).length === 0 && (
                <p className="px-3 py-1.5 text-xs text-alfred-muted">No conversations</p>
              )}
            </div>
          )}
        </div>
      ))}

      {projects.length === 0 && !showCreate && (
        <p className="px-4 py-2 text-xs text-alfred-muted">No folders yet</p>
      )}
    </div>
  )
}
