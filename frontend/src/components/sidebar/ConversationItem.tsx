import { useState, useCallback } from 'react'
import { Trash2, Pencil, Check, X } from 'lucide-react'
import { type Conversation } from '../../api/conversations'
import { useChatStore } from '../../stores/chatStore'
import { useSidebarStore } from '../../stores/sidebarStore'
import { relativeDate } from '../../lib/dates'

interface ConversationItemProps {
  conversation: Conversation
}

export function ConversationItem({ conversation }: ConversationItemProps) {
  const currentId = useChatStore(s => s.currentConversationId)
  const loadConversation = useChatStore(s => s.loadConversation)
  const newChat = useChatStore(s => s.newChat)
  const closeSidebar = useSidebarStore(s => s.close)
  const { deleteConversation, renameConversation } = useSidebarStore()
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(conversation.title)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const isActive = currentId === conversation.id

  const handleClick = useCallback(() => {
    loadConversation(conversation.id)
    closeSidebar()
  }, [conversation.id, loadConversation, closeSidebar])

  const handleRename = useCallback(async () => {
    if (editTitle.trim() && editTitle !== conversation.title) {
      await renameConversation(conversation.id, editTitle.trim())
    }
    setIsEditing(false)
  }, [editTitle, conversation.id, conversation.title, renameConversation])

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirmDelete) {
      setConfirmDelete(true)
      setTimeout(() => setConfirmDelete(false), 3000)
      return
    }
    await deleteConversation(conversation.id)
    if (isActive) newChat()
  }, [conversation.id, confirmDelete, deleteConversation, isActive, newChat])

  return (
    <div
      onClick={isEditing ? undefined : handleClick}
      className={`group flex items-center gap-2 px-4 py-2 cursor-pointer transition-colors ${
        isActive ? 'bg-alfred-hover' : 'hover:bg-alfred-hover/50'
      }`}
    >
      <div className="flex-1 min-w-0">
        {isEditing ? (
          <div className="flex items-center gap-1">
            <input
              value={editTitle}
              onChange={e => setEditTitle(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRename()}
              autoFocus
              className="flex-1 bg-alfred-bg border border-alfred-border rounded px-2 py-0.5 text-sm text-white outline-none"
            />
            <button onClick={handleRename} className="p-0.5 text-green-400"><Check size={14} /></button>
            <button onClick={() => setIsEditing(false)} className="p-0.5 text-alfred-muted"><X size={14} /></button>
          </div>
        ) : (
          <>
            <p className="text-sm text-alfred-text truncate">{conversation.title || 'New Chat'}</p>
            <p className="text-[11px] text-alfred-muted">{relativeDate(conversation.updated_at || conversation.created_at)}</p>
          </>
        )}
      </div>

      {!isEditing && (
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
          <button
            onClick={(e) => { e.stopPropagation(); setIsEditing(true); setEditTitle(conversation.title) }}
            className="p-1 rounded text-alfred-muted hover:text-white hover:bg-alfred-hover transition-colors"
            title="Rename"
          >
            <Pencil size={12} />
          </button>
          <button
            onClick={handleDelete}
            className={`p-1 rounded transition-colors ${
              confirmDelete
                ? 'text-red-400 bg-red-400/10'
                : 'text-alfred-muted hover:text-red-400 hover:bg-alfred-hover'
            }`}
            title={confirmDelete ? 'Click again to delete' : 'Delete'}
          >
            <Trash2 size={12} />
          </button>
        </div>
      )}
    </div>
  )
}
