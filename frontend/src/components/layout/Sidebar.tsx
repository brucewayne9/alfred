import { useEffect } from 'react'
import { X, Plus } from 'lucide-react'
import { useSidebarStore } from '../../stores/sidebarStore'
import { useChatStore } from '../../stores/chatStore'
import { ConversationList } from '../sidebar/ConversationList'
import { ProjectList } from '../sidebar/ProjectList'
import { SearchBar } from '../sidebar/SearchBar'

export function Sidebar() {
  const { isOpen, close } = useSidebarStore()
  const newChat = useChatStore(s => s.newChat)

  // Close sidebar on escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) close()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, close])

  const handleNewChat = () => {
    newChat()
    close()
  }

  return (
    <>
      {/* Backdrop on mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={close}
        />
      )}

      <aside
        className={`fixed lg:relative top-0 left-0 h-full z-40 w-72 bg-alfred-surface border-r border-alfred-border flex flex-col transition-transform duration-200 ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0 lg:w-0 lg:border-0 lg:overflow-hidden'
        }`}
      >
        <div className="flex items-center justify-between p-3 border-b border-alfred-border">
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-alfred-accent/10 text-alfred-accent hover:bg-alfred-accent/20 transition-colors text-sm font-medium"
          >
            <Plus size={16} /> New Chat
          </button>
          <button
            onClick={close}
            className="p-1.5 rounded-lg hover:bg-alfred-hover text-alfred-muted hover:text-white transition-colors lg:hidden"
          >
            <X size={18} />
          </button>
        </div>

        <SearchBar />

        <div className="flex-1 overflow-y-auto">
          <ConversationList />
          <ProjectList />
        </div>
      </aside>
    </>
  )
}
