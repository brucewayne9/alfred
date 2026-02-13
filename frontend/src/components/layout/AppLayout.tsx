import { useEffect } from 'react'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { ChatArea } from '../chat/ChatArea'
import { ChatInput } from '../input/ChatInput'
import { WelcomeScreen } from '../chat/WelcomeScreen'
import { ToastContainer } from '../notifications/ToastContainer'
import { VadStatus } from '../voice/VadStatus'
import { useChatStore } from '../../stores/chatStore'
import { useSidebarStore } from '../../stores/sidebarStore'
import { useWebSocket } from '../../hooks/useWebSocket'
import { usePushNotifications } from '../../hooks/usePushNotifications'
import { useHandsFree } from '../../hooks/useHandsFree'
import { useWakeWord } from '../../hooks/useWakeWord'

export function AppLayout() {
  const { messages, currentConversationId } = useChatStore()
  const { loadConversations, loadProjects } = useSidebarStore()
  const { subscribe } = usePushNotifications()

  useWebSocket()
  useHandsFree()
  useWakeWord()

  useEffect(() => {
    loadConversations()
    loadProjects()
    subscribe().catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const hasMessages = messages.length > 0 || currentConversationId

  return (
    <div className="h-full flex flex-col bg-alfred-bg safe-top">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 flex flex-col min-w-0">
          {hasMessages ? (
            <>
              <ChatArea />
              <ChatInput />
            </>
          ) : (
            <WelcomeScreen />
          )}
        </main>
      </div>
      <VadStatus />
      <ToastContainer />
    </div>
  )
}
