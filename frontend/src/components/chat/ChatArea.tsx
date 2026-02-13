import { useEffect, useRef } from 'react'
import { useChatStore } from '../../stores/chatStore'
import { MessageBubble } from './MessageBubble'
import { ThinkingIndicator } from './ThinkingIndicator'

export function ChatArea() {
  const { messages, isThinking } = useChatStore()
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      <div className="max-w-3xl mx-auto space-y-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isThinking && (() => {
          const last = messages[messages.length - 1]
          // Show thinking when waiting or assistant message is still empty (streaming hasn't started)
          return !last || last.role !== 'assistant' || (last.isStreaming && !last.content)
        })() && (
          <ThinkingIndicator />
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
