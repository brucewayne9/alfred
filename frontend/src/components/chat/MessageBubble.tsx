import { useState } from 'react'
import { type ChatMessage } from '../../stores/chatStore'
import { MarkdownRenderer } from './MarkdownRenderer'
import { MessageActions } from './MessageActions'
import { User, Bot } from 'lucide-react'

interface MessageBubbleProps {
  message: ChatMessage
}

const tierColors: Record<string, string> = {
  local: 'bg-green-500/20 text-green-400',
  cloud: 'bg-blue-500/20 text-blue-400',
  openai: 'bg-emerald-500/20 text-emerald-400',
  claude_code: 'bg-purple-500/20 text-purple-400',
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [showActions, setShowActions] = useState(false)
  const isUser = message.role === 'user'

  // Hide empty streaming placeholder — ThinkingIndicator covers this state
  if (!isUser && message.isStreaming && !message.content) return null

  return (
    <div
      className={`group flex gap-3 ${isUser ? 'justify-end' : ''}`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {!isUser && (
        <div className="shrink-0 w-7 h-7 rounded-full bg-alfred-accent/20 flex items-center justify-center mt-1">
          <Bot size={14} className="text-alfred-accent" />
        </div>
      )}

      <div className={`relative max-w-[85%] ${isUser ? 'order-1' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 ${
            isUser
              ? 'bg-alfred-hover text-white rounded-br-md'
              : 'bg-transparent text-alfred-text'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-sm">{message.content}</p>
          ) : (
            <>
              <MarkdownRenderer content={message.content || ''} />
              {message.isStreaming && (
                <span className="inline-block w-1.5 h-4 bg-alfred-accent animate-pulse ml-0.5 align-middle" />
              )}
            </>
          )}

          {/* Tier badge */}
          {!isUser && message.tier && (
            <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded mt-2 ${tierColors[message.tier] || 'bg-gray-500/20 text-gray-400'}`}>
              {message.tier === 'claude_code' ? 'max' : message.tier === 'openai' ? 'gpt' : message.tier}
            </span>
          )}

          {/* Images */}
          {message.images?.map((img, i) => (
            <div key={i} className="mt-2">
              <img
                src={img.download_url || `data:image/png;base64,${img.base64}`}
                alt={img.filename}
                className="rounded-lg max-w-full max-h-80"
              />
              <p className="text-xs text-alfred-muted mt-1">{img.filename}</p>
            </div>
          ))}
        </div>

        {!isUser && showActions && !message.isStreaming && message.content && (
          <MessageActions content={message.content} />
        )}
      </div>

      {isUser && (
        <div className="shrink-0 w-7 h-7 rounded-full bg-alfred-border flex items-center justify-center mt-1 order-2">
          <User size={14} className="text-alfred-muted" />
        </div>
      )}
    </div>
  )
}
