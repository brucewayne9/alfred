import { useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Copy, Check } from 'lucide-react'
import { useState } from 'react'

interface MarkdownRendererProps {
  content: string
}

function CodeBlock({ className, children, ...props }: React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode }) {
  const [copied, setCopied] = useState(false)
  const match = /language-(\w+)/.exec(className || '')
  const isInline = !match && typeof children === 'string' && !children.includes('\n')

  const handleCopy = useCallback(() => {
    const text = String(children).replace(/\n$/, '')
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [children])

  if (isInline) {
    return <code className={className} {...props}>{children}</code>
  }

  return (
    <div className="relative group/code my-2">
      {match && (
        <div className="flex items-center justify-between px-4 py-1.5 bg-[#1a1a1a] border-b border-alfred-border rounded-t-lg">
          <span className="text-xs text-alfred-muted">{match[1]}</span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 text-xs text-alfred-muted hover:text-white transition-colors"
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      )}
      {!match && (
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1 rounded bg-alfred-hover text-alfred-muted hover:text-white opacity-0 group-hover/code:opacity-100 transition-opacity"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
        </button>
      )}
      <code className={`${className} ${!match ? 'rounded-lg' : 'rounded-b-lg rounded-t-none'}`} {...props}>
        {children}
      </code>
    </div>
  )
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code: CodeBlock,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
