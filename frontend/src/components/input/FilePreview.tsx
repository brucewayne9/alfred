import { X, FileText } from 'lucide-react'

interface FilePreviewProps {
  name: string
  onRemove: () => void
}

export function FilePreview({ name, onRemove }: FilePreviewProps) {
  return (
    <div className="flex items-center gap-2 mb-2 p-2 bg-alfred-surface rounded-lg border border-alfred-border">
      <FileText size={16} className="text-alfred-accent shrink-0" />
      <span className="text-sm text-alfred-text truncate flex-1">{name}</span>
      <button
        onClick={onRemove}
        className="p-0.5 rounded text-alfred-muted hover:text-white transition-colors"
      >
        <X size={14} />
      </button>
    </div>
  )
}
