import { useState, useRef, useCallback } from 'react'
import { Upload, Loader2 } from 'lucide-react'

interface DropZoneProps {
  onFile: (file: File) => void
  isUploading: boolean
}

const ACCEPT = '.pdf,.txt,.md,.docx,.doc'

export function DropZone({ onFile, isUploading }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file) onFile(file)
    },
    [onFile],
  )

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
    e.target.value = ''
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !isUploading && inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-3 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-colors ${
        dragOver
          ? 'border-alfred-accent bg-alfred-accent/10'
          : 'border-alfred-border hover:border-alfred-muted'
      } ${isUploading ? 'opacity-60 pointer-events-none' : ''}`}
    >
      {isUploading ? (
        <Loader2 size={32} className="text-alfred-accent animate-spin" />
      ) : (
        <Upload size={32} className={dragOver ? 'text-alfred-accent' : 'text-alfred-muted'} />
      )}
      <p className="text-sm text-alfred-muted text-center">
        {isUploading ? 'Uploading & indexing...' : 'Drop a file here or click to browse'}
      </p>
      <p className="text-xs text-alfred-muted/60">PDF, TXT, MD, DOCX</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        onChange={handleChange}
        className="hidden"
      />
    </div>
  )
}
