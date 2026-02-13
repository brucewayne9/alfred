export function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-3 py-1">
      <img
        src="/glabs-logo.jpg"
        alt="Thinking"
        className="thinking-logo w-8 h-8 rounded-full object-cover"
      />
      <span className="text-alfred-muted text-sm animate-pulse">
        Thinking...
      </span>
    </div>
  )
}
