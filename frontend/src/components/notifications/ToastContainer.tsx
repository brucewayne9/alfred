import { X } from 'lucide-react'
import { useNotificationStore, type Toast } from '../../stores/notificationStore'

const typeStyles: Record<string, string> = {
  info: 'border-blue-500/30 bg-blue-500/10',
  success: 'border-green-500/30 bg-green-500/10',
  error: 'border-red-500/30 bg-red-500/10',
  warning: 'border-yellow-500/30 bg-yellow-500/10',
}

function ToastItem({ toast }: { toast: Toast }) {
  const removeToast = useNotificationStore(s => s.removeToast)

  return (
    <div className={`toast-enter flex items-start gap-3 p-3 rounded-xl border ${typeStyles[toast.type] || typeStyles.info} backdrop-blur-sm shadow-lg max-w-sm`}>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-white">{toast.title}</p>
        {toast.message && (
          <p className="text-xs text-alfred-muted mt-0.5 truncate">{toast.message}</p>
        )}
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="p-0.5 text-alfred-muted hover:text-white transition-colors shrink-0"
      >
        <X size={14} />
      </button>
    </div>
  )
}

export function ToastContainer() {
  const toasts = useNotificationStore(s => s.toasts)

  if (toasts.length === 0) return null

  return (
    <div className="fixed top-14 right-4 z-50 space-y-2">
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  )
}
