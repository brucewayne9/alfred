import { ChatInput } from '../input/ChatInput'

export function WelcomeScreen() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4">
      <div className="max-w-2xl w-full text-center mb-8">
        <img src="/alfred-icon.jpg" alt="Alfred" className="w-16 h-16 rounded-full mx-auto mb-4 opacity-80" />
        <h2 className="text-2xl font-semibold text-white mb-2">What can I help with?</h2>
        <p className="text-alfred-muted text-sm">
          Ask me anything — manage your servers, schedule meetings, search the web, write code, analyze documents, or just chat.
        </p>
      </div>
      <div className="w-full max-w-2xl">
        <ChatInput />
      </div>
    </div>
  )
}
