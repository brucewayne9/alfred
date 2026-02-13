import { useSidebarStore } from '../../stores/sidebarStore'
import { ConversationItem } from './ConversationItem'
import { groupByDate } from '../../lib/dates'

export function ConversationList() {
  const { conversations, searchQuery, searchResults, isSearching } = useSidebarStore()

  const items = searchQuery.length >= 2 ? searchResults : conversations
  const groups = groupByDate(items)

  if (isSearching) {
    return <div className="px-4 py-8 text-center text-alfred-muted text-sm">Searching...</div>
  }

  if (searchQuery.length >= 2 && items.length === 0) {
    return <div className="px-4 py-8 text-center text-alfred-muted text-sm">No results found</div>
  }

  if (items.length === 0) {
    return <div className="px-4 py-8 text-center text-alfred-muted text-sm">No conversations yet</div>
  }

  return (
    <div className="py-2">
      {groups.map(group => (
        <div key={group.label}>
          <h3 className="px-4 py-1.5 text-[11px] font-medium text-alfred-muted uppercase tracking-wider">
            {group.label}
          </h3>
          {group.items.map(conv => (
            <ConversationItem key={conv.id} conversation={conv} />
          ))}
        </div>
      ))}
    </div>
  )
}
