import { useCallback } from 'react'
import { Search, X } from 'lucide-react'
import { useSidebarStore } from '../../stores/sidebarStore'

export function SearchBar() {
  const { searchQuery, search, clearSearch } = useSidebarStore()

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    search(e.target.value)
  }, [search])

  return (
    <div className="px-3 py-2 border-b border-alfred-border">
      <div className="relative">
        <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-alfred-muted" />
        <input
          type="text"
          value={searchQuery}
          onChange={handleChange}
          placeholder="Search conversations..."
          className="w-full pl-8 pr-8 py-1.5 bg-alfred-bg border border-alfred-border rounded-lg text-sm text-white placeholder-alfred-muted outline-none focus:border-alfred-accent/50 transition-colors"
        />
        {searchQuery && (
          <button
            onClick={clearSearch}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-alfred-muted hover:text-white transition-colors"
          >
            <X size={14} />
          </button>
        )}
      </div>
    </div>
  )
}
