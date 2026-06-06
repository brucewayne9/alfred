import { useCastingStore } from '../../stores/castingStore'
import { castingApi } from '../../api/casting'

export function Library() {
  const { djs, loading } = useCastingStore()
  if (loading) return <p className="casting-empty">Loading…</p>
  if (!djs.length) return <p className="casting-empty">No DJs yet. Create one to get started.</p>
  return (
    <div className="casting-grid">
      {djs.map(dj => (
        <div className="dj-card" key={dj.id}>
          <div className="dj-card-top">
            <span className="dj-name">{dj.name}</span>
            <span className={`dj-status dj-status-${dj.status}`}>{dj.status}</span>
          </div>
          <div className="dj-tags">{dj.archetype_tags.join(' · ') || 'no archetype'}</div>
          <div className="dj-moods">
            <span className="dj-moods-count">{dj.moods_present.length}/8</span> moods captured
          </div>
          <audio className="dj-preview" controls preload="none" src={castingApi.previewUrl(dj.id)} />
        </div>
      ))}
    </div>
  )
}
