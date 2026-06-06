import { useEffect, useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { castingApi } from '../../api/casting'
import type { Assignment } from '../../api/casting'

const STATION_ID = 22

export function DeploySchedule() {
  const { djs } = useCastingStore()
  const [assignments, setAssignments] = useState<Assignment[]>([])
  const [djId, setDjId] = useState<number | ''>('')
  const [slot, setSlot] = useState('10a-2p')
  const [effectiveAt, setEffectiveAt] = useState('')
  const reload = () => castingApi.listAssignments(STATION_ID).then(setAssignments)
  useEffect(() => { reload() }, [])

  const schedule = async () => {
    if (!djId || !effectiveAt) return
    await castingApi.createAssignment(Number(djId), STATION_ID, slot, effectiveAt)
    await reload()
  }
  const deployable = djs.filter(d => d.status !== 'draft')

  return (
    <div className="deploy-schedule">
      <h3>Schedule a host onto News Muse (station {STATION_ID})</h3>
      <div className="deploy-form">
        <select value={djId} onChange={e => setDjId(e.target.value ? Number(e.target.value) : '')}>
          <option value="">Pick a DJ…</option>
          {deployable.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <input value={slot} onChange={e => setSlot(e.target.value)} placeholder="slot e.g. 10a-2p" />
        <input type="datetime-local" value={effectiveAt} onChange={e => setEffectiveAt(e.target.value)} />
        <button className="btn-primary" disabled={!djId || !effectiveAt} onClick={schedule}>Schedule swap</button>
      </div>

      <h4>Upcoming lineup</h4>
      {assignments.length === 0
        ? <p className="casting-empty">Nothing queued yet.</p>
        : (
          <ul className="lineup">
            {assignments.map(a => (
              <li key={a.id}>
                <strong>{a.slot}</strong> → {a.dj_name} @ {a.effective_at}
                <span className={`lineup-state${a.applied ? ' live' : ''}`}>{a.applied ? 'live' : 'queued'}</span>
              </li>
            ))}
          </ul>
        )}
    </div>
  )
}
