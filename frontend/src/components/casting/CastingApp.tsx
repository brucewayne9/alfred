import { useEffect, useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { Library } from './Library'
import { CreateDJ } from './CreateDJ'
import { DeploySchedule } from './DeploySchedule'
import './casting.css'

type Tab = 'library' | 'create' | 'deploy'

export function CastingApp() {
  const [tab, setTab] = useState<Tab>('library')
  const { loadStatic, refresh } = useCastingStore()
  useEffect(() => { loadStatic(); refresh() }, [loadStatic, refresh])
  return (
    <div className="casting-app">
      <header className="casting-header">
        <div className="casting-title">
          <h1>Central Casting</h1>
          <p className="casting-sub">Cast, voice, and deploy AI radio hosts onto News Muse.</p>
        </div>
        <nav>
          <button className={tab === 'library' ? 'active' : ''} onClick={() => setTab('library')}>Library</button>
          <button className={tab === 'create' ? 'active' : ''} onClick={() => setTab('create')}>Create a DJ</button>
          <button className={tab === 'deploy' ? 'active' : ''} onClick={() => setTab('deploy')}>Deploy</button>
        </nav>
      </header>
      <div className="casting-body">
        {tab === 'library' && <Library />}
        {tab === 'create' && <CreateDJ onDone={() => setTab('library')} />}
        {tab === 'deploy' && <DeploySchedule />}
      </div>
    </div>
  )
}
