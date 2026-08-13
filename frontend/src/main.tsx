import React, { Component, ReactNode, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Page = 'Dashboard' | 'Status Review' | 'Models' | 'History' | 'Environment' | 'Thresholds';
type User = { username: string; role: string; mock_mode?: boolean };
type LiveTick = {
  machine_id: string;
  timestamp?: string;
  health_state: number;
  anomaly_score: number;
  model_version: string;
  maintenance: { level: string; reason?: string; trigger?: string };
  a_rms_mps2?: number;
  v_rms_mms?: number;
  a_peak_mps2?: number;
  crest_factor?: number;
  temperature_c?: number;
  mock_mode?: boolean;
};

const SENSOR_META: Record<string, { label: string; unit: string; detail: string }> = {
  a_rms_mps2: { label: 'Acceleration RMS', unit: 'm/s²', detail: 'Overall vibration acceleration energy.' },
  v_rms_mms: { label: 'Velocity RMS', unit: 'mm/s', detail: 'Overall vibration velocity severity.' },
  a_peak_mps2: { label: 'Acceleration Peak', unit: 'm/s²', detail: 'Peak instantaneous acceleration.' },
  crest_factor: { label: 'Crest Factor', unit: '', detail: 'Peak-to-RMS ratio; useful for impulsive behavior.' },
  temperature_c: { label: 'Temperature', unit: '°C', detail: 'Measured spindle temperature.' },
};

async function api(path: string, opts: RequestInit = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      credentials: 'include',
      ...opts,
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
  } finally { window.clearTimeout(timer); }
}

const PAGE_SIZES = [10, 25, 50, 100];
type Paged<T> = { items: T[]; total: number; limit: number; offset: number };
function usePagination(pageSize0 = 25) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSizeRaw] = useState(pageSize0);
  const setPageSize = (n: number) => { setPageSizeRaw(n); setPage(1); };
  const offset = (page - 1) * pageSize;
  return { page, setPage, pageSize, setPageSize, offset };
}
function cn(...values: Array<string | false | null | undefined>) { return values.filter(Boolean).join(' '); }
function fmt(value: any, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits) : String(value);
}
function shortTime(value?: string) {
  if (!value) return '—';
  const d = new Date(value); return Number.isNaN(d.getTime()) ? value : d.toLocaleString([], { month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit' });
}
function tone(level?: string) {
  const x = String(level || '').toUpperCase();
  if (x === 'CRITICAL') return 'critical';
  if (x === 'WARN' || x === 'WARNING') return 'warning';
  if (x === 'OK' || x === 'NORMAL' || x === 'ACTIVE') return 'normal';
  return 'neutral';
}
function triggerLabel(trigger?: string) {
  return ({ health_threshold:'Health threshold', health_inspect:'Inspection threshold', trend_probability:'Trend forecast', none:'No trigger' } as Record<string,string>)[trigger || ''] || (trigger || 'Unknown');
}
function reviewLabel(status?: string) {
  return ({ pending:'Pending', confirmed_anomaly:'Confirmed anomaly', confirmed_normal:'Confirmed normal', acknowledged:'Acknowledged', flagged:'Flagged' } as Record<string,string>)[status || ''] || (status || 'Unknown');
}
function reviewTone(status?: string) {
  if (status === 'confirmed_anomaly' || status === 'flagged') return 'warning';
  if (status === 'confirmed_normal' || status === 'acknowledged') return 'normal';
  if (status === 'pending') return 'neutral';
  return 'neutral';
}
function historyOutcome(row: any): {text: string; tone: string} | null {
  if (row.alert_status) return {text: `Alert · ${reviewLabel(row.alert_status)}`, tone: reviewTone(row.alert_status)};
  if (row.near_miss_status) return {text: `Near miss · ${reviewLabel(row.near_miss_status)}`, tone: reviewTone(row.near_miss_status)};
  return null;
}

const Icon = ({ name }: { name: string }) => {
  const p: Record<string, ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    alert: <><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></>,
    model: <><path d="M4 7h16M7 4v6m10-6v6M5 13h14v7H5z"/><path d="M9 16h6"/></>,
    trend: <><path d="M3 17l5-5 4 3 8-9"/><path d="M15 6h5v5"/></>,
    history: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    sliders: <><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
    close: <path d="m6 6 12 12M18 6 6 18"/>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    logout: <><path d="M9 5H5v14h4M14 8l4 4-4 4M18 12H9"/></>,
    pulse: <><path d="M3 12h4l2-5 4 10 2-5h6"/></>,
    shield: <><path d="M12 3 5 6v5c0 4.6 2.9 7.7 7 10 4.1-2.3 7-5.4 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/></>,
    more: <><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></>,
    check: <path d="M5 12.5 10 17l9-10"/>,
    trash: <><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/></>,
  };
  return <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{p[name]}</svg>;
};

class ErrorBoundary extends Component<{ children: ReactNode }, { error?: Error }> {
  state: { error?: Error } = {};
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) return <main className="center-screen"><section className="state-panel"><div className="state-icon danger">!</div><h1>Interface error</h1><p>{this.state.error.message}</p><p className="muted">Check the terminal running <code>start_project.ps1</code>, then refresh this page.</p></section></main>;
    return this.props.children;
  }
}

function Login({ done, apiError }: { done: () => void; apiError?: string }) {
  const [u,setU]=useState(''); const [p,setP]=useState(''); const [e,setE]=useState(''); const [busy,setBusy]=useState(false);
  return <main className="login-screen">
    <section className="login-brand-panel">
      <div className="wordmark">AKEBONO</div>
      <div className="login-visual"><div className="orb one"/><div className="orb two"/><div className="signal-line"><span/><span/><span/><span/><span/></div></div>
      <div><p className="eyebrow light">FLEET CONDITION INTELLIGENCE</p><h1>Know the spindle<br/>before it stops.</h1><p className="login-copy">Continuous vibration, temperature, anomaly, and maintenance-state visibility in one operator console.</p></div>
      <div className="login-foot">Industrial monitoring · Human-reviewed decisions</div>
    </section>
    <section className="login-form-panel"><form onSubmit={async ev=>{ev.preventDefault();setBusy(true);setE('');try{await api('/api/login',{method:'POST',body:JSON.stringify({username:u,password:p})});done();}catch(err:any){setE(err?.name==='AbortError'?'API request timed out. Check the backend terminal.':String(err?.message||err));}finally{setBusy(false);}}}>
      <div className="login-mark"><span>SCM</span></div><p className="eyebrow">SECURE OPERATOR ACCESS</p><h2>Welcome back</h2><p className="muted">Sign in to open the live monitoring console.</p>
      {apiError&&<Notice tone="warning">Backend check failed: {apiError}</Notice>}
      <label className="field"><span>Username</span><input autoFocus value={u} onChange={x=>setU(x.target.value)} placeholder="Enter username" autoComplete="username"/></label>
      <label className="field"><span>Password</span><input type="password" value={p} onChange={x=>setP(x.target.value)} placeholder="Enter password" autoComplete="current-password"/></label>
      {e&&<Notice tone="critical">{e}</Notice>}
      <button className="primary wide" disabled={busy}>{busy?'Signing in…':'Sign in'}<Icon name="chevron"/></button>
      <p className="login-help">Local deployment · credentials are validated by the backend</p>
    </form></section>
  </main>
}

function Notice({ children, tone: t='neutral' }: { children:ReactNode; tone?:string }) { return <div className={cn('notice',`notice-${t}`)}>{children}</div> }

function App(){
  const [me,setMe]=useState<User|false|undefined>(); const [apiError,setApiError]=useState<string>(); const [page,setPage]=useState<Page>('Dashboard'); const [collapsed,setCollapsed]=useState(false);
  const [machines,setMachines]=useState<string[]>(['VVB001']); const [machineId,setMachineId]=useState(localStorage.getItem('machine_id')||'VVB001');
  const refreshMe=()=>api('/api/me',{},5000).then((x:User)=>{setMe(x);setApiError(undefined)}).catch((err:any)=>{setMe(false);if(err?.name==='AbortError')setApiError('API timed out on /api/me');else if(!String(err?.message||'').includes('401'))setApiError(String(err?.message||err));});
  useEffect(()=>{refreshMe()},[]);
  useEffect(()=>{if(me)api('/api/machines').then((r:{items:string[];default:string})=>{const items=r.items.length?r.items:[r.default];setMachines(items);setMachineId(current=>items.includes(current)?current:(r.default||items[0]))}).catch(()=>{})},[me]);
  useEffect(()=>{localStorage.setItem('machine_id',machineId)},[machineId]);
  if(me===undefined)return <main className="center-screen"><section className="state-panel"><div className="spinner"/><p className="eyebrow">INITIALIZING CONSOLE</p><h1>Connecting to the local API</h1><p className="muted">If this takes more than a few seconds, check <code>localhost:8000/docs</code>.</p></section></main>;
  if(!me)return <Login done={refreshMe} apiError={apiError}/>;
  const nav:Array<{page:Page;icon:string;label:string}>=[
    {page:'Dashboard',icon:'grid',label:'Overview'},{page:'Status Review',icon:'alert',label:'Status review'},{page:'Models',icon:'model',label:'Models'},{page:'History',icon:'history',label:'History'},{page:'Thresholds',icon:'sliders',label:'Thresholds'},...(me.role==='admin'?[{page:'Environment' as Page,icon:'database',label:'Environment'}]:[])
  ];
  return <div className={cn('app-shell',collapsed&&'nav-collapsed')}>
    <aside className="sidebar">
      <div className="brand-lockup"><div className="brand-symbol"><span/></div><div className="brand-copy"><small>AKEBONO</small><strong>Spindle Monitor</strong><span>{machineId}</span></div></div>
      <button className="collapse-btn" onClick={()=>setCollapsed(v=>!v)} title={collapsed?'Expand navigation':'Collapse navigation'}><Icon name="chevron"/></button>
      <nav>{nav.map(n=><button key={n.page} className={cn('nav-item',page===n.page&&'active')} onClick={()=>setPage(n.page)} title={n.label}><Icon name={n.icon}/><span>{n.label}</span>{page===n.page&&<i/>}</button>)}</nav>
      <div className="sidebar-bottom">
        {me.mock_mode&&<button className="demo-chip" title="This session is using temporary synthetic data"><span className="dot"/><span>Mock data</span></button>}
        <div className="user-chip"><div className="avatar">{me.username.slice(0,2).toUpperCase()}</div><div><strong>{me.username}</strong><span>{me.role}</span></div><button title="Sign out" onClick={async()=>{await api('/api/logout',{method:'POST'});setMe(false)}}><Icon name="logout"/></button></div>
      </div>
    </aside>
    <main className="workspace">
      <header className="topbar"><div><p className="breadcrumb">{machineId} <span>/</span> {page}</p></div><div className="topbar-meta"><Select value={machineId} onChange={setMachineId} options={machines.map(id=>[id,id] as [string,string])}/><span className="system-pill"><span className="pulse-dot"/>System online</span><span className="clock">{new Date().toLocaleDateString([], {weekday:'short',month:'short',day:'numeric'})}</span></div></header>
      <div className="content"><PageView key={machineId} page={page} role={me.role} mock={!!me.mock_mode} machineId={machineId}/></div>
    </main>
  </div>
}

function PageHeader({ eyebrow, title, description, actions }: { eyebrow:string;title:string;description:string;actions?:ReactNode }) {
  return <div className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>{actions&&<div className="page-actions">{actions}</div>}</div>
}

function PageView({page,role,mock,machineId}:{page:Page;role:string;mock:boolean;machineId:string}){
  if(page==='Dashboard')return <Dashboard mock={mock} machineId={machineId}/>;
  if(page==='Status Review')return <StatusReview machineId={machineId}/>;
  if(page==='Models')return <Models admin={role==='admin'} machineId={machineId}/>;
  if(page==='History')return <HistoryPage machineId={machineId}/>;
  if(page==='Thresholds')return <Thresholds admin={role==='admin'}/>;
  return <Environment/>;
}

function Sparkline({values, inverse=false}:{values:number[];inverse?:boolean}){
  const clean=values.filter(Number.isFinite); if(clean.length<2)return <svg className="sparkline" viewBox="0 0 200 54"/>;
  const min=Math.min(...clean),max=Math.max(...clean),span=max-min||1;
  const pts=clean.map((v,i)=>`${(i/(clean.length-1))*200},${48-((v-min)/span)*40}`).join(' ');
  return <svg className={cn('sparkline',inverse&&'inverse')} viewBox="0 0 200 54" preserveAspectRatio="none"><polyline points={pts}/></svg>
}

function Dashboard({mock,machineId}:{mock:boolean;machineId:string}){
  const [live,setLive]=useState<LiveTick>(); const [history,setHistory]=useState<LiveTick[]>([]); const [wsState,setWsState]=useState<'connecting'|'live'|'offline'>('connecting'); const [detail,setDetail]=useState<string|null>(null); const [menu,setMenu]=useState(false);
  useEffect(()=>{setLive(undefined);setHistory([]);setWsState('connecting');const proto=location.protocol==='https:'?'wss':'ws';const ws=new WebSocket(`${proto}://${location.host}/ws/live?machine_id=${encodeURIComponent(machineId)}`);ws.onopen=()=>setWsState('live');ws.onmessage=e=>{const row=JSON.parse(e.data);setLive(row);setHistory(h=>[...h.slice(-39),row]);setWsState('live')};ws.onerror=()=>setWsState('offline');ws.onclose=()=>setWsState('offline');return()=>ws.close()},[machineId]);
  const level=live?.maintenance?.level||'WAITING'; const health=Number(live?.health_state||0); const anomaly=Number(live?.anomaly_score||0);
  const sensors=Object.keys(SENSOR_META).map(k=>({key:k,value:(live as any)?.[k],...SENSOR_META[k]}));
  return <>
    <PageHeader eyebrow="LIVE CONDITION" title="Machine overview" description="Current spindle condition, anomaly behavior, and sensor context from the production monitoring path." actions={<><span className={cn('connection-badge',wsState)}><span/>{wsState==='live'?'Live stream':wsState==='connecting'?'Connecting…':'Stream offline'}</span><div className="dropdown-wrap"><button className="icon-button" onClick={()=>setMenu(v=>!v)}><Icon name="more"/></button>{menu&&<div className="dropdown-menu right"><button onClick={()=>{setDetail('system');setMenu(false)}}>System details</button><button onClick={()=>{setDetail('model');setMenu(false)}}>Model context</button><button onClick={()=>location.reload()}>Reconnect interface</button></div>}</div></>}/>
    {mock&&<div className="demo-ribbon"><span>DEMO</span><p>Temporary synthetic signal is driving this interface. ML inference and plant PostgreSQL are bypassed.</p><button onClick={()=>setDetail('demo')}>What is simulated?</button></div>}
    <section className={cn('status-board',`tone-${tone(level)}`)}>
      <div className="status-main">
        <div className="status-kicker"><span className="machine-dot"/>{machineId} · SPINDLE CONDITION</div>
        <div className="status-line"><div><span className="status-label">Current state</span><h2>{level}</h2></div><button className="text-button" onClick={()=>setDetail('status')}><Icon name="info"/>Why this status?</button></div>
        <p className="status-reason">{live?.maintenance?.reason||'Waiting for the first monitoring tick from the backend.'}</p>
        <div className="health-row"><div><span>Health state</span><strong>{live?fmt(health,1):'—'}<small>%</small></strong></div><div className="health-track"><i style={{width:`${Math.max(0,Math.min(100,health))}%`}}/></div><span className="health-caption">0 critical <b>·</b> 100 healthy</span></div>
      </div>
      <div className="status-side">
        <div className="metric-stack"><span>Anomaly score<button className="mini-info" onClick={()=>setDetail('anomaly')}><Icon name="info"/></button></span><strong>{live?fmt(anomaly,4):'—'}</strong><Sparkline values={history.map(x=>Number(x.anomaly_score))}/></div>
        <div className="vertical-divider"/>
        <div className="metric-stack"><span>Model</span><strong className="model-name">{live?.model_version||'—'}</strong><small>{shortTime(live?.timestamp)}</small></div>
      </div>
    </section>
    <section className="sensor-panel">
      <div className="panel-heading"><div><p className="eyebrow">LIVE SENSOR CHANNELS</p><h3>Signal snapshot</h3></div><button className="ghost-button" onClick={()=>setDetail('sensors')}>Channel guide <Icon name="chevron"/></button></div>
      <div className="sensor-grid">{sensors.map(s=><button key={s.key} className="sensor-item" onClick={()=>setDetail(`sensor:${s.key}`)}><div><span>{s.label}</span><small>{s.unit||'ratio'}</small></div><strong>{s.value===undefined?'—':fmt(s.value,s.key==='temperature_c'?1:3)}</strong><Sparkline values={history.map(x=>Number((x as any)[s.key])).filter(Number.isFinite)}/><div className="sensor-foot"><span>{s.detail}</span><Icon name="chevron"/></div></button>)}</div>
    </section>
    {detail&&<DashboardModal kind={detail} live={live} onClose={()=>setDetail(null)}/>} 
  </>
}

function DashboardModal({kind,live,onClose}:{kind:string;live?:LiveTick;onClose:()=>void}){
  let title='Details',body:ReactNode=null;
  if(kind==='status'){title='Maintenance status evidence';body=<div className="detail-list"><DetailRow label="State" value={live?.maintenance?.level||'—'} badge={tone(live?.maintenance?.level)}/><DetailRow label="Trigger" value={triggerLabel(live?.maintenance?.trigger)}/><DetailRow label="Reason" value={live?.maintenance?.reason||'No reason received.'}/><DetailRow label="Health" value={`${fmt(live?.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(live?.anomaly_score,4)}/></div>}
  else if(kind==='anomaly'){title='How to read anomaly score';body=<><p>The anomaly score summarizes how unusual the current feature pattern is relative to the active model reference. The frontend only displays this value; it does not calculate or modify it.</p><div className="scale"><span>Lower deviation</span><i/><span>Higher deviation</span></div><p className="muted">Current score: <b>{fmt(live?.anomaly_score,4)}</b></p></>}
  else if(kind==='model'||kind==='system'){title=kind==='model'?'Active model context':'Runtime context';body=<div className="detail-list"><DetailRow label="Model version" value={live?.model_version||'—'}/><DetailRow label="Latest tick" value={shortTime(live?.timestamp)}/><DetailRow label="Data path" value="Backend → WebSocket → browser"/><DetailRow label="Frontend role" value="Visualization only"/></div>}
  else if(kind==='demo'){title='Temporary mock mode';body=<><p>This session is driven by a local SQLite demo database and synthetic multi-machine ticks so the web experience can be tested without plant infrastructure.</p><div className="callout-grid"><div><b>Simulated</b><span>Sensor values, predictions, alerts, model metadata</span></div><div><b>Real application path</b><span>Authentication, API calls, review actions, WebSocket UI flow</span></div><div><b>Bypassed</b><span>Plant PostgreSQL and production ML inference worker</span></div></div></>}
  else if(kind==='sensors'){title='Sensor channel guide';body=<div className="detail-list">{Object.entries(SENSOR_META).map(([k,m])=><DetailRow key={k} label={`${m.label} (${k})`} value={`${m.detail}${m.unit?` Unit: ${m.unit}.`:''}`}/>)}</div>}
  else if(kind.startsWith('sensor:')){const k=kind.split(':')[1],m=SENSOR_META[k];title=m?.label||k;body=<div className="detail-list"><DetailRow label="Raw key" value={k}/><DetailRow label="Current value" value={`${fmt((live as any)?.[k],3)} ${m?.unit||''}`}/><DetailRow label="Meaning" value={m?.detail||'Sensor channel'}/><DetailRow label="Source" value="Backend live tick"/></div>}
  return <Modal title={title} onClose={onClose}>{body}</Modal>
}

function DetailRow({label,value,badge}:{label:string;value:any;badge?:string}){return <div className="detail-row"><span>{label}</span>{badge?<StatusBadge value={String(value)} forced={badge}/>:<strong>{String(value)}</strong>}</div>}
function StatusBadge({value,forced}:{value:string;forced?:string}){return <span className={cn('status-badge',forced||tone(value))}><i/>{value}</span>}

function Modal({title,onClose,children,wide=false}:{title:string;onClose:()=>void;children:ReactNode;wide?:boolean}){
  useEffect(()=>{const f=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};window.addEventListener('keydown',f);return()=>window.removeEventListener('keydown',f)},[onClose]);
  return <div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)onClose()}}><section className={cn('modal',wide&&'modal-wide')}><header><div><p className="eyebrow">DETAIL VIEW</p><h2>{title}</h2></div><button className="icon-button" onClick={onClose}><Icon name="close"/></button></header><div className="modal-body">{children}</div></section></div>
}

function StatusReview({machineId}:{machineId:string}){
  const [tab,setTab]=useState<'alert'|'near'>('alert');
  return <>
    <PageHeader eyebrow="HUMAN REVIEW" title="Status review" description="Confirm maintenance alerts and healthy-state near-misses that need a human decision, in one place." actions={<div className="tab-switch"><button className={cn('tab-btn',tab==='alert'&&'active')} onClick={()=>setTab('alert')}>Alert review</button><button className={cn('tab-btn',tab==='near'&&'active')} onClick={()=>setTab('near')}>Near miss</button></div>}/>
    {tab==='alert'?<AlertPanel machineId={machineId}/>:<NearMissPanel machineId={machineId}/>}
  </>
}

function AlertPanel({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[status,setStatus]=useState('pending'),[trigger,setTrigger]=useState(''),[selected,setSelected]=useState<any|null>(null),[context,setContext]=useState<any[]>([]),[busy,setBusy]=useState(false);
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/alerts?machine_id=${encodeURIComponent(machineId)}&status=${encodeURIComponent(status)}${trigger?`&trigger=${encodeURIComponent(trigger)}`:''}&limit=${pageSize}&offset=${offset}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[status,trigger,pageSize,offset]);
  useEffect(()=>{setPage(1)},[status,trigger]);
  const inspect=async(a:any)=>{setSelected(a);setContext([]);try{setContext(await api(`/api/alerts/${a.id}/context?hours=3`))}catch{setContext([])}};
  const decide=async(decision:string)=>{if(!selected)return;setBusy(true);try{await api(`/api/alerts/${selected.id}/review`,{method:'POST',body:JSON.stringify({decision})});setSelected(null);load()}finally{setBusy(false)}};
  return <>
  <div className="filter-bar sub-filter-bar"><Select value={status} onChange={setStatus} options={[['pending','Pending'],['confirmed_anomaly','Confirmed anomaly'],['confirmed_normal','Confirmed normal']]}/><Select value={trigger} onChange={setTrigger} options={[['','All triggers'],['health_threshold','Health threshold'],['health_inspect','Inspection threshold'],['trend_probability','Trend forecast']]}/></div>
  {error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{total} result{total===1?'':'s'}</span><small>Click any row to inspect ±3 hour context</small></div>{rows.length===0?<Empty title="No alerts in this view" text="Try another review status or trigger filter."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>Level</th><th>Trigger</th><th>Health</th><th>Anomaly</th><th>Review state</th><th/></tr></thead><tbody>{rows.map(a=><tr key={a.id} onClick={()=>inspect(a)}><td><strong>{shortTime(a.tick_timestamp)}</strong><small>#{a.id}</small></td><td><StatusBadge value={a.level}/></td><td>{triggerLabel(a.trigger)}</td><td>{fmt(a.health_state,1)}%</td><td>{fmt(a.anomaly_score,4)}</td><td><span className="review-state">{String(a.status).replaceAll('_',' ')}</span></td><td><Icon name="chevron"/></td></tr>)}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title={`Alert #${selected.id}`} wide onClose={()=>setSelected(null)}><div className="split-detail"><div><div className="detail-list"><DetailRow label="Level" value={selected.level} badge={tone(selected.level)}/><DetailRow label="Trigger" value={triggerLabel(selected.trigger)}/><DetailRow label="Health" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/></div><Disclosure title="Raw sensor snapshot"><JsonGrid value={selected.raw_reading}/></Disclosure></div><div><p className="section-label">SURROUNDING CONTEXT</p><MiniContextChart rows={context}/><p className="muted small">Context is fetched from the backend around the alert timestamp. This view does not recalculate the maintenance decision.</p></div></div>{selected.status==='pending'&&<div className="modal-actions"><button className="secondary danger-text" disabled={busy} onClick={()=>decide('confirmed_normal')}>Confirm normal</button><button className="primary" disabled={busy} onClick={()=>decide('confirmed_anomaly')}>Confirm anomaly</button></div>}</Modal>}
  </>
}

function NearMissPanel({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[reviewStatus,setReviewStatus]=useState('pending'),[selected,setSelected]=useState<any|null>(null),[busy,setBusy]=useState(false),[actionError,setActionError]=useState('');
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/near-miss?machine_id=${encodeURIComponent(machineId)}&status=${encodeURIComponent(reviewStatus)}&limit=${pageSize}&offset=${offset}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[reviewStatus,pageSize,offset]);
  useEffect(()=>{setPage(1)},[reviewStatus]);
  const decide=async(decision:string)=>{
    if(!selected)return; setBusy(true); setActionError('');
    try{await api(`/api/near-miss/${selected.id}/review`,{method:'POST',body:JSON.stringify({decision})});setSelected(null);load()}
    catch(e:any){setActionError(String(e.message||e))}
    finally{setBusy(false)}
  };
  return <>
  <div className="filter-bar sub-filter-bar"><Select value={reviewStatus} onChange={setReviewStatus} options={[['pending','Pending'],['acknowledged','Acknowledged'],['flagged','Flagged']]}/></div>
  {error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{total} record{total===1?'':'s'}</span><small>Ranked by backend near-miss query</small></div>{rows.length===0?<Empty title="No matching records" text="Try another review status."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>State</th><th>Health</th><th>Anomaly</th><th>Model</th><th>Review state</th><th/></tr></thead><tbody>{rows.map((r,i)=><tr key={r.id||`${r.tick_timestamp}-${i}`} onClick={()=>setSelected(r)}><td><strong>{shortTime(r.tick_timestamp)}</strong></td><td><StatusBadge value={r.maintenance_level||'OK'}/></td><td>{fmt(r.health_state,1)}%</td><td>{fmt(r.anomaly_score,4)}</td><td>{r.model_version||'—'}</td><td><span className="review-state">{String(r.review_status||'pending').replaceAll('_',' ')}</span></td><td><Icon name="chevron"/></td></tr>)}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title="Near-miss record" wide onClose={()=>setSelected(null)}><div className="split-detail"><div className="detail-list"><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/><DetailRow label="Maintenance state" value={selected.maintenance_level||'—'} badge={tone(selected.maintenance_level)}/><DetailRow label="Health" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Review state" value={String(selected.review_status||'pending').replaceAll('_',' ')}/></div><div><p className="section-label">SENSOR SNAPSHOT</p><JsonGrid value={selected.raw_reading}/><Disclosure title="Full backend record"><pre className="code-panel compact">{JSON.stringify(selected,null,2)}</pre></Disclosure></div></div>
  {(selected.review_status||'pending')==='pending'&&<div className="modal-actions">{actionError&&<Notice tone="critical">{actionError}</Notice>}<button className="secondary" disabled={busy} onClick={()=>decide('acknowledged')}>Acknowledge</button><button className="primary" disabled={busy} onClick={()=>decide('flagged')}>Flag for follow-up</button></div>}
  {(selected.review_status||'pending')==='flagged'&&<p className="muted small flag-note">Flagging opens a suspected-false-negative regression check for this time window (see Models → retraining gate).</p>}
  </Modal>}
  </>
}

function Select({value,onChange,options}:{value:string;onChange:(v:string)=>void;options:Array<[string,string]>}){return <label className="select-wrap"><select value={value} onChange={e=>onChange(e.target.value)}>{options.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select><span>⌄</span></label>}
function Switch({checked,onChange,disabled}:{checked:boolean;onChange:()=>void;disabled?:boolean}){return <button type="button" role="switch" aria-checked={checked} disabled={disabled} className={cn('switch',checked&&'on')} onClick={onChange}><span className="switch-thumb"/></button>}
function InUseBadge(){return <span className="in-use-badge"><Icon name="check"/>In use</span>}

function Pagination({page,pageSize,total,onPage,onPageSize,sizes=PAGE_SIZES}:{page:number;pageSize:number;total:number;onPage:(p:number)=>void;onPageSize:(n:number)=>void;sizes?:number[]}){
  const pages=Math.max(1,Math.ceil(total/pageSize));
  const from=total===0?0:(page-1)*pageSize+1; const to=Math.min(total,page*pageSize);
  return <div className="pagination">
    <span className="pagination-info">{total===0?'No records':`${from}–${to} of ${total}`}</span>
    <div className="pagination-controls">
      <button className="icon-button" disabled={page<=1} onClick={()=>onPage(1)} title="First page">«</button>
      <button className="icon-button" disabled={page<=1} onClick={()=>onPage(page-1)} title="Previous page">‹</button>
      <span className="pagination-page">Page {page} of {pages}</span>
      <button className="icon-button" disabled={page>=pages} onClick={()=>onPage(page+1)} title="Next page">›</button>
      <button className="icon-button" disabled={page>=pages} onClick={()=>onPage(pages)} title="Last page">»</button>
    </div>
    <Select value={String(pageSize)} onChange={v=>onPageSize(Number(v))} options={sizes.map(s=>[String(s),`${s} / page`] as [string,string])}/>
  </div>
}

function MiniContextChart({rows}:{rows:any[]}){const values=rows.map(r=>Number(r.health_state)).filter(Number.isFinite);return <div className="context-chart"><Sparkline values={values}/><div className="context-axis"><span>{rows[0]?shortTime(rows[0].tick_timestamp):'No data'}</span><span>Health trend</span><span>{rows.length?shortTime(rows[rows.length-1].tick_timestamp):''}</span></div></div>}
function JsonGrid({value}:{value:any}){const obj=typeof value==='object'&&value?value:{};return <div className="json-grid">{Object.entries(obj).map(([k,v])=><div key={k}><span>{SENSOR_META[k]?.label||k}</span><strong>{String(v)}</strong></div>)}</div>}
function Disclosure({title,children,defaultOpen=false}:{title:string;children:ReactNode;defaultOpen?:boolean}){const[o,setO]=useState(defaultOpen);return <div className={cn('disclosure',o&&'open')}><button onClick={()=>setO(v=>!v)}><span>{title}</span><Icon name="chevron"/></button>{o&&<div className="disclosure-body">{children}</div>}</div>}

function Models({admin,machineId}:{admin:boolean;machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[error,setError]=useState(''),[selected,setSelected]=useState<any|null>(null),[retrain,setRetrain]=useState<any>(),[panel,setPanel]=useState<'training'|'calibration'|null>(null);
  const load=()=>Promise.all([api('/api/models'),api('/api/retrain/status')]).then(([m,s])=>{setRows(m);setRetrain(s)}).catch(e=>setError(String(e.message||e)));useEffect(()=>{load()},[]);
  const activeVersion=rows.find(r=>r.status==='active');
  const deleteVersion=async(versionId:string)=>{if(!window.confirm(`Delete model version ${versionId}? This removes its artifacts and recalibration history and cannot be undone.`))return;try{await api(`/api/models/${versionId}`,{method:'DELETE'});setSelected(null);load()}catch(e){setError(String((e as any).message||e))}};
  return <><PageHeader eyebrow="MODEL LIFECYCLE" title="Models & retraining" description="Review active and shadow bundles, validation evidence, and the human-controlled promotion path." actions={<><button className="secondary" onClick={()=>setPanel('training')}>Training options</button>{activeVersion&&<button className="secondary" onClick={()=>setPanel('calibration')}>Health% calibration</button>}{admin&&<button className="primary" onClick={async()=>{await api('/api/retrain/trigger',{method:'POST'});load()}}>Run shadow retrain</button>}</>}/>{error&&<Notice tone="critical">{error}</Notice>}
  {retrain&&<div className="inline-summary"><div><span>Reference candidates</span><strong>{retrain.candidate_count??0}</strong></div><div><span>Batch target</span><strong>{retrain.batch_size??'—'}</strong></div><div><span>Retrain state</span><StatusBadge value={retrain.due?'DUE':'NOT DUE'} forced={retrain.due?'warning':'normal'}/></div><button className="text-button" onClick={()=>setSelected({__retrain:true,...retrain})}>How scheduling works <Icon name="info"/></button></div>}
  <section className="data-surface">{rows.length===0?<Empty title="No model versions" text="No registered model bundles were returned by the backend."/>:<div className="responsive-table"><table><thead><tr><th>Version</th><th>Status</th><th>Created</th><th>Validation</th><th>Promoted by</th><th/></tr></thead><tbody>{rows.map(m=><tr key={m.version_id} onClick={()=>setSelected(m)}><td><strong>{m.version_id}</strong><small>{m.reference_signature||'No reference signature'}</small></td><td><StatusBadge value={m.status}/></td><td>{shortTime(m.created_at)}</td><td><span className={cn('validation-dot',m.validation_report?.passed===false&&'bad')}/>{m.validation_report?.passed===false?'Failed':'Passed / available'}</td><td>{m.promoted_by||'—'}</td><td><Icon name="chevron"/></td></tr>)}</tbody></table></div>}</section>
  {selected&&<Modal title={selected.__retrain?'Retraining schedule':selected.version_id} onClose={()=>setSelected(null)} wide={!selected.__retrain}>{selected.__retrain?<div className="detail-list"><DetailRow label="Confirmed-normal candidates" value={selected.candidate_count??0}/><DetailRow label="Batch threshold" value={selected.batch_size??'—'}/><DetailRow label="Currently due" value={selected.due?'Yes':'No'}/><p className="muted">The backend owns the retraining decision. This popup only exposes the current scheduler state.</p></div>:<><div className="split-detail"><div className="detail-list"><DetailRow label="Status" value={selected.status} badge={tone(selected.status)}/><DetailRow label="Artifact path" value={selected.artifact_path||'—'}/><DetailRow label="Reference signature" value={selected.reference_signature||'—'}/><DetailRow label="Created" value={shortTime(selected.created_at)}/><DetailRow label="Promoted" value={shortTime(selected.promoted_at)}/><DetailRow label="Promoted by" value={selected.promoted_by||'—'}/></div><div><p className="section-label">VALIDATION REPORT</p><pre className="code-panel">{JSON.stringify(selected.validation_report||{},null,2)}</pre></div></div>{admin&&<div className="modal-actions">{selected.status!=='active'&&selected.status!=='rejected'&&<button className="primary" onClick={async()=>{await api(`/api/models/${selected.version_id}/promote`,{method:'POST'});setSelected(null);load()}}>Promote to active</button>}{selected.status!=='active'&&<button className="secondary danger-text" onClick={()=>deleteVersion(selected.version_id)}>Delete version</button>}</div>}</>}</Modal>}
  {panel==='training'&&<Modal title="Training options" onClose={()=>setPanel(null)}><TrainingOptions admin={admin}/></Modal>}
  {panel==='calibration'&&activeVersion&&<Modal title={`Health% calibration — ${machineId} (${activeVersion.version_id})`} onClose={()=>setPanel(null)} wide><CalibrationPanel admin={admin} versionId={activeVersion.version_id} machineId={machineId}/></Modal>}
  </>
}

function TrainingOptions({admin}:{admin:boolean}){
  const [v,setV]=useState<any>(),[saved,setSaved]=useState(false),[error,setError]=useState('');
  useEffect(()=>{api('/api/config/training').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  const meta:Record<string,{label:string;desc:string;unit:string}>={
    RETRAIN_BATCH_SIZE:{label:'Retrain batch size',desc:'Confirmed-normal alerts needed before a shadow retrain is due.',unit:'alerts'},
    RETRAIN_TIME_CAP_DAYS:{label:'Retrain time cap',desc:'Retrain becomes due after this many days even below the batch size.',unit:'days'},
    REFERENCE_WINDOW_MONTHS:{label:'Reference window',desc:'How much history is kept in the reference set used for calibration and dedup.',unit:'months'},
    REFERENCE_DEDUP_WINDOW_HOURS:{label:'Dedup window',desc:'Candidates within this many hours of an existing reference row are treated as duplicates.',unit:'hours'},
    REFERENCE_COSINE_SIMILARITY:{label:'Dedup similarity',desc:'Cosine-similarity threshold above which a candidate is considered a duplicate.',unit:'0–1'},
  };
  if(error)return <Notice tone="critical">{error}</Notice>;
  if(!v)return <Loading/>;
  return <div className="settings-surface">
  <div className="setting-row"><div><strong>Automatic retraining</strong><span>Hourly background check that runs a shadow retrain once enough confirmed-normal candidates are available. Turning this off doesn't disable the manual "Run shadow retrain" button above.</span><code>AUTO_RETRAIN_ENABLED</code></div><div className="setting-control toggle-control"><Switch checked={!!v.AUTO_RETRAIN_ENABLED} disabled={!admin} onChange={()=>setV({...v,AUTO_RETRAIN_ENABLED:!v.AUTO_RETRAIN_ENABLED})}/><span className={cn('toggle-state',v.AUTO_RETRAIN_ENABLED?'on':'off')}>{v.AUTO_RETRAIN_ENABLED?'On':'Off'}</span></div></div>
  {Object.keys(v).filter(k=>k!=='AUTO_RETRAIN_ENABLED').map(k=><div className="setting-row" key={k}><div><strong>{meta[k]?.label||k}</strong><span>{meta[k]?.desc||k}</span><code>{k}</code></div><div className="setting-control"><input type="number" step="any" value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}/><span>{meta[k]?.unit}</span></div></div>)}
  {admin&&<div className="modal-actions"><button className="primary" onClick={async()=>{setSaved(false);await api('/api/config/training',{method:'PUT',body:JSON.stringify(v)});setSaved(true)}}>Save training options</button></div>}
  {saved&&<Notice tone="normal">Saved. The next shadow retrain — manual or scheduled — uses these values; nothing retrains automatically just from saving.</Notice>}
  <p className="muted">These already drove every shadow retrain — this just makes them editable instead of a direct database write.</p>
  </div>
}

function CalibrationPanel({admin,versionId,machineId}:{admin:boolean;versionId:string;machineId:string}){
  const [data,setData]=useState<any>(),[error,setError]=useState(''),[busy,setBusy]=useState(false),[hours,setHours]=useState(24),[minRows,setMinRows]=useState(200),[source,setSource]=useState<'spec_bounds'|'threshold'>('spec_bounds');
  const [specBounds,setSpecBounds]=useState<Record<string,number|null>|null>(null);
  const [thresholds,setThresholds]=useState<any>(null),[thBusy,setThBusy]=useState(false),[thSaved,setThSaved]=useState(false);
  const load=()=>api(`/api/models/${versionId}/calibrations?machine_id=${encodeURIComponent(machineId)}`).then(setData).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[versionId]);
  useEffect(()=>{api('/api/config/spec-bounds').then(r=>setSpecBounds(r.bounds)).catch(()=>{});api('/api/config/thresholds').then(setThresholds).catch(()=>{})},[]);
  const activate=async(id:number|null)=>{setBusy(true);setError('');try{await api(`/api/models/${versionId}/calibration/activate?machine_id=${encodeURIComponent(machineId)}`,{method:'POST',body:JSON.stringify({calibration_id:id})});await load()}catch(e){setError(String((e as any).message||e))}finally{setBusy(false)}};
  const recalibrate=async()=>{setBusy(true);setError('');try{await api(`/api/models/${versionId}/recalibrate`,{method:'POST',body:JSON.stringify({hours,min_rows:minRows,source,machine_id:machineId})});await load()}catch(e){setError(String((e as any).message||e))}finally{setBusy(false)}};
  const removeCalibration=async(id:number)=>{if(!window.confirm('Delete this recalibration run? This cannot be undone.'))return;setBusy(true);setError('');try{await api(`/api/models/${versionId}/calibration/${id}?machine_id=${encodeURIComponent(machineId)}`,{method:'DELETE'});await load()}catch(e){setError(String((e as any).message||e))}finally{setBusy(false)}};
  const saveThresholds=async()=>{setThBusy(true);setThSaved(false);try{await api('/api/config/thresholds',{method:'PUT',body:JSON.stringify(thresholds)});setThSaved(true)}catch(e){setError(String((e as any).message||e))}finally{setThBusy(false)}};
  if(error)return <Notice tone="critical">{error}</Notice>;
  if(!data)return <Loading/>;
  const activeId=data.active_calibration_id;
  return <div className="settings-surface">
    <div className={cn('setting-row',activeId==null&&'in-use-row')}><div><strong>Normal (pooled default)</strong><span>The health% anchor baked into this model's own bundle, unmodified.</span></div><div className="calibration-actions">{activeId==null?<InUseBadge/>:(admin?<button className="text-button" disabled={busy} onClick={()=>activate(null)}>Use this</button>:<span className="muted">Not in use</span>)}</div></div>
    {data.items.length===0&&<p className="muted">No recalibration runs yet for this model version.</p>}
    {data.items.map((c:any)=><div className={cn('setting-row',activeId===c.id&&'in-use-row')} key={c.id}><div><strong>Recalibrated — {shortTime(c.created_at)}</strong><span>{c.source_description||'—'} · {c.source_rows} rows · by {c.created_by||'—'}</span></div><div className="calibration-actions">{activeId===c.id?<InUseBadge/>:(admin?<button className="text-button" disabled={busy} onClick={()=>activate(c.id)}>Use this</button>:<span className="muted">Not in use</span>)}{admin&&<button className="icon-button danger" disabled={busy||activeId===c.id} title={activeId===c.id?'Switch off this calibration before deleting it':'Delete this recalibration'} onClick={()=>removeCalibration(c.id)}><Icon name="trash"/></button>}</div></div>)}
    {admin&&<Disclosure title="Recalibrate now"><div className="explain-grid">
      <label className="env-field"><span>Live window</span><input type="number" min={1} step="any" value={hours} onChange={e=>setHours(Number(e.target.value))}/><small>Hours of recent readings to compute the new baseline from.</small></label>
      <label className="env-field"><span>Minimum rows required</span><input type="number" min={1} value={minRows} onChange={e=>setMinRows(Number(e.target.value))}/><small>Recalibration is refused if fewer normal rows than this are available in the window.</small></label>
    </div>
    <div className="setting-row"><div><strong>Reference source</strong><span>What counts as "normal" for this run.</span></div><div className="setting-control"><Select value={source} onChange={v=>setSource(v as 'spec_bounds'|'threshold')} options={[['spec_bounds','Spec bounds'],['threshold','Maintenance threshold']]}/></div></div>
    <p className="muted">{source==='spec_bounds'?'Every raw sensor reading stays within the fixed rated bounds (config.SPEC_MAX) — a static definition, independent of the maintenance policy.':'The row\'s own live prediction came back OK — i.e. health stayed above the Thresholds page\'s inspection/failure limits. Tracks whatever the maintenance policy is currently tuned to instead of a fixed sensor range.'}</p>
    {source==='spec_bounds'?<div className="source-detail">
      <div className="source-detail-head"><span>config.SPEC_MAX</span><span className="static-tag">Fixed in code, not editable here</span></div>
      {specBounds?<div className="json-grid">{Object.entries(specBounds).map(([k,val])=><div key={k}><span>{SENSOR_META[k]?.label||k}</span><strong>{val==null?'Not set':`≤ ${val} ${SENSOR_META[k]?.unit||''}`}</strong></div>)}</div>:<Loading/>}
    </div>:<div className="source-detail">
      <div className="source-detail-head"><span>Maintenance thresholds</span>{admin&&thresholds&&<button className="text-button" disabled={thBusy} onClick={saveThresholds}>{thBusy?'Saving…':'Save changes'}</button>}</div>
      {thresholds?<div className="explain-grid">
        <label className="env-field"><span>Inspection health threshold</span><input type="number" step="any" value={thresholds.MAINTENANCE_HEALTH_INSPECT} disabled={!admin} onChange={e=>setThresholds({...thresholds,MAINTENANCE_HEALTH_INSPECT:Number(e.target.value)})}/><small>Health % below which inspection is recommended.</small></label>
        <label className="env-field"><span>Failure health threshold</span><input type="number" step="any" value={thresholds.FAILURE_HEALTH_THRESHOLD} disabled={!admin} onChange={e=>setThresholds({...thresholds,FAILURE_HEALTH_THRESHOLD:Number(e.target.value)})}/><small>Health % below which maintenance becomes critical.</small></label>
      </div>:<Loading/>}
      {thSaved&&<Notice tone="normal">Saved — this is the same value the Thresholds page uses, so it updates there too.</Notice>}
    </div>}
    <div className="modal-actions"><button className="primary" disabled={busy} onClick={recalibrate}>{busy?'Recalibrating…':'Recalibrate now'}</button></div><p className="muted">Only the active model version can be recalibrated here. Computing a recalibration doesn't change what's deployed by itself — pick "Use this" above to activate it, or leave it on Normal to discard it.</p></Disclosure>}
  </div>
}

function HistoryPage({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[q,setQ]=useState(''),[level,setLevel]=useState(''),[selected,setSelected]=useState<any|null>(null);
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/history?machine_id=${encodeURIComponent(machineId)}&limit=${pageSize}&offset=${offset}${level?`&level=${encodeURIComponent(level)}`:''}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[level,pageSize,offset]);
  useEffect(()=>{setPage(1)},[level]);
  const filtered=useMemo(()=>rows.filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q.toLowerCase())),[rows,q]);
  return <><PageHeader eyebrow="AUDIT TRAIL" title="Prediction history" description="Searchable record of backend prediction ticks and their maintenance outputs." actions={<div className="filter-bar"><label className="search-field"><Icon name="search"/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search this page"/></label><Select value={level} onChange={setLevel} options={[['','All states'],['OK','OK'],['WARN','WARN'],['CRITICAL','CRITICAL']]}/></div>}/>{error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{q?`${filtered.length} of ${rows.length} on this page`:`${total} record${total===1?'':'s'}`}</span><small>Most recent first</small></div>{filtered.length===0?<Empty title="No matching records" text="Change the search or state filter."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>State</th><th>Health</th><th>Anomaly</th><th>Remaining</th><th>Model</th><th>Outcome</th><th/></tr></thead><tbody>{filtered.map((r,i)=>{const outcome=historyOutcome(r); return <tr key={r.id||`${r.tick_timestamp}-${i}`} onClick={()=>setSelected(r)}><td><strong>{shortTime(r.tick_timestamp)}</strong></td><td><StatusBadge value={r.maintenance_level||'OK'}/></td><td>{fmt(r.health_state,1)}%</td><td>{fmt(r.anomaly_score,4)}</td><td>{r.remaining_days!==undefined?`${fmt(r.remaining_days,1)} d`:'—'}</td><td>{r.model_version||'—'}</td><td>{outcome?<StatusBadge value={outcome.text} forced={outcome.tone}/>:<span className="muted">—</span>}</td><td><Icon name="chevron"/></td></tr>})}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title="Prediction record" wide onClose={()=>setSelected(null)}><div className="split-detail"><div className="detail-list"><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/><DetailRow label="Maintenance state" value={selected.maintenance_level||'—'} badge={tone(selected.maintenance_level)}/><DetailRow label="Maintenance reason" value={selected.maintenance_reason||'—'}/><DetailRow label="Trigger" value={triggerLabel(selected.maintenance_trigger)}/><DetailRow label="Health" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Remaining days" value={fmt(selected.remaining_days,2)}/>{selected.alert_status&&<><DetailRow label="Alert outcome" value={reviewLabel(selected.alert_status)} badge={reviewTone(selected.alert_status)}/><DetailRow label="Alert level / trigger" value={`${selected.alert_level||'—'} · ${triggerLabel(selected.alert_trigger)}`}/>{selected.alert_reviewed_by&&<DetailRow label="Reviewed by" value={`${selected.alert_reviewed_by} · ${shortTime(selected.alert_reviewed_at)}`}/>}</>}{!selected.alert_status&&selected.near_miss_status&&<><DetailRow label="Near-miss outcome" value={reviewLabel(selected.near_miss_status)} badge={reviewTone(selected.near_miss_status)}/>{selected.near_miss_reviewed_by&&<DetailRow label="Reviewed by" value={`${selected.near_miss_reviewed_by} · ${shortTime(selected.near_miss_reviewed_at)}`}/>}</>}{!selected.alert_status&&!selected.near_miss_status&&<DetailRow label="Outcome" value="Not surfaced in Alert review or Near miss"/>}</div><div><p className="section-label">SENSOR SNAPSHOT</p><JsonGrid value={selected.raw_reading}/><Disclosure title="Failure probability"><JsonGrid value={selected.failure_probability}/></Disclosure><Disclosure title="Full backend record"><pre className="code-panel compact">{JSON.stringify(selected,null,2)}</pre></Disclosure></div></div>
  </Modal>}
  </>
}

function Thresholds({admin}:{admin:boolean}){
  const [v,setV]=useState<any>(),[saved,setSaved]=useState(false),[error,setError]=useState('');
  useEffect(()=>{api('/api/config/thresholds').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  const meta:Record<string,{label:string;desc:string;unit:string}>={MAINTENANCE_PROB_URGENT:{label:'Urgent probability',desc:'Forecast probability boundary for urgent maintenance.',unit:'0–1'},MAINTENANCE_PROB_PLAN:{label:'Plan probability',desc:'Forecast probability boundary for planned inspection.',unit:'0–1'},FAILURE_HEALTH_THRESHOLD:{label:'Failure health threshold',desc:'Health state below which maintenance becomes critical.',unit:'health %'},MAINTENANCE_HEALTH_INSPECT:{label:'Inspection health threshold',desc:'Health state below which inspection is recommended.',unit:'health %'}};
  if(error)return <><PageHeader eyebrow="POLICY" title="Maintenance thresholds" description="Runtime decision thresholds controlled by the backend."/><Notice tone="critical">{error}</Notice></>;
  if(!v)return <Loading/>;
  return <><PageHeader eyebrow="POLICY" title="Maintenance thresholds" description="Tune the small set of runtime maintenance-policy values without changing model artifacts." actions={admin?<button className="primary" onClick={async()=>{setSaved(false);await api('/api/config/thresholds',{method:'PUT',body:JSON.stringify(v)});setSaved(true)}}>Save & hot reload</button>:<StatusBadge value="READ ONLY"/>}/>{saved&&<Notice tone="normal">Thresholds saved. The backend will apply the updated policy without retraining the model.</Notice>}
  <section className="settings-surface">{Object.keys(v).map(k=><div className="setting-row" key={k}><div><strong>{meta[k]?.label||k}</strong><span>{meta[k]?.desc||k}</span><code>{k}</code></div><div className="setting-control"><input type="number" step="any" value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}/><span>{meta[k]?.unit}</span></div></div>)}<Disclosure title="How these thresholds are applied"><div className="explain-grid"><div><b>Backend authority</b><span>Ordering and validation rules are checked again server-side before values are accepted.</span></div><div><b>No model retrain</b><span>These are maintenance-policy settings, not learned model parameters.</span></div><div><b>Safe escalation order</b><span>Critical health thresholds are evaluated before inspection thresholds; urgent probability remains above plan probability.</span></div></div></Disclosure></section>
  </>
}

function Environment(){
  const [v,setV]=useState<any>(),[error,setError]=useState(''),[saved,setSaved]=useState<any>(); useEffect(()=>{api('/api/env').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  if(error)return <><PageHeader eyebrow="SYSTEM" title="Environment" description="Local runtime and database configuration."/><Notice tone="critical">{error}</Notice></>;
  if(!v)return <Loading/>;
  const primary=['PG_HOST','PG_PORT','PG_DATABASE','PG_USER','PG_PASSWORD','PG_TABLE']; const extra=Object.keys(v).filter(k=>!primary.includes(k));
  const field=(k:string)=><label className="env-field" key={k}><span>{k.replaceAll('_',' ')}</span><input type={k.includes('PASSWORD')?'password':'text'} value={v[k]??''} onChange={e=>setV({...v,[k]:e.target.value})}/><small>{k==='PG_PASSWORD'?'Stored password is masked in the browser.':k==='PG_HOST'?'Database server hostname or IP address.':k==='PG_PORT'?'PostgreSQL defaults to 5432.':''}</small></label>;
  return <><PageHeader eyebrow="SYSTEM" title="Environment" description="Configure the local database connection used by the backend and worker." actions={<button className="primary" onClick={async()=>setSaved(await api('/api/env',{method:'PUT',body:JSON.stringify({values:v})}))}>Save environment</button>}/>{saved&&<Notice tone="normal">Saved {saved.changed_keys?.length||0} changed key(s). {saved.worker_restart_requested?'Worker restart requested.':'No worker restart required.'}</Notice>}
  <section className="settings-surface"><div className="environment-grid">{primary.filter(k=>k in v).map(field)}</div>{extra.length>0&&<Disclosure title={`Advanced variables (${extra.length})`}>{<div className="environment-grid advanced">{extra.map(field)}</div>}</Disclosure>}<div className="env-note"><Icon name="shield"/><div><b>Credential handling</b><span>The browser receives a masked password value. Connection changes are validated and persisted by the backend.</span></div></div></section>
  </>
}

function Empty({title,text}:{title:string;text:string}){return <div className="empty"><div className="empty-icon"><Icon name="pulse"/></div><strong>{title}</strong><span>{text}</span></div>}
function Loading(){return <div className="loading-row"><div className="spinner small"/><span>Loading from backend…</span></div>}

createRoot(document.getElementById('root')!).render(<ErrorBoundary><App/></ErrorBoundary>);
