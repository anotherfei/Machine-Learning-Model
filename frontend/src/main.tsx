import React, { Component, ReactNode, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import {
  ChartLineUpIcon, CheckIcon, ClockCounterClockwiseIcon, CubeIcon, DatabaseIcon,
  DotsThreeIcon, FactoryIcon, GaugeIcon, GearIcon, HouseIcon, InfoIcon,
  MagnifyingGlassIcon, ShieldCheckIcon, SignOutIcon, SlidersHorizontalIcon,
  SquaresFourIcon, TrashIcon, WarningIcon, WaveSineIcon, XIcon, CaretDownIcon, CaretRightIcon, CopyIcon,
} from '@phosphor-icons/react';
import './styles.css';

type Page = 'Home' | 'Simulation' | 'Machine Overview' | 'Status Review' | 'Models' | 'History' | 'Environment' | 'Thresholds';
type User = { username: string; role: string; mock_mode?: boolean };
type LiveTick = {
  machine_id: string;
  timestamp?: string;
  prediction_timestamp?: string;
  prediction_available?: boolean;
  source?: string;
  source_table?: string;
  source_age_seconds?: number;
  prediction_lag_seconds?: number;
  prediction_matches_source?: boolean;
  prediction_delayed?: boolean;
  allowed_prediction_lag_seconds?: number;
  worker_poll_seconds?: number;
  prediction_wait_reason?: string;
  operating_state?: string;
  operating_state_reason?: string;
  operating_state_confidence?: number;
  operating_state_source?: string;
  detected_operating_state?: string;
  detected_operating_state_reason?: string;
  operating_override?: { state:string; set_by?:string; set_at?:string; expires_at?:string; note?:string } | null;
  operating_state_changed_at?: string;
  operating_state_activity?: number;
  operating_state_stop_threshold?: number;
  operating_state_run_threshold?: number;
  health_state?: number;
  anomaly_score?: number;
  model_version?: string;
  maintenance?: { level: string; reason?: string; trigger?: string; stabilizing?: boolean; candidate_level?: string; candidate_elapsed_minutes?: number; candidate_required_minutes?: number };
  a_rms_mps2?: number;
  v_rms_mms?: number;
  a_peak_mps2?: number;
  crest_factor?: number;
  temperature_c?: number;
  mock_mode?: boolean;
};
type FleetTrendPoint = { timestamp:string; health_state:number; samples:number };
type FleetLatestResponse = { items:LiveTick[]; unavailable:Array<{machine_id:string;detail?:string}> };
type BackfillStatus = {
  run_id?:string; status:string; progress?:number; machine_id?:string;
  phase?:string;
  machine_number?:number; machine_count?:number; source_rows?:number;
  source_rows_total?:number; source_rows_total_estimated?:boolean;
  predictions_written?:number; skipped_existing?:number; rows_per_second?:number;
  eta_seconds?:number|null; model_version?:string; range_start?:string; range_end?:string;
  started_at?:string; finished_at?:string; updated_at?:string; error?:string; message?:string;
};
const LIVE_RENDER_INTERVAL_MS = 5000;
type FleetTrendResponse = {
  days:number;
  requested_days?:number;
  coverage_seconds?:number;
  metric:string;
  start:string;
  end:string;
  series:Array<{ machine_id:string; points:FleetTrendPoint[] }>;
};

const SENSOR_MODEL = 'ifm VVB001';

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
      cache: 'no-store',
      ...opts,
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      let message = text;
      try { message = JSON.parse(text).detail || text; } catch { /* keep a non-JSON backend response */ }
      throw new Error(message || `Request failed (${response.status})`);
    }
    return await response.json();
  } catch (error) {
    if (controller.signal.aborted) {
      const timeout = new Error(`Request timed out after ${(timeoutMs/1000).toFixed(0)} seconds`);
      timeout.name = 'AbortError';
      throw timeout;
    }
    throw error;
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
  if (x === 'CRITICAL' || x === 'FAILED' || x === 'REJECTED' || x === 'FAIL' || x === 'ERROR') return 'critical';
  if (x === 'SENSOR_FAULT' || x === 'NO_DATA') return 'critical';
  if (x === 'STARTING' || x === 'STOPPED' || x === 'QUEUED') return 'warning';
  if (x === 'WARN' || x === 'WARNING') return 'warning';
  if (x === 'OK' || x === 'NORMAL' || x === 'ACTIVE' || x === 'RUNNING' || x === 'PASSED' || x === 'PASS' || x === 'COMPLETED') return 'normal';
  return 'neutral';
}
function triggerLabel(trigger?: string) {
  return ({ health_threshold:'Condition threshold', health_inspect:'Inspection threshold', trend_probability:'Trend forecast', none:'No trigger' } as Record<string,string>)[trigger || ''] || (trigger || 'Unknown');
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

const ICONS:Record<string,React.ComponentType<any>>={
  home:HouseIcon,grid:SquaresFourIcon,alert:WarningIcon,model:CubeIcon,trend:ChartLineUpIcon,
  history:ClockCounterClockwiseIcon,sliders:SlidersHorizontalIcon,database:DatabaseIcon,
  chevron:CaretRightIcon,info:InfoIcon,close:XIcon,search:MagnifyingGlassIcon,
  logout:SignOutIcon,pulse:WaveSineIcon,shield:ShieldCheckIcon,more:DotsThreeIcon,
  check:CheckIcon,trash:TrashIcon,copy:CopyIcon,machine:FactoryIcon,gauge:GaugeIcon,settings:GearIcon,simulation:ChartLineUpIcon,
};
const Icon=({name}:{name:string})=>{const Glyph=ICONS[name]||GaugeIcon;return <Glyph className="icon" size={19} weight="regular" aria-hidden="true"/>};

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

function compactDuration(value?:number|null){
  if(value===undefined||value===null||!Number.isFinite(Number(value)))return 'Calculating';
  const seconds=Math.max(0,Math.round(Number(value)));const minutes=Math.floor(seconds/60);const hours=Math.floor(minutes/60);
  if(hours)return `${hours}h ${String(minutes%60).padStart(2,'0')}m`;
  if(minutes)return `${minutes}m ${String(seconds%60).padStart(2,'0')}s`;
  return `${seconds}s`;
}
function BackfillProgressPopup({status,onDismiss}:{status?:BackfillStatus;onDismiss:(runId:string)=>void}){
  if(!status||status.status==='idle')return null;
  const state=String(status.status).toLowerCase();const terminal=['completed','failed'].includes(state);
  if(state==='completed'&&status.finished_at&&Date.now()-new Date(status.finished_at).getTime()>60000)return null;
  const progress=Math.max(0,Math.min(1,Number(status.progress)||0));
  const title=state==='completed'?'Caught up; realtime monitoring active':state==='failed'?'Historical catch-up failed':state==='restarting'?'Restarting catch-up with the active model':state==='draining'?'Closing the realtime gap':state==='launching'?'Starting historical catch-up':state==='connecting'?'Connecting catch-up worker':state==='preparing'?'Preparing historical catch-up':'Catching up before realtime monitoring';
  const detail=state==='failed'?(status.error||'Catch-up stopped before realtime handoff. Check the worker output and restart.'):state==='restarting'?(status.message||'The model or inference settings changed; catch-up is restarting safely.'):state==='draining'?(status.message||'Draining readings that arrived during the historical replay.'):state==='launching'?(status.message||'Starting the production worker.'):state==='connecting'?(status.message||'Connecting to PostgreSQL.'):state==='preparing'?(status.message||'Discovering machines and the newest source watermark.'):state==='completed'?`${Number(status.predictions_written??status.source_rows??0).toLocaleString()} predictions added / ${Number(status.skipped_existing||0).toLocaleString()} existing preserved`:(status.message||`${status.machine_id||'Preparing source'}${status.machine_count?` / machine ${status.machine_number||1} of ${status.machine_count}`:''}`);
  const processedRows=Number(status.source_rows||0),totalRows=Number(status.source_rows_total||0);
  const rowProgress=state==='draining'
    ?`${totalRows.toLocaleString()} historical rows complete · ${Math.max(0,processedRows-totalRows).toLocaleString()} newer rows drained`
    :status.phase==='motion_profile'
      ?`Profiling motion regimes${totalRows?` · ${status.source_rows_total_estimated?'~':''}${totalRows.toLocaleString()} rows queued`:''}`
      :`${processedRows.toLocaleString()}${totalRows?` / ${status.source_rows_total_estimated?'~':''}${totalRows.toLocaleString()}`:''} rows · ${Math.round(Number(status.rows_per_second)||0).toLocaleString()}/s`;
  return <aside className={cn('backfill-toast',`backfill-${state}`)} role="status" aria-live="polite">
    <div className="backfill-toast-icon"><Icon name={state==='failed'?'alert':state==='completed'?'check':'database'}/></div>
    <div className="backfill-toast-body"><div className="backfill-toast-title"><strong>{title}</strong><span>{Math.round(progress*100)}%</span></div><p>{detail}</p>
      {!terminal&&<><div className="backfill-progress-track"><i style={{width:`${progress*100}%`}}/></div><div className="backfill-progress-meta"><span>{rowProgress}</span><span>{state==='draining'?'Waiting for source queue to empty':`ETA ${compactDuration(status.eta_seconds)}`}</span></div></>}
      {status.model_version&&<small>Model {status.model_version}</small>}
    </div>
    {terminal&&status.run_id&&<button className="backfill-toast-close" aria-label="Dismiss backfill notification" onClick={()=>onDismiss(status.run_id!)}><Icon name="close"/></button>}
  </aside>;
}

function App(){
  const [me,setMe]=useState<User|false|undefined>(); const [apiError,setApiError]=useState<string>(); const [page,setPage]=useState<Page>('Home'); const [collapsed,setCollapsed]=useState(false);
  const [machines,setMachines]=useState<string[]>([]); const [machinesLoaded,setMachinesLoaded]=useState(false); const [machineId,setMachineId]=useState(localStorage.getItem('machine_id')||''); const [sourceError,setSourceError]=useState('');
  const [fleetRows,setFleetRows]=useState<LiveTick[]>([]); const [fleetLoading,setFleetLoading]=useState(true); const [fleetUnavailable,setFleetUnavailable]=useState<string[]>([]);
  const [backfillStatus,setBackfillStatus]=useState<BackfillStatus>(); const [dismissedBackfill,setDismissedBackfill]=useState(localStorage.getItem('dismissed_backfill_run')||'');
  const refreshMe=()=>api('/api/me',{},5000).then((x:User)=>{setMe(x);setApiError(undefined)}).catch((err:any)=>{setMe(false);if(err?.name==='AbortError')setApiError('API timed out on /api/me');else if(!String(err?.message||'').includes('401'))setApiError(String(err?.message||err));});
  useEffect(()=>{refreshMe()},[]);
  useEffect(()=>{
    let disposed=false;const preferred=localStorage.getItem('machine_id')||'';
    setMachines([]);setMachineId('');setMachinesLoaded(false);setSourceError('');
    if(!me)return()=>{disposed=true};
    api('/api/machines').then((r:{items:string[];default:string})=>{
      if(disposed)return;
      const items=r.items||[];
      setMachines(items);setMachineId(items.includes(preferred)?preferred:items.includes(r.default)?r.default:(items[0]||''));setSourceError('');
    }).catch(e=>{if(!disposed){setMachines([]);setMachineId('');setSourceError(String(e.message||e))}})
      .finally(()=>{if(!disposed)setMachinesLoaded(true)});
    return()=>{disposed=true};
  },[me]);
  useEffect(()=>{if(machineId)localStorage.setItem('machine_id',machineId);else localStorage.removeItem('machine_id')},[machineId]);
  useEffect(()=>{
    let disposed=false;
    if(!me||me.mock_mode){setBackfillStatus(undefined);return()=>{disposed=true}};
    const load=()=>api('/api/backfill/status',{},5000).then((value:BackfillStatus)=>{if(!disposed)setBackfillStatus(value)}).catch(()=>{});
    load();const timer=window.setInterval(load,2000);
    return()=>{disposed=true;window.clearInterval(timer)};
  },[me]);
  useEffect(()=>{
    let disposed=false,inFlight=false;
    if(!me||!machines.length){setFleetRows([]);setFleetUnavailable([]);setFleetLoading(false);return()=>{disposed=true}}
    setFleetLoading(true);setFleetUnavailable([]);
    const load=async()=>{
      if(inFlight)return;inFlight=true;
      try{
        const response:FleetLatestResponse=await api(`/api/fleet/latest?machine_ids=${encodeURIComponent(machines.join(','))}`,{},30000);
        if(disposed)return;
        setFleetRows((response.items||[]).sort((a,b)=>a.machine_id.localeCompare(b.machine_id)));
        setFleetUnavailable((response.unavailable||[]).map(item=>item.machine_id));
      }catch{
        if(!disposed)setFleetUnavailable([...machines]);
      }finally{
        inFlight=false;if(!disposed)setFleetLoading(false);
      }
    };
    load();
    const timer=window.setInterval(load,60000);
    return()=>{disposed=true;window.clearInterval(timer)};
  },[me,machines.join('|')]);
  if(me===undefined)return <main className="center-screen"><section className="state-panel"><div className="spinner"/><p className="eyebrow">INITIALIZING CONSOLE</p><h1>Connecting to the local API</h1><p className="muted">If this takes more than a few seconds, check <code>localhost:8000/docs</code>.</p></section></main>;
  if(!me)return <Login done={refreshMe} apiError={apiError}/>;
  const fleetNav:Array<{page:Page;icon:string;label:string}>=[
    {page:'Home',icon:'home',label:'Home'},{page:'Simulation',icon:'simulation',label:'Accuracy simulation'},{page:'Models',icon:'model',label:'Models & retraining'},{page:'Thresholds',icon:'sliders',label:'Global thresholds'},...(me.role==='admin'?[{page:'Environment' as Page,icon:'database',label:'Environment'}]:[])
  ];
  const machineNav:Array<{page:Page;icon:string;label:string}>=[
    {page:'Machine Overview',icon:'gauge',label:'Overview'},{page:'Status Review',icon:'alert',label:'Status review'},{page:'History',icon:'history',label:'History'},
  ];
  const machineScoped=machineNav.some(item=>item.page===page);
  const fleetError=fleetUnavailable.length?`${fleetUnavailable.length} machine${fleetUnavailable.length===1?' is':'s are'} temporarily unavailable.`:'';
  const goMachine=(id:string)=>{setMachineId(id);setPage('Machine Overview')};
  const pageLabel=page==='Machine Overview'?'Overview':page;
  return <div className={cn('app-shell',collapsed&&'nav-collapsed')}>
    <aside className="sidebar">
      <div className="brand-lockup"><div className="brand-symbol"><Icon name="pulse"/></div><div className="brand-copy"><small>AKEBONO</small><strong>Spindle Monitor</strong><span>Fleet condition console</span></div></div>
      <div className="mobile-machine-picker"><Select ariaLabel="Select machine" value={machineId} onChange={goMachine} options={(machines.length?machines:['']).map(id=>[id,id||'No database machines'] as [string,string])}/></div>
      <button type="button" className="collapse-btn" onClick={()=>setCollapsed(v=>!v)} title={collapsed?'Expand navigation':'Collapse navigation'} aria-label={collapsed?'Expand navigation':'Collapse navigation'} aria-expanded={!collapsed} data-label={collapsed?'Expand':'Collapse'}><Icon name="chevron"/></button>
      <nav className="scope-nav">
        <section className="nav-group"><p>FLEET</p>{fleetNav.map(n=><button key={n.page} className={cn('nav-item',page===n.page&&'active')} onClick={()=>setPage(n.page)} title={n.label}><Icon name={n.icon}/><span>{n.label}</span>{page===n.page&&<i/>}</button>)}</section>
        <section className="nav-group machine-group"><div className="nav-group-title"><p>MACHINES</p><span>{machines.length}</span></div><MachineNavList machines={machines} rows={fleetRows} unavailable={fleetUnavailable} selected={machineId} active={machineScoped} onSelect={goMachine}/>{machineId&&<div className="machine-subnav"><small>{machineId} WORKSPACE</small>{machineNav.map(n=><button key={n.page} className={cn('nav-item',page===n.page&&'active')} onClick={()=>setPage(n.page)} title={`${n.label} · ${machineId}`}><Icon name={n.icon}/><span>{n.label}</span>{page===n.page&&<i/>}</button>)}</div>}</section>
      </nav>
      <div className="sidebar-bottom">
        {me.mock_mode&&<button className="demo-chip" title="This session is using temporary synthetic data"><span className="dot"/><span>Mock data</span></button>}
        <div className="user-chip"><div className="avatar">{me.username.slice(0,2).toUpperCase()}</div><div><strong>{me.username}</strong><span>{me.role}</span></div><button title="Sign out" onClick={async()=>{await api('/api/logout',{method:'POST'});setMe(false)}}><Icon name="logout"/></button></div>
      </div>
    </aside>
    <main className="workspace">
      <header className="topbar"><p className="breadcrumb">{machineScoped?<>MACHINES <span>/</span> {machineId} <span>/</span> {pageLabel}</>:<>FLEET <span>/</span> {pageLabel}</>}</p><div className="topbar-meta"><span className="data-freshness"><Icon name="database"/>{me.mock_mode?'Demo source':'Production source'}</span><span className="system-pill"><span className="pulse-dot"/>API online</span><span className="clock">{new Date().toLocaleDateString([], {weekday:'short',month:'short',day:'numeric'})}</span></div></header>
      <div className="content">{page==='Environment'?<Environment/>:!machinesLoaded?<Loading/>:sourceError?<><PageHeader eyebrow="PRODUCTION SOURCE" title="Machine discovery unavailable" description="The monitoring API could not return machine IDs from the configured raw source." actions={me.role==='admin'?<button className="primary" onClick={()=>setPage('Environment')}>Check environment</button>:undefined}/><Notice tone="critical">{sourceError}</Notice></>:page==='Home'?<FleetHome machines={machines} rows={fleetRows} loading={fleetLoading} error={fleetError} openMachine={goMachine}/>:page==='Simulation'?<SimulationPage machines={machines} admin={me.role==='admin'} mock={!!me.mock_mode}/>:machineId?<PageView key={`${machineId}-${page}`} page={page} role={me.role} mock={!!me.mock_mode} machineId={machineId} live={fleetRows.find(row=>row.machine_id===machineId)} liveError={fleetUnavailable.includes(machineId)?`${machineId} is temporarily unavailable.`:''}/>:<Loading/>}</div>
    </main>
    {backfillStatus?.run_id!==dismissedBackfill&&<BackfillProgressPopup status={backfillStatus} onDismiss={runId=>{localStorage.setItem('dismissed_backfill_run',runId);setDismissedBackfill(runId)}}/>}
  </div>
}

function PageHeader({ eyebrow, title, description, actions }: { eyebrow:string;title:string;description:string;actions?:ReactNode }) {
  return <div className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>{actions&&<div className="page-actions">{actions}</div>}</div>
}

function ScopeBanner({scope,machineId}:{scope:'fleet'|'machine';machineId?:string}){
  return <div className={cn('scope-banner',`scope-${scope}`)}><Icon name={scope==='fleet'?'shield':'machine'}/><div><strong>{scope==='fleet'?'Fleet scope':`${machineId} scope`}</strong><span>{scope==='fleet'?'Changes here apply across all machines.':'Data and settings here apply only to this machine.'}</span></div></div>;
}

function MachineNavList({machines,rows,unavailable,selected,active,onSelect}:{machines:string[];rows:LiveTick[];unavailable:string[];selected:string;active:boolean;onSelect:(id:string)=>void}){
  const states=Object.fromEntries([
    ...unavailable.map(machineId=>[machineId,'NO_DATA']),
    ...rows.map(row=>[row.machine_id,String(row.operating_state||'UNKNOWN')]),
  ]);
  return <div className="machine-list">{machines.map(id=><button key={id} className={cn('machine-link',id===selected&&active&&'active')} onClick={()=>onSelect(id)} title={`${id} · ${states[id]||'Loading state'}`}><span className={cn('machine-state-dot',tone(states[id]))}/><span>{id}</span><Icon name="chevron"/></button>)}</div>;
}

const FLEET_LINE_COLORS=['#1769c2','#d47b12','#23875e','#b43b64','#7257c8','#17889b','#bf4b32','#596b82','#8a6d13','#2672a4'];
const fleetLineColor=(index:number)=>FLEET_LINE_COLORS[index]||`hsl(${Math.round((index*137.508+211)%360)} 64% 42%)`;
function fleetTrendPeriodLabel(data?:FleetTrendResponse){
  if(!data)return 'Condition trend (up to 7 days)';
  const seconds=Math.max(0,Number(data.coverage_seconds??data.days*86400));
  if(seconds<=60*60)return 'Latest-hour condition trend';
  if(seconds<24*60*60)return `${Math.max(1,Math.ceil(seconds/3600))}-hour condition trend`;
  return `${Math.min(7,Math.max(1,Math.ceil(seconds/86400)))}-day condition trend`;
}
function FleetTrendChart({machines,data}:{machines:string[];data:FleetTrendResponse}){
  const width=1000,height=258,left=48,right=18,top=16,bottom=38;
  const byMachine=new Map(data.series.map(series=>[series.machine_id,series.points]));
  const ids=Array.from(new Set([...machines,...data.series.map(series=>series.machine_id)])).sort();
  const observedTimes=data.series.flatMap(series=>series.points.map(point=>new Date(point.timestamp).getTime())).filter(Number.isFinite).sort((a,b)=>a-b);
  const validCount=observedTimes.length;
  if(validCount===0)return <Empty title="No condition history in the last seven days" text="The worker has not stored condition predictions in this period."/>;
  let start=observedTimes[0]??new Date(data.start).getTime(),end=observedTimes[observedTimes.length-1]??new Date(data.end).getTime();
  if(end<=start){start-=30*60*1000;end+=30*60*1000}
  const span=Math.max(1,end-start),plotW=width-left-right,plotH=height-top-bottom;
  const x=(timestamp:string)=>left+Math.max(0,Math.min(1,(new Date(timestamp).getTime()-start)/span))*plotW;
  const y=(health:number)=>top+(1-Math.max(0,Math.min(100,health))/100)*plotH;
  const hour=60*60*1000,day=24*hour;
  const tickCount=span<=6*hour?Math.max(2,Math.min(7,Math.ceil(span/hour)+1)):span<=2*day?Math.max(2,Math.min(9,Math.ceil(span/(6*hour))+1)):Math.max(2,Math.min(8,Math.ceil(span/day)+1));
  const timeTicks=Array.from({length:tickCount},(_,index)=>start+(span*index/(tickCount-1)));
  const includeTime=span<=2*day;
  return <div className="fleet-trend-body">
    <div className="fleet-trend-scroll"><svg className="fleet-trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Condition trend for all machines, capped at seven days">
      {([0,25,50,75,100] as number[]).map(value=><g key={value}><line className="trend-grid-line" x1={left} x2={width-right} y1={y(value)} y2={y(value)}/><text className="trend-y-label" x={left-10} y={y(value)+3} textAnchor="end">{value}</text></g>)}
      {timeTicks.map((tick,index)=>{const tickX=left+(index/(tickCount-1))*plotW;return <g key={tick}><line className="trend-day-line" x1={tickX} x2={tickX} y1={top} y2={height-bottom}/><text className="trend-x-label" x={tickX} y={height-13} textAnchor={index===0?'start':index===tickCount-1?'end':'middle'}>{includeTime?new Date(tick).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):new Date(tick).toLocaleDateString([], {month:'short',day:'numeric'})}</text></g>})}
      <text className="trend-axis-title" transform={`translate(12 ${top+plotH/2}) rotate(-90)`} textAnchor="middle">Condition %</text>
      {ids.map((machineId,index)=>{const points=(byMachine.get(machineId)||[]).filter(point=>Number.isFinite(Number(point.health_state))&&!Number.isNaN(new Date(point.timestamp).getTime()));if(points.length===0)return null;const color=fleetLineColor(index);const path=points.map((point,pointIndex)=>{const previous=points[pointIndex-1],gap=previous?new Date(point.timestamp).getTime()-new Date(previous.timestamp).getTime():0;return `${pointIndex&&gap<=2*60*60*1000?'L':'M'} ${x(point.timestamp).toFixed(2)} ${y(Number(point.health_state)).toFixed(2)}`}).join(' ');const latest=points[points.length-1];return <g key={machineId}><path className="fleet-machine-line" d={path} stroke={color}><title>{machineId}</title></path><circle className="fleet-machine-endpoint" cx={x(latest.timestamp)} cy={y(Number(latest.health_state))} r="3.5" fill={color}><title>{`${machineId}: ${fmt(latest.health_state,1)}% at ${shortTime(latest.timestamp)}`}</title></circle></g>})}
    </svg></div>
    <div className="fleet-trend-legend" aria-label="Machine color key">{ids.map((machineId,index)=>{const points=byMachine.get(machineId)||[],latest=points[points.length-1];return <span key={machineId} className={points.length?'':'unavailable'}><i style={{background:fleetLineColor(index)}}/><b>{machineId}</b><small>{latest?`${fmt(latest.health_state,1)}% latest`:'No data'}</small></span>})}</div>
    <p>Each line is one machine’s hourly average condition score. Machines are never averaged together; gaps indicate periods without stored predictions.</p>
  </div>;
}

function PageView({page,role,mock,machineId,live,liveError}:{page:Page;role:string;mock:boolean;machineId:string;live?:LiveTick;liveError:string}){
  if(page==='Machine Overview')return <Dashboard admin={role==='admin'} mock={mock} machineId={machineId} snapshot={live} snapshotError={liveError}/>;
  if(page==='Status Review')return <StatusReview machineId={machineId} live={live}/>;
  if(page==='Models')return <Models admin={role==='admin'}/>;
  if(page==='History')return <HistoryPage machineId={machineId}/>;
  if(page==='Thresholds')return <Thresholds admin={role==='admin'}/>;
  return <Environment/>;
}

function FleetHome({machines,rows,loading,error,openMachine}:{machines:string[];rows:LiveTick[];loading:boolean;error:string;openMachine:(id:string)=>void}){
  const [trend,setTrend]=useState<FleetTrendResponse>(),[trendLoading,setTrendLoading]=useState(true),[trendError,setTrendError]=useState('');
  useEffect(()=>{
    let disposed=false;
    const load=()=>api('/api/fleet/condition-trend?days=7',{},15000).then((response:FleetTrendResponse)=>{if(!disposed){setTrend(response);setTrendError('')}}).catch(e=>{if(!disposed)setTrendError(String(e.message||e))}).finally(()=>{if(!disposed)setTrendLoading(false)});
    load();const timer=window.setInterval(load,300000);return()=>{disposed=true;window.clearInterval(timer)};
  },[]);
  const running=rows.filter(r=>String(r.operating_state).toUpperCase()==='RUNNING').length;
  const monitored=rows.filter(r=>r.prediction_available===true&&!!r.maintenance?.level).length;
  const monitoredUnknown=rows.filter(r=>r.prediction_available===true&&String(r.operating_state).toUpperCase()==='UNKNOWN').length;
  const stopped=rows.filter(r=>String(r.operating_state).toUpperCase()==='STOPPED').length;
  const critical=rows.filter(r=>String(r.maintenance?.level).toUpperCase()==='CRITICAL').length;
  const attention=rows.filter(r=>['WARN','CRITICAL'].includes(String(r.maintenance?.level).toUpperCase())||['NO_DATA','SENSOR_FAULT'].includes(String(r.operating_state).toUpperCase()));
  return <>
    <PageHeader eyebrow="FLEET HOME" title="Operations dashboard" description="Shared fleet visibility across every machine in the production source. Open a machine to work in its individual context."/>
    {error&&<Notice tone="warning">{error}</Notice>}
    <section className="fleet-summary" aria-label="Fleet summary">
      <div><strong>{machines.length}</strong><span>Total machines</span></div><div className="normal"><strong>{monitored}</strong><span>Actively monitored</span><small>{running} confirmed running{monitoredUnknown?` · ${monitoredUnknown} motion unknown`:''}</small></div><div className="warning"><strong>{stopped}</strong><span>Confirmed stopped</span></div><div className={critical?'critical':'normal'}><strong>{critical}</strong><span>Critical</span></div>
    </section>
    <section className="fleet-table data-surface">
      <div className="surface-heading"><span>Machine status</span><small>Shared overview · select a row to enter its workspace</small></div>
      {loading?<Loading/>:rows.length===0?<Empty title="No live machine snapshots" text="The production source did not return a current machine state."/>:<div className="responsive-table"><table><thead><tr><th>Machine</th><th>Operating state</th><th>Maintenance</th><th>Condition</th><th>Latest data</th><th>Active model</th><th/></tr></thead><tbody>{rows.map(row=>{const state=String(row.operating_state||'UNKNOWN').toUpperCase();return <tr key={row.machine_id} onClick={()=>openMachine(row.machine_id)}><td><strong>{row.machine_id}</strong><small>Machine workspace</small></td><td><StatusBadge value={state}/>{row.operating_state_source==='operator'?<small>Operator confirmed · expires {shortTime(row.operating_override?.expires_at)}</small>:state==='UNKNOWN'&&row.prediction_available===true?<small>ML active · motion unconfirmed</small>:null}</td><td><StatusBadge value={row.maintenance?.level||(row.prediction_wait_reason?'PROCESSING':'AWAITING MODEL')}/></td><td><strong>{row.prediction_available!==false&&row.health_state!==undefined?`${fmt(row.health_state,1)}%`:'—'}</strong></td><td>{shortTime(row.timestamp)}</td><td>{row.model_version||'—'}</td><td><button className="row-action" onClick={e=>{e.stopPropagation();openMachine(row.machine_id)}}>Open <Icon name="chevron"/></button></td></tr>})}</tbody></table></div>}
    </section>
    <div className="fleet-support-grid">
      <section className="fleet-trend data-surface"><div className="surface-heading"><span>{fleetTrendPeriodLabel(trend)}</span><small>Up to 7 days · one line per machine</small></div>{trendError&&!trend&&<Notice tone="warning">Condition history unavailable: {trendError}</Notice>}{trendLoading&&!trend?<Loading/>:trend?<FleetTrendChart machines={machines} data={trend}/>:null}</section>
      <section className="attention-list data-surface"><div className="surface-heading"><span>Needs attention</span><small>{attention.length} machine{attention.length===1?'':'s'}</small></div>{attention.length===0?<div className="attention-empty"><Icon name={rows.length?'check':'database'}/><div><strong>{rows.length?'No fleet exceptions':'Fleet status unavailable'}</strong><span>{rows.length?'All connected machines are within the current policy.':'Waiting for current machine snapshots from the backend.'}</span></div></div>:attention.map(row=><button key={row.machine_id} onClick={()=>openMachine(row.machine_id)}><StatusBadge value={row.maintenance?.level||row.operating_state||'CHECK'}/><div><strong>{row.machine_id}</strong><span>{row.operating_state_reason||row.maintenance?.reason}</span></div><Icon name="chevron"/></button>)}</section>
    </div>
  </>;
}

function Sparkline({values, inverse=false}:{values:number[];inverse?:boolean}){
  const clean=values.filter(Number.isFinite); if(clean.length<2)return <svg className="sparkline" viewBox="0 0 200 54"/>;
  const min=Math.min(...clean),max=Math.max(...clean),span=max-min||1;
  const pts=clean.map((v,i)=>`${(i/(clean.length-1))*200},${48-((v-min)/span)*40}`).join(' ');
  return <svg className={cn('sparkline',inverse&&'inverse')} viewBox="0 0 200 54" preserveAspectRatio="none"><polyline points={pts}/></svg>
}

function Dashboard({admin,mock,machineId,snapshot,snapshotError}:{admin:boolean;mock:boolean;machineId:string;snapshot?:LiveTick;snapshotError:string}){
  const [live,setLive]=useState<LiveTick>(); const [history,setHistory]=useState<LiveTick[]>([]); const [wsState,setWsState]=useState<'connecting'|'live'|'offline'>('connecting'); const [detail,setDetail]=useState<string|null>(null); const [menu,setMenu]=useState(false); const [confirmingMotion,setConfirmingMotion]=useState(false);
  useEffect(()=>{
    if(!snapshot)return;
    setLive(snapshot);
    setHistory(items=>{const withoutSame=items.filter(item=>item.timestamp!==snapshot.timestamp);return[...withoutSame.slice(-39),snapshot]});
  },[snapshot]);
  useEffect(()=>{
    let disposed=false;
    let pendingRow:LiveTick|undefined;
    let renderedOnce=false;
    let lastRenderedTime=-Infinity;
    setLive(snapshot);setHistory(snapshot?[snapshot]:[]);setWsState('connecting');
    const record=(row:LiveTick)=>{if(disposed)return;setLive(row);setHistory(items=>{const withoutSame=items.filter(item=>item.timestamp!==row.timestamp);return[...withoutSame.slice(-39),row]})};
    const rowTime=(row:LiveTick)=>{const value=new Date(row.timestamp||'').getTime();return Number.isFinite(value)?value:-Infinity};
    const flush=()=>{if(!pendingRow)return;const row=pendingRow;pendingRow=undefined;const timestamp=rowTime(row);if(timestamp<lastRenderedTime)return;lastRenderedTime=timestamp;renderedOnce=true;record(row)};
    const queue=(row:LiveTick)=>{if(disposed||rowTime(row)<lastRenderedTime)return;if(!pendingRow||rowTime(row)>=rowTime(pendingRow))pendingRow=row;if(!renderedOnce)flush()};
    const renderTimer=window.setInterval(flush,LIVE_RENDER_INTERVAL_MS);
    const proto=location.protocol==='https:'?'wss':'ws';const ws=new WebSocket(`${proto}://${location.host}/ws/live?machine_id=${encodeURIComponent(machineId)}`);
    ws.onopen=()=>setWsState('live');
    ws.onmessage=e=>{const parsed=JSON.parse(e.data);const row={...parsed,prediction_available:parsed.prediction_available!==false,source:mock?'mock':'postgresql'};queue(row);setWsState('live')};
    ws.onerror=()=>setWsState('offline');ws.onclose=()=>setWsState('offline');
    return()=>{disposed=true;window.clearInterval(renderTimer);ws.close()};
  },[machineId,mock]);
  const operatingState=String(live?.operating_state||'UNKNOWN').toUpperCase();
  const sourceState=live?'live':wsState==='offline'?'offline':'connecting';
  const operatingGate=['STOPPED','STARTING','SENSOR_FAULT','NO_DATA'].includes(operatingState);
  const predictionAvailable=!operatingGate&&!!live?.maintenance&&live?.prediction_available!==false;
  const level=operatingGate?operatingState:predictionAvailable?(live?.maintenance?.level||'WAITING'):(live?.prediction_wait_reason?'PROCESSING':live&&operatingState!=='UNKNOWN'?operatingState:live?'AWAITING MODEL':'WAITING'); const health=Number(live?.health_state||0); const anomaly=Number(live?.anomaly_score||0);
  const statusReason=predictionAvailable?live?.maintenance?.reason:(live?.prediction_wait_reason||live?.operating_state_reason||'Real sensor data is connected. Waiting for the ML worker to produce a prediction.');
  const sensors=Object.keys(SENSOR_META).map(k=>({key:k,value:(live as any)?.[k],...SENSOR_META[k]}));
  const hasConfirmation=!!live?.operating_override;
  const operatorConfirmed=live?.operating_state_source==='operator'&&hasConfirmation;
  const confirmationBlocked=['NO_DATA','SENSOR_FAULT'].includes(operatingState);
  const refreshLive=async()=>{const row:LiveTick=await api(`/api/live/latest?machine_id=${encodeURIComponent(machineId)}`,{},15000);setLive(row);setHistory(items=>[...items.filter(item=>item.timestamp!==row.timestamp).slice(-39),row])};
  return <>
    <PageHeader eyebrow="MACHINE OVERVIEW" title={machineId} description="Latest sensor readings from the configured PostgreSQL source, combined with ML results when applicable." actions={<><span className={cn('connection-badge',sourceState)}><span/>{sourceState==='live'?(operatingState==='NO_DATA'?'No recent data':mock?'Mock live':wsState==='live'?'PostgreSQL + worker live':'PostgreSQL live'):sourceState==='connecting'?'Connecting…':'Source unavailable'}</span><div className="dropdown-wrap"><button className="icon-button" onClick={()=>setMenu(v=>!v)}><Icon name="more"/></button>{menu&&<div className="dropdown-menu right"><button onClick={()=>{setDetail('system');setMenu(false)}}>System details</button><button onClick={()=>{setDetail('model');setMenu(false)}}>Model context</button><button onClick={()=>location.reload()}>Reconnect interface</button></div>}</div></>}/>
    <ScopeBanner scope="machine" machineId={machineId}/>
    {snapshotError&&<Notice tone="critical">Production data sync unavailable: {snapshotError} The monitoring API may be restarting or unable to query PostgreSQL.</Notice>}
    {mock&&<div className="demo-ribbon"><span>DEMO</span><p>Temporary synthetic signal is driving this interface. ML inference and plant PostgreSQL are bypassed.</p><button onClick={()=>setDetail('demo')}>What is simulated?</button></div>}
    <section className={cn('motion-confirmation',operatorConfirmed&&'operator-confirmed')}>
      <div><span className="motion-confirmation-label">Operating state</span><StatusBadge value={operatingState}/><p>{operatorConfirmed?`Confirmed by ${live?.operating_override?.set_by||'an operator'} until ${shortTime(live?.operating_override?.expires_at)}.`:hasConfirmation?`The saved confirmation is temporarily superseded by ${operatingState}; it can still be cleared.`:operatingState==='UNKNOWN'?'The vibration detector cannot yet prove whether the spindle is rotating.':`Automatically inferred from this machine's vibration history.`}</p></div>
      {admin&&!mock&&<button className="secondary" disabled={confirmationBlocked&&!hasConfirmation} title={confirmationBlocked&&!hasConfirmation?'Fresh, valid sensor data is required before a motion confirmation can control monitoring.':''} onClick={()=>setConfirmingMotion(true)}>{hasConfirmation?'Manage confirmation':'Confirm manually'}</button>}
    </section>
    <section className={cn('status-board',`tone-${tone(level)}`)}>
      <div className="status-main">
        <div className="status-kicker"><span className="machine-dot"/>{machineId} · SPINDLE CONDITION</div>
        <div className="status-line"><div><span className="status-label">Current state</span><h2>{level}</h2></div><button className="text-button" onClick={()=>setDetail('status')}><Icon name="info"/>Why this status?</button></div>
        <p className="status-reason">{statusReason}</p>
        <div className="health-row"><div><span>Condition score</span><strong>{predictionAvailable?fmt(health,1):'—'}<small>{predictionAvailable?'%':''}</small></strong></div><div className="health-track"><i style={{width:`${predictionAvailable?Math.max(0,Math.min(100,health)):0}%`}}/></div><span className="health-caption">0 severe deviation <b>·</b> 100 baseline-like</span></div>
      </div>
      <div className="status-side">
        <div className="metric-stack"><span>Anomaly score<button className="mini-info" onClick={()=>setDetail('anomaly')}><Icon name="info"/></button></span><strong>{predictionAvailable?fmt(anomaly,4):'—'}</strong><Sparkline values={history.map(x=>Number(x.anomaly_score))}/></div>
        <div className="vertical-divider"/>
        <div className="metric-stack"><span>Model</span><strong className="model-name">{predictionAvailable?live?.model_version:operatingGate?'Scoring paused':'Awaiting worker'}</strong><small>{shortTime(live?.prediction_timestamp||live?.timestamp)}</small></div>
      </div>
    </section>
    <section className="sensor-panel">
      <div className="panel-heading"><div><p className="eyebrow">LIVE SENSOR CHANNELS</p><h3>Signal snapshot</h3></div><button className="ghost-button" onClick={()=>setDetail('sensors')}>Channel guide <Icon name="chevron"/></button></div>
      <div className="sensor-grid">{sensors.map(s=><button key={s.key} className="sensor-item" onClick={()=>setDetail(`sensor:${s.key}`)}><div><span>{s.label}</span><small>{s.unit||'ratio'}</small></div><strong>{s.value===undefined?'—':fmt(s.value,s.key==='temperature_c'?1:3)}</strong><Sparkline values={history.map(x=>Number((x as any)[s.key])).filter(Number.isFinite)}/><div className="sensor-foot"><span>{s.detail}</span><Icon name="chevron"/></div></button>)}</div>
    </section>
    {detail&&<DashboardModal kind={detail} live={live} onClose={()=>setDetail(null)}/>} 
    {confirmingMotion&&<OperatingStateOverrideModal machineId={machineId} live={live} onClose={()=>setConfirmingMotion(false)} onUpdated={async()=>{await refreshLive();setConfirmingMotion(false)}}/>}
  </>
}

function OperatingStateOverrideModal({machineId,live,onClose,onUpdated}:{machineId:string;live?:LiveTick;onClose:()=>void;onUpdated:()=>Promise<void>}){
  const current=live?.operating_override; const [state,setState]=useState(current?.state||'RUNNING'); const [duration,setDuration]=useState('480'); const [note,setNote]=useState(current?.note||''); const [busy,setBusy]=useState(false); const [error,setError]=useState('');
  const save=async()=>{setBusy(true);setError('');try{await api(`/api/machines/${encodeURIComponent(machineId)}/operating-override`,{method:'POST',body:JSON.stringify({state,expires_minutes:Number(duration),note})},15000);await onUpdated()}catch(e:any){setError(String(e.message||e))}finally{setBusy(false)}};
  const clear=async()=>{setBusy(true);setError('');try{await api(`/api/machines/${encodeURIComponent(machineId)}/operating-override`,{method:'DELETE'},15000);await onUpdated()}catch(e:any){setError(String(e.message||e))}finally{setBusy(false)}};
  return <Modal title="Confirm operating state" onClose={onClose}><p>Use this when an operator can physically verify the spindle state but no PLC run signal is available. It controls live inference only; it does not rewrite stored history or simulations.</p>
    <div className="confirmation-choice" role="group" aria-label="Confirmed operating state"><button className={cn(state==='RUNNING'&&'selected')} onClick={()=>setState('RUNNING')}><StatusBadge value="RUNNING"/><span>Spindle is rotating</span></button><button className={cn(state==='STOPPED'&&'selected')} onClick={()=>setState('STOPPED')}><StatusBadge value="STOPPED"/><span>Spindle is stationary</span></button></div>
    <div className="field"><span>Confirmation duration</span><Select ariaLabel="Confirmation duration" value={duration} onChange={setDuration} options={[["15","15 minutes"],["60","1 hour"],["240","4 hours"],["480","8 hours (shift)"],["1440","24 hours"]]}/><small>The confirmation expires automatically so an old operator decision cannot remain active indefinitely.</small></div>
    <label className="field"><span>Operator note (optional)</span><input maxLength={240} value={note} onChange={event=>setNote(event.target.value)} placeholder="For example: visually checked at the machine"/></label>
    <Notice tone="warning">A manual RUNNING confirmation permits ML scoring even when vibration regimes are inconclusive. Invalid or stale sensor data still takes priority and cannot be overridden.</Notice>
    {error&&<Notice tone="critical">{error}</Notice>}
    <div className="modal-actions">{current&&<button className="secondary danger-text" disabled={busy} onClick={clear}>Clear confirmation</button>}<button className="secondary" disabled={busy} onClick={onClose}>Cancel</button><button className="primary" disabled={busy} onClick={save}>{busy?'Saving…':'Confirm state'}</button></div>
  </Modal>
}

function DashboardModal({kind,live,onClose}:{kind:string;live?:LiveTick;onClose:()=>void}){
  let title='Details',body:ReactNode=null;
  if(kind==='status'){title='Machine state evidence';body=<div className="detail-list"><DetailRow label="Effective operating state" value={live?.operating_state||'UNKNOWN'} badge={tone(live?.operating_state)}/><DetailRow label="Decision source" value={live?.operating_state_source==='operator'?'Operator confirmation':'Automatic vibration detector'}/><DetailRow label="Automatic estimate" value={live?.detected_operating_state||live?.operating_state||'UNKNOWN'} badge={tone(live?.detected_operating_state||live?.operating_state)}/><DetailRow label="Automatic evidence" value={live?.detected_operating_state_reason||live?.operating_state_reason||'No state reason received.'}/>{live?.operating_override&&<><DetailRow label="Confirmed by" value={live.operating_override.set_by||'—'}/><DetailRow label="Confirmation expires" value={shortTime(live.operating_override.expires_at)}/><DetailRow label="Operator note" value={live.operating_override.note||'No note supplied.'}/></>}<DetailRow label="State confidence" value={fmt(live?.operating_state_confidence,3)}/><DetailRow label="Activity / stop / run" value={`${fmt(live?.operating_state_activity,3)} / ${fmt(live?.operating_state_stop_threshold,3)} / ${fmt(live?.operating_state_run_threshold,3)}`}/><DetailRow label="State changed" value={shortTime(live?.operating_state_changed_at)}/><DetailRow label="Maintenance state" value={live?.maintenance?.level||'—'} badge={tone(live?.maintenance?.level)}/><DetailRow label="Maintenance trigger" value={triggerLabel(live?.maintenance?.trigger)}/><DetailRow label="Condition score" value={live?.prediction_available===false?'—':`${fmt(live?.health_state,2)}%`}/><DetailRow label="Anomaly score" value={live?.prediction_available===false?'—':fmt(live?.anomaly_score,4)}/></div>}
  else if(kind==='anomaly'){title='How to read anomaly score';body=<><p>The anomaly score summarizes how unusual the current feature pattern is relative to the active model reference. The frontend only displays this value; it does not calculate or modify it.</p><div className="scale"><span>Lower deviation</span><i/><span>Higher deviation</span></div><p className="muted">Current score: <b>{fmt(live?.anomaly_score,4)}</b></p></>}
  else if(kind==='model'||kind==='system'){title=kind==='model'?'Active model context':'Runtime context';body=<div className="detail-list"><DetailRow label="Machine ID" value={live?.machine_id||'—'}/><DetailRow label="Sensor model" value={SENSOR_MODEL}/><DetailRow label="Operating state" value={live?.operating_state||'UNKNOWN'} badge={tone(live?.operating_state)}/><DetailRow label="Model version" value={live?.model_version||'—'}/><DetailRow label="Latest source tick" value={shortTime(live?.timestamp)}/><DetailRow label="Latest prediction tick" value={shortTime(live?.prediction_timestamp)}/><DetailRow label="Prediction lag" value={live?.prediction_lag_seconds===undefined?'—':`${fmt(live.prediction_lag_seconds,1)} seconds`}/><DetailRow label="Worker poll interval" value={live?.worker_poll_seconds===undefined?'—':`${fmt(live.worker_poll_seconds,0)} seconds`}/><DetailRow label="Dashboard refresh" value={`${LIVE_RENDER_INTERVAL_MS/1000} seconds (latest result)`}/><DetailRow label="Source age" value={live?.source_age_seconds===undefined?'—':`${fmt(live.source_age_seconds,0)} seconds`}/><DetailRow label="Data path" value="Backend → WebSocket → browser"/><DetailRow label="Frontend role" value="Visualization only"/></div>}
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

type EditableSettingMeta={label:string;desc:string;unit:string;purpose:string;higher:string;lower:string;step?:number;min?:number;max?:number;choices?:number[];higherLabel?:string;lowerLabel?:string};
function SettingTitle({item,onInfo}:{item:EditableSettingMeta;onInfo:()=>void}){return <div className="setting-title"><strong>{item.label}</strong><button type="button" className="setting-info-button" aria-label={`About ${item.label}`} title={`About ${item.label}`} onClick={onInfo}><Icon name="info"/></button></div>}
function SettingInfoDialog({item,value,onClose}:{item:EditableSettingMeta;value:any;onClose:()=>void}){return createPortal(<Modal title={item.label} onClose={onClose}><p className="setting-purpose">{item.purpose}</p><div className="current-setting-value"><span>Current value</span><div><strong>{String(value)}</strong><small>{item.unit}</small></div></div><div className="setting-impact-grid"><section><span>{item.higherLabel||'If increased'}</span><p>{item.higher}</p></section><section><span>{item.lowerLabel||'If decreased'}</span><p>{item.lower}</p></section></div></Modal>,document.body)}

function StatusReview({machineId,live}:{machineId:string;live?:LiveTick}){
  const [tab,setTab]=useState<'alert'|'near'>('alert');
  return <>
    <PageHeader eyebrow="MACHINE REVIEW" title="Status review" description={`Review model decisions for ${machineId}. Stopped and starting readings are excluded before they can enter either queue.`} actions={<div className="tab-switch"><button className={cn('tab-btn',tab==='alert'&&'active')} onClick={()=>setTab('alert')}>Alert review</button><button className={cn('tab-btn',tab==='near'&&'active')} onClick={()=>setTab('near')}>Near miss</button></div>}/>
    <div className="review-context"><div><span>Machine</span><strong>{machineId}</strong></div><div><span>Operating state</span><StatusBadge value={live?.operating_state||'UNKNOWN'}/></div><div><span>Maintenance state</span><StatusBadge value={live?.maintenance?.level||(live?.prediction_wait_reason?'PROCESSING':'AWAITING MODEL')}/></div><p>{live?.prediction_wait_reason||live?.operating_state_reason||'Waiting for the latest machine state.'}</p></div>
    {tab==='alert'?<AlertPanel machineId={machineId}/>:<NearMissPanel machineId={machineId}/>}
  </>
}

function AlertPanel({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[status,setStatus]=useState('pending'),[trigger,setTrigger]=useState(''),[contextHours,setContextHours]=useState('3'),[selected,setSelected]=useState<any|null>(null),[context,setContext]=useState<any[]>([]),[busy,setBusy]=useState(false);
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/alerts?machine_id=${encodeURIComponent(machineId)}&status=${encodeURIComponent(status)}${trigger?`&trigger=${encodeURIComponent(trigger)}`:''}&limit=${pageSize}&offset=${offset}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[status,trigger,pageSize,offset]);
  useEffect(()=>{setPage(1)},[status,trigger]);
  const inspect=async(a:any)=>{setSelected(a);setContext([]);try{setContext(await api(`/api/alerts/${a.id}/context?hours=${encodeURIComponent(contextHours)}`))}catch{setContext([])}};
  const decide=async(decision:string)=>{if(!selected)return;setBusy(true);try{await api(`/api/alerts/${selected.id}/review`,{method:'POST',body:JSON.stringify({decision})});setSelected(null);load()}finally{setBusy(false)}};
  return <>
  <div className="filter-bar sub-filter-bar"><Select value={status} onChange={setStatus} options={[['pending','Pending'],['confirmed_anomaly','Confirmed anomaly'],['confirmed_normal','Confirmed normal']]}/><Select value={trigger} onChange={setTrigger} options={[['','All triggers'],['health_threshold','Condition threshold'],['health_inspect','Inspection threshold'],['trend_probability','Trend forecast']]}/><Select ariaLabel="Alert context window" value={contextHours} onChange={setContextHours} options={[['1','Context ±1 h'],['3','Context ±3 h'],['6','Context ±6 h'],['12','Context ±12 h'],['24','Context ±24 h']]}/></div>
  {error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{total} result{total===1?'':'s'}</span><small>Click any row to inspect ±{contextHours} hour context</small></div>{rows.length===0?<Empty title="No alerts in this view" text="Try another review status or trigger filter."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>Level</th><th>Trigger</th><th>Condition</th><th>Anomaly</th><th>Review state</th><th/></tr></thead><tbody>{rows.map(a=><tr key={a.id} onClick={()=>inspect(a)}><td><strong>{shortTime(a.tick_timestamp)}</strong><small>#{a.id}</small></td><td><StatusBadge value={a.level}/></td><td>{triggerLabel(a.trigger)}</td><td>{fmt(a.health_state,1)}%</td><td>{fmt(a.anomaly_score,4)}</td><td><span className="review-state">{String(a.status).replaceAll('_',' ')}</span></td><td><Icon name="chevron"/></td></tr>)}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title={`Alert #${selected.id}`} wide onClose={()=>setSelected(null)}><div className="split-detail"><div><div className="detail-list"><DetailRow label="Level" value={selected.level} badge={tone(selected.level)}/><DetailRow label="Trigger" value={triggerLabel(selected.trigger)}/><DetailRow label="Condition score" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/></div><Disclosure title="Raw sensor snapshot"><JsonGrid value={selected.raw_reading}/></Disclosure></div><div><p className="section-label">SURROUNDING CONTEXT</p><MiniContextChart rows={context}/><p className="muted small">Context is fetched from the backend around the alert timestamp. This view does not recalculate the maintenance decision.</p></div></div>{selected.status==='pending'&&<div className="modal-actions"><button className="secondary danger-text" disabled={busy} onClick={()=>decide('confirmed_normal')}>Confirm normal</button><button className="primary" disabled={busy} onClick={()=>decide('confirmed_anomaly')}>Confirm anomaly</button></div>}</Modal>}
  </>
}

function NearMissPanel({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[reviewStatus,setReviewStatus]=useState('pending'),[trendHours,setTrendHours]=useState(''),[selected,setSelected]=useState<any|null>(null),[busy,setBusy]=useState(false),[actionError,setActionError]=useState('');
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/near-miss?machine_id=${encodeURIComponent(machineId)}&status=${encodeURIComponent(reviewStatus)}${trendHours?`&hours=${encodeURIComponent(trendHours)}`:''}&limit=${pageSize}&offset=${offset}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[reviewStatus,trendHours,pageSize,offset]);
  useEffect(()=>{setPage(1)},[reviewStatus,trendHours]);
  const decide=async(decision:string)=>{
    if(!selected)return; setBusy(true); setActionError('');
    try{await api(`/api/near-miss/${selected.id}/review`,{method:'POST',body:JSON.stringify({decision})});setSelected(null);load()}
    catch(e:any){setActionError(String(e.message||e))}
    finally{setBusy(false)}
  };
  return <>
  <div className="filter-bar sub-filter-bar"><Select value={reviewStatus} onChange={setReviewStatus} options={[['pending','Pending'],['acknowledged','Acknowledged'],['flagged','Flagged']]}/><Select ariaLabel="Near-miss trend window" value={trendHours} onChange={setTrendHours} options={[['','Policy trend window'],['1','1-hour trend'],['3','3-hour trend'],['6','6-hour trend'],['12','12-hour trend'],['24','24-hour trend']]}/></div>
  {error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{total} record{total===1?'':'s'}</span><small>Ranked by backend near-miss query</small></div>{rows.length===0?<Empty title="No matching records" text="Try another review status."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>State</th><th>Condition</th><th>Anomaly</th><th>Model</th><th>Review state</th><th/></tr></thead><tbody>{rows.map((r,i)=><tr key={r.id||`${r.tick_timestamp}-${i}`} onClick={()=>setSelected(r)}><td><strong>{shortTime(r.tick_timestamp)}</strong></td><td><StatusBadge value={r.maintenance_level||'OK'}/></td><td>{fmt(r.health_state,1)}%</td><td>{fmt(r.anomaly_score,4)}</td><td>{r.model_version||'—'}</td><td><span className="review-state">{String(r.review_status||'pending').replaceAll('_',' ')}</span></td><td><Icon name="chevron"/></td></tr>)}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title="Near-miss record" wide onClose={()=>setSelected(null)}><div className="split-detail"><div className="detail-list"><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/><DetailRow label="Maintenance state" value={selected.maintenance_level||'—'} badge={tone(selected.maintenance_level)}/><DetailRow label="Condition score" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Review state" value={String(selected.review_status||'pending').replaceAll('_',' ')}/></div><div><p className="section-label">SENSOR SNAPSHOT</p><JsonGrid value={selected.raw_reading}/><Disclosure title="Full backend record"><pre className="code-panel compact">{JSON.stringify(selected,null,2)}</pre></Disclosure></div></div>
  {(selected.review_status||'pending')==='pending'&&<div className="modal-actions">{actionError&&<Notice tone="critical">{actionError}</Notice>}<button className="secondary" disabled={busy} onClick={()=>decide('acknowledged')}>Acknowledge</button><button className="primary" disabled={busy} onClick={()=>decide('flagged')}>Flag for follow-up</button></div>}
  {(selected.review_status||'pending')==='flagged'&&<p className="muted small flag-note">Flagging records this suspected false negative for audit. Add its surrounding period to Accuracy simulation when you want to compare it with the model.</p>}
  </Modal>}
  </>
}

function Select({value,onChange,options,ariaLabel}:{value:string;onChange:(v:string)=>void;options:Array<[string,string]>;ariaLabel?:string}){return <label className="select-wrap"><select aria-label={ariaLabel} value={value} onChange={e=>onChange(e.target.value)}>{options.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select><CaretDownIcon className="select-chevron" size={14} weight="bold" aria-hidden="true"/></label>}
function Switch({checked,onChange,disabled}:{checked:boolean;onChange:()=>void;disabled?:boolean}){return <button type="button" role="switch" aria-checked={checked} disabled={disabled} className={cn('switch',checked&&'on')} onClick={onChange}><span className="switch-thumb"/></button>}

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

function MiniContextChart({rows}:{rows:any[]}){const values=rows.map(r=>Number(r.health_state)).filter(Number.isFinite);return <div className="context-chart"><Sparkline values={values}/><div className="context-axis"><span>{rows[0]?shortTime(rows[0].tick_timestamp):'No data'}</span><span>Condition trend</span><span>{rows.length?shortTime(rows[rows.length-1].tick_timestamp):''}</span></div></div>}
function JsonGrid({value}:{value:any}){const obj=typeof value==='object'&&value?value:{};return <div className="json-grid">{Object.entries(obj).map(([k,v])=><div key={k}><span>{SENSOR_META[k]?.label||k}</span><strong>{String(v)}</strong></div>)}</div>}
function Disclosure({title,children,defaultOpen=false}:{title:string;children:ReactNode;defaultOpen?:boolean}){const[o,setO]=useState(defaultOpen);return <div className={cn('disclosure',o&&'open')}><button onClick={()=>setO(v=>!v)}><span>{title}</span><Icon name="chevron"/></button>{o&&<div className="disclosure-body">{children}</div>}</div>}

function parsedObject(value:any){if(value&&typeof value==='object')return value;if(typeof value==='string')try{return JSON.parse(value)}catch{return {}}return {}}
function validationLabel(model:any){const report=parsedObject(model.validation_report);if(report.passed===true)return 'Passed';if(report.passed===false)return 'Review';return model.status==='active'?'Legacy / active':'Not recorded'}

function ValidationReport({report:raw}:{report:any}){
  const report=parsedObject(raw),machines=report.reference_fp_by_machine||report.per_machine||{};
  if(!Object.keys(report).length)return <Notice>No validation evidence is stored for this model version.</Notice>;
  const verdict=report.passed===true?'SCORE OK':report.passed===false?'REVIEW':'NOT RECORDED';
  return <div className="validation-report"><div className="validation-verdict"><StatusBadge value={verdict} forced={verdict==='SCORE OK'?'normal':verdict==='REVIEW'?'warning':'neutral'}/><span>{verdict==='SCORE OK'?'Validation scores are within their configured targets.':verdict==='REVIEW'?'At least one score is outside its target; activation and promotion remain available to the administrator.':'This older report does not contain a final score verdict.'}</span></div>
  {!!Object.keys(machines).length&&<Disclosure title="Per-machine held-out false-positive gates" defaultOpen><div className="validation-list">{Object.entries(machines).map(([machineId,value]:[string,any])=>{const commissioning=value?.holdout_false_positive_rate!==undefined;return <div key={machineId}><strong>{machineId}</strong><span>{typeof value==='object'?(commissioning?`${value.holdout_rows??'—'} held out · alert rate ${fmt(value.holdout_false_positive_rate,3)} · limit ${fmt(value.maximum_allowed_false_positive_rate,3)}`:`${value.holdout_rows??'—'} held out · active ${fmt(value.active_reference_fp??value.old_rate,3)} → shadow ${fmt(value.shadow_reference_fp??value.new_rate,3)}`):String(value)}</span>{typeof value==='object'&&(value.pass!==undefined||value.passed!==undefined)&&<StatusBadge value={(value.pass??value.passed)?'PASS':'FAIL'} forced={(value.pass??value.passed)?'normal':'critical'}/>}</div>})}</div></Disclosure>}
  {report.labelled_simulation_evaluation&&<Disclosure title="Human-labelled simulation score (advisory)" defaultOpen><div className="detail-list"><DetailRow label="Configured suite" value={report.labelled_simulation_evaluation.configured?(report.labelled_simulation_evaluation.suite_name||`Suite #${report.labelled_simulation_evaluation.suite_id}`):'None'}/><DetailRow label="Evaluation result" value={report.labelled_simulation_evaluation.configured?(report.labelled_simulation_evaluation.passed?'All checks matched':'Review mismatches'):'Not configured'} badge={report.labelled_simulation_evaluation.configured?(report.labelled_simulation_evaluation.passed?'normal':'warning'):'neutral'}/><DetailRow label="Event accuracy" value={report.labelled_simulation_evaluation.summary?.event_accuracy===undefined?'—':`${fmt(Number(report.labelled_simulation_evaluation.summary.event_accuracy)*100,1)}%`}/><DetailRow label="Timing compliance" value={report.labelled_simulation_evaluation.summary?.timing_compliance===undefined?'—':`${fmt(Number(report.labelled_simulation_evaluation.summary.timing_compliance)*100,1)}%`}/><p className="muted small">This labelled score is evidence for the operator; it does not block model activation or promotion.</p></div></Disclosure>}
  <Disclosure title="Technical report"><pre className="code-panel compact">{JSON.stringify(report,null,2)}</pre></Disclosure></div>
}

type SimulationDraftCase={key:number;machine_id:string;description:string;start:string;end:string;expected_status:string};
const simulationCase=(machineId:string):SimulationDraftCase=>({key:Date.now()+Math.random(),machine_id:machineId,description:'',start:'',end:'',expected_status:'OK'});

const SIMULATION_STATUSES=['OK','WARN','CRITICAL','STOPPED','STARTING','SENSOR_FAULT','NO_PREDICTION','NO_DATA'];
const simulationCaseReady=(item:SimulationDraftCase)=>!!item.machine_id&&!!item.description.trim()&&!!item.start&&!!item.end&&new Date(item.end).getTime()>new Date(item.start).getTime();
function SimulationEventEditor({item,index,machines,expanded,onToggle,onChange,onDuplicate,onRemove,canRemove}:{item:SimulationDraftCase;index:number;machines:string[];expanded:boolean;onToggle:()=>void;onChange:(values:Partial<SimulationDraftCase>)=>void;onDuplicate:()=>void;onRemove:()=>void;canRemove:boolean}){
  const ready=simulationCaseReady(item);
  const range=item.start&&item.end?`${shortTime(item.start)} – ${shortTime(item.end)}`:'Time range incomplete';
  return <section className={cn('simulation-case-editor',expanded&&'expanded',ready&&'ready')}>
    <div className="case-index">
      <button type="button" className="case-toggle" onClick={onToggle} aria-expanded={expanded} aria-controls={`simulation-event-${item.key}`}>
        <span className="case-number">{index+1}</span>
        <span className="case-summary">
          <strong>{item.description.trim()||`Untitled event ${index+1}`}</strong>
          <small><b>{item.machine_id||'No machine'}</b><i/> {range}</small>
        </span>
        <StatusBadge value={item.expected_status}/>
        <span className={cn('case-completion',ready&&'complete')}><Icon name={ready?'check':'more'}/>{ready?'Ready':'Needs details'}</span>
        <Icon name="chevron"/>
      </button>
      <div className="case-actions">
        <button type="button" className="icon-button" onClick={onDuplicate} title="Duplicate event" aria-label={`Duplicate event ${index+1}`}><Icon name="copy"/></button>
        {canRemove&&<button type="button" className="icon-button danger-text" onClick={onRemove} title="Remove event" aria-label={`Remove event ${index+1}`}><Icon name="trash"/></button>}
      </div>
    </div>
    {expanded&&<div className="case-fields" id={`simulation-event-${item.key}`}>
      <div className="field"><span>Machine</span><Select value={item.machine_id} onChange={machine_id=>onChange({machine_id})} options={machines.map(id=>[id,id] as [string,string])}/></div>
      <div className="field"><span>Actual status throughout event</span><Select value={item.expected_status} onChange={expected_status=>onChange({expected_status})} options={SIMULATION_STATUSES.map(value=>[value,value.replaceAll('_',' ')] as [string,string])}/></div>
      <label className="field case-description"><span>Description / known event</span><input required maxLength={500} value={item.description} onChange={e=>onChange({description:e.target.value})} placeholder="What was known to be true during this range?"/></label>
      <label className="field"><span>Event start</span><input required type="datetime-local" max={item.end||undefined} value={item.start} onChange={e=>onChange({start:e.target.value})}/></label>
      <label className="field"><span>Event end</span><input required type="datetime-local" min={item.start||undefined} value={item.end} onChange={e=>onChange({end:e.target.value})}/></label>
    </div>}
  </section>;
}

function SimulationTrace({result,policy}:{result:any;policy:any}){
  const trace=Array.isArray(result?.trace)?result.trace:[];
  const evidence=result?.representative_tick||result?.target_tick;
  const points=trace.filter((point:any)=>Number.isFinite(Number(point.condition)));
  const width=780,height=210,left=42,right=16,top=15,bottom=30,plotW=width-left-right,plotH=height-top-bottom;
  const x=(index:number)=>left+(points.length===1?0:index/(points.length-1))*plotW;
  const y=(value:number)=>top+(1-Math.max(0,Math.min(100,value))/100)*plotH;
  const path=points.map((point:any,index:number)=>`${index?'L':'M'} ${x(index).toFixed(2)} ${y(Number(point.condition)).toFixed(2)}`).join(' ');
  const thresholds=[['Inspection',Number(policy?.MAINTENANCE_HEALTH_INSPECT),'#c88612'],['Critical',Number(policy?.FAILURE_HEALTH_THRESHOLD),'#c33b3b']] as Array<[string,number,string]>;
  const chart=points.length?<svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Condition score throughout the causal lead-in and labelled event">
    {([0,25,50,75,100] as number[]).map(value=><g key={value}><line x1={left} x2={width-right} y1={y(value)} y2={y(value)} className="simulation-grid"/><text x={left-8} y={y(value)+3} textAnchor="end">{value}</text></g>)}
    {thresholds.filter(([,value])=>Number.isFinite(value)).map(([label,value,color])=><g key={label}><line x1={left} x2={width-right} y1={y(value)} y2={y(value)} stroke={color} className="simulation-threshold"/><text x={width-right-3} y={y(value)-5} textAnchor="end" fill={color}>{label} {value}%</text></g>)}
    <path d={path} className="simulation-condition-line"/>
    <circle cx={x(points.length-1)} cy={y(Number(points[points.length-1].condition))} r="4" className="simulation-endpoint"/>
    <text x={left} y={height-8} textAnchor="start">{shortTime(points[0].timestamp)}</text><text x={width-right} y={height-8} textAnchor="end">{shortTime(points[points.length-1].timestamp)}</text>
  </svg>:<Empty title="No condition trace" text="This lead window produced no maintenance prediction. Its sampled operating-state output remains available below."/>;
  return <div className="simulation-trace">{chart}{points.length>0&&<p>The trace is sampled for display only. Range scoring uses every source row inside the labelled event; lead-time checks use the complete causal replay.</p>}{evidence?.sensor_snapshot&&<Disclosure title="Representative sensor snapshot"><JsonGrid value={evidence.sensor_snapshot}/></Disclosure>}{trace.length>0&&<Disclosure title={`Sampled maintenance output (${trace.length} points)`}><div className="responsive-table"><table><thead><tr><th>Time</th><th>Operating state</th><th>Maintenance output</th><th>Condition</th><th>Model score</th></tr></thead><tbody>{trace.map((point:any,index:number)=><tr key={`${point.timestamp}-${index}`}><td>{shortTime(point.timestamp)}</td><td><StatusBadge value={point.operating_state}/></td><td>{point.maintenance_status?<StatusBadge value={point.maintenance_status}/>:<span className="muted">Not applicable</span>}</td><td>{point.condition===null||point.condition===undefined?'—':`${fmt(point.condition,2)}%`}</td><td>{fmt(point.anomaly_score,5)}</td></tr>)}</tbody></table></div></Disclosure>}</div>;
}

function SimulationPage({machines,admin,mock}:{machines:string[];admin:boolean;mock:boolean}){
  const [runs,setRuns]=useState<any[]>([]),[selected,setSelected]=useState<any|null>(null),[selectedCase,setSelectedCase]=useState<any|null>(null);
  const [workspace,setWorkspace]=useState<'build'|'results'>(admin?'build':'results'),[showFilters,setShowFilters]=useState(false),[showHistory,setShowHistory]=useState(false);
  const [name,setName]=useState('Historical maintenance check'),[cases,setCases]=useState<SimulationDraftCase[]>([simulationCase(machines[0]||'')]);
  const [templates,setTemplates]=useState<any[]>([]),[templateId,setTemplateId]=useState<number|null>(null),[expandedCases,setExpandedCases]=useState<Set<number>>(new Set());
  const [availableModels,setAvailableModels]=useState<any[]>([]),[modelVersion,setModelVersion]=useState('');
  const [caseSearch,setCaseSearch]=useState(''),[caseMachine,setCaseMachine]=useState(''),[caseStatus,setCaseStatus]=useState('');
  const [loading,setLoading]=useState(true),[submitting,setSubmitting]=useState(false),[savingTemplate,setSavingTemplate]=useState(false),[error,setError]=useState('');
  const loadRuns=()=>api('/api/simulations?limit=25',{},15000).then(data=>{setRuns(data);setError('')}).catch(e=>setError(String(e.message||e))).finally(()=>setLoading(false));
  const loadTemplates=()=>api('/api/simulation-templates?limit=50',{},15000).then(setTemplates).catch(e=>setError(String(e.message||e)));
  const loadModels=()=>api('/api/models',{},15000).then(items=>{const usable=items.filter((item:any)=>Number(item.commissioned_machines||0)>0);setAvailableModels(usable);setModelVersion(current=>current&&usable.some((item:any)=>item.version_id===current)?current:(usable.find((item:any)=>item.status==='active')||usable[0])?.version_id||'')}).catch(e=>setError(String(e.message||e)));
  const openRun=async(id:number)=>{try{const detail=await api(`/api/simulations/${id}`,{},15000);setSelected(detail);setWorkspace('results');setShowHistory(false)}catch(e){setError(String((e as any).message||e))}};
  useEffect(()=>{if(mock){setLoading(false);return}loadRuns();loadTemplates();loadModels();const timer=window.setInterval(loadRuns,5000);return()=>window.clearInterval(timer)},[mock]);
  useEffect(()=>{if(mock||!selected?.id)return;const refresh=()=>api(`/api/simulations/${selected.id}`,{},15000).then(setSelected).catch(()=>{});const timer=window.setInterval(refresh,4000);return()=>window.clearInterval(timer)},[mock,selected?.id]);
  useEffect(()=>{if(!machines.length)return;setCases(current=>current.map(item=>({...item,machine_id:item.machine_id||machines[0]})))},[machines.join('|')]);
  useEffect(()=>{setExpandedCases(current=>current.size?current:new Set(cases.slice(0,1).map(item=>item.key)))},[]);
  const updateCase=(key:number,values:Partial<SimulationDraftCase>)=>setCases(current=>current.map(item=>item.key===key?{...item,...values}:item));
  const addCase=()=>{const item=simulationCase(machines[0]||'');setCases(current=>[...current,item]);setExpandedCases(new Set([item.key]))};
  const duplicateCase=(source:SimulationDraftCase)=>{const item={...source,key:Date.now()+Math.random(),description:source.description?source.description+' (copy)':''};setCases(current=>{const index=current.findIndex(row=>row.key===source.key);return[...current.slice(0,index+1),item,...current.slice(index+1)]});setExpandedCases(new Set([item.key]))};
  const removeCase=(key:number)=>{setCases(current=>current.filter(row=>row.key!==key));setExpandedCases(current=>{const next=new Set(current);next.delete(key);return next})};
  const toggleCase=(key:number)=>setExpandedCases(current=>current.has(key)?new Set():new Set([key]));
  const draftBody=()=>({name,cases:cases.map(({key,...item})=>item)});
  const saveTemplate=async()=>{setSavingTemplate(true);setError('');try{const saved=await api(templateId?`/api/simulation-templates/${templateId}`:'/api/simulation-templates',{method:templateId?'PUT':'POST',body:JSON.stringify(draftBody())},15000);setTemplateId(saved.id);await loadTemplates()}catch(e){setError(String((e as any).message||e))}finally{setSavingTemplate(false)}};
  const inputDateTime=(value:any)=>{if(!value)return'';const date=new Date(value);if(Number.isNaN(date.getTime()))return String(value).slice(0,16);return new Date(date.getTime()-date.getTimezoneOffset()*60000).toISOString().slice(0,16)};
  const reusableCases=(items:any[])=>items.map(item=>({key:Date.now()+Math.random(),machine_id:String(item.machine_id||''),description:String(item.description||''),start:inputDateTime(item.start),end:inputDateTime(item.end),expected_status:String(item.expected_status||'OK')}));
  const loadTemplate=(template:any)=>{const loaded=reusableCases(template.cases||[]);if(!loaded.length)return;setName(template.name);setCases(loaded);setTemplateId(template.id);setExpandedCases(new Set([loaded[0].key]));setCaseSearch('');setCaseMachine('');setCaseStatus('')};
  const reuseRun=(saved:any)=>{const loaded=reusableCases(saved.cases||[]);if(!loaded.length)return;setName(saved.name+' – reused');setCases(loaded);setTemplateId(null);if(saved.model_version&&availableModels.some(item=>item.version_id===saved.model_version))setModelVersion(saved.model_version);setExpandedCases(new Set([loaded[0].key]));setCaseSearch('');setCaseMachine('');setCaseStatus('');document.querySelector('.simulation-builder')?.scrollIntoView({behavior:'smooth',block:'start'})};
  const newTemplate=()=>{const item=simulationCase(machines[0]||'');setName('Historical maintenance check');setCases([item]);setTemplateId(null);setExpandedCases(new Set([item.key]));setCaseSearch('');setCaseMachine('');setCaseStatus('')};
  const deleteTemplate=async()=>{if(!templateId||!window.confirm('Delete this reusable event list?'))return;try{await api(`/api/simulation-templates/${templateId}`,{method:'DELETE'});newTemplate();await loadTemplates()}catch(e){setError(String((e as any).message||e))}};
  const toggleValidationSuite=async()=>{if(!templateId)return;const current=templates.find(item=>item.id===templateId);try{await api(`/api/simulation-templates/${templateId}/validation-suite?enabled=${current?.is_validation_suite?'false':'true'}`,{method:'PUT'});await loadTemplates()}catch(e){setError(String((e as any).message||e))}};
  const run=async(ev:any)=>{ev.preventDefault();setSubmitting(true);setError('');try{if(!modelVersion)throw new Error('Choose a model to evaluate.');if(incompatibleCases.length)throw new Error(`The selected model is not commissioned for ${incompatibleCases.map(item=>item.machine_id).filter((value,index,list)=>list.indexOf(value)===index).join(', ')}.`);const invalid=cases.find(item=>!simulationCaseReady(item));if(invalid){setExpandedCases(new Set([invalid.key]));throw new Error('Complete the highlighted event before running the simulation.')}const body={name,model_version:modelVersion,cases:cases.map(({key,...item})=>({...item,start:new Date(item.start).toISOString(),end:new Date(item.end).toISOString()}))};const queued=await api('/api/simulations',{method:'POST',body:JSON.stringify(body)},15000);await loadRuns();await openRun(queued.id)}catch(e){setError(String((e as any).message||e))}finally{setSubmitting(false)}};
  const removeRun=async(id:number)=>{if(!window.confirm(`Delete simulation run #${id}?`))return;try{await api(`/api/simulations/${id}`,{method:'DELETE'});if(selected?.id===id)setSelected(null);loadRuns()}catch(e){setError(String((e as any).message||e))}};
  const summary=selected?.summary||{},confusion=summary.confusion_matrix||{};
  const simulationProgress=selected?.progress||{};
  const simulationProgressFraction=Math.max(0,Math.min(1,Number(simulationProgress.overall_progress??((selected?.completed_cases||0)/Math.max(1,selected?.total_cases||1)))||0));
  const simulationHeartbeatAge=simulationProgress.updated_at?Math.max(0,(Date.now()-new Date(simulationProgress.updated_at).getTime())/1000):null;
  const simulationHeartbeatStale=simulationHeartbeatAge!==null&&simulationHeartbeatAge>60;
  const confusionLabels=Array.from(new Set([...Object.keys(confusion),...Object.values(confusion).flatMap((row:any)=>Object.keys(row as object))]));
  const active=runs.find(run=>['queued','running'].includes(run.status));
  const readyCases=cases.filter(simulationCaseReady).length;
  const selectedTemplate=templates.find(item=>item.id===templateId);
  const selectedModel=availableModels.find(item=>item.version_id===modelVersion);
  const incompatibleCases=selectedModel?cases.filter(item=>!selectedModel.commissioned_machine_ids?.includes(item.machine_id)):[];
  const modelLabel=(version:string)=>{const model=availableModels.find(item=>item.version_id===version);return model?.display_name&&model.display_name!==version?`${model.display_name} (${version})`:version};
  const query=caseSearch.trim().toLowerCase();
  const visibleCases=cases.filter(item=>(!caseMachine||item.machine_id===caseMachine)&&(!caseStatus||item.expected_status===caseStatus)&&(!query||[item.description,item.machine_id,item.expected_status,item.start,item.end].some(value=>String(value).toLowerCase().includes(query))));
  const representative=selectedCase?.result?.representative_tick||selectedCase?.result?.target_tick||{};
  const selectedTiming=selectedCase?.result?.timing||{};
  if(mock)return <><PageHeader eyebrow="HISTORICAL EVALUATION" title="Accuracy simulation" description="Label known machine event ranges and compare them with production maintenance output."/><ScopeBanner scope="fleet"/><Notice tone="warning">Accuracy simulation is unavailable in mock mode because synthetic labels and predictions would not measure real model accuracy. Start the production backend to replay PostgreSQL history.</Notice></>;
  return <><PageHeader eyebrow="HISTORICAL EVALUATION" title="Accuracy simulation" description="Compare human-labelled event ranges with the production model's causal historical output." actions={<div className="simulation-page-actions"><div className="simulation-view-switch" role="tablist" aria-label="Simulation workspace">{admin&&<button type="button" className={workspace==='build'?'active':''} onClick={()=>setWorkspace('build')}>Build event set</button>}<button type="button" className={workspace==='results'?'active':''} onClick={()=>setWorkspace('results')}>Results <span>{runs.length}</span></button></div>{active&&<StatusBadge value={active.status}/>}</div>}/><ScopeBanner scope="fleet"/>
    <details className="simulation-guide data-surface"><summary><span><Icon name="info"/><b>How this evaluation works</b></span><small>Read-only · seven-day lead-in · selected model and current policy</small><Icon name="chevron"/></summary><div><p><strong>Causal replay</strong> reads raw source rows only through each event end and never creates production predictions or alerts.</p><p><strong>Every source row is evaluated.</strong> The preceding seven days warm the same feature, selected model, condition, forecast, and maintenance path used live.</p><p><strong>Two checks stay separate:</strong> dominant labelled-range status coverage and whether WARN/CRITICAL timing follows the configured maintenance horizons.</p></div></details>
    {error&&<Notice tone="critical">{error}</Notice>}
    {workspace==='build'&&admin&&<form className="simulation-builder data-surface" onSubmit={run}>
      <div className="surface-heading simulation-builder-heading"><div><span>Event set</span><small>{readyCases}/{cases.length} ready</small></div><small>Up to 50 labelled ranges</small></div>
      <div className="simulation-name">
        <div className="simulation-setup-fields"><label className="field"><span>List name</span><input required maxLength={200} value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. July bearing incidents and healthy periods"/></label><label className="field"><span>Model to evaluate</span><Select ariaLabel="Model to evaluate" value={modelVersion} onChange={setModelVersion} options={availableModels.map(model=>[model.version_id,`${model.display_name||model.version_id} · ${String(model.status).replaceAll('_',' ')} · ${model.commissioned_machines} machines`] as [string,string])}/><small>The selected immutable version and current policy are pinned to this run.</small></label></div>
        <div className="template-controls">
          <Select ariaLabel="Load reusable event list" value={templateId===null?'':String(templateId)} onChange={value=>{const template=templates.find(item=>String(item.id)===value);if(template)loadTemplate(template)}} options={[['','Load saved list'],...templates.map(item=>[String(item.id),`${item.name} (${item.case_count||item.cases?.length||0})`] as [string,string])]}/>
          <button type="button" className="secondary" disabled={savingTemplate} onClick={saveTemplate}>{savingTemplate?'Saving…':templateId?'Update saved list':'Save list'}</button>
          <details className="simulation-action-menu"><summary className="secondary">More <Icon name="chevron"/></summary><div><button type="button" onClick={newTemplate}>Start a new list</button>{templateId&&<button type="button" onClick={toggleValidationSuite}>{selectedTemplate?.is_validation_suite?'Remove as model validation suite':'Use as model validation suite'}</button>}{templateId&&<button type="button" className="danger-text" onClick={deleteTemplate}>Delete saved list</button>}</div></details>
          <button type="button" className="primary" disabled={cases.length>=50} onClick={addCase}>Add event</button>
        </div>
      </div>
      {!!incompatibleCases.length&&<Notice tone="warning">The selected model is not commissioned for {incompatibleCases.map(item=>item.machine_id).filter((value,index,list)=>list.indexOf(value)===index).join(', ')}. Choose another model or remove those events.</Notice>}
      <div className="simulation-list-tools"><div><strong>Labelled events</strong><span>{cases.length} total · {readyCases} ready</span></div><button type="button" className={cn('secondary',showFilters&&'active')} onClick={()=>setShowFilters(value=>!value)}><Icon name="sliders"/>{showFilters?'Hide filters':'Find and filter'}</button></div>
      {showFilters&&<div className="simulation-case-toolbar">
        <label className="search-field"><Icon name="search"/><input value={caseSearch} onChange={e=>setCaseSearch(e.target.value)} placeholder="Find by event, machine, status, or date…" aria-label="Search labelled events"/></label>
        <Select ariaLabel="Filter events by machine" value={caseMachine} onChange={setCaseMachine} options={[['','All machines'],...machines.map(id=>[id,id] as [string,string])]}/>
        <Select ariaLabel="Filter events by status" value={caseStatus} onChange={setCaseStatus} options={[['','All statuses'],...SIMULATION_STATUSES.map(value=>[value,value.replaceAll('_',' ')] as [string,string])]}/>
        <span className="case-list-count">Showing <b>{visibleCases.length}</b> of {cases.length}</span>
        <button type="button" className="text-button" onClick={()=>setExpandedCases(new Set(visibleCases.map(item=>item.key)))}>Expand shown</button>
        <button type="button" className="text-button" onClick={()=>setExpandedCases(new Set())}>Collapse all</button>
      </div>}
      <div className="simulation-case-list">
        {visibleCases.length?visibleCases.map(item=><SimulationEventEditor key={item.key} item={item} index={cases.findIndex(row=>row.key===item.key)} machines={machines} expanded={expandedCases.has(item.key)} onToggle={()=>toggleCase(item.key)} onChange={values=>updateCase(item.key,values)} onDuplicate={()=>duplicateCase(item)} onRemove={()=>removeCase(item.key)} canRemove={cases.length>1}/>):<div className="case-list-empty"><Icon name="search"/><strong>No matching events</strong><span>Clear a filter to return to the full list.</span></div>}
      </div>
      <div className="simulation-submit"><p><b>{readyCases} of {cases.length} events ready.</b> Hidden filters never change what is saved or submitted.</p><button className="primary" disabled={submitting||!!active||!machines.length||!modelVersion||!!incompatibleCases.length||readyCases!==cases.length}>{submitting?'Queueing simulation…':active?'Another simulation is active':'Run accuracy simulation'}</button></div>
    </form>}
    {workspace==='build'&&!admin&&<Notice>Viewing is available to all signed-in users. An administrator must create a simulation because replay can use significant database and CPU capacity.</Notice>}
    {workspace==='results'&&<div className="simulation-results-workspace"><section className="simulation-results-toolbar data-surface"><div><strong>{selected?`Run #${selected.id} · ${selected.name}`:'Simulation results'}</strong><span>{selected?'The model and policy are pinned to this saved report.':'Choose a completed or active run to inspect.'}</span></div><div>{selected&&admin&&!!selected.cases?.length&&<button type="button" className="secondary" onClick={()=>{reuseRun(selected);setWorkspace('build')}}><Icon name="copy"/>Reuse event set</button>}{selected&&<button type="button" className={cn('secondary',showHistory&&'active')} onClick={()=>setShowHistory(value=>!value)}>{showHistory?'Hide run history':'Browse runs'}<span className="count-pill">{runs.length}</span></button>}</div></section>
    {(showHistory||!selected)&&<section className="data-surface simulation-runs"><div className="surface-heading"><span>Run history</span><small>Select a run to inspect its saved report</small></div>{loading?<Loading/>:runs.length===0?<Empty title="No simulations yet" text="Build an event set, then run the historical replay."/>:<div className="responsive-table"><table><thead><tr><th>Run</th><th>Status</th><th>Events</th><th>Accuracy</th><th>Created</th><th/></tr></thead><tbody>{runs.map(run=>{const runSummary=run.summary||{},accuracy=runSummary.event_accuracy??runSummary.target_accuracy??runSummary.accuracy,runProgress=run.progress||{};return <tr key={run.id} className={cn(selected?.id===run.id&&'selected-row')} onClick={()=>openRun(run.id)}><td><strong>#{run.id} · {run.name}</strong><small>{modelLabel(run.model_version)} · {run.created_by||'administrator'}</small></td><td><StatusBadge value={run.status}/></td><td><strong>{run.completed_cases??0} / {run.total_cases??run.case_count??0}</strong>{run.status==='running'&&<small>{fmt(Number(runProgress.overall_progress||0)*100,1)}% · {String(runProgress.phase||'starting').replaceAll('_',' ')}</small>}</td><td>{accuracy===null||accuracy===undefined?'—':`${fmt(Number(accuracy)*100,1)}%`}</td><td>{shortTime(run.created_at)}</td><td>{admin&&['completed','failed'].includes(run.status)?<button className="icon-button danger-text" onClick={e=>{e.stopPropagation();removeRun(run.id)}} title="Delete simulation"><Icon name="trash"/></button>:<Icon name="chevron"/>}</td></tr>})}</tbody></table></div>}</section>}
    {selected&&<section className="data-surface simulation-report"><div className="simulation-report-meta"><StatusBadge value={selected.status}/><span>Created {shortTime(selected.created_at)}</span><span>Model {modelLabel(selected.model_version)}</span><span>Policy captured at queue time</span>{selected.error&&<strong>{selected.error}</strong>}</div>{['queued','running'].includes(selected.status)?<div className="simulation-progress"><div className="simulation-progress-heading"><div><strong>{simulationProgress.message||'Waiting for the replay worker to report progress.'}</strong><span>{simulationProgress.description||selected.name}</span></div><b>{fmt(simulationProgressFraction*100,1)}%</b></div><div className="simulation-progress-track"><i style={{width:`${simulationProgressFraction*100}%`}}/></div><div className="simulation-progress-details"><span><b>Event</b>{simulationProgress.event_position||Math.min((selected.completed_cases||0)+1,selected.total_cases||1)} of {simulationProgress.event_count||selected.total_cases||1}</span><span><b>Phase</b>{String(simulationProgress.phase||selected.status).replaceAll('_',' ')}</span><span><b>Rows processed</b>{Number(simulationProgress.processed_rows||0).toLocaleString()}</span><span><b>Speed</b>{Number(simulationProgress.rows_per_second||0)>0?`${Math.round(Number(simulationProgress.rows_per_second)).toLocaleString()} rows/s`:'Calculating'}</span><span><b>Elapsed</b>{compactDuration(simulationProgress.elapsed_seconds)}</span><span><b>Event ETA</b>{compactDuration(simulationProgress.eta_seconds)}</span></div><div className={cn('simulation-heartbeat',simulationHeartbeatStale&&'stale')}><i/><span>{simulationHeartbeatAge===null?'Waiting for the first backend heartbeat.':simulationHeartbeatStale?`No progress update for ${compactDuration(simulationHeartbeatAge)}; the current batch may still be scoring.`:`Backend heartbeat ${compactDuration(simulationHeartbeatAge)} ago.`}</span>{simulationProgress.current_timestamp&&<small>Reached source time {shortTime(simulationProgress.current_timestamp)}</small>}</div></div>:selected.status==='completed'?<><div className="simulation-scorecards"><div><span>Event accuracy</span><strong>{(summary.event_accuracy??summary.target_accuracy??summary.accuracy)===null||(summary.event_accuracy??summary.target_accuracy??summary.accuracy)===undefined?'—':`${fmt(Number(summary.event_accuracy??summary.target_accuracy??summary.accuracy)*100,1)}%`}</strong></div><div><span>Labelled coverage</span><strong>{summary.mean_status_coverage===null||summary.mean_status_coverage===undefined?'—':`${fmt(Number(summary.mean_status_coverage)*100,1)}%`}</strong></div><div><span>Timing compliance</span><strong>{summary.timing_compliance===null||summary.timing_compliance===undefined?'—':`${fmt(Number(summary.timing_compliance)*100,1)}%`}</strong></div><div><span>Correct events</span><strong>{summary.correct_ranges??summary.correct_targets??0} / {summary.evaluated_ranges??summary.evaluated_targets??0}</strong><small>{summary.premature_critical_cases??0} premature critical · {summary.false_alert_cases??0} false alerts</small></div></div></>:null}
      {!!selected.cases?.length&&<div className="responsive-table simulation-case-results"><table><thead><tr><th>Labelled event</th><th>Expected → output</th><th>Verdict</th><th>Coverage</th><th>Timing</th><th/></tr></thead><tbody>{selected.cases.map((item:any)=>{const result=item.result||{},timing=result.timing||{},start=item.start||item.target||item.end,end=item.end||item.target||item.start;return <tr key={item.id} onClick={()=>setSelectedCase(item)}><td><strong>{item.description}</strong><small>{item.machine_id} · {shortTime(start)} to {shortTime(end)}</small></td><td><div className="simulation-status-flow"><StatusBadge value={item.expected_status}/><Icon name="chevron"/><StatusBadge value={result.predicted_status||(item.error?'ERROR':'PENDING')}/></div></td><td>{result.evaluable?<StatusBadge value={result.match?'MATCH':'MISMATCH'} forced={result.match?'normal':'critical'}/>:<StatusBadge value="UNSCORED"/>}</td><td>{result.expected_status_coverage===null||result.expected_status_coverage===undefined?<span className="muted">—</span>:<strong>{fmt(Number(result.expected_status_coverage)*100,1)}%</strong>}</td><td>{timing.timing_evaluable?<StatusBadge value={timing.timing_pass?'PASS':'CHECK'} forced={timing.timing_pass?'normal':'warning'}/>:<span className="muted">Not applicable</span>}</td><td><Icon name="chevron"/></td></tr>})}</tbody></table></div>}
      {selected.status==='completed'&&(confusionLabels.length>0||selected.error)&&<Disclosure title="Advanced report"><>{selected.error&&<Notice tone="critical">{selected.error}</Notice>}{confusionLabels.length>0&&<div className="responsive-table simulation-confusion"><table><thead><tr><th>Actual ↓ / Dominant output →</th>{confusionLabels.map(label=><th key={label}>{label}</th>)}</tr></thead><tbody>{confusionLabels.map(expected=><tr key={expected}><th>{expected}</th>{confusionLabels.map(predicted=><td key={predicted}>{confusion[expected]?.[predicted]||0}</td>)}</tr>)}</tbody></table></div>}<p className="muted">Accuracy compares each labelled range with its dominant maintenance output. Coverage measures how much of that range carried the expected status; timing separately checks WARN and CRITICAL lead-time policy.</p></></Disclosure>}</section>}
    </div>}
    {selectedCase&&<Modal title={selectedCase.description} wide onClose={()=>setSelectedCase(null)}><div className="simulation-case-detail"><div className="simulation-verdict"><div><span>Actual throughout event</span><StatusBadge value={selectedCase.expected_status}/></div><Icon name="chevron"/><div><span>Dominant production output</span><StatusBadge value={selectedCase.result?.predicted_status||'UNSCORED'}/></div><StatusBadge value={selectedCase.result?.evaluable?(selectedCase.result?.match?'MATCH':'MISMATCH'):'UNSCORED'} forced={selectedCase.result?.evaluable?(selectedCase.result?.match?'normal':'critical'):'neutral'}/></div><Notice tone={selectedCase.result?.evaluable?(selectedCase.result?.match?'normal':'critical'):'neutral'}>{selectedCase.result?.explanation||selectedCase.error||'No replay evidence is available.'}</Notice><SimulationTrace result={selectedCase.result} policy={selected?.policy_snapshot}/><div className="split-detail"><div className="detail-list"><DetailRow label="Machine" value={selectedCase.machine_id}/><DetailRow label="Event range" value={`${shortTime(selectedCase.start||selectedCase.result?.event_start||selectedCase.target)} to ${shortTime(selectedCase.end||selectedCase.result?.event_end||selectedCase.target)}`}/><DetailRow label="Causal replay window" value={selectedCase.result?.window_start?`${shortTime(selectedCase.result.window_start)} to ${shortTime(selectedCase.result.event_end||selectedCase.result.target_timestamp)}`:'—'}/><DetailRow label="Event evaluation rows" value={selectedCase.result?.evaluation_rows??selectedCase.result?.target_rows??'—'}/><DetailRow label="Lead-in replay rows" value={selectedCase.result?.replay_rows??'—'}/><DetailRow label="Warm-up context rows" value={selectedCase.result?.context_rows??'—'}/><DetailRow label="ML prediction coverage" value={selectedCase.result?.coverage===null||selectedCase.result?.coverage===undefined?'—':`${fmt(Number(selectedCase.result.coverage)*100,1)}%`}/><DetailRow label="Labelled-status coverage" value={selectedCase.result?.expected_status_coverage===null||selectedCase.result?.expected_status_coverage===undefined?'—':`${fmt(Number(selectedCase.result.expected_status_coverage)*100,1)}%`}/><DetailRow label="Dominant-status coverage" value={selectedCase.result?.dominant_status_coverage===null||selectedCase.result?.dominant_status_coverage===undefined?'—':`${fmt(Number(selectedCase.result.dominant_status_coverage)*100,1)}%`}/><DetailRow label="Source age at event end" value={(selectedCase.result?.source_age_seconds_at_event_end??selectedCase.result?.source_age_seconds_at_target)===null||(selectedCase.result?.source_age_seconds_at_event_end??selectedCase.result?.source_age_seconds_at_target)===undefined?'—':`${fmt(selectedCase.result?.source_age_seconds_at_event_end??selectedCase.result?.source_age_seconds_at_target,1)} seconds`}/><DetailRow label="First WARN / lead" value={selectedTiming.first_warn_timestamp?`${shortTime(selectedTiming.first_warn_timestamp)} · ${fmt(selectedTiming.first_warn_lead_hours,1)} h before onset`:'Not observed'}/><DetailRow label="First CRITICAL / lead" value={selectedTiming.first_critical_timestamp?`${shortTime(selectedTiming.first_critical_timestamp)} · ${fmt(selectedTiming.first_critical_lead_hours,1)} h before onset`:'Not observed'}/><DetailRow label="Planning warning" value={selectedTiming.planning_warning_pass===null||selectedTiming.planning_warning_pass===undefined?'Not applicable':selectedTiming.planning_warning_pass?'Passed':'Missed'}/><DetailRow label="Urgent escalation" value={selectedTiming.urgent_critical_pass===null||selectedTiming.urgent_critical_pass===undefined?'Not applicable':selectedTiming.urgent_critical_pass?'Passed':'Failed or premature'}/><DetailRow label="Representative evidence" value={shortTime(representative.timestamp)}/><DetailRow label="Condition / risk" value={representative.condition===undefined?'—':`${fmt(representative.condition,2)}% / ${fmt(Number(representative.condition_risk)*100,1)}%`}/><DetailRow label="Raw anomaly score" value={fmt(representative.anomaly_score,5)}/><DetailRow label="Decision trigger" value={triggerLabel(representative.maintenance_trigger)}/><DetailRow label="Trend / remaining" value={representative.trend_slope_per_day===undefined?'—':`${fmt(representative.trend_slope_per_day,3)}%/day · ${fmt(representative.remaining_days,1)} days`}/></div><div><p className="section-label">RANGE DECISION AND TIMING EVIDENCE</p><Disclosure title="Lead-time assessment" defaultOpen><JsonGrid value={selectedTiming}/></Disclosure><Disclosure title="Event pipeline-output distribution"><JsonGrid value={selectedCase.result?.pipeline_outcome_distribution}/></Disclosure><Disclosure title="Event maintenance distribution"><JsonGrid value={selectedCase.result?.maintenance_distribution}/></Disclosure><Disclosure title="Event operating-state distribution"><JsonGrid value={selectedCase.result?.operating_state_distribution}/></Disclosure><Disclosure title="Lead-in maintenance distribution"><JsonGrid value={selectedCase.result?.lead_window_maintenance_distribution}/></Disclosure><Disclosure title="Top feature contributors"><pre className="code-panel compact">{JSON.stringify(representative.top_contributors||[],null,2)}</pre></Disclosure><Disclosure title="Captured policy"><pre className="code-panel compact">{JSON.stringify(selected?.policy_snapshot||{},null,2)}</pre></Disclosure></div></div></div></Modal>}
  </>;
}

function ModelNameEditor({model,onSaved}:{model:any;onSaved:(name:string)=>void}){
  const [name,setName]=useState(model.display_name||model.version_id),[saving,setSaving]=useState(false),[error,setError]=useState('');
  useEffect(()=>{setName(model.display_name||model.version_id);setError('')},[model.version_id,model.display_name]);
  const save=async()=>{const cleaned=name.trim().replace(/\s+/g,' ');if(!cleaned){setError('Enter a model name.');return}setSaving(true);setError('');try{const result=await api(`/api/models/${model.version_id}/name`,{method:'PATCH',body:JSON.stringify({name:cleaned})});setName(result.display_name);onSaved(result.display_name)}catch(e){setError(String((e as any).message||e))}finally{setSaving(false)}};
  return <div className="model-name-editor"><label><span>Display name</span><input maxLength={80} value={name} onChange={e=>setName(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();save()}}}/></label><button className="secondary" disabled={saving||name.trim()===(model.display_name||model.version_id)} onClick={save}>{saving?'Saving…':'Save name'}</button>{error&&<small>{error}</small>}<p>The technical version ID and artifact directory remain unchanged.</p></div>;
}

function Models({admin}:{admin:boolean}){
  const [rows,setRows]=useState<any[]>([]),[jobs,setJobs]=useState<any[]>([]),[error,setError]=useState(''),[selected,setSelected]=useState<any|null>(null),[renaming,setRenaming]=useState<any|null>(null),[retrain,setRetrain]=useState<any>(),[panel,setPanel]=useState<'training'|null>(null),[queueing,setQueueing]=useState(false);
  const load=()=>Promise.all([api('/api/models'),api('/api/retrain/status'),api('/api/retrain/jobs?limit=10')]).then(([m,s,j])=>{setRows(m);setRetrain(s);setJobs(j);setSelected(current=>{const updated=current?.__job?j.find((job:any)=>job.id===current.id):null;return updated?{...updated,__job:true}:current});setError('')}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{let disposed=false;const refresh=()=>{if(!disposed)load()};refresh();const timer=window.setInterval(refresh,10000);return()=>{disposed=true;window.clearInterval(timer)}},[]);
  const deleteVersion=async(versionId:string)=>{if(!window.confirm(`Delete model version ${versionId}? Its staged candidates will become eligible again. Artifact deletion cannot be undone.`))return;try{await api(`/api/models/${versionId}`,{method:'DELETE'});setSelected(null);load()}catch(e){setError(String((e as any).message||e))}};
  const runRetrain=async()=>{setQueueing(true);setError('');try{await api('/api/retrain/trigger',{method:'POST'});await load()}catch(e){setError(String((e as any).message||e))}finally{setQueueing(false)}};
  const activeJob=retrain?.active_job,pendingShadow=retrain?.pending_shadow,eligibleCandidates=retrain?.eligible_pending??retrain?.pending??0,jobBusy=queueing||eligibleCandidates===0||!!pendingShadow||activeJob?.status==='queued'||activeJob?.status==='running';
  return <><PageHeader eyebrow="FLEET MODELS" title="Models & retraining" description="Shared fleet model lifecycle, held-out safety gates, and persistent retraining jobs." actions={<><button className="secondary" onClick={()=>setPanel('training')}>Retraining policy</button>{admin&&<button className="primary" disabled={jobBusy} onClick={runRetrain}>{queueing?'Queueing...':pendingShadow?'Shadow awaiting decision':activeJob?.status==='queued'?'Retrain queued':activeJob?.status==='running'?'Retraining...':eligibleCandidates===0?'No eligible candidates':'Run shadow retrain'}</button>}</>}/><ScopeBanner scope="fleet"/>{error&&<Notice tone="critical">{error}</Notice>}
  {retrain&&<div className="inline-summary"><div><span>Eligible candidates</span><strong>{eligibleCandidates}</strong></div><div><span>Per-machine target</span><strong>{retrain.batch_size_per_machine??retrain.batch_size??'—'}</strong></div><div><span>Eligibility</span><StatusBadge value={retrain.due?'DUE':'NOT DUE'} forced={retrain.due?'warning':'normal'}/></div><div><span>Latest job</span>{retrain.active_job?<StatusBadge value={retrain.active_job.status}/>:retrain.last_job?<StatusBadge value={retrain.last_job.status}/>:<strong>None</strong>}</div><button className="text-button" onClick={()=>setSelected({__retrain:true,...retrain})}>Details <Icon name="info"/></button></div>}
  <section className="data-surface"><div className="surface-heading"><span>Model versions</span><small>Commissioning and shadow validation scores are advisory</small></div>{rows.length===0?<Empty title="No model versions" text="No registered model bundles were returned by the backend."/>:<div className="responsive-table"><table><thead><tr><th>Model</th><th>Status</th><th>Created</th><th>Validation score</th><th>Promoted by</th><th/></tr></thead><tbody>{rows.map(m=>{const label=validationLabel(m);return <tr key={m.version_id} onClick={()=>setSelected(m)}><td><strong>{m.display_name||m.version_id}</strong><small>{m.display_name?m.version_id:(m.reference_signature||'No reference signature')}</small></td><td><StatusBadge value={m.status}/></td><td>{shortTime(m.created_at)}</td><td><span className={cn('validation-dot',label==='Review'&&'bad',!['Passed','Review'].includes(label)&&'unknown')}/>{label}</td><td>{m.promoted_by||'—'}</td><td>{admin?<button className="text-button" onClick={event=>{event.stopPropagation();setRenaming(m)}}>Rename</button>:<Icon name="chevron"/>}</td></tr>})}</tbody></table></div>}</section>
  <section className="data-surface retrain-jobs"><div className="surface-heading"><span>Retraining jobs</span><small>Persistent backend history · latest 10</small></div>{jobs.length===0?<Empty title="No retraining jobs" text="Manual and scheduled attempts will appear here."/>:<div className="responsive-table"><table><thead><tr><th>Job</th><th>Trigger</th><th>Status</th><th>Requested</th><th>Model</th><th>Result</th></tr></thead><tbody>{jobs.map(job=><tr key={job.id} onClick={()=>setSelected({__job:true,...job})}><td><strong>#{job.id}</strong><small>{job.requested_by||'scheduler'}</small></td><td>{job.trigger}</td><td><StatusBadge value={job.status}/></td><td>{shortTime(job.created_at)}</td><td>{job.model_version_id||'—'}</td><td>{job.error||parsedObject(job.result).reason||'—'}</td></tr>)}</tbody></table></div>}</section>
  {selected&&<Modal title={selected.__retrain?'Retraining eligibility':selected.__job?`Retraining job #${selected.id}`:selected.version_id} onClose={()=>setSelected(null)} wide={!selected.__retrain}>{selected.__retrain?<div className="detail-list"><DetailRow label="Eligible candidates" value={selected.pending??selected.candidate_count??0}/><DetailRow label="Per-machine batch threshold" value={selected.batch_size_per_machine??selected.batch_size??'—'}/><DetailRow label="Currently due" value={selected.due?'Yes':'No'}/><DetailRow label="Next automatic check" value={shortTime(selected.next_check_at)}/>{selected.pending_by_machine&&<div className="candidate-list">{Object.entries(selected.pending_by_machine).map(([id,value]:[string,any])=><div key={id}><strong>{id}</strong><span>{value.pending??0} candidates · oldest {fmt(value.oldest_age_days??0,1)} days</span>{value.requires_commissioning&&<StatusBadge value="COMMISSIONING REQUIRED" forced="warning"/>}</div>)}</div>}<p className="muted">A commissioned machine reaching its count or age threshold starts one balanced shared-model job. Candidates stay unconsumed until a validated shadow is promoted.</p></div>:selected.__job?<><div className="detail-list"><DetailRow label="Status" value={selected.status} badge={tone(selected.status)}/><DetailRow label="Trigger" value={selected.trigger}/><DetailRow label="Requested by" value={selected.requested_by||'scheduler'}/><DetailRow label="Requested" value={shortTime(selected.created_at)}/><DetailRow label="Started" value={shortTime(selected.started_at)}/><DetailRow label="Finished" value={shortTime(selected.finished_at)}/><DetailRow label="Model version" value={selected.model_version_id||'—'}/><DetailRow label="Error" value={selected.error||'—'}/></div><Disclosure title="Job result"><pre className="code-panel compact">{JSON.stringify(parsedObject(selected.result),null,2)}</pre></Disclosure></>:<><div className="split-detail"><div className="detail-list"><DetailRow label="Status" value={selected.status} badge={tone(selected.status)}/><DetailRow label="Artifact path" value={selected.artifact_path||'—'}/><DetailRow label="Reference signature" value={selected.reference_signature||'—'}/><DetailRow label="Created" value={shortTime(selected.created_at)}/><DetailRow label="Promoted" value={shortTime(selected.promoted_at)}/><DetailRow label="Promoted by" value={selected.promoted_by||'—'}/></div><div><p className="section-label">VALIDATION REPORT</p><ValidationReport report={selected.validation_report}/></div></div>{admin&&<div className="modal-actions">{['shadow','retired'].includes(selected.status)&&<button className="primary" onClick={async()=>{try{await api(`/api/models/${selected.version_id}/promote`,{method:'POST'});setSelected(null);load()}catch(e){setError(String((e as any).message||e))}}}>Promote to active</button>}{selected.status!=='active'&&<button className="secondary danger-text" onClick={()=>deleteVersion(selected.version_id)}>Delete version</button>}</div>}</>}</Modal>}
  {renaming&&<Modal title="Rename model" onClose={()=>setRenaming(null)}><ModelNameEditor model={renaming} onSaved={name=>{setRows(current=>current.map(item=>item.version_id===renaming.version_id?{...item,display_name:name}:item));setSelected(current=>current?.version_id===renaming.version_id?{...current,display_name:name}:current);setRenaming(null)}}/></Modal>}
  {panel==='training'&&<Modal title="Training options" onClose={()=>setPanel(null)}><TrainingOptions admin={admin}/></Modal>}
  </>
}

function TrainingOptions({admin}:{admin:boolean}){
  const [v,setV]=useState<any>(),[saved,setSaved]=useState(false),[error,setError]=useState(''),[saveError,setSaveError]=useState(''),[saving,setSaving]=useState(false),[infoKey,setInfoKey]=useState<string|null>(null);
  useEffect(()=>{api('/api/config/training').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  const automatic:EditableSettingMeta={label:'Automatic retraining',desc:'Scheduled check for a due balanced shadow retrain.',unit:'state',purpose:'Controls whether the scheduler may start a shadow retrain when a commissioned machine reaches its candidate-count or age threshold. Manual shadow retraining remains available.',higher:'The scheduler checks eligibility and may create a validated shadow model automatically.',lower:'Only an administrator pressing Run shadow retrain can start retraining.',higherLabel:'When enabled',lowerLabel:'When disabled'};
  const meta:Record<string,EditableSettingMeta>={
    RETRAIN_BATCH_SIZE:{label:'Retrain batch size',desc:'Confirmed-normal alerts required from one machine before retraining is due.',unit:'alerts / machine',purpose:'Sets the per-machine confirmed-normal candidate count that makes the shared-model shadow retrain due.',higher:'Retraining happens less often and waits for more reviewed evidence.',lower:'Retraining becomes due sooner with less new evidence.',step:1,min:1,max:100000},
    RETRAIN_TIME_CAP_DAYS:{label:'Retrain time cap',desc:'Maximum candidate age before retraining becomes due.',unit:'days',purpose:'Makes retraining due when eligible candidates have waited this many days, even if the batch-size target has not been reached. It cannot exceed the candidate window.',higher:'Allows candidates to wait longer and reduces retraining frequency.',lower:'Refreshes the model sooner when candidate volume is low.',step:1,min:1,max:3650},
    REFERENCE_WINDOW_MONTHS:{label:'Candidate window',desc:'How far back confirmed-normal candidates remain eligible.',unit:'months',purpose:'Limits candidate eligibility to reviewed normal samples collected within this recent time window. It must be long enough to contain the retrain time cap.',higher:'Keeps more historical evidence but may include behavior that no longer reflects current operation.',lower:'Focuses on recent operation but may leave fewer samples for retraining.',step:1,min:1,max:120},
    REFERENCE_DEDUP_WINDOW_HOURS:{label:'Deduplication window',desc:'Time span in which similar same-machine candidates are collapsed.',unit:'hours',purpose:'Compares candidates from the same machine within this time span and removes near-duplicates before retraining.',higher:'Removes repeated behavior across a longer period, improving diversity but retaining fewer samples.',lower:'Only removes close-in-time duplicates, retaining more potentially repetitive samples.',step:1,min:0,max:8760},
    REFERENCE_COSINE_SIMILARITY:{label:'Deduplication similarity',desc:'Similarity required for two candidates to count as duplicates.',unit:'0–1',purpose:'Treats two same-machine candidates as duplicates when their cosine similarity in machine-relative feature space reaches this value.',higher:'Requires candidates to be almost identical before deduplicating, so more samples remain.',lower:'Deduplicates more aggressively, improving diversity but potentially discarding useful variation.',step:.01,min:0,max:1},
    RETRAIN_CHECK_INTERVAL_MINUTES:{label:'Retraining check interval',desc:'How often automatic retraining eligibility is evaluated.',unit:'minutes',purpose:'Sets the scheduler interval used to check whether balanced candidate count or age rules make a shadow retrain due.',higher:'Checks less often and reduces scheduler activity, but a due retrain may start later.',lower:'Responds to newly due retraining sooner, with more frequent database checks.',step:1,min:5,max:1440},
    RETRAIN_RETRY_COOLDOWN_HOURS:{label:'Failed-job retry cooldown',desc:'Wait before the scheduler retries an unchanged job that failed unexpectedly.',unit:'hours',purpose:'Limits retries after an operational failure such as a database or filesystem interruption. A safety-gate rejection remains suppressed until candidates, material policy, or retraining code changes; an administrator can still run manually.',higher:'Reduces repeated compute after transient failures, but waits longer before an automatic recovery attempt.',lower:'Retries operational failures sooner, with a greater risk of repeating the same interruption.',step:.25,min:.25,max:720},
    RETRAIN_MAX_FP_RATE_INCREASE:{label:'Maximum false-positive increase',desc:'Largest allowed per-machine false-positive-rate increase for a shadow model.',unit:'rate (0–0.10)',purpose:'Rejects the shared shadow if any machine exceeds its active-model false-positive rate by more than this amount.',higher:'Makes promotion easier but permits a larger increase in false alarms.',lower:'Protects each machine more strictly, but rejects more potentially useful shadows.',step:.005,min:0,max:.1},
  };
  if(error)return <Notice tone="critical">{error}</Notice>;
  if(!v)return <Loading/>;
  const save=async()=>{setSaved(false);setSaveError('');setSaving(true);try{const next=await api('/api/config/training',{method:'PUT',body:JSON.stringify(v)});setV(next);setSaved(true)}catch(e){setSaveError(String((e as any).message||e))}finally{setSaving(false)}};
  const selectedInfo=infoKey==='AUTO_RETRAIN_ENABLED'?automatic:(infoKey?meta[infoKey]:undefined);
  return <div className="settings-surface settings-surface-modal">
  <div className="settings-intro"><strong>Retraining policy</strong><span>Controls when reviewed normal evidence may propose a new shared model. Every proposal still has to pass validation before promotion.</span></div>
  <div className="setting-row"><div><SettingTitle item={automatic} onInfo={()=>setInfoKey('AUTO_RETRAIN_ENABLED')}/><span>{automatic.desc}</span><code>AUTO_RETRAIN_ENABLED</code></div><div className="setting-value"><span>Current state</span><div className="setting-control toggle-control"><Switch checked={!!v.AUTO_RETRAIN_ENABLED} disabled={!admin} onChange={()=>setV({...v,AUTO_RETRAIN_ENABLED:!v.AUTO_RETRAIN_ENABLED})}/><span className={cn('toggle-state',v.AUTO_RETRAIN_ENABLED?'on':'off')}>{v.AUTO_RETRAIN_ENABLED?'On':'Off'}</span></div></div></div>
  {Object.keys(v).filter(k=>k!=='AUTO_RETRAIN_ENABLED').map(k=>{const item=meta[k];return <div className="setting-row" key={k}><div><SettingTitle item={item} onInfo={()=>setInfoKey(k)}/><span>{item.desc}</span><code>{k}</code></div><div className="setting-value"><span>Current value</span><div className="setting-control"><input type="number" step={item.step??'any'} min={item.min} max={item.max} value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}/><span>{item.unit}</span></div></div></div>})}
  {saved&&<Notice tone="normal">Saved. The scheduler interval is refreshed immediately, and the next manual or scheduled shadow retrain uses this policy.</Notice>}{saveError&&<Notice tone="critical">{saveError}</Notice>}
  {admin&&<div className="modal-actions"><button className="primary" disabled={saving} onClick={save}>{saving?'Saving…':'Save retraining policy'}</button></div>}
  {selectedInfo&&<SettingInfoDialog item={selectedInfo} value={infoKey==='AUTO_RETRAIN_ENABLED'?(v.AUTO_RETRAIN_ENABLED?'On':'Off'):v[infoKey||'']} onClose={()=>setInfoKey(null)}/>}</div>
}

function HistoryPage({machineId}:{machineId:string}){
  const [rows,setRows]=useState<any[]>([]),[total,setTotal]=useState(0),[error,setError]=useState(''),[q,setQ]=useState(''),[level,setLevel]=useState(''),[selected,setSelected]=useState<any|null>(null);
  const {page,setPage,pageSize,setPageSize,offset}=usePagination(25);
  const load=()=>api(`/api/history?machine_id=${encodeURIComponent(machineId)}&limit=${pageSize}&offset=${offset}${level?`&level=${encodeURIComponent(level)}`:''}`).then((r:Paged<any>)=>{setRows(r.items);setTotal(r.total)}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{load()},[level,pageSize,offset]);
  useEffect(()=>{setPage(1)},[level]);
  const filtered=useMemo(()=>rows.filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q.toLowerCase())),[rows,q]);
  return <><PageHeader eyebrow="MACHINE HISTORY" title="Prediction history" description={`Backend prediction and review history for ${machineId}.`} actions={<div className="filter-bar"><label className="search-field"><Icon name="search"/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search this page"/></label><Select value={level} onChange={setLevel} options={[['','All states'],['OK','OK'],['WARN','WARN'],['CRITICAL','CRITICAL']]}/></div>}/><ScopeBanner scope="machine" machineId={machineId}/>{error&&<Notice tone="critical">{error}</Notice>}
  <section className="data-surface"><div className="surface-heading"><span>{q?`${filtered.length} of ${rows.length} on this page`:`${total} record${total===1?'':'s'}`}</span><small>Most recent first</small></div>{filtered.length===0?<Empty title="No matching records" text="Change the search or state filter."/>:<div className="responsive-table"><table><thead><tr><th>Time</th><th>State</th><th>Condition</th><th>Anomaly</th><th>Remaining</th><th>Model</th><th>Outcome</th><th/></tr></thead><tbody>{filtered.map((r,i)=>{const outcome=historyOutcome(r); return <tr key={r.id||`${r.tick_timestamp}-${i}`} onClick={()=>setSelected(r)}><td><strong>{shortTime(r.tick_timestamp)}</strong></td><td><StatusBadge value={r.maintenance_level||'OK'}/></td><td>{fmt(r.health_state,1)}%</td><td>{fmt(r.anomaly_score,4)}</td><td>{r.remaining_days!==undefined?`${fmt(r.remaining_days,1)} d`:'—'}</td><td>{r.model_version||'—'}</td><td>{outcome?<StatusBadge value={outcome.text} forced={outcome.tone}/>:<span className="muted">—</span>}</td><td><Icon name="chevron"/></td></tr>})}</tbody></table></div>}
  <Pagination page={page} pageSize={pageSize} total={total} onPage={setPage} onPageSize={setPageSize}/></section>
  {selected&&<Modal title="Prediction record" wide onClose={()=>setSelected(null)}><div className="split-detail"><div className="detail-list"><DetailRow label="Timestamp" value={shortTime(selected.tick_timestamp)}/><DetailRow label="Maintenance state" value={selected.maintenance_level||'—'} badge={tone(selected.maintenance_level)}/><DetailRow label="Maintenance reason" value={selected.maintenance_reason||'—'}/><DetailRow label="Trigger" value={triggerLabel(selected.maintenance_trigger)}/><DetailRow label="Condition score" value={`${fmt(selected.health_state,2)}%`}/><DetailRow label="Anomaly score" value={fmt(selected.anomaly_score,4)}/><DetailRow label="Remaining days" value={fmt(selected.remaining_days,2)}/>{selected.alert_status&&<><DetailRow label="Alert outcome" value={reviewLabel(selected.alert_status)} badge={reviewTone(selected.alert_status)}/><DetailRow label="Alert level / trigger" value={`${selected.alert_level||'—'} · ${triggerLabel(selected.alert_trigger)}`}/>{selected.alert_reviewed_by&&<DetailRow label="Reviewed by" value={`${selected.alert_reviewed_by} · ${shortTime(selected.alert_reviewed_at)}`}/>}</>}{!selected.alert_status&&selected.near_miss_status&&<><DetailRow label="Near-miss outcome" value={reviewLabel(selected.near_miss_status)} badge={reviewTone(selected.near_miss_status)}/>{selected.near_miss_reviewed_by&&<DetailRow label="Reviewed by" value={`${selected.near_miss_reviewed_by} · ${shortTime(selected.near_miss_reviewed_at)}`}/>}</>}{!selected.alert_status&&!selected.near_miss_status&&<DetailRow label="Outcome" value="Not surfaced in Alert review or Near miss"/>}</div><div><p className="section-label">SENSOR SNAPSHOT</p><JsonGrid value={selected.raw_reading}/><Disclosure title="Forecast boundary-crossing risk"><JsonGrid value={selected.failure_probability}/><p className="muted small">Model-based first-passage risk under the current linear-drift assumption; validate against real outcomes before treating it as calibrated probability.</p></Disclosure><Disclosure title="Full backend record"><pre className="code-panel compact">{JSON.stringify(selected,null,2)}</pre></Disclosure></div></div>
  </Modal>}
  </>
}

function Thresholds({admin}:{admin:boolean}){
  const [v,setV]=useState<any>(),[saved,setSaved]=useState(false),[error,setError]=useState(''),[saveError,setSaveError]=useState(''),[saving,setSaving]=useState(false),[infoKey,setInfoKey]=useState<string|null>(null);
  useEffect(()=>{api('/api/config/thresholds').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  const meta:Record<string,EditableSettingMeta>={
    FAILURE_HEALTH_THRESHOLD:{label:'Critical condition threshold',desc:'Immediate CRITICAL boundary for the smoothed condition score.',unit:'condition %',purpose:'Declares the machine critical when its smoothed condition score is at or below this value. It is also the boundary used by remaining-life and first-passage forecast-risk calculations.',higher:'Triggers CRITICAL earlier and increases model-estimated boundary-crossing risk. This is more sensitive but can produce more false alarms.',lower:'Requires more severe deviation before CRITICAL and lowers model-estimated crossing risk. This reduces alarms but may delay detection.',step:1,min:0,max:99},
    MAINTENANCE_HEALTH_INSPECT:{label:'Inspection condition threshold',desc:'Immediate WARN boundary for the smoothed condition score.',unit:'condition %',purpose:'Requests inspection when the condition score is at or below this value but remains above the critical boundary.',higher:'Requests inspection sooner and increases warning sensitivity.',lower:'Waits for a worse condition score before warning, reducing warnings but delaying inspection.',step:1,min:1,max:100},
    HEALTH_SENSITIVITY_STD:{label:'Condition sensitivity',desc:'Robust anomaly spreads required to reduce condition from 100% to 0%.',unit:'robust spreads',purpose:'Controls how strongly model anomaly-score deviation changes the condition percentage for every machine. Saving this resets in-memory condition histories so old and new scales are never blended.',higher:'Condition decreases more slowly, reducing sensitivity and usually producing fewer condition alerts.',lower:'Condition decreases faster, increasing sensitivity and potentially producing more alerts.',step:.1,min:.5,max:20},
    KALMAN_INIT_SAMPLES:{label:'Condition warm-up readings',desc:'Raw condition readings averaged before smoothed condition is first emitted.',unit:'readings',purpose:'Sets how many valid running readings seed the condition smoothing filter after startup, restart, or a condition-processing policy reset.',higher:'Produces a more stable initial condition but keeps the machine in warm-up longer.',lower:'Shows condition sooner but makes the initial value more sensitive to startup noise.',step:1,min:1,max:10000},
    MAINTENANCE_PROB_PLAN:{label:'Planned-maintenance risk',desc:'Model-estimated crossing risk required for a trend-based WARN.',unit:'0–1',purpose:'Issues a trend-based WARN when the Brownian first-passage estimate of crossing the critical condition boundary within the selected horizon reaches this value.',higher:'Requires stronger model evidence, producing fewer but later WARN alerts.',lower:'Warns on weaker model evidence, producing earlier but potentially noisier alerts.',step:.01,min:0,max:1},
    MAINTENANCE_PROB_URGENT:{label:'Urgent crossing risk',desc:'Model-estimated crossing risk required for a trend-based CRITICAL alert.',unit:'0–1',purpose:'Issues a trend-based CRITICAL alert when the estimated first-passage risk of crossing the critical condition boundary within the selected horizon reaches this value.',higher:'Requires stronger evidence before CRITICAL, reducing false alarms but delaying escalation.',lower:'Escalates earlier with less evidence, increasing sensitivity and false-alarm risk.',step:.01,min:0,max:1},
    MAINTENANCE_HORIZON_DAYS:{label:'Planned-maintenance horizon',desc:'Look-ahead used for the trend-based WARN decision.',unit:'days',purpose:'Selects how far ahead the model estimates critical-boundary crossing risk for planned maintenance. Supported horizons extend to 7 days; this is a model-based trend estimate and must be checked against real labelled outcomes.',higher:'Provides earlier planning visibility but increases uncertainty and can create more WARN candidates.',lower:'Focuses planning on nearer-term risk, usually warning later with less extrapolation.',choices:[.25,.5,.75,1,7]},
    MAINTENANCE_URGENT_HORIZON_DAYS:{label:'Urgent forecast horizon',desc:'Near-term look-ahead used only for trend-based CRITICAL decisions.',unit:'days',purpose:'Keeps urgent decisions within the next day and strictly shorter than the planning horizon. Seven-day risk can therefore produce WARN, never CRITICAL.',higher:'Uses more of the next 24 hours for urgent escalation, increasing sensitivity.',lower:'Requires the forecast risk to be more imminent before CRITICAL escalation.',choices:[.25,.5,.75,1]},
    TREND_MIN_POINTS:{label:'Minimum trend readings',desc:'Condition readings required before the first trend fit.',unit:'readings',purpose:'Prevents the trend model from fitting until this many smoothed condition readings are available.',higher:'Builds the trend from more evidence and reduces unstable early fits, but forecasting starts later.',lower:'Starts forecasting sooner with less evidence and more noise sensitivity.',step:1,min:3,max:720},
    TREND_LOOKBACK_MINUTES:{label:'Trend lookback',desc:'Actual elapsed-time history retained for degradation forecasting.',unit:'minutes',purpose:'Keeps smoothed condition readings whose database timestamps fall within this real elapsed-time window. It no longer assumes one row equals one minute.',higher:'Uses more history for a steadier but slower-changing forecast.',lower:'Reacts faster to recent changes but becomes more sensitive to short-term noise.',step:5,min:5,max:43200},
    TREND_SLOPE_Z_THRESHOLD:{label:'Trend confidence threshold',desc:'Evidence required before a trend may drive an alert.',unit:'z-score',purpose:'Requires the estimated degradation slope to be sufficiently large compared with its statistical uncertainty before forecast alerts are trusted.',higher:'Rejects more noisy trends and reduces false forecast alerts, but may miss slow early degradation.',lower:'Accepts weaker trends sooner, improving sensitivity but increasing noise-driven alerts.',step:.1,min:.1,max:10},
    TREND_SETTLE_TICKS:{label:'Trend stabilization readings',desc:'Trusted trend delay after enough trend points first become available.',unit:'readings',purpose:'Waits this many additional prediction readings before forecast crossing risk may trigger maintenance after the trend model first becomes available.',higher:'Provides more warm-up protection but delays trend-based alerts.',lower:'Enables forecasting sooner but increases startup-transient alert risk.',step:1,min:0},
    MAINTENANCE_WARN_CONFIRM_MINUTES:{label:'Warning confirmation',desc:'Continuous alert evidence required before WARN is shown.',unit:'minutes',purpose:'Suppresses brief candidates using source timestamps. Sustained CRITICAL evidence also enters WARN at this point while its longer critical confirmation continues.',higher:'Reduces warning noise but delays the first visible maintenance alert.',lower:'Shows warnings sooner but makes short disturbances more influential.',step:.5,min:0,max:1440},
    MAINTENANCE_CRITICAL_CONFIRM_MINUTES:{label:'Critical confirmation',desc:'Continuous CRITICAL evidence required before final escalation.',unit:'minutes',purpose:'Advances a sustained critical candidate from its warning stage to CRITICAL. It must be at least the warning confirmation duration and no longer than alert recovery.',higher:'Reduces critical noise but keeps a genuine dangerous condition at WARN for longer.',lower:'Escalates critical conditions sooner after WARN, but is more sensitive to persistent short disturbances.',step:.5,min:0,max:1440},
    MAINTENANCE_RECOVERY_MINUTES:{label:'Alert recovery',desc:'Continuous lower-severity evidence required before clearing.',unit:'minutes',purpose:'Prevents WARN and CRITICAL states from flapping when the condition oscillates around a decision boundary.',higher:'Keeps alerts latched longer and requires stronger recovery evidence.',lower:'Clears alerts sooner but can alternate states during noisy operation.',step:.5,min:0,max:1440},
    OPERATING_STATE_STOP_CONFIRM_TICKS:{label:'Shutdown confirmation readings',desc:'Consecutive stationary readings required to declare STOPPED.',unit:'readings',purpose:'Confirms an automatically learned low-vibration state for this many consecutive source readings before suppressing model scoring and clearing the machine monitor history.',higher:'Avoids false shutdown detection during short vibration dips, but recognizes a real shutdown later.',lower:'Stops scoring sooner after vibration falls, but brief dips can be mistaken for shutdown.',step:1,min:1},
    OPERATING_STATE_START_CONFIRM_TICKS:{label:'Restart confirmation readings',desc:'Consecutive rotating readings required to begin STARTING.',unit:'readings',purpose:'Confirms that vibration has returned to the learned rotating regime before beginning the restart warm-up sequence.',higher:'Requires a more persistent restart signal and rejects short motion bursts, but delays restart recognition.',lower:'Recognizes restart sooner, but short vibration bursts can begin an unnecessary warm-up.',step:1,min:1},
    SOURCE_STALE_SECONDS:{label:'Source stale timeout',desc:'Maximum source-row age before the website reports NO DATA.',unit:'seconds',purpose:'Marks a machine as having no fresh data when the newest database sensor row is older than this timeout.',higher:'Tolerates longer ingestion delays but can make an interrupted stream appear online longer.',lower:'Detects missing data sooner but may report NO DATA during normal ingestion delays.',step:1,min:1},
    WORKER_POLL_SECONDS:{label:'Database polling interval',desc:'How often the production worker checks for newly ingested source rows.',unit:'seconds',purpose:'Controls database polling frequency independently of the sensor sample cadence. All newly found rows are still processed in timestamp order.',higher:'Reduces database polling load but delays new readings appearing in monitoring.',lower:'Shows new rows sooner but performs more frequent database queries.',step:1,min:1,max:3600},
    NEAR_MISS_TREND_WINDOW_HOURS:{label:'Near-miss trend window',desc:'Default elapsed-time window used to identify declining anomaly behavior.',unit:'hours',purpose:'Defines the real timestamp range used by Status Review and History when deciding whether an otherwise-OK prediction is a near miss.',higher:'Uses broader context and detects slower changes, but may mix different operating periods.',lower:'Focuses on recent changes, reacting faster but becoming more noise-sensitive.',step:.25,min:.25,max:168},
  };
  const groups=[
    {title:'Condition response',desc:'How anomaly deviation becomes a stable condition score and immediate decision.',keys:['HEALTH_SENSITIVITY_STD','KALMAN_INIT_SAMPLES','FAILURE_HEALTH_THRESHOLD','MAINTENANCE_HEALTH_INSPECT']},
    {title:'Failure forecast',desc:'Separate planning and urgent horizons keep a one-week outlook from becoming an urgent alarm.',keys:['MAINTENANCE_PROB_PLAN','MAINTENANCE_HORIZON_DAYS','MAINTENANCE_PROB_URGENT','MAINTENANCE_URGENT_HORIZON_DAYS']},
    {title:'Status stability',desc:'Timestamp-based confirmation and recovery applied consistently to direct condition and forecast decisions.',keys:['MAINTENANCE_WARN_CONFIRM_MINUTES','MAINTENANCE_CRITICAL_CONFIRM_MINUTES','MAINTENANCE_RECOVERY_MINUTES']},
    {title:'Trend reliability',desc:'Elapsed history, evidence, and warm-up required before forecast risk may influence status.',keys:['TREND_LOOKBACK_MINUTES','TREND_MIN_POINTS','TREND_SLOPE_Z_THRESHOLD','TREND_SETTLE_TICKS']},
    {title:'Machine availability',desc:'Source polling, shutdown, restart, and missing-data timing.',keys:['WORKER_POLL_SECONDS','OPERATING_STATE_STOP_CONFIRM_TICKS','OPERATING_STATE_START_CONFIRM_TICKS','SOURCE_STALE_SECONDS']},
    {title:'Review analysis',desc:'Default timestamp-based evidence window used by the near-miss queue and History.',keys:['NEAR_MISS_TREND_WINDOW_HOURS']},
  ];
  if(error)return <><PageHeader eyebrow="FLEET POLICY" title="Global thresholds" description="Maintenance policy applied consistently to every machine."/><ScopeBanner scope="fleet"/><Notice tone="critical">{error}</Notice></>;
  if(!v)return <Loading/>;
  const save=async()=>{setSaved(false);setSaveError('');setSaving(true);try{const next=await api('/api/config/thresholds',{method:'PUT',body:JSON.stringify(v)});setV(next);setSaved(true)}catch(e){setSaveError(String((e as any).message||e))}finally{setSaving(false)}};
  const info=infoKey?meta[infoKey]:undefined;
  return <><PageHeader eyebrow="FLEET POLICY" title="Global thresholds" description="Tune writable runtime policy applied consistently to every machine." actions={admin?<button className="primary" disabled={saving} onClick={save}>{saving?'Saving…':'Save fleet policy'}</button>:<StatusBadge value="READ ONLY"/>}/><ScopeBanner scope="fleet"/>{saved&&<Notice tone="normal">Policy saved without model retraining. Changes to condition sensitivity, warm-up, or trend history reset active monitors and warm them up again.</Notice>}{saveError&&<Notice tone="critical">{saveError}</Notice>}
  <section className="settings-layout">{groups.map(group=><section className="settings-surface settings-group" key={group.title}><header className="settings-group-header"><div><strong>{group.title}</strong><span>{group.desc}</span></div><small>{group.keys.length} setting{group.keys.length===1?'':'s'}</small></header>{group.keys.map(k=>{const item=meta[k];const choices=item.choices?.filter(choice=>k==='MAINTENANCE_HORIZON_DAYS'?choice>Number(v.MAINTENANCE_URGENT_HORIZON_DAYS):k==='MAINTENANCE_URGENT_HORIZON_DAYS'?choice<Number(v.MAINTENANCE_HORIZON_DAYS)&&choice<=1:true);return <div className="setting-row" key={k}><div><SettingTitle item={item} onInfo={()=>setInfoKey(k)}/><span>{item.desc}</span><code>{k}</code></div><div className="setting-value"><span>Current value</span><div className="setting-control">{choices?<select value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}>{choices.map(choice=><option key={choice} value={choice}>{choice}</option>)}</select>:<input type="number" step={item.step??'any'} min={item.min} max={item.max} value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}/>}<span>{item.unit}</span></div></div></div>})}</section>)}<section className="settings-surface settings-help"><Disclosure title="How these thresholds are applied"><div className="explain-grid"><div><b>Backend authority</b><span>Values are range-checked and relationship rules are validated again before saving.</span></div><div><b>No model retrain</b><span>These are runtime policy controls, not learned model or normalization parameters.</span></div><div><b>Fleet scope</b><span>The same saved policy applies to every commissioned machine.</span></div></div></Disclosure></section></section>
  {info&&<SettingInfoDialog item={info} value={infoKey?v[infoKey]:''} onClose={()=>setInfoKey(null)}/>}
  </>
}

function Environment(){
  const [v,setV]=useState<any>(),[error,setError]=useState(''),[saved,setSaved]=useState<any>(); useEffect(()=>{api('/api/env').then(setV).catch(e=>setError(String(e.message||e)))},[]);
  if(error)return <><PageHeader eyebrow="FLEET SYSTEM" title="Environment" description="Shared database and runtime configuration for the fleet."/><ScopeBanner scope="fleet"/><Notice tone="critical">{error}</Notice></>;
  if(!v)return <Loading/>;
  const primary=['PG_HOST','PG_PORT','PG_DATABASE','PG_USER','PG_PASSWORD','PG_TABLE']; const extra=Object.keys(v).filter(k=>!primary.includes(k));
  const envHelp:Record<string,string>={
    PG_PASSWORD:'Stored database password is masked and never returned to the browser.',
    APP_SECRET_KEY:'Signing secret is masked, must contain at least 32 characters, and invalidates existing sessions when changed.',
    BOOTSTRAP_ADMIN_PASSWORD:'Masked credential used only when no application user exists; it must be unique and at least 12 characters.',
    APP_SESSION_SECONDS:'Login lifetime for newly issued sessions: 300–604800 seconds.',
    PG_HOST:'Database server hostname or IP address.',
    PG_PORT:'PostgreSQL defaults to 5432.',
    COOKIE_SECURE:'Use true when the website is served over HTTPS.',
  };
  const field=(k:string)=>{const secret=k.includes('PASSWORD')||k.includes('SECRET');return <label className="env-field" key={k}><span>{k.replaceAll('_',' ')}</span><input type={secret?'password':k==='APP_SESSION_SECONDS'?'number':'text'} min={k==='APP_SESSION_SECONDS'?300:undefined} max={k==='APP_SESSION_SECONDS'?604800:undefined} value={v[k]??''} autoComplete={secret?'new-password':undefined} onChange={e=>setV({...v,[k]:e.target.value})}/><small>{envHelp[k]||''}</small></label>};
  return <><PageHeader eyebrow="FLEET SYSTEM" title="Environment" description="Shared database and runtime configuration used by the API and all machine workers." actions={<button className="primary" onClick={async()=>setSaved(await api('/api/env',{method:'PUT',body:JSON.stringify({values:v})}))}>Save environment</button>}/><ScopeBanner scope="fleet"/>{saved&&<Notice tone="normal">Saved {saved.changed_keys?.length||0} changed key(s). {saved.worker_restart_requested?'Worker restart requested.':'No worker restart required.'}</Notice>}
  <section className="settings-surface"><div className="environment-grid">{primary.filter(k=>k in v).map(field)}</div>{extra.length>0&&<Disclosure title={`Advanced variables (${extra.length})`}>{<div className="environment-grid advanced">{extra.map(field)}</div>}</Disclosure>}<div className="env-note"><Icon name="shield"/><div><b>Credential handling</b><span>Database passwords, application signing secrets, and bootstrap passwords are masked and never returned to the browser.</span></div></div></section>
  </>
}

function Empty({title,text}:{title:string;text:string}){return <div className="empty"><div className="empty-icon"><Icon name="pulse"/></div><strong>{title}</strong><span>{text}</span></div>}
function Loading(){return <div className="loading-row"><div className="spinner small"/><span>Loading from backend…</span></div>}

const appRoot=createRoot(document.getElementById('root')!);
appRoot.render(<ErrorBoundary><App/></ErrorBoundary>);

// A full root teardown prevents effects from an older development bundle
// continuing to poll after Vite replaces this entry module.
if(import.meta.hot){
  import.meta.hot.dispose(()=>appRoot.unmount());
}
