import React, { Component, ReactNode, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import {
  ChartLineUpIcon, CheckIcon, ClockCounterClockwiseIcon, CubeIcon, DatabaseIcon,
  DotsThreeIcon, FactoryIcon, GaugeIcon, GearIcon, HouseIcon, InfoIcon,
  MagnifyingGlassIcon, ShieldCheckIcon, SignOutIcon, SlidersHorizontalIcon,
  SquaresFourIcon, TrashIcon, WarningIcon, WaveSineIcon, XIcon, CaretRightIcon,
} from '@phosphor-icons/react';
import './styles.css';

type Page = 'Home' | 'Machine Overview' | 'Status Review' | 'Models' | 'History' | 'Environment' | 'Thresholds';
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
  prediction_wait_reason?: string;
  operating_state?: string;
  operating_state_reason?: string;
  operating_state_confidence?: number;
  operating_state_changed_at?: string;
  operating_state_activity?: number;
  operating_state_stop_threshold?: number;
  operating_state_run_threshold?: number;
  health_state?: number;
  anomaly_score?: number;
  model_version?: string;
  maintenance?: { level: string; reason?: string; trigger?: string };
  a_rms_mps2?: number;
  v_rms_mms?: number;
  a_peak_mps2?: number;
  crest_factor?: number;
  temperature_c?: number;
  mock_mode?: boolean;
};
type FleetTrendPoint = { timestamp:string; health_state:number; samples:number };
type FleetTrendResponse = {
  days:number;
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
  if (x === 'CRITICAL' || x === 'FAILED' || x === 'REJECTED' || x === 'FAIL') return 'critical';
  if (x === 'SENSOR_FAULT' || x === 'NO_DATA') return 'critical';
  if (x === 'STARTING' || x === 'STOPPED' || x === 'QUEUED') return 'warning';
  if (x === 'WARN' || x === 'WARNING') return 'warning';
  if (x === 'OK' || x === 'NORMAL' || x === 'ACTIVE' || x === 'RUNNING' || x === 'PASSED' || x === 'PASS') return 'normal';
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
  check:CheckIcon,trash:TrashIcon,machine:FactoryIcon,gauge:GaugeIcon,settings:GearIcon,
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

function App(){
  const [me,setMe]=useState<User|false|undefined>(); const [apiError,setApiError]=useState<string>(); const [page,setPage]=useState<Page>('Home'); const [collapsed,setCollapsed]=useState(false);
  const [machines,setMachines]=useState<string[]>([]); const [machinesLoaded,setMachinesLoaded]=useState(false); const [machineId,setMachineId]=useState(localStorage.getItem('machine_id')||''); const [sourceError,setSourceError]=useState('');
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
  if(me===undefined)return <main className="center-screen"><section className="state-panel"><div className="spinner"/><p className="eyebrow">INITIALIZING CONSOLE</p><h1>Connecting to the local API</h1><p className="muted">If this takes more than a few seconds, check <code>localhost:8000/docs</code>.</p></section></main>;
  if(!me)return <Login done={refreshMe} apiError={apiError}/>;
  const fleetNav:Array<{page:Page;icon:string;label:string}>=[
    {page:'Home',icon:'home',label:'Home'},{page:'Models',icon:'model',label:'Models & retraining'},{page:'Thresholds',icon:'sliders',label:'Global thresholds'},...(me.role==='admin'?[{page:'Environment' as Page,icon:'database',label:'Environment'}]:[])
  ];
  const machineNav:Array<{page:Page;icon:string;label:string}>=[
    {page:'Machine Overview',icon:'gauge',label:'Overview'},{page:'Status Review',icon:'alert',label:'Status review'},{page:'History',icon:'history',label:'History'},
  ];
  const machineScoped=machineNav.some(item=>item.page===page);
  const goMachine=(id:string)=>{setMachineId(id);setPage('Machine Overview')};
  const pageLabel=page==='Machine Overview'?'Overview':page;
  return <div className={cn('app-shell',collapsed&&'nav-collapsed')}>
    <aside className="sidebar">
      <div className="brand-lockup"><div className="brand-symbol"><Icon name="pulse"/></div><div className="brand-copy"><small>AKEBONO</small><strong>Spindle Monitor</strong><span>Fleet condition console</span></div></div>
      <div className="mobile-machine-picker"><Select ariaLabel="Select machine" value={machineId} onChange={goMachine} options={(machines.length?machines:['']).map(id=>[id,id||'No database machines'] as [string,string])}/></div>
      <button type="button" className="collapse-btn" onClick={()=>setCollapsed(v=>!v)} title={collapsed?'Expand navigation':'Collapse navigation'} aria-label={collapsed?'Expand navigation':'Collapse navigation'} aria-expanded={!collapsed} data-label={collapsed?'Expand':'Collapse'}><Icon name="chevron"/></button>
      <nav className="scope-nav">
        <section className="nav-group"><p>FLEET</p>{fleetNav.map(n=><button key={n.page} className={cn('nav-item',page===n.page&&'active')} onClick={()=>setPage(n.page)} title={n.label}><Icon name={n.icon}/><span>{n.label}</span>{page===n.page&&<i/>}</button>)}</section>
        <section className="nav-group machine-group"><div className="nav-group-title"><p>MACHINES</p><span>{machines.length}</span></div><MachineNavList machines={machines} selected={machineId} active={machineScoped} onSelect={goMachine}/>{machineId&&<div className="machine-subnav"><small>{machineId} WORKSPACE</small>{machineNav.map(n=><button key={n.page} className={cn('nav-item',page===n.page&&'active')} onClick={()=>setPage(n.page)} title={`${n.label} · ${machineId}`}><Icon name={n.icon}/><span>{n.label}</span>{page===n.page&&<i/>}</button>)}</div>}</section>
      </nav>
      <div className="sidebar-bottom">
        {me.mock_mode&&<button className="demo-chip" title="This session is using temporary synthetic data"><span className="dot"/><span>Mock data</span></button>}
        <div className="user-chip"><div className="avatar">{me.username.slice(0,2).toUpperCase()}</div><div><strong>{me.username}</strong><span>{me.role}</span></div><button title="Sign out" onClick={async()=>{await api('/api/logout',{method:'POST'});setMe(false)}}><Icon name="logout"/></button></div>
      </div>
    </aside>
    <main className="workspace">
      <header className="topbar"><p className="breadcrumb">{machineScoped?<>MACHINES <span>/</span> {machineId} <span>/</span> {pageLabel}</>:<>FLEET <span>/</span> {pageLabel}</>}</p><div className="topbar-meta"><span className="data-freshness"><Icon name="database"/>{me.mock_mode?'Demo source':'Production source'}</span><span className="system-pill"><span className="pulse-dot"/>API online</span><span className="clock">{new Date().toLocaleDateString([], {weekday:'short',month:'short',day:'numeric'})}</span></div></header>
      <div className="content">{page==='Environment'?<Environment/>:!machinesLoaded?<Loading/>:sourceError?<><PageHeader eyebrow="DATABASE SOURCE" title="PostgreSQL connection error" description="The website could not discover machine IDs from the configured raw source." actions={me.role==='admin'?<button className="primary" onClick={()=>setPage('Environment')}>Check environment</button>:undefined}/><Notice tone="critical">{sourceError}</Notice></>:page==='Home'?<FleetHome machines={machines} openMachine={goMachine}/>:machineId?<PageView key={`${machineId}-${page}`} page={page} role={me.role} mock={!!me.mock_mode} machineId={machineId}/>:<Loading/>}</div>
    </main>
  </div>
}

function PageHeader({ eyebrow, title, description, actions }: { eyebrow:string;title:string;description:string;actions?:ReactNode }) {
  return <div className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>{actions&&<div className="page-actions">{actions}</div>}</div>
}

function ScopeBanner({scope,machineId}:{scope:'fleet'|'machine';machineId?:string}){
  return <div className={cn('scope-banner',`scope-${scope}`)}><Icon name={scope==='fleet'?'shield':'machine'}/><div><strong>{scope==='fleet'?'Fleet scope':`${machineId} scope`}</strong><span>{scope==='fleet'?'Changes here apply across all machines.':'Data and settings here apply only to this machine.'}</span></div></div>;
}

function MachineNavList({machines,selected,active,onSelect}:{machines:string[];selected:string;active:boolean;onSelect:(id:string)=>void}){
  const [states,setStates]=useState<Record<string,string>>({});
  useEffect(()=>{let disposed=false;const load=()=>Promise.allSettled(machines.map(id=>api(`/api/live/latest?machine_id=${encodeURIComponent(id)}`,{},5000))).then(results=>{if(disposed)return;const next:Record<string,string>={};results.forEach((result,index)=>{next[machines[index]]=result.status==='fulfilled'?String((result.value as LiveTick).operating_state||'UNKNOWN'):'NO_DATA'});setStates(next)});load();const timer=window.setInterval(load,15000);return()=>{disposed=true;window.clearInterval(timer)}},[machines.join('|')]);
  return <div className="machine-list">{machines.map(id=><button key={id} className={cn('machine-link',id===selected&&active&&'active')} onClick={()=>onSelect(id)} title={`${id} · ${states[id]||'Loading state'}`}><span className={cn('machine-state-dot',tone(states[id]))}/><span>{id}</span><Icon name="chevron"/></button>)}</div>;
}

const FLEET_LINE_COLORS=['#1769c2','#d47b12','#23875e','#b43b64','#7257c8','#17889b','#bf4b32','#596b82','#8a6d13','#2672a4'];
const fleetLineColor=(index:number)=>FLEET_LINE_COLORS[index]||`hsl(${Math.round((index*137.508+211)%360)} 64% 42%)`;
function FleetTrendChart({machines,data}:{machines:string[];data:FleetTrendResponse}){
  const width=1000,height=258,left=48,right=18,top=16,bottom=38;
  const start=new Date(data.start).getTime(),end=new Date(data.end).getTime();
  const span=Math.max(1,end-start),plotW=width-left-right,plotH=height-top-bottom;
  const byMachine=new Map(data.series.map(series=>[series.machine_id,series.points]));
  const ids=Array.from(new Set([...machines,...data.series.map(series=>series.machine_id)])).sort();
  const validCount=data.series.reduce((sum,series)=>sum+series.points.length,0);
  const x=(timestamp:string)=>left+Math.max(0,Math.min(1,(new Date(timestamp).getTime()-start)/span))*plotW;
  const y=(health:number)=>top+(1-Math.max(0,Math.min(100,health))/100)*plotH;
  const dayTicks=Array.from({length:data.days+1},(_,index)=>start+(span*index/data.days));
  if(validCount===0)return <Empty title="No seven-day condition history" text="The worker has not stored condition predictions in this period."/>;
  return <div className="fleet-trend-body">
    <div className="fleet-trend-scroll"><svg className="fleet-trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Seven-day condition trend for all machines">
      {([0,25,50,75,100] as number[]).map(value=><g key={value}><line className="trend-grid-line" x1={left} x2={width-right} y1={y(value)} y2={y(value)}/><text className="trend-y-label" x={left-10} y={y(value)+3} textAnchor="end">{value}</text></g>)}
      {dayTicks.map((tick,index)=><g key={tick}><line className="trend-day-line" x1={left+(index/data.days)*plotW} x2={left+(index/data.days)*plotW} y1={top} y2={height-bottom}/><text className="trend-x-label" x={left+(index/data.days)*plotW} y={height-13} textAnchor={index===0?'start':index===data.days?'end':'middle'}>{new Date(tick).toLocaleDateString([], {month:'short',day:'numeric'})}</text></g>)}
      <text className="trend-axis-title" transform={`translate(12 ${top+plotH/2}) rotate(-90)`} textAnchor="middle">Condition %</text>
      {ids.map((machineId,index)=>{const points=(byMachine.get(machineId)||[]).filter(point=>Number.isFinite(Number(point.health_state))&&!Number.isNaN(new Date(point.timestamp).getTime()));if(points.length===0)return null;const color=fleetLineColor(index);const path=points.map((point,pointIndex)=>{const previous=points[pointIndex-1],gap=previous?new Date(point.timestamp).getTime()-new Date(previous.timestamp).getTime():0;return `${pointIndex&&gap<=2*60*60*1000?'L':'M'} ${x(point.timestamp).toFixed(2)} ${y(Number(point.health_state)).toFixed(2)}`}).join(' ');const latest=points[points.length-1];return <g key={machineId}><path className="fleet-machine-line" d={path} stroke={color}><title>{machineId}</title></path><circle className="fleet-machine-endpoint" cx={x(latest.timestamp)} cy={y(Number(latest.health_state))} r="3.5" fill={color}><title>{`${machineId}: ${fmt(latest.health_state,1)}% at ${shortTime(latest.timestamp)}`}</title></circle></g>})}
    </svg></div>
    <div className="fleet-trend-legend" aria-label="Machine color key">{ids.map((machineId,index)=>{const points=byMachine.get(machineId)||[],latest=points[points.length-1];return <span key={machineId} className={points.length?'':'unavailable'}><i style={{background:fleetLineColor(index)}}/><b>{machineId}</b><small>{latest?`${fmt(latest.health_state,1)}% latest`:'No data'}</small></span>})}</div>
    <p>Each line is one machine’s hourly average condition score. Machines are never averaged together; gaps indicate periods without stored predictions.</p>
  </div>;
}

function PageView({page,role,mock,machineId}:{page:Page;role:string;mock:boolean;machineId:string}){
  if(page==='Machine Overview')return <Dashboard mock={mock} machineId={machineId}/>;
  if(page==='Status Review')return <StatusReview machineId={machineId}/>;
  if(page==='Models')return <Models admin={role==='admin'}/>;
  if(page==='History')return <HistoryPage machineId={machineId}/>;
  if(page==='Thresholds')return <Thresholds admin={role==='admin'}/>;
  return <Environment/>;
}

function FleetHome({machines,openMachine}:{machines:string[];openMachine:(id:string)=>void}){
  const [rows,setRows]=useState<LiveTick[]>([]),[loading,setLoading]=useState(true),[error,setError]=useState('');
  const [trend,setTrend]=useState<FleetTrendResponse>(),[trendLoading,setTrendLoading]=useState(true),[trendError,setTrendError]=useState('');
  useEffect(()=>{
    let disposed=false;setLoading(true);setError('');
    const load=()=>Promise.allSettled(machines.map(id=>api(`/api/live/latest?machine_id=${encodeURIComponent(id)}`,{},5000)))
      .then(results=>{if(disposed)return;const ok=results.filter((r):r is PromiseFulfilledResult<LiveTick>=>r.status==='fulfilled').map(r=>r.value);setRows(ok.sort((a,b)=>a.machine_id.localeCompare(b.machine_id)));const failed=results.length-ok.length;setError(failed?`${failed} machine${failed===1?' is':'s are'} temporarily unavailable.`:'')})
      .finally(()=>{if(!disposed)setLoading(false)});
    load();const timer=window.setInterval(load,10000);return()=>{disposed=true;window.clearInterval(timer)};
  },[machines.join('|')]);
  useEffect(()=>{
    let disposed=false;
    const load=()=>api('/api/fleet/condition-trend?days=7',{},15000).then((response:FleetTrendResponse)=>{if(!disposed){setTrend(response);setTrendError('')}}).catch(e=>{if(!disposed)setTrendError(String(e.message||e))}).finally(()=>{if(!disposed)setTrendLoading(false)});
    load();const timer=window.setInterval(load,300000);return()=>{disposed=true;window.clearInterval(timer)};
  },[]);
  const running=rows.filter(r=>String(r.operating_state).toUpperCase()==='RUNNING').length;
  const stopped=rows.filter(r=>String(r.operating_state).toUpperCase()==='STOPPED').length;
  const critical=rows.filter(r=>String(r.maintenance?.level).toUpperCase()==='CRITICAL').length;
  const attention=rows.filter(r=>['WARN','CRITICAL'].includes(String(r.maintenance?.level).toUpperCase())||['NO_DATA','SENSOR_FAULT'].includes(String(r.operating_state).toUpperCase()));
  return <>
    <PageHeader eyebrow="FLEET HOME" title="Operations dashboard" description="Shared fleet visibility across every machine in the production source. Open a machine to work in its individual context."/>
    {error&&<Notice tone="warning">{error}</Notice>}
    <section className="fleet-summary" aria-label="Fleet summary">
      <div><strong>{machines.length}</strong><span>Total machines</span></div><div className="normal"><strong>{running}</strong><span>Running</span></div><div className="warning"><strong>{stopped}</strong><span>Stopped</span></div><div className={critical?'critical':'normal'}><strong>{critical}</strong><span>Critical</span></div>
    </section>
    <section className="fleet-table data-surface">
      <div className="surface-heading"><span>Machine status</span><small>Shared overview · select a row to enter its workspace</small></div>
      {loading?<Loading/>:rows.length===0?<Empty title="No live machine snapshots" text="The production source did not return a current machine state."/>:<div className="responsive-table"><table><thead><tr><th>Machine</th><th>Operating state</th><th>Maintenance</th><th>Condition</th><th>Latest data</th><th>Active model</th><th/></tr></thead><tbody>{rows.map(row=><tr key={row.machine_id} onClick={()=>openMachine(row.machine_id)}><td><strong>{row.machine_id}</strong><small>Machine workspace</small></td><td><StatusBadge value={row.operating_state||'UNKNOWN'}/></td><td><StatusBadge value={row.maintenance?.level||(row.prediction_wait_reason?'PROCESSING':'AWAITING MODEL')}/></td><td><strong>{row.prediction_available!==false&&row.health_state!==undefined?`${fmt(row.health_state,1)}%`:'—'}</strong></td><td>{shortTime(row.timestamp)}</td><td>{row.model_version||'—'}</td><td><button className="row-action" onClick={e=>{e.stopPropagation();openMachine(row.machine_id)}}>Open <Icon name="chevron"/></button></td></tr>)}</tbody></table></div>}
    </section>
    <div className="fleet-support-grid">
      <section className="fleet-trend data-surface"><div className="surface-heading"><span>7-day condition trend</span><small>Hourly fleet recap · one line per machine</small></div>{trendError&&!trend&&<Notice tone="warning">Condition history unavailable: {trendError}</Notice>}{trendLoading&&!trend?<Loading/>:trend?<FleetTrendChart machines={machines} data={trend}/>:null}</section>
      <section className="attention-list data-surface"><div className="surface-heading"><span>Needs attention</span><small>{attention.length} machine{attention.length===1?'':'s'}</small></div>{attention.length===0?<div className="attention-empty"><Icon name="check"/><div><strong>No fleet exceptions</strong><span>All connected machines are within the current policy.</span></div></div>:attention.map(row=><button key={row.machine_id} onClick={()=>openMachine(row.machine_id)}><StatusBadge value={row.maintenance?.level||row.operating_state||'CHECK'}/><div><strong>{row.machine_id}</strong><span>{row.operating_state_reason||row.maintenance?.reason}</span></div><Icon name="chevron"/></button>)}</section>
    </div>
  </>;
}

function Sparkline({values, inverse=false}:{values:number[];inverse?:boolean}){
  const clean=values.filter(Number.isFinite); if(clean.length<2)return <svg className="sparkline" viewBox="0 0 200 54"/>;
  const min=Math.min(...clean),max=Math.max(...clean),span=max-min||1;
  const pts=clean.map((v,i)=>`${(i/(clean.length-1))*200},${48-((v-min)/span)*40}`).join(' ');
  return <svg className={cn('sparkline',inverse&&'inverse')} viewBox="0 0 200 54" preserveAspectRatio="none"><polyline points={pts}/></svg>
}

function Dashboard({mock,machineId}:{mock:boolean;machineId:string}){
  const [live,setLive]=useState<LiveTick>(); const [history,setHistory]=useState<LiveTick[]>([]); const [wsState,setWsState]=useState<'connecting'|'live'|'offline'>('connecting'); const [sourceState,setSourceState]=useState<'connecting'|'live'|'offline'>('connecting'); const [sourceError,setSourceError]=useState(''); const [detail,setDetail]=useState<string|null>(null); const [menu,setMenu]=useState(false);
  useEffect(()=>{
    let disposed=false;
    setLive(undefined);setHistory([]);setWsState('connecting');setSourceState('connecting');setSourceError('');
    const record=(row:LiveTick)=>{if(disposed)return;setLive(row);setHistory(items=>{const withoutSame=items.filter(item=>item.timestamp!==row.timestamp);return[...withoutSame.slice(-39),row]})};
    const loadSnapshot=()=>api(`/api/live/latest?machine_id=${encodeURIComponent(machineId)}`,{},5000).then((row:LiveTick)=>{record(row);setSourceState('live');setSourceError('')}).catch(e=>{if(!disposed){setSourceState('offline');setSourceError(String(e.message||e))}});
    loadSnapshot();const poll=window.setInterval(loadSnapshot,5000);
    const proto=location.protocol==='https:'?'wss':'ws';const ws=new WebSocket(`${proto}://${location.host}/ws/live?machine_id=${encodeURIComponent(machineId)}`);
    ws.onopen=()=>setWsState('live');
    ws.onmessage=e=>{const parsed=JSON.parse(e.data);const row={...parsed,prediction_available:parsed.prediction_available!==false,source:mock?'mock':'postgresql'};record(row);setWsState('live')};
    ws.onerror=()=>setWsState('offline');ws.onclose=()=>setWsState('offline');
    return()=>{disposed=true;window.clearInterval(poll);ws.close()};
  },[machineId,mock]);
  const operatingState=String(live?.operating_state||'UNKNOWN').toUpperCase();
  const operatingGate=['STOPPED','STARTING','SENSOR_FAULT','NO_DATA'].includes(operatingState);
  const predictionAvailable=!operatingGate&&!!live?.maintenance&&live?.prediction_available!==false;
  const level=operatingGate?operatingState:predictionAvailable?(live?.maintenance?.level||'WAITING'):(live?.prediction_wait_reason?'PROCESSING':live&&operatingState!=='UNKNOWN'?operatingState:live?'AWAITING MODEL':'WAITING'); const health=Number(live?.health_state||0); const anomaly=Number(live?.anomaly_score||0);
  const statusReason=predictionAvailable?live?.maintenance?.reason:(live?.prediction_wait_reason||live?.operating_state_reason||'Real sensor data is connected. Waiting for the ML worker to produce a prediction.');
  const sensors=Object.keys(SENSOR_META).map(k=>({key:k,value:(live as any)?.[k],...SENSOR_META[k]}));
  return <>
    <PageHeader eyebrow="MACHINE OVERVIEW" title={machineId} description="Latest sensor readings from the configured PostgreSQL source, combined with ML results when applicable." actions={<><span className={cn('connection-badge',sourceState)}><span/>{sourceState==='live'?(operatingState==='NO_DATA'?'No recent data':mock?'Mock live':wsState==='live'?'PostgreSQL + worker live':'PostgreSQL live'):sourceState==='connecting'?'Connecting…':'Database offline'}</span><div className="dropdown-wrap"><button className="icon-button" onClick={()=>setMenu(v=>!v)}><Icon name="more"/></button>{menu&&<div className="dropdown-menu right"><button onClick={()=>{setDetail('system');setMenu(false)}}>System details</button><button onClick={()=>{setDetail('model');setMenu(false)}}>Model context</button><button onClick={()=>location.reload()}>Reconnect interface</button></div>}</div></>}/>
    <ScopeBanner scope="machine" machineId={machineId}/>
    {sourceError&&<Notice tone="critical">Real database sync failed: {sourceError}</Notice>}
    {mock&&<div className="demo-ribbon"><span>DEMO</span><p>Temporary synthetic signal is driving this interface. ML inference and plant PostgreSQL are bypassed.</p><button onClick={()=>setDetail('demo')}>What is simulated?</button></div>}
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
  </>
}

function DashboardModal({kind,live,onClose}:{kind:string;live?:LiveTick;onClose:()=>void}){
  let title='Details',body:ReactNode=null;
  if(kind==='status'){title='Machine state evidence';body=<div className="detail-list"><DetailRow label="Operating state" value={live?.operating_state||'UNKNOWN'} badge={tone(live?.operating_state)}/><DetailRow label="State reason" value={live?.operating_state_reason||'No state reason received.'}/><DetailRow label="State confidence" value={fmt(live?.operating_state_confidence,3)}/><DetailRow label="Activity / stop / run" value={`${fmt(live?.operating_state_activity,3)} / ${fmt(live?.operating_state_stop_threshold,3)} / ${fmt(live?.operating_state_run_threshold,3)}`}/><DetailRow label="State changed" value={shortTime(live?.operating_state_changed_at)}/><DetailRow label="Maintenance state" value={live?.maintenance?.level||'—'} badge={tone(live?.maintenance?.level)}/><DetailRow label="Maintenance trigger" value={triggerLabel(live?.maintenance?.trigger)}/><DetailRow label="Condition score" value={live?.prediction_available===false?'—':`${fmt(live?.health_state,2)}%`}/><DetailRow label="Anomaly score" value={live?.prediction_available===false?'—':fmt(live?.anomaly_score,4)}/></div>}
  else if(kind==='anomaly'){title='How to read anomaly score';body=<><p>The anomaly score summarizes how unusual the current feature pattern is relative to the active model reference. The frontend only displays this value; it does not calculate or modify it.</p><div className="scale"><span>Lower deviation</span><i/><span>Higher deviation</span></div><p className="muted">Current score: <b>{fmt(live?.anomaly_score,4)}</b></p></>}
  else if(kind==='model'||kind==='system'){title=kind==='model'?'Active model context':'Runtime context';body=<div className="detail-list"><DetailRow label="Machine ID" value={live?.machine_id||'—'}/><DetailRow label="Sensor model" value={SENSOR_MODEL}/><DetailRow label="Operating state" value={live?.operating_state||'UNKNOWN'} badge={tone(live?.operating_state)}/><DetailRow label="Model version" value={live?.model_version||'—'}/><DetailRow label="Latest source tick" value={shortTime(live?.timestamp)}/><DetailRow label="Latest prediction tick" value={shortTime(live?.prediction_timestamp)}/><DetailRow label="Prediction lag" value={live?.prediction_lag_seconds===undefined?'—':`${fmt(live.prediction_lag_seconds,0)} seconds`}/><DetailRow label="Source age" value={live?.source_age_seconds===undefined?'—':`${fmt(live.source_age_seconds,0)} seconds`}/><DetailRow label="Data path" value="Backend → WebSocket → browser"/><DetailRow label="Frontend role" value="Visualization only"/></div>}
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

type EditableSettingMeta={label:string;desc:string;unit:string;purpose:string;higher:string;lower:string;step?:number;min?:number;max?:number;higherLabel?:string;lowerLabel?:string};
function SettingTitle({item,onInfo}:{item:EditableSettingMeta;onInfo:()=>void}){return <div className="setting-title"><strong>{item.label}</strong><button type="button" className="setting-info-button" aria-label={`About ${item.label}`} title={`About ${item.label}`} onClick={onInfo}><Icon name="info"/></button></div>}
function SettingInfoDialog({item,value,onClose}:{item:EditableSettingMeta;value:any;onClose:()=>void}){return createPortal(<Modal title={item.label} onClose={onClose}><p className="setting-purpose">{item.purpose}</p><div className="current-setting-value"><span>Current value</span><div><strong>{String(value)}</strong><small>{item.unit}</small></div></div><div className="setting-impact-grid"><section><span>{item.higherLabel||'If increased'}</span><p>{item.higher}</p></section><section><span>{item.lowerLabel||'If decreased'}</span><p>{item.lower}</p></section></div></Modal>,document.body)}

function StatusReview({machineId}:{machineId:string}){
  const [tab,setTab]=useState<'alert'|'near'>('alert'); const [live,setLive]=useState<LiveTick>();
  useEffect(()=>{let disposed=false;const load=()=>api(`/api/live/latest?machine_id=${encodeURIComponent(machineId)}`,{},5000).then((row:LiveTick)=>{if(!disposed)setLive(row)}).catch(()=>{});load();const timer=window.setInterval(load,5000);return()=>{disposed=true;window.clearInterval(timer)}},[machineId]);
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
  {(selected.review_status||'pending')==='flagged'&&<p className="muted small flag-note">Flagging creates a suspected-false-negative regression check pinned to this exact prediction timestamp (see Models → validation tests).</p>}
  </Modal>}
  </>
}

function Select({value,onChange,options,ariaLabel}:{value:string;onChange:(v:string)=>void;options:Array<[string,string]>;ariaLabel?:string}){return <label className="select-wrap"><select aria-label={ariaLabel} value={value} onChange={e=>onChange(e.target.value)}>{options.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select><span>⌄</span></label>}
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
function validationLabel(model:any){const report=parsedObject(model.validation_report);if(report.passed===true)return 'Passed';if(report.passed===false)return 'Failed';return model.status==='active'?'Legacy / active':'Not recorded'}

function ValidationReport({report:raw}:{report:any}){
  const report=parsedObject(raw),machines=report.reference_fp_by_machine||report.per_machine||{},tests=report.regression_tests||[];
  if(!Object.keys(report).length)return <Notice>No validation evidence is stored for this model version.</Notice>;
  const verdict=report.passed===true?'PASSED':report.passed===false?'FAILED':'NOT RECORDED';
  return <div className="validation-report"><div className="validation-verdict"><StatusBadge value={verdict} forced={verdict==='PASSED'?'normal':verdict==='FAILED'?'critical':'neutral'}/><span>{verdict==='PASSED'?'All required fleet gates passed.':verdict==='FAILED'?'At least one required validation gate failed.':'This older report does not contain a final gate verdict.'}</span></div>
  {!!Object.keys(machines).length&&<Disclosure title="Per-machine held-out false-positive gates" defaultOpen><div className="validation-list">{Object.entries(machines).map(([machineId,value]:[string,any])=><div key={machineId}><strong>{machineId}</strong><span>{typeof value==='object'?`${value.holdout_rows??'—'} held out · active ${fmt(value.active_reference_fp??value.old_rate,3)} → shadow ${fmt(value.shadow_reference_fp??value.new_rate,3)}`:String(value)}</span>{typeof value==='object'&&(value.pass!==undefined||value.passed!==undefined)&&<StatusBadge value={(value.pass??value.passed)?'PASS':'FAIL'} forced={(value.pass??value.passed)?'normal':'critical'}/>}</div>)}</div></Disclosure>}
  {Array.isArray(tests)&&tests.length>0&&<Disclosure title={`Regression tests (${tests.length})`}><div className="validation-list">{tests.map((test:any,index:number)=><div key={test.id||index}><strong>{test.description||`Test ${test.id||index+1}`}</strong><span>{test.machine_id||'Fleet'} · {test.evaluation==='exact_flagged_prediction'?'exact flagged tick':'window peak'} risk {fmt(test.observed_anomaly_risk??test.max_anomaly_risk??test.anomaly_risk??test.score,3)} / minimum {fmt(test.minimum,3)}</span><StatusBadge value={(test.pass??test.passed)?'PASS':'FAIL'} forced={(test.pass??test.passed)?'normal':'critical'}/></div>)}</div></Disclosure>}
  <Disclosure title="Technical report"><pre className="code-panel compact">{JSON.stringify(report,null,2)}</pre></Disclosure></div>
}

function RegressionTests({admin,rows,reload}:{admin:boolean;rows:any[];reload:()=>void}){
  const [machines,setMachines]=useState<string[]>([]),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const [form,setForm]=useState({machine_id:'',description:'',start:'',end:'',minimum_anomaly_risk:.6});
  useEffect(()=>{api('/api/machines').then((r:any)=>{const items=r.items||[];setMachines(items);setForm(v=>({...v,machine_id:v.machine_id||r.default||items[0]||''}))}).catch(e=>setError(String(e.message||e)))},[]);
  const submit=async(ev:any)=>{ev.preventDefault();setBusy(true);setError('');try{await api('/api/regression-tests',{method:'POST',body:JSON.stringify({...form,start:new Date(form.start).toISOString(),end:new Date(form.end).toISOString()})});setForm(v=>({...v,description:'',start:'',end:''}));reload()}catch(e){setError(String((e as any).message||e))}finally{setBusy(false)}};
  const disable=async(id:number)=>{if(!window.confirm('Disable this regression test? Its audit record will be kept.'))return;setBusy(true);setError('');try{await api(`/api/regression-tests/${id}`,{method:'DELETE'});reload()}catch(e){setError(String((e as any).message||e))}finally{setBusy(false)}};
  return <div className="regression-manager"><Notice tone="neutral">Manual tests require a detectable peak within their saved period. Tests created from a flagged near miss validate that exact prediction timestamp, so unrelated activity elsewhere in the window cannot hide it.</Notice>{error&&<Notice tone="critical">{error}</Notice>}
  {admin&&<form className="regression-form" onSubmit={submit}><div className="field"><span>Machine</span><Select value={form.machine_id} onChange={machine_id=>setForm({...form,machine_id})} options={machines.map(id=>[id,id] as [string,string])}/></div><label className="field"><span>Description</span><input required value={form.description} onChange={e=>setForm({...form,description:e.target.value})} placeholder="Known bearing event or false negative"/></label><label className="field"><span>Start</span><input required type="datetime-local" value={form.start} onChange={e=>setForm({...form,start:e.target.value})}/></label><label className="field"><span>End</span><input required type="datetime-local" value={form.end} onChange={e=>setForm({...form,end:e.target.value})}/></label><label className="field"><span>Minimum anomaly risk (0-1)</span><input required type="number" min="0" max="1" step="0.01" value={form.minimum_anomaly_risk} onChange={e=>setForm({...form,minimum_anomaly_risk:Number(e.target.value)})}/></label><button className="primary" disabled={busy||!form.machine_id}>{busy?'Saving...':'Add validation test'}</button></form>}
  <div className="regression-list">{rows.length===0?<Empty title="No regression tests" text="Add historical periods that every shadow model must continue to detect."/>:rows.map(row=><div className={cn('regression-item',row.disabled_at&&'disabled')} key={row.id}><div><strong>{row.description}</strong><span>{row.machine_id} · {row.target_timestamp?`Exact tick ${shortTime(row.target_timestamp)}`:`${shortTime(row.start)} to ${shortTime(row.end)}`}</span><small>Minimum risk {fmt(row.minimum_anomaly_risk,2)} · {row.target_prediction_id?`Prediction #${row.target_prediction_id}`:'Manual window'} · Added by {row.created_by||'legacy process'}{row.disabled_at?` · Disabled by ${row.disabled_by||'administrator'}`:''}</small></div><StatusBadge value={row.disabled_at?'DISABLED':'ACTIVE'} forced={row.disabled_at?'neutral':'normal'}/>{admin&&!row.disabled_at&&<button className="icon-button danger-text" disabled={busy} onClick={()=>disable(row.id)} title="Disable validation test"><Icon name="trash"/></button>}</div>)}</div></div>
}

function Models({admin}:{admin:boolean}){
  const [rows,setRows]=useState<any[]>([]),[jobs,setJobs]=useState<any[]>([]),[tests,setTests]=useState<any[]>([]),[error,setError]=useState(''),[selected,setSelected]=useState<any|null>(null),[retrain,setRetrain]=useState<any>(),[panel,setPanel]=useState<'training'|'validation'|null>(null),[queueing,setQueueing]=useState(false);
  const load=()=>Promise.all([api('/api/models'),api('/api/retrain/status'),api('/api/retrain/jobs?limit=10'),api('/api/regression-tests')]).then(([m,s,j,t])=>{setRows(m);setRetrain(s);setJobs(j);setTests(t);setSelected(current=>{const updated=current?.__job?j.find((job:any)=>job.id===current.id):null;return updated?{...updated,__job:true}:current});setError('')}).catch(e=>setError(String(e.message||e)));
  useEffect(()=>{let disposed=false;const refresh=()=>{if(!disposed)load()};refresh();const timer=window.setInterval(refresh,10000);return()=>{disposed=true;window.clearInterval(timer)}},[]);
  const deleteVersion=async(versionId:string)=>{if(!window.confirm(`Delete model version ${versionId}? Its staged candidates will become eligible again. Artifact deletion cannot be undone.`))return;try{await api(`/api/models/${versionId}`,{method:'DELETE'});setSelected(null);load()}catch(e){setError(String((e as any).message||e))}};
  const runRetrain=async()=>{setQueueing(true);setError('');try{await api('/api/retrain/trigger',{method:'POST'});await load()}catch(e){setError(String((e as any).message||e))}finally{setQueueing(false)}};
  const activeJob=retrain?.active_job,pendingShadow=retrain?.pending_shadow,eligibleCandidates=retrain?.eligible_pending??retrain?.pending??0,jobBusy=queueing||eligibleCandidates===0||!!pendingShadow||activeJob?.status==='queued'||activeJob?.status==='running';
  return <><PageHeader eyebrow="FLEET MODELS" title="Models & retraining" description="Shared fleet model lifecycle, validation gates, and persistent retraining jobs." actions={<><button className="secondary" onClick={()=>setPanel('validation')}>Validation tests</button><button className="secondary" onClick={()=>setPanel('training')}>Retraining policy</button>{admin&&<button className="primary" disabled={jobBusy} onClick={runRetrain}>{queueing?'Queueing...':pendingShadow?'Shadow awaiting decision':activeJob?.status==='queued'?'Retrain queued':activeJob?.status==='running'?'Retraining...':eligibleCandidates===0?'No eligible candidates':'Run shadow retrain'}</button>}</>}/><ScopeBanner scope="fleet"/>{error&&<Notice tone="critical">{error}</Notice>}
  {retrain&&<div className="inline-summary"><div><span>Eligible candidates</span><strong>{eligibleCandidates}</strong></div><div><span>Per-machine target</span><strong>{retrain.batch_size_per_machine??retrain.batch_size??'—'}</strong></div><div><span>Eligibility</span><StatusBadge value={retrain.due?'DUE':'NOT DUE'} forced={retrain.due?'warning':'normal'}/></div><div><span>Latest job</span>{retrain.active_job?<StatusBadge value={retrain.active_job.status}/>:retrain.last_job?<StatusBadge value={retrain.last_job.status}/>:<strong>None</strong>}</div><button className="text-button" onClick={()=>setSelected({__retrain:true,...retrain})}>Details <Icon name="info"/></button></div>}
  <section className="data-surface"><div className="surface-heading"><span>Model versions</span><small>A shadow must pass every gate before promotion</small></div>{rows.length===0?<Empty title="No model versions" text="No registered model bundles were returned by the backend."/>:<div className="responsive-table"><table><thead><tr><th>Version</th><th>Status</th><th>Created</th><th>Validation</th><th>Promoted by</th><th/></tr></thead><tbody>{rows.map(m=>{const label=validationLabel(m);return <tr key={m.version_id} onClick={()=>setSelected(m)}><td><strong>{m.version_id}</strong><small>{m.reference_signature||'No reference signature'}</small></td><td><StatusBadge value={m.status}/></td><td>{shortTime(m.created_at)}</td><td><span className={cn('validation-dot',label==='Failed'&&'bad',!['Passed','Failed'].includes(label)&&'unknown')}/>{label}</td><td>{m.promoted_by||'—'}</td><td><Icon name="chevron"/></td></tr>})}</tbody></table></div>}</section>
  <section className="data-surface retrain-jobs"><div className="surface-heading"><span>Retraining jobs</span><small>Persistent backend history · latest 10</small></div>{jobs.length===0?<Empty title="No retraining jobs" text="Manual and scheduled attempts will appear here."/>:<div className="responsive-table"><table><thead><tr><th>Job</th><th>Trigger</th><th>Status</th><th>Requested</th><th>Model</th><th>Result</th></tr></thead><tbody>{jobs.map(job=><tr key={job.id} onClick={()=>setSelected({__job:true,...job})}><td><strong>#{job.id}</strong><small>{job.requested_by||'scheduler'}</small></td><td>{job.trigger}</td><td><StatusBadge value={job.status}/></td><td>{shortTime(job.created_at)}</td><td>{job.model_version_id||'—'}</td><td>{job.error||parsedObject(job.result).reason||'—'}</td></tr>)}</tbody></table></div>}</section>
  {selected&&<Modal title={selected.__retrain?'Retraining eligibility':selected.__job?`Retraining job #${selected.id}`:selected.version_id} onClose={()=>setSelected(null)} wide={!selected.__retrain}>{selected.__retrain?<div className="detail-list"><DetailRow label="Eligible candidates" value={selected.pending??selected.candidate_count??0}/><DetailRow label="Per-machine batch threshold" value={selected.batch_size_per_machine??selected.batch_size??'—'}/><DetailRow label="Currently due" value={selected.due?'Yes':'No'}/><DetailRow label="Next automatic check" value={shortTime(selected.next_check_at)}/>{selected.pending_by_machine&&<div className="candidate-list">{Object.entries(selected.pending_by_machine).map(([id,value]:[string,any])=><div key={id}><strong>{id}</strong><span>{value.pending??0} candidates · oldest {fmt(value.oldest_age_days??0,1)} days</span>{value.requires_commissioning&&<StatusBadge value="COMMISSIONING REQUIRED" forced="warning"/>}</div>)}</div>}<p className="muted">A commissioned machine reaching its count or age threshold starts one balanced shared-model job. Candidates stay unconsumed until a validated shadow is promoted.</p></div>:selected.__job?<><div className="detail-list"><DetailRow label="Status" value={selected.status} badge={tone(selected.status)}/><DetailRow label="Trigger" value={selected.trigger}/><DetailRow label="Requested by" value={selected.requested_by||'scheduler'}/><DetailRow label="Requested" value={shortTime(selected.created_at)}/><DetailRow label="Started" value={shortTime(selected.started_at)}/><DetailRow label="Finished" value={shortTime(selected.finished_at)}/><DetailRow label="Model version" value={selected.model_version_id||'—'}/><DetailRow label="Error" value={selected.error||'—'}/></div><Disclosure title="Job result"><pre className="code-panel compact">{JSON.stringify(parsedObject(selected.result),null,2)}</pre></Disclosure></>:<><div className="split-detail"><div className="detail-list"><DetailRow label="Status" value={selected.status} badge={tone(selected.status)}/><DetailRow label="Artifact path" value={selected.artifact_path||'—'}/><DetailRow label="Reference signature" value={selected.reference_signature||'—'}/><DetailRow label="Created" value={shortTime(selected.created_at)}/><DetailRow label="Promoted" value={shortTime(selected.promoted_at)}/><DetailRow label="Promoted by" value={selected.promoted_by||'—'}/></div><div><p className="section-label">VALIDATION REPORT</p><ValidationReport report={selected.validation_report}/></div></div>{admin&&<div className="modal-actions">{['shadow','retired'].includes(selected.status)&&<button className="primary" onClick={async()=>{try{await api(`/api/models/${selected.version_id}/promote`,{method:'POST'});setSelected(null);load()}catch(e){setError(String((e as any).message||e))}}}>Promote to active</button>}{selected.status!=='active'&&<button className="secondary danger-text" onClick={()=>deleteVersion(selected.version_id)}>Delete version</button>}</div>}</>}</Modal>}
  {panel==='training'&&<Modal title="Training options" onClose={()=>setPanel(null)}><TrainingOptions admin={admin}/></Modal>}
  {panel==='validation'&&<Modal title="Validation tests" wide onClose={()=>setPanel(null)}><RegressionTests admin={admin} rows={tests} reload={load}/></Modal>}
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
    NEAR_MISS_REGRESSION_WINDOW_HOURS:{label:'Near-miss regression window',desc:'History captured around a flagged near miss for shadow-model validation.',unit:'hours each side',purpose:'Creates a regression test covering this much time before and after a human-flagged suspected false negative.',higher:'Preserves more surrounding behavior for validation, but can include unrelated operating changes.',lower:'Focuses validation tightly on the flagged event, but may omit useful lead-up or recovery context.',step:.25,min:.25,max:168},
    RETRAIN_CHECK_INTERVAL_MINUTES:{label:'Retraining check interval',desc:'How often automatic retraining eligibility is evaluated.',unit:'minutes',purpose:'Sets the scheduler interval used to check whether balanced candidate count or age rules make a shadow retrain due.',higher:'Checks less often and reduces scheduler activity, but a due retrain may start later.',lower:'Responds to newly due retraining sooner, with more frequent database checks.',step:1,min:5,max:1440},
    RETRAIN_RETRY_COOLDOWN_HOURS:{label:'Failed-job retry cooldown',desc:'Wait before the scheduler retries an unchanged job that failed unexpectedly.',unit:'hours',purpose:'Limits retries after an operational failure such as a database or filesystem interruption. A validation rejection is deterministic and remains suppressed until candidates, material policy, or regression tests change; an administrator can still run manually.',higher:'Reduces repeated compute after transient failures, but waits longer before an automatic recovery attempt.',lower:'Retries operational failures sooner, with a greater risk of repeating the same interruption.',step:.25,min:.25,max:720},
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
    MAINTENANCE_HORIZON_DAYS:{label:'Forecast-risk horizon',desc:'Future interval used by trend-based maintenance rules.',unit:'days',purpose:'Selects how far ahead the first-passage model estimates critical-boundary crossing risk. Supported values are 0.25, 0.5, 0.75, and 1 day.',higher:'Looks farther ahead, usually increasing estimated crossing risk and causing earlier alerts.',lower:'Focuses on nearer-term risk, usually reducing alerts until the boundary is closer.',step:.25,min:.25,max:1},
    TREND_MIN_POINTS:{label:'Minimum trend readings',desc:'Condition readings required before the first trend fit.',unit:'readings',purpose:'Prevents the trend model from fitting until this many smoothed condition readings are available.',higher:'Builds the trend from more evidence and reduces unstable early fits, but forecasting starts later.',lower:'Starts forecasting sooner with less evidence and more noise sensitivity.',step:1,min:3,max:720},
    TREND_LOOKBACK_MINUTES:{label:'Trend lookback',desc:'Actual elapsed-time history retained for degradation forecasting.',unit:'minutes',purpose:'Keeps smoothed condition readings whose database timestamps fall within this real elapsed-time window. It no longer assumes one row equals one minute.',higher:'Uses more history for a steadier but slower-changing forecast.',lower:'Reacts faster to recent changes but becomes more sensitive to short-term noise.',step:5,min:5,max:43200},
    TREND_SLOPE_Z_THRESHOLD:{label:'Trend confidence threshold',desc:'Evidence required before a trend may drive an alert.',unit:'z-score',purpose:'Requires the estimated degradation slope to be sufficiently large compared with its statistical uncertainty before forecast alerts are trusted.',higher:'Rejects more noisy trends and reduces false forecast alerts, but may miss slow early degradation.',lower:'Accepts weaker trends sooner, improving sensitivity but increasing noise-driven alerts.',step:.1,min:.1,max:10},
    TREND_SETTLE_TICKS:{label:'Trend stabilization readings',desc:'Trusted trend delay after enough trend points first become available.',unit:'readings',purpose:'Waits this many additional prediction readings before forecast crossing risk may trigger maintenance after the trend model first becomes available.',higher:'Provides more warm-up protection but delays trend-based alerts.',lower:'Enables forecasting sooner but increases startup-transient alert risk.',step:1,min:0},
    MAINTENANCE_TREND_DEBOUNCE_TICKS:{label:'Trend alert confirmation',desc:'Consecutive trend-alert readings required to escalate.',unit:'readings',purpose:'Requires the same trend-based WARN or CRITICAL result for this many consecutive prediction readings before showing the escalation.',higher:'Filters short spikes more strongly but delays genuine trend alerts.',lower:'Escalates faster but makes single short disturbances more influential.',step:1,min:1},
    MAINTENANCE_TREND_RECOVERY_TICKS:{label:'Trend recovery confirmation',desc:'Consecutive improved readings required to clear a trend alert.',unit:'readings',purpose:'Requires this many consecutive improved prediction readings before clearing a previously reported trend-based alert.',higher:'Prevents alert flapping and clears more slowly.',lower:'Clears alerts sooner but may alternate between states during noisy periods.',step:1,min:1},
    OPERATING_STATE_STOP_CONFIRM_TICKS:{label:'Shutdown confirmation readings',desc:'Consecutive stationary readings required to declare STOPPED.',unit:'readings',purpose:'Confirms an automatically learned low-vibration state for this many consecutive source readings before suppressing model scoring and clearing the machine monitor history.',higher:'Avoids false shutdown detection during short vibration dips, but recognizes a real shutdown later.',lower:'Stops scoring sooner after vibration falls, but brief dips can be mistaken for shutdown.',step:1,min:1},
    OPERATING_STATE_START_CONFIRM_TICKS:{label:'Restart confirmation readings',desc:'Consecutive rotating readings required to begin STARTING.',unit:'readings',purpose:'Confirms that vibration has returned to the learned rotating regime before beginning the restart warm-up sequence.',higher:'Requires a more persistent restart signal and rejects short motion bursts, but delays restart recognition.',lower:'Recognizes restart sooner, but short vibration bursts can begin an unnecessary warm-up.',step:1,min:1},
    SOURCE_STALE_SECONDS:{label:'Source stale timeout',desc:'Maximum source-row age before the website reports NO DATA.',unit:'seconds',purpose:'Marks a machine as having no fresh data when the newest database sensor row is older than this timeout.',higher:'Tolerates longer ingestion delays but can make an interrupted stream appear online longer.',lower:'Detects missing data sooner but may report NO DATA during normal ingestion delays.',step:1,min:1},
    WORKER_POLL_SECONDS:{label:'Database polling interval',desc:'How often the production worker checks for newly ingested source rows.',unit:'seconds',purpose:'Controls database polling frequency independently of the sensor sample cadence. All newly found rows are still processed in timestamp order.',higher:'Reduces database polling load but delays new readings appearing in monitoring.',lower:'Shows new rows sooner but performs more frequent database queries.',step:1,min:1,max:3600},
    NEAR_MISS_TREND_WINDOW_HOURS:{label:'Near-miss trend window',desc:'Default elapsed-time window used to identify declining anomaly behavior.',unit:'hours',purpose:'Defines the real timestamp range used by Status Review and History when deciding whether an otherwise-OK prediction is a near miss.',higher:'Uses broader context and detects slower changes, but may mix different operating periods.',lower:'Focuses on recent changes, reacting faster but becoming more noise-sensitive.',step:.25,min:.25,max:168},
  };
  const groups=[
    {title:'Condition response',desc:'How anomaly deviation becomes a stable condition score and immediate decision.',keys:['HEALTH_SENSITIVITY_STD','KALMAN_INIT_SAMPLES','FAILURE_HEALTH_THRESHOLD','MAINTENANCE_HEALTH_INSPECT']},
    {title:'Failure forecast',desc:'How predicted near-term risk becomes planned or urgent maintenance.',keys:['MAINTENANCE_PROB_PLAN','MAINTENANCE_PROB_URGENT','MAINTENANCE_HORIZON_DAYS']},
    {title:'Trend reliability',desc:'Elapsed history, evidence, warm-up, and persistence required before trend alerts change state.',keys:['TREND_LOOKBACK_MINUTES','TREND_MIN_POINTS','TREND_SLOPE_Z_THRESHOLD','TREND_SETTLE_TICKS','MAINTENANCE_TREND_DEBOUNCE_TICKS','MAINTENANCE_TREND_RECOVERY_TICKS']},
    {title:'Machine availability',desc:'Source polling, shutdown, restart, and missing-data timing.',keys:['WORKER_POLL_SECONDS','OPERATING_STATE_STOP_CONFIRM_TICKS','OPERATING_STATE_START_CONFIRM_TICKS','SOURCE_STALE_SECONDS']},
    {title:'Review analysis',desc:'Default timestamp-based evidence window used by the near-miss queue and History.',keys:['NEAR_MISS_TREND_WINDOW_HOURS']},
  ];
  if(error)return <><PageHeader eyebrow="FLEET POLICY" title="Global thresholds" description="Maintenance policy applied consistently to every machine."/><ScopeBanner scope="fleet"/><Notice tone="critical">{error}</Notice></>;
  if(!v)return <Loading/>;
  const save=async()=>{setSaved(false);setSaveError('');setSaving(true);try{const next=await api('/api/config/thresholds',{method:'PUT',body:JSON.stringify(v)});setV(next);setSaved(true)}catch(e){setSaveError(String((e as any).message||e))}finally{setSaving(false)}};
  const info=infoKey?meta[infoKey]:undefined;
  return <><PageHeader eyebrow="FLEET POLICY" title="Global thresholds" description="Tune writable runtime policy applied consistently to every machine." actions={admin?<button className="primary" disabled={saving} onClick={save}>{saving?'Saving…':'Save fleet policy'}</button>:<StatusBadge value="READ ONLY"/>}/><ScopeBanner scope="fleet"/>{saved&&<Notice tone="normal">Policy saved without model retraining. Changes to condition sensitivity, warm-up, or trend history reset active monitors and warm them up again.</Notice>}{saveError&&<Notice tone="critical">{saveError}</Notice>}
  <section className="settings-layout">{groups.map(group=><section className="settings-surface settings-group" key={group.title}><header className="settings-group-header"><div><strong>{group.title}</strong><span>{group.desc}</span></div><small>{group.keys.length} setting{group.keys.length===1?'':'s'}</small></header>{group.keys.map(k=>{const item=meta[k];return <div className="setting-row" key={k}><div><SettingTitle item={item} onInfo={()=>setInfoKey(k)}/><span>{item.desc}</span><code>{k}</code></div><div className="setting-value"><span>Current value</span><div className="setting-control"><input type="number" step={item.step??'any'} min={item.min} max={item.max} value={v[k]} disabled={!admin} onChange={e=>setV({...v,[k]:Number(e.target.value)})}/><span>{item.unit}</span></div></div></div>})}</section>)}<section className="settings-surface settings-help"><Disclosure title="How these thresholds are applied"><div className="explain-grid"><div><b>Backend authority</b><span>Values are range-checked and relationship rules are validated again before saving.</span></div><div><b>No model retrain</b><span>These are runtime policy controls, not learned model or normalization parameters.</span></div><div><b>Fleet scope</b><span>The same saved policy applies to every commissioned machine.</span></div></div></Disclosure></section></section>
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

createRoot(document.getElementById('root')!).render(<ErrorBoundary><App/></ErrorBoundary>);
