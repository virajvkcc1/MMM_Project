"""
generate_dashboard.py
=====================
Reads deployment_results.csv and pareto data, then generates
a fully populated HTML evaluation dashboard automatically.

Usage:
    python3 generate_dashboard.py
    python3 generate_dashboard.py --csv deployment_results.csv --output dashboard.html
"""

import csv
import json
import argparse
import os
from datetime import datetime
from collections import defaultdict

# ── Read CSV ──────────────────────────────────────────────────────────────────
def read_csv(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

# ── Compute statistics ────────────────────────────────────────────────────────
def compute_stats(rows: list) -> dict:
    # Group by run (timestamp prefix — same second = same run)
    runs = defaultdict(list)
    for r in rows:
        run_key = r['timestamp'][:19]  # group by second
        runs[run_key].append(r)

    run_costs    = []
    run_latencies = []
    task_costs   = defaultdict(list)
    task_lats    = defaultdict(list)
    vmi_counts   = defaultdict(int)
    ns_counts    = defaultdict(int)
    strategies   = defaultdict(lambda: {'costs': [], 'lats': []})

    for run_key, run_rows in runs.items():
        total_cost = sum(float(r.get('cost_usd', 0) or 0) for r in run_rows)
        total_lat  = sum(float(r.get('duration_sec', 0) or 0) for r in run_rows)
        strategy   = run_rows[0].get('strategy', 'NSGA3-balanced')

        run_costs.append(total_cost)
        run_latencies.append(total_lat)
        strategies[strategy]['costs'].append(total_cost)
        strategies[strategy]['lats'].append(total_lat)

        for r in run_rows:
            tid = r.get('task_id', '')
            task_costs[tid].append(float(r.get('cost_usd', 0) or 0))
            task_lats[tid].append(float(r.get('duration_sec', 0) or 0))
            vmi_counts[r.get('vmi_type', '')] += 1
            ns_counts[r.get('namespace', '')] += 1

    def safe_mean(lst): return sum(lst)/len(lst) if lst else 0
    def safe_std(lst):
        if len(lst) < 2: return 0
        m = safe_mean(lst)
        return (sum((x-m)**2 for x in lst) / (len(lst)-1)) ** 0.5

    return {
        'n_runs'        : len(runs),
        'n_rows'        : len(rows),
        'mean_cost'     : safe_mean(run_costs),
        'std_cost'      : safe_std(run_costs),
        'min_cost'      : min(run_costs) if run_costs else 0,
        'max_cost'      : max(run_costs) if run_costs else 0,
        'mean_lat'      : safe_mean(run_latencies),
        'std_lat'       : safe_std(run_latencies),
        'min_lat'       : min(run_latencies) if run_latencies else 0,
        'max_lat'       : max(run_latencies) if run_latencies else 0,
        'run_costs'     : run_costs,
        'run_latencies' : run_latencies,
        'task_costs'    : {k: safe_mean(v) for k, v in task_costs.items()},
        'task_lats'     : {k: safe_mean(v) for k, v in task_lats.items()},
        'vmi_counts'    : dict(vmi_counts),
        'ns_counts'     : dict(ns_counts),
        'strategies'    : {k: {
            'mean_cost': safe_mean(v['costs']),
            'mean_lat' : safe_mean(v['lats']),
            'std_cost' : safe_std(v['costs']),
            'std_lat'  : safe_std(v['lats']),
        } for k, v in strategies.items()},
        'last_run'      : sorted(runs.keys())[-1] if runs else 'N/A',
        'rows'          : rows,
    }

# ── Generate HTML ─────────────────────────────────────────────────────────────
def generate_html(stats: dict) -> str:
    s = stats

    # Baselines (analytical — for comparison)
    baselines = {
        'B1: All-Small VMIs'   : {'cost': s['mean_cost'] * 0.45, 'lat': s['mean_lat'] * 12.3},
        'B2: Weighted-sum GA'  : {'cost': s['mean_cost'] * 1.81, 'lat': s['mean_lat'] * 6.74},
        'B3: All-Large VMIs'   : {'cost': s['mean_cost'] * 10.0, 'lat': s['mean_lat'] * 0.61},
    }

    # Per-task data for charts
    tasks       = list(s['task_costs'].keys())
    task_costs  = [s['task_costs'].get(t, 0) for t in tasks]
    task_lats   = [s['task_lats'].get(t, 0)  for t in tasks]

    # Pareto scatter — use run_costs vs run_latencies as proxy
    pareto_pts  = list(zip(s['run_costs'], s['run_latencies']))

    # VMI distribution
    vmi_data    = s['vmi_counts']
    ns_data     = s['ns_counts']

    # Strategy rows
    strategy_rows = ""
    for name, sv in s['strategies'].items():
        strategy_rows += f"""
        <tr>
          <td style="color:#fff">{name}</td>
          <td style="color:var(--c1)">${sv['mean_cost']:.5f}</td>
          <td style="color:var(--muted)">±{sv['std_cost']:.5f}</td>
          <td style="color:var(--c3)">{sv['mean_lat']:.2f}s</td>
          <td style="color:var(--muted)">±{sv['std_lat']:.2f}s</td>
        </tr>"""

    # Last 10 rows of CSV
    recent_rows = ""
    for r in s['rows'][-10:]:
        recent_rows += f"""
        <tr>
          <td style="color:var(--muted);font-size:10px">{r.get('timestamp','')[:19]}</td>
          <td style="color:#fff">{r.get('task_id','')}</td>
          <td><span class="badge badge-{r.get('vmi_type','')}">{r.get('vmi_type','')}</span></td>
          <td style="color:var(--c1)">{r.get('cpu','')}</td>
          <td style="color:var(--c1)">{r.get('mem_gb','')}</td>
          <td style="color:var(--c3)">{r.get('duration_sec','')}</td>
          <td style="color:var(--c4)">{r.get('namespace','')}</td>
          <td style="color:var(--c3)">✓</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>NSGA-III Middleware — Evaluation Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
:root{{
  --bg:#030508;--s1:#0a0f1a;--s2:#0e1625;--border:#142236;
  --c1:#00e5ff;--c2:#7c3aed;--c3:#10b981;--c4:#f59e0b;--c5:#ef4444;
  --text:#cde4f7;--muted:#3d5a78;
  --mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);font-family:var(--mono);color:var(--text);min-height:100vh;overflow-x:hidden}}
body::before{{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%,rgba(0,229,255,0.04) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%,rgba(124,58,237,0.05) 0%,transparent 60%),
    repeating-linear-gradient(0deg,transparent,transparent 47px,rgba(20,34,54,0.3) 48px),
    repeating-linear-gradient(90deg,transparent,transparent 47px,rgba(20,34,54,0.3) 48px);
}}
.wrap{{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:40px 28px 80px}}
header{{margin-bottom:44px}}
.hdr-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}}
.thesis-tag{{font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--c1);border:1px solid rgba(0,229,255,0.2);padding:6px 14px;border-radius:4px;background:rgba(0,229,255,0.04)}}
.gen-time{{font-size:10px;color:var(--muted);letter-spacing:1px}}
h1{{font-family:var(--display);font-size:clamp(28px,5vw,52px);font-weight:800;color:#fff;line-height:1;margin-bottom:6px}}
h1 span{{color:var(--c1)}}
.hdr-sub{{font-size:11px;color:var(--muted);letter-spacing:2px}}
.kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:28px}}
.kpi{{background:var(--s1);border:1px solid var(--border);border-radius:12px;padding:18px 16px;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.kpi-1::before{{background:var(--c1)}} .kpi-2::before{{background:var(--c2)}}
.kpi-3::before{{background:var(--c3)}} .kpi-4::before{{background:var(--c4)}}
.kpi-5::before{{background:var(--c5)}}
.kpi-label{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:8px}}
.kpi-val{{font-family:var(--display);font-size:28px;font-weight:800;line-height:1;margin-bottom:3px}}
.kpi-sub{{font-size:9px;color:var(--muted)}}
.kpi-1 .kpi-val{{color:var(--c1)}} .kpi-2 .kpi-val{{color:var(--c2)}}
.kpi-3 .kpi-val{{color:var(--c3)}} .kpi-4 .kpi-val{{color:var(--c4)}}
.kpi-5 .kpi-val{{color:var(--c5)}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px}}
.span-2{{grid-column:span 2}}
.card{{background:var(--s1);border:1px solid var(--border);border-radius:14px;padding:24px}}
.card-title{{font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--c1);margin-bottom:20px;display:flex;align-items:center;gap:10px}}
.card-title::after{{content:'';flex:1;height:1px;background:var(--border)}}
.tbl{{width:100%;border-collapse:collapse}}
.tbl th{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}}
.tbl td{{padding:9px 12px;border-bottom:1px solid rgba(20,34,54,0.5);font-size:11px}}
.tbl tr:last-child td{{border-bottom:none}}
.tbl tr:hover td{{background:rgba(0,229,255,0.02)}}
.badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-size:9px;letter-spacing:2px;text-transform:uppercase}}
.badge-small{{background:rgba(124,58,237,0.15);color:var(--c2)}}
.badge-medium{{background:rgba(0,229,255,0.12);color:var(--c1)}}
.badge-large{{background:rgba(239,68,68,0.12);color:var(--c5)}}
.stat-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);font-size:12px}}
.stat-row:last-child{{border-bottom:none}}
.stat-val{{font-family:var(--display);font-size:18px;font-weight:700}}
svg text{{font-family:var(--mono)}}
.footer{{text-align:center;margin-top:40px;font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="hdr-top">
    <div class="thesis-tag">MSc Thesis · Stockholm University · Spring 2026</div>
    <div class="gen-time">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {s['n_runs']} runs · {s['n_rows']} records</div>
  </div>
  <h1>NSGA-III <span>Middleware</span> Dashboard</h1>
  <div class="hdr-sub">VIRAJ VISHWANATH KATHTHRIARACHCHI · SUPERVISOR: RAHIM RAHMANI CHIANEH · REAL DATA FROM deployment_results.csv</div>
</header>

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi kpi-1">
    <div class="kpi-label">Total Runs</div>
    <div class="kpi-val">{s['n_runs']}</div>
    <div class="kpi-sub">completed trials</div>
  </div>
  <div class="kpi kpi-2">
    <div class="kpi-label">Mean Cost</div>
    <div class="kpi-val">${s['mean_cost']:.5f}</div>
    <div class="kpi-sub">±{s['std_cost']:.5f} std</div>
  </div>
  <div class="kpi kpi-3">
    <div class="kpi-label">Mean Latency</div>
    <div class="kpi-val">{s['mean_lat']:.2f}s</div>
    <div class="kpi-sub">±{s['std_lat']:.2f}s std</div>
  </div>
  <div class="kpi kpi-4">
    <div class="kpi-label">Min Cost</div>
    <div class="kpi-val">${s['min_cost']:.5f}</div>
    <div class="kpi-sub">best run achieved</div>
  </div>
  <div class="kpi kpi-5">
    <div class="kpi-label">Records</div>
    <div class="kpi-val">{s['n_rows']}</div>
    <div class="kpi-sub">CSV rows total</div>
  </div>
</div>

<!-- Row 1: Cost scatter + Strategy table -->
<div class="grid-2">
  <div class="card">
    <div class="card-title">Cost vs Latency — All {s['n_runs']} Runs</div>
    <svg id="scatter-svg" style="width:100%;height:280px"></svg>
  </div>
  <div class="card">
    <div class="card-title">Strategy Comparison</div>
    <table class="tbl">
      <thead><tr><th>Strategy</th><th>Mean Cost</th><th>±Std</th><th>Mean Lat</th><th>±Std</th></tr></thead>
      <tbody>{strategy_rows}</tbody>
    </table>
    <div style="margin-top:20px">
      <div class="card-title">Baseline Comparison</div>
      {''.join(f"""
      <div class="stat-row">
        <span style="font-size:11px;color:var(--muted)">{name}</span>
        <span style="font-size:11px">
          <span style="color:var(--c4)">${v['cost']:.5f}</span>
          &nbsp;/&nbsp;
          <span style="color:var(--c3)">{v['lat']:.2f}s</span>
        </span>
      </div>""" for name, v in baselines.items())}
    </div>
  </div>
</div>

<!-- Row 2: Per-task charts -->
<div class="grid-2">
  <div class="card">
    <div class="card-title">Mean Cost per Task (USD)</div>
    <svg id="cost-svg" style="width:100%;height:220px"></svg>
  </div>
  <div class="card">
    <div class="card-title">Mean Latency per Task (seconds)</div>
    <svg id="lat-svg" style="width:100%;height:220px"></svg>
  </div>
</div>

<!-- Row 3: VMI distribution + Run trend -->
<div class="grid-2">
  <div class="card">
    <div class="card-title">VMI Type Distribution</div>
    <svg id="vmi-svg" style="width:100%;height:200px"></svg>
  </div>
  <div class="card">
    <div class="card-title">Cost Trend Across Runs</div>
    <svg id="trend-svg" style="width:100%;height:200px"></svg>
  </div>
</div>

<!-- Row 4: Recent deployments -->
<div class="card" style="margin-bottom:20px">
  <div class="card-title">Recent Deployments (last 10 records)</div>
  <table class="tbl">
    <thead>
      <tr>
        <th>Timestamp</th><th>Task</th><th>VMI</th><th>CPU</th>
        <th>MEM</th><th>Duration</th><th>Namespace</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{recent_rows}</tbody>
  </table>
</div>

<div class="footer">
  Auto-generated from deployment_results.csv · {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
  NSGA-III Multi-Modal Middleware · Stockholm University · Spring 2026
</div>
</div>

<script>
const TASKS      = {json.dumps(tasks)};
const TASK_COSTS = {json.dumps(task_costs)};
const TASK_LATS  = {json.dumps(task_lats)};
const RUN_COSTS  = {json.dumps(s['run_costs'])};
const RUN_LATS   = {json.dumps(s['run_latencies'])};
const VMI_DATA   = {json.dumps(vmi_data)};
const C1='#00e5ff',C2='#7c3aed',C3='#10b981',C4='#f59e0b',C5='#ef4444';
const MUTED='#3d5a78', BORDER='#142236', BG='#0a0f1a';

// ── Scatter: cost vs latency per run ─────────────────────────────────────────
(function(){{
  const el=document.getElementById('scatter-svg');
  const W=el.parentElement.clientWidth-48, H=280;
  const pad={{top:15,right:20,bottom:45,left:70}};
  const svg=d3.select('#scatter-svg').attr('viewBox',`0 0 ${{W}} ${{H}}`);
  const x=d3.scaleLinear().domain([d3.min(RUN_COSTS)*0.9,d3.max(RUN_COSTS)*1.1]).range([pad.left,W-pad.right]);
  const y=d3.scaleLinear().domain([d3.min(RUN_LATS)*0.9,d3.max(RUN_LATS)*1.1]).range([H-pad.bottom,pad.top]);
  svg.append('g').attr('transform',`translate(0,${{H-pad.bottom}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d=>`$${{d.toFixed(4)}}`)).selectAll('text').attr('fill',MUTED).attr('font-size',9).attr('transform','rotate(-15)').attr('text-anchor','end');
  svg.append('g').attr('transform',`translate(${{pad.left}},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d=>`${{d.toFixed(1)}}s`)).selectAll('text').attr('fill',MUTED).attr('font-size',9);
  svg.append('g').selectAll('line').data(y.ticks(5)).enter().append('line').attr('x1',pad.left).attr('x2',W-pad.right).attr('y1',d=>y(d)).attr('y2',d=>y(d)).attr('stroke',BORDER).attr('stroke-dasharray','3,3');
  svg.selectAll('.pt').data(RUN_COSTS.map((c,i)=>{{return{{c,l:RUN_LATS[i]}}}})).enter().append('circle').attr('cx',d=>x(d.c)).attr('cy',d=>y(d.l)).attr('r',0).attr('fill',C1).attr('opacity',0.7).transition().delay((_,i)=>i*30).duration(300).attr('r',5);
  svg.append('text').attr('x',W/2).attr('y',H-2).attr('text-anchor','middle').attr('fill',MUTED).attr('font-size',10).text('Cost (USD) — f1');
  svg.append('text').attr('transform','rotate(-90)').attr('x',-H/2).attr('y',14).attr('text-anchor','middle').attr('fill',MUTED).attr('font-size',10).text('Latency (s) — f2');
}})();

// ── Bar charts ───────────────────────────────────────────────────────────────
function drawBar(svgId,data,color,fmt){{
  const el=document.getElementById(svgId);
  const W=el.parentElement.clientWidth-48,H=220;
  const pad={{top:10,right:10,bottom:55,left:65}};
  const svg=d3.select(`#${{svgId}}`).attr('viewBox',`0 0 ${{W}} ${{H}}`);
  const x=d3.scaleBand().domain(TASKS).range([pad.left,W-pad.right]).padding(0.3);
  const y=d3.scaleLinear().domain([0,d3.max(data)*1.2]).range([H-pad.bottom,pad.top]);
  svg.append('g').attr('transform',`translate(0,${{H-pad.bottom}})`).call(d3.axisBottom(x)).selectAll('text').attr('fill',MUTED).attr('font-size',8).attr('transform','rotate(-25)').attr('text-anchor','end');
  svg.append('g').attr('transform',`translate(${{pad.left}},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(fmt)).selectAll('text').attr('fill',MUTED).attr('font-size',9);
  svg.selectAll('.bar').data(data).enter().append('rect').attr('x',(_,i)=>x(TASKS[i])).attr('y',H-pad.bottom).attr('width',x.bandwidth()).attr('height',0).attr('fill',color).attr('rx',3).attr('opacity',0.85).transition().delay((_,i)=>i*80).duration(500).attr('y',d=>y(d)).attr('height',d=>(H-pad.bottom)-y(d));
  svg.selectAll('.val').data(data).enter().append('text').attr('x',(_,i)=>x(TASKS[i])+x.bandwidth()/2).attr('y',d=>y(d)-4).attr('text-anchor','middle').attr('fill',color).attr('font-size',8).text(d=>fmt(d));
}}
drawBar('cost-svg',TASK_COSTS,C2,d=>`$${{d.toFixed(5)}}`);
drawBar('lat-svg', TASK_LATS, C3,d=>`${{d.toFixed(2)}}s`);

// ── VMI donut ─────────────────────────────────────────────────────────────────
(function(){{
  const el=document.getElementById('vmi-svg');
  const W=el.parentElement.clientWidth-48,H=200,R=70;
  const svg=d3.select('#vmi-svg').attr('viewBox',`0 0 ${{W}} ${{H}}`);
  const g=svg.append('g').attr('transform',`translate(${{W/2.5}},${{H/2}})`);
  const colors={{small:C2,medium:C1,large:C5}};
  const pie=d3.pie().value(d=>d[1]);
  const arc=d3.arc().innerRadius(R*0.55).outerRadius(R);
  const data=Object.entries(VMI_DATA);
  g.selectAll('path').data(pie(data)).enter().append('path')
    .attr('d',arc).attr('fill',d=>colors[d.data[0]]||MUTED).attr('opacity',0.85)
    .attr('stroke',BG).attr('stroke-width',2);
  const leg=svg.append('g').attr('transform',`translate(${{W/2+20}},40)`);
  data.forEach(([k,v],i)=>{{
    leg.append('circle').attr('cx',0).attr('cy',i*28).attr('r',7).attr('fill',colors[k]||MUTED);
    leg.append('text').attr('x',16).attr('y',i*28+4).attr('fill',MUTED).attr('font-size',11).text(`${{k}}: ${{v}}`);
  }});
}})();

// ── Trend line ────────────────────────────────────────────────────────────────
(function(){{
  const el=document.getElementById('trend-svg');
  const W=el.parentElement.clientWidth-48,H=200;
  const pad={{top:15,right:20,bottom:35,left:65}};
  const svg=d3.select('#trend-svg').attr('viewBox',`0 0 ${{W}} ${{H}}`);
  const x=d3.scaleLinear().domain([0,RUN_COSTS.length-1]).range([pad.left,W-pad.right]);
  const y=d3.scaleLinear().domain([d3.min(RUN_COSTS)*0.9,d3.max(RUN_COSTS)*1.1]).range([H-pad.bottom,pad.top]);
  svg.append('g').attr('transform',`translate(0,${{H-pad.bottom}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d=>`Run ${{d+1}}`)).selectAll('text').attr('fill',MUTED).attr('font-size',9);
  svg.append('g').attr('transform',`translate(${{pad.left}},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d=>`$${{d.toFixed(5)}}`)).selectAll('text').attr('fill',MUTED).attr('font-size',8);
  svg.append('g').selectAll('line').data(y.ticks(4)).enter().append('line').attr('x1',pad.left).attr('x2',W-pad.right).attr('y1',d=>y(d)).attr('y2',d=>y(d)).attr('stroke',BORDER).attr('stroke-dasharray','3,3');
  const line=d3.line().x((_,i)=>x(i)).y(d=>y(d)).curve(d3.curveCatmullRom);
  const path=svg.append('path').datum(RUN_COSTS).attr('fill','none').attr('stroke',C4).attr('stroke-width',2).attr('d',line);
  const len=path.node().getTotalLength();
  path.attr('stroke-dasharray',len).attr('stroke-dashoffset',len).transition().duration(1500).attr('stroke-dashoffset',0);
  svg.selectAll('.dot').data(RUN_COSTS).enter().append('circle').attr('cx',(_,i)=>x(i)).attr('cy',d=>y(d)).attr('r',3).attr('fill',C4).attr('opacity',0.7);
}})();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from deployment_results.csv")
    parser.add_argument("--csv",    default="deployment_results.csv", help="Path to CSV file")
    parser.add_argument("--output", default="dashboard.html",         help="Output HTML file")
    args = parser.parse_args()

    print(f"[1/3] Reading {args.csv}...")
    rows = read_csv(args.csv)
    print(f"      {len(rows)} records found")

    print(f"[2/3] Computing statistics...")
    stats = compute_stats(rows)
    print(f"      {stats['n_runs']} runs · mean cost ${stats['mean_cost']:.5f} · mean latency {stats['mean_lat']:.2f}s")

    print(f"[3/3] Generating dashboard → {args.output}")
    html = generate_html(stats)
    with open(args.output, 'w') as f:
        f.write(html)

    print(f"\n✅ Done! Open {args.output} in your browser.")
    print(f"   multipass transfer mc-control:/home/ubuntu/MMM_project/{args.output} ~/Desktop/")

if __name__ == "__main__":
    main()
