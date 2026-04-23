"""
generate_architecture.py
Run:  python generate_architecture.py
Saves architecture_diagram.png  (180 dpi, suitable for thesis)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colours ────────────────────────────────────────────────────────────────────
C = {
    'input' : '#EAF4FB', 'input_b' : '#2E86C1',
    'l1'    : '#EAFAF1', 'l1_b'    : '#1E8449',
    'l2'    : '#EBF5FB', 'l2_b'    : '#1A5276',
    'l3'    : '#FEF9E7', 'l3_b'    : '#9A7D0A',
    'out'   : '#F5EEF8', 'out_b'   : '#6C3483',
    'text'  : '#1A252F', 'sub'     : '#2C3E50',
    'arrow' : '#34495E',
}

fig, ax = plt.subplots(figsize=(13, 16), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(0, 13)
ax.set_ylim(0, 16)
ax.axis('off')

# ── Helpers ────────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, fc, ec, title, rows=None, title_size=11):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=2,
    ))
    top = y + h - 0.28
    ax.text(x + w / 2, top, title,
            ha='center', va='top', fontsize=title_size,
            fontweight='bold', color=ec, zorder=3)
    if rows:
        for i, row in enumerate(rows):
            ax.text(x + w / 2, top - 0.52 - i * 0.40, row,
                    ha='center', va='top', fontsize=8.8,
                    color=C['sub'], zorder=3)

def draw_arrow(ax, x, y_from, y_to, label=''):
    ax.annotate(
        '', xy=(x, y_to), xytext=(x, y_from),
        arrowprops=dict(arrowstyle='->', color=C['arrow'],
                        lw=2.2, mutation_scale=20),
        zorder=4,
    )
    if label:
        ax.text(x + 0.25, (y_from + y_to) / 2, label,
                ha='left', va='center', fontsize=8.2,
                color='#666', style='italic')

# ── Main title ─────────────────────────────────────────────────────────────────
ax.text(6.5, 15.65,
        'NSGA-III Multi-Modal Middleware — Three-Layer Architecture',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color=C['text'])

# ── INPUT: Pipeline Definition ─────────────────────────────────────────────────
draw_box(ax, 1.5, 13.7, 10, 1.55,
         C['input'], C['input_b'],
         'Pipeline Definition  ·  pipeline.yaml',
         rows=[
             'Lambda Architecture Tasks  ·  VMI Catalogue  ·  Dependency Graph  ·  Namespace Mapping',
             'Data volume (data_gb)  ·  Processing mode (stream / batch / serve)  ·  Container images',
         ],
         title_size=11)

draw_arrow(ax, 6.5, 13.7, 13.05, 'YAML config')

# ── LAYER 1: Logical Pipeline Manager ─────────────────────────────────────────
draw_box(ax, 0.5, 10.3, 12, 2.65,
         C['l1'], C['l1_b'],
         'LAYER 1  —  Logical Pipeline Manager  (lpm.py)',
         rows=[
             'YAML Parser  ·  NetworkX DAG Constructor  ·  Cycle / Acyclicity Validator',
             'Topological Sort  ·  Source-to-sink Path Enumeration  ·  Task Metadata Enrichment',
             'VMI Option Filtering  ·  Workload Scaling (Low × 0.5 / Medium × 1.0 / High × 2.0)',
             'Outputs:  NetworkX DiGraph  ·  VMI Catalogue  ·  All DAG Paths  ·  Task Order',
         ],
         title_size=11)

draw_arrow(ax, 6.5, 10.3, 9.65, 'DAG + task metadata + paths')

# ── LAYER 2: Optimization Engine ──────────────────────────────────────────────
draw_box(ax, 0.5, 6.0, 12, 3.50,
         C['l2'], C['l2_b'],
         'LAYER 2  —  Orchestration Optimization Engine  (optimizer.py)',
         rows=[
             'Decision variables per task:  VMI index  ·  CPU fraction [0.5–1.0]  ·  MEM fraction [0.5–1.0]',
             'Objective f₁: Total execution cost (USD)             Objective f₂: Critical-path latency (s)',
             'Algorithm: NSGA-III (pymoo)  ·  Population = 100  ·  Generations = 100  ·  Seed varied (0–29)',
             'Fraction encoding: all solutions feasible by construction — no constraint violations',
             'Output: Pareto front  →  weighted plan selection (cost_weight)  ·  Hypervolume tracking',
             'Baseline B2: Weighted-Sum GA — single-objective scalarisation for comparison',
         ],
         title_size=11)

draw_arrow(ax, 6.5, 6.0, 5.35, 'Orchestration plan  (VMI type + CPU + MEM per task)')

# ── LAYER 3: KubeVirt Execution Adapter ───────────────────────────────────────
draw_box(ax, 0.5, 2.6, 12, 2.60,
         C['l3'], C['l3_b'],
         'LAYER 3  —  KubeVirt Execution Adapter  (executor.py)',
         rows=[
             'VMI Manifest Builder  ·  kubevirt.io/v1 VirtualMachineInstance objects',
             'Namespace Manager  ·  Topological deployment order (dependency-safe)',
             'Dry-run mode (simulation)  ·  Live mode (real cluster API calls)',
             'CSV event logger  ·  Outputs: deployment_results.csv  ·  deployment_plan.yaml',
         ],
         title_size=11)

draw_arrow(ax, 6.5, 2.6, 1.95, 'VMI manifests  (K8s CustomObjectsApi)')

# ── OUTPUT: Execution Environment ─────────────────────────────────────────────
draw_box(ax, 1.5, 0.2, 10, 1.60,
         C['out'], C['out_b'],
         'Execution Environment  —  Canonical MicroCloud  (KubeVirt)',
         rows=[
             'Namespace: speed-layer  (stream tasks)  ·  batch-layer  (batch tasks)  ·  serve-layer  (serve tasks)',
             'Distributed VirtualMachineInstances  ·  Lambda Architecture workloads',
         ],
         title_size=11)

# ── Side annotation: evaluation loop ──────────────────────────────────────────
ax.annotate(
    '', xy=(0.5, 13.7), xytext=(0.5, 2.6),
    arrowprops=dict(arrowstyle='<->', color='#AAB7B8',
                    lw=1.5, linestyle='dashed', mutation_scale=14),
    zorder=1,
)
ax.text(0.18, 8.15, 'Evaluation\nloop\n(30 runs\n× 3 workloads)',
        ha='center', va='center', fontsize=7.5,
        color='#888', rotation=90)

plt.tight_layout(pad=0.4)
plt.savefig('architecture_diagram.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved → architecture_diagram.png")
