"""
main.py — Middleware Runner
============================
Wires together all three middleware layers:
  Layer 1 → lpm.py       (Logical Pipeline Manager)
  Layer 2 → optimizer.py (Orchestration Optimization Engine)
  Layer 3 → executor.py  (KubeVirt Execution Adapter)

Usage:
  # Dry-run (no cluster needed — for local testing)
  python main.py --dry-run

  # Live cluster run
  python main.py

  # Live run with latency preference
  python main.py --cost-weight 0.2

  # Full evaluation run (balanced + cost + latency strategies)
  python main.py --dry-run --evaluate

Thesis reference: Section 3.6 — Demonstration
"""

import argparse
import sys
import json
from pathlib import Path

# ── Import middleware layers ───────────────────────────────────────────────────
from lpm       import LogicalPipelineManager
from optimizer import OrchestrationOptimizationEngine
from executor  import KubeVirtAdapter


PIPELINE_YAML = "pipeline.yaml"


def run_middleware(dry_run: bool = True,
                   cost_weight: float = 0.5,
                   pop_size: int = 100,
                   n_gen: int = 100,
                   save_outputs: bool = True) -> dict:
    """
    Full middleware pipeline:
      1. Parse YAML → build DAG
      2. Run NSGA-III → get Pareto front
      3. Select best plan
      4. Deploy VMIs to cluster (or simulate in dry-run)

    Returns the deployment result dict.
    """

    print("\n" + "═"*60)
    print("  NSGA-III MULTI-MODAL MIDDLEWARE")
    print("  MSc Thesis — Stockholm University, Spring 2026")
    print("═"*60)

    # ── LAYER 1: Build DAG ────────────────────────────────────────────────────
    print("\n[1/3] Logical Pipeline Manager — Building DAG...")
    lpm = LogicalPipelineManager(PIPELINE_YAML)
    lpm.build_dag()
    lpm.summary()

    if save_outputs:
        lpm.visualize(save_path="dag_visualization.png")

    # ── LAYER 2: Optimise ─────────────────────────────────────────────────────
    print("[2/3] Optimization Engine — Running NSGA-III...")
    engine = OrchestrationOptimizationEngine(lpm, pop_size=pop_size, n_gen=n_gen)
    opt_result = engine.run()

    if opt_result['status'] != 'success':
        print(f"  [ERROR] Optimisation failed.")
        return {"status": "error"}

    # Select the best plan with the requested cost/latency preference
    plan = engine.select_plan(cost_weight=cost_weight)
    engine.print_plan(plan)

    if save_outputs:
        engine.save_pareto_plot(save_path="pareto_front.png")

    # ── LAYER 3: Deploy ───────────────────────────────────────────────────────
    print("[3/3] KubeVirt Adapter — Deploying pipeline...")
    adapter = KubeVirtAdapter(dry_run=dry_run)

    if save_outputs:
        adapter.save_plan(plan, path="deployment_plan.yaml")

    deploy_result = adapter.deploy_pipeline(
        plan        = plan,
        task_order  = lpm.get_task_order(),
    )

    print("═"*60)
    print(f"  COMPLETE  |  Status: {deploy_result['status'].upper()}")
    print(f"  Overhead  : {opt_result['overhead_sec']}s (NSGA-III)")
    print(f"  Deploy    : {deploy_result['total_duration']}s")
    print("═"*60 + "\n")

    return {
        "optimization": opt_result,
        "plan"        : plan,
        "deployment"  : deploy_result,
    }


WORKLOAD_LEVELS = [('Low', 0.5), ('Medium', 1.0), ('High', 2.0)]


def run_evaluation(dry_run: bool = True, n_gen: int = 100, n_runs: int = 30):
    """
    Full statistical evaluation (thesis Sections 3.7.2 and 3.7.3):
      - 30 independent NSGA-III runs per workload level (seeds 0–29)
      - B2 Weighted-Sum GA: 30 runs per workload level
      - B1 All-Small, B3 All-Large: deterministic static baselines
      - Workload levels: Low (0.5×), Medium (1.0×), High (2.0×) data_gb scaling
      - Shared-reference Hypervolume indicator for fair NSGA-III vs B2 comparison
      - Wilcoxon signed-rank test on HV distributions (α = 0.05)
    """
    import numpy as np
    import csv
    from scipy.stats import wilcoxon
    from pymoo.indicators.hv import HV
    from optimizer import WeightedSumBaseline, _task_latency, _task_cost

    print("\n" + "═"*68)
    print(f"  EVALUATION — {n_runs} runs × {len(WORKLOAD_LEVELS)} workload levels")
    print("═"*68)

    csv_rows  = []
    all_data  = {}   # keyed by wl_name — used later by _plot_chapter5()

    for wl_name, wl_scale in WORKLOAD_LEVELS:
        print(f"\n  ── Workload: {wl_name} (data_gb × {wl_scale}) {'─'*30}")

        lpm = LogicalPipelineManager(PIPELINE_YAML)
        lpm.build_dag()
        lpm.scale_workload(wl_scale)

        # ── NSGA-III: n_runs independent seeds ────────────────────────
        print(f"  [EVAL] NSGA-III × {n_runs} runs...")
        nsga_costs, nsga_lats, nsga_overheads, nsga_npareto = [], [], [], []
        nsga_pareto_Fs = []
        best_engine, best_npareto_val = None, -1   # track run with richest Pareto front
        for seed in range(n_runs):
            engine = OrchestrationOptimizationEngine(lpm, pop_size=100, n_gen=n_gen)
            r = engine.run(seed=seed)
            p = engine.select_plan(cost_weight=0.5)
            nsga_costs.append(p['total_cost_usd'])
            nsga_lats.append(p['total_latency_sec'])
            nsga_overheads.append(r['overhead_sec'])
            nsga_npareto.append(r['n_pareto'])
            nsga_pareto_Fs.append(r['pareto_solutions'])
            if r['n_pareto'] > best_npareto_val:
                best_npareto_val = r['n_pareto']
                best_engine = engine

        # ── B2 Weighted-Sum GA: n_runs independent seeds ──────────────
        print(f"  [EVAL] B2 Weighted-Sum GA × {n_runs} runs...")
        b2_costs, b2_lats = [], []
        for seed in range(n_runs):
            b2 = WeightedSumBaseline(lpm, cost_weight=0.5, pop_size=100, n_gen=n_gen)
            r2 = b2.run(seed=seed)
            b2_costs.append(r2['cost'])
            b2_lats.append(r2['latency'])

        # ── B1 All-Small, B3 All-Large (deterministic) ────────────────
        static = {}
        for bl_label, vmi_name in [('B1-AllSmall', 'small'), ('B3-AllLarge', 'large')]:
            vmi_spec   = lpm.vmi_catalogue[vmi_name]
            total_cost = 0.0
            task_lats  = {}
            for tid in lpm.get_task_order():
                node = lpm.dag.nodes[tid]
                lat  = _task_latency(node['data_gb'], node['mode'],
                                     vmi_spec['cpu'], vmi_spec['mem_gb'])
                task_lats[tid] = lat
                total_cost    += _task_cost(lpm.vmi_catalogue, vmi_name, lat)
            paths    = lpm.get_all_paths()
            makespan = max(sum(task_lats[t] for t in p) for p in paths)
            static[bl_label] = {'cost': total_cost, 'latency': makespan}

        # ── Plot best Pareto front with baselines overlaid ────────────
        b2_mean_cost = sum(b2_costs) / len(b2_costs)
        b2_mean_lat  = sum(b2_lats)  / len(b2_lats)
        best_engine.save_pareto_plot(
            save_path      = f"pareto_{wl_name.lower()}.png",
            workload_label = wl_name,
            baselines      = {
                'B1 All-Small': (static['B1-AllSmall']['cost'],
                                 static['B1-AllSmall']['latency']),
                'B2 W-Sum GA':  (b2_mean_cost, b2_mean_lat),
                'B3 All-Large': (static['B3-AllLarge']['cost'],
                                 static['B3-AllLarge']['latency']),
            },
        )

        # ── Shared reference point for fair HV comparison ─────────────
        # Reference must dominate the worst point across ALL strategies
        ref_cost = max(static['B3-AllLarge']['cost'],
                       max(nsga_costs), max(b2_costs)) * 1.1
        ref_lat  = max(static['B1-AllSmall']['latency'],
                       max(nsga_lats), max(b2_lats)) * 1.1
        ref_point = np.array([ref_cost, ref_lat])
        hv_calc   = HV(ref_point=ref_point)

        nsga_hvs = [hv_calc(F) for F in nsga_pareto_Fs]
        b2_hvs   = [hv_calc(np.array([[c, l]])) for c, l in zip(b2_costs, b2_lats)]

        # ── Wilcoxon signed-rank test: NSGA-III HV vs B2 HV ──────────
        w_stat, p_val = wilcoxon(nsga_hvs, b2_hvs, alternative='greater')
        sig_label = ('***' if p_val < 0.001 else
                     '**'  if p_val < 0.01  else
                     '*'   if p_val < 0.05  else 'ns')

        # ── Print statistics table ─────────────────────────────────────
        W = 24
        print(f"\n  {'Strategy':<22} {'Cost mean±std':>{W}} {'Latency mean±std':>{W}} {'HV mean±std':>{W}}")
        print(f"  {'─'*22} {'─'*W} {'─'*W} {'─'*W}")
        print(f"  {'NSGA-III':<22}"
              f"  ${np.mean(nsga_costs):.4f}±{np.std(nsga_costs):.4f}"
              f"  {np.mean(nsga_lats):.1f}±{np.std(nsga_lats):.1f}s"
              f"  {np.mean(nsga_hvs):.4f}±{np.std(nsga_hvs):.4f}")
        print(f"  {'B2 Weighted-Sum GA':<22}"
              f"  ${np.mean(b2_costs):.4f}±{np.std(b2_costs):.4f}"
              f"  {np.mean(b2_lats):.1f}±{np.std(b2_lats):.1f}s"
              f"  {np.mean(b2_hvs):.4f}±{np.std(b2_hvs):.4f}")
        for bl, v in static.items():
            print(f"  {bl:<22}  ${v['cost']:.4f} (det.)    {v['latency']:.1f}s (det.)    — ")
        print(f"\n  HV reference point         : cost={ref_point[0]:.5f}, lat={ref_point[1]:.2f}s")
        print(f"  Wilcoxon HV (NSGA-III > B2): W={w_stat:.1f}, p={p_val:.6e} {sig_label}")
        print(f"  NSGA-III mean Pareto size  : {np.mean(nsga_npareto):.1f} ± {np.std(nsga_npareto):.1f} solutions")
        print(f"  NSGA-III best Pareto size  : {best_npareto_val} solutions (used for plot)")
        print(f"  NSGA-III mean overhead     : {np.mean(nsga_overheads):.3f} ± {np.std(nsga_overheads):.3f}s")

        all_data[wl_name] = {
            'nsga'   : {'costs': nsga_costs, 'lats': nsga_lats,
                        'hvs': nsga_hvs, 'overheads': nsga_overheads,
                        'npareto': nsga_npareto},
            'b2'     : {'costs': b2_costs, 'lats': b2_lats, 'hvs': b2_hvs},
            'b1'     : static['B1-AllSmall'],
            'b3'     : static['B3-AllLarge'],
            'wilcoxon': {'stat': w_stat, 'p_val': p_val, 'sig': sig_label},
        }

        # ── CSV rows ───────────────────────────────────────────────────
        for i in range(n_runs):
            csv_rows.append([wl_name, 'NSGA-III', i,
                             nsga_costs[i], nsga_lats[i], nsga_hvs[i],
                             nsga_overheads[i], nsga_npareto[i]])
        for i in range(n_runs):
            csv_rows.append([wl_name, 'B2-WeightedSumGA', i,
                             b2_costs[i], b2_lats[i], b2_hvs[i], '', ''])
        csv_rows.append([wl_name, 'B1-AllSmall', 0,
                         static['B1-AllSmall']['cost'],
                         static['B1-AllSmall']['latency'], '', '', ''])
        csv_rows.append([wl_name, 'B3-AllLarge', 0,
                         static['B3-AllLarge']['cost'],
                         static['B3-AllLarge']['latency'], '', '', ''])

    # ── Save raw data CSV ──────────────────────────────────────────────
    csv_path = "evaluation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['workload', 'strategy', 'run', 'cost_usd', 'latency_sec',
                    'hypervolume', 'overhead_sec', 'n_pareto'])
        w.writerows(csv_rows)
    print(f"\n  [EVAL] Raw results saved → {csv_path}")

    # ── Generate all Chapter 5 figures ────────────────────────────────
    _plot_chapter5(all_data)

    # ── Deploy balanced Medium plan ────────────────────────────────────
    print(f"\n[EVAL] Deploying Medium balanced plan ({('DRY-RUN' if dry_run else 'LIVE')})...")
    lpm_med = LogicalPipelineManager(PIPELINE_YAML)
    lpm_med.build_dag()
    lpm_med.scale_workload(1.0)
    engine_med = OrchestrationOptimizationEngine(lpm_med, pop_size=100, n_gen=n_gen)
    engine_med.run(seed=0)
    plan_med = engine_med.select_plan(cost_weight=0.5)
    KubeVirtAdapter(dry_run=dry_run).deploy_pipeline(plan_med, lpm_med.get_task_order())


def _plot_chapter5(all_data: dict):
    """
    Generates three publication-quality figures for Chapter 5:
      Figure 5.2 — Box plots: cost and latency per workload (chapter5_boxplots.png)
      Figure 5.3 — Hypervolume bar chart with Wilcoxon significance (chapter5_hv.png)
      Figure 5.4 — Scalability: cost and latency vs workload level (chapter5_scalability.png)
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workloads  = list(all_data.keys())
    COLORS = {
        'NSGA-III'        : '#2DC653',
        'B2 W-Sum GA'     : '#2176AE',
        'B1 All-Small'    : '#FF6B35',
        'B3 All-Large'    : '#9B59B6',
    }

    # ── Figure 5.2: Box plots (cost + latency, 2 rows × 3 workloads) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle('Cost and Latency Distribution Across Strategies',
                 fontsize=13, fontweight='bold', y=1.01)

    box_colors = [COLORS['NSGA-III'], COLORS['B2 W-Sum GA'],
                  COLORS['B1 All-Small'], COLORS['B3 All-Large']]
    labels = ['NSGA-III', 'B2', 'B1', 'B3']

    for col, wl in enumerate(workloads):
        d = all_data[wl]
        cost_data = [d['nsga']['costs'], d['b2']['costs'],
                     [d['b1']['cost']], [d['b3']['cost']]]
        lat_data  = [d['nsga']['lats'],  d['b2']['lats'],
                     [d['b1']['latency']], [d['b3']['latency']]]

        for row, (data, ylabel, title_suffix) in enumerate([
            (cost_data, 'Cost (USD)',   'Cost'),
            (lat_data,  'Latency (s)', 'Latency'),
        ]):
            ax = axes[row, col]
            ax.set_facecolor('white')
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 2})
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_title(f'{wl} — {title_suffix}', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, axis='y', color='#dddddd', linewidth=0.7)
            ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('chapter5_boxplots.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [EVAL] Saved → chapter5_boxplots.png")

    # ── Figure 5.3: HV bar chart with significance markers ─────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    ax.set_facecolor('white')
    x     = np.arange(len(workloads))
    width = 0.35

    nsga_hv_m = [np.mean(all_data[wl]['nsga']['hvs']) for wl in workloads]
    nsga_hv_s = [np.std(all_data[wl]['nsga']['hvs'])  for wl in workloads]
    b2_hv_m   = [np.mean(all_data[wl]['b2']['hvs'])   for wl in workloads]
    b2_hv_s   = [np.std(all_data[wl]['b2']['hvs'])    for wl in workloads]

    ax.bar(x - width/2, nsga_hv_m, width, yerr=nsga_hv_s,
           label='NSGA-III', color=COLORS['NSGA-III'],
           capsize=5, alpha=0.85, edgecolor='white')
    ax.bar(x + width/2, b2_hv_m, width, yerr=b2_hv_s,
           label='B2 Weighted-Sum GA', color=COLORS['B2 W-Sum GA'],
           capsize=5, alpha=0.85, edgecolor='white')

    # Significance markers
    for i, wl in enumerate(workloads):
        sig   = all_data[wl]['wilcoxon']['sig']
        y_top = max(nsga_hv_m[i], b2_hv_m[i]) + \
                max(nsga_hv_s[i], b2_hv_s[i]) * 1.5
        ax.text(i, y_top, sig, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=11)
    ax.set_xlabel('Workload Level', fontsize=11)
    ax.set_ylabel('Hypervolume Indicator', fontsize=11)
    ax.set_title('Hypervolume: NSGA-III vs B2 Weighted-Sum GA\n'
                 '(* p<0.05  ** p<0.01  *** p<0.001  ns = not significant)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', color='#dddddd', linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color('#aaaaaa')

    plt.tight_layout()
    plt.savefig('chapter5_hv.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [EVAL] Saved → chapter5_hv.png")

    # ── Figure 5.4: Scalability across workload levels ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    scale_labels = workloads
    nsga_c = [np.mean(all_data[wl]['nsga']['costs']) for wl in workloads]
    nsga_l = [np.mean(all_data[wl]['nsga']['lats'])  for wl in workloads]
    b2_c   = [np.mean(all_data[wl]['b2']['costs'])   for wl in workloads]
    b2_l   = [np.mean(all_data[wl]['b2']['lats'])    for wl in workloads]
    b1_c   = [all_data[wl]['b1']['cost']             for wl in workloads]
    b1_l   = [all_data[wl]['b1']['latency']          for wl in workloads]
    b3_c   = [all_data[wl]['b3']['cost']             for wl in workloads]
    b3_l   = [all_data[wl]['b3']['latency']          for wl in workloads]

    for ax, (nsga_v, b2_v, b1_v, b3_v, ylabel, title) in zip(axes, [
        (nsga_c, b2_c, b1_c, b3_c, 'Mean Cost (USD)',    'Cost Scalability'),
        (nsga_l, b2_l, b1_l, b3_l, 'Mean Latency (s)',   'Latency Scalability'),
    ]):
        ax.set_facecolor('white')
        ax.plot(scale_labels, nsga_v, 'o-', color=COLORS['NSGA-III'],
                lw=2, ms=8, label='NSGA-III')
        ax.plot(scale_labels, b2_v,   's-', color=COLORS['B2 W-Sum GA'],
                lw=2, ms=8, label='B2 W-Sum GA')
        ax.plot(scale_labels, b1_v,   '^--', color=COLORS['B1 All-Small'],
                lw=1.5, ms=7, label='B1 All-Small', alpha=0.8)
        ax.plot(scale_labels, b3_v,   'v--', color=COLORS['B3 All-Large'],
                lw=1.5, ms=7, label='B3 All-Large', alpha=0.8)
        ax.set_xlabel('Workload Level', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, color='#dddddd', linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#aaaaaa')

    plt.tight_layout()
    plt.savefig('chapter5_scalability.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [EVAL] Saved → chapter5_scalability.png")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSGA-III Multi-Modal Middleware — Phase 2"
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = "Simulate K8s API calls without a real cluster (default: False)"
    )
    parser.add_argument(
        "--cost-weight",
        type    = float,
        default = 0.5,
        help    = "Pareto plan preference: 0.0=min latency, 1.0=min cost (default: 0.5)"
    )
    parser.add_argument(
        "--pop-size",
        type    = int,
        default = 100,
        help    = "NSGA-III population size (default: 100)"
    )
    parser.add_argument(
        "--n-gen",
        type    = int,
        default = 100,
        help    = "NSGA-III generations (default: 100)"
    )
    parser.add_argument(
        "--evaluate",
        action  = "store_true",
        default = False,
        help    = "Run full baseline comparison evaluation"
    )

    args = parser.parse_args()

    if args.evaluate:
        run_evaluation(dry_run=args.dry_run, n_gen=args.n_gen)
    else:
        run_middleware(
            dry_run     = args.dry_run,
            cost_weight = args.cost_weight,
            pop_size    = args.pop_size,
            n_gen       = args.n_gen,
        )
