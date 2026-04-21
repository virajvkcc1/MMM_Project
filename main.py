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
    Full statistical evaluation (thesis Section 3.7.2 and 3.7.3):
      - 30 independent NSGA-III runs per workload level (seeds 0–29)
      - B2 Weighted-Sum GA: 30 runs per workload level
      - B1 All-Small, B3 All-Large: deterministic static baselines
      - Workload levels: Low (0.5×), Medium (1.0×), High (2.0×) data_gb scaling

    Prints mean ± std tables per workload and saves all raw runs to CSV.
    """
    import numpy as np
    import csv
    from optimizer import WeightedSumBaseline, _task_latency, _task_cost

    print("\n" + "═"*68)
    print(f"  EVALUATION — {n_runs} runs × {len(WORKLOAD_LEVELS)} workload levels")
    print("═"*68)

    csv_rows = []

    for wl_name, wl_scale in WORKLOAD_LEVELS:
        print(f"\n  ── Workload: {wl_name} (data_gb × {wl_scale}) {'─'*30}")

        lpm = LogicalPipelineManager(PIPELINE_YAML)
        lpm.build_dag()
        lpm.scale_workload(wl_scale)

        # ── NSGA-III: n_runs independent seeds ────────────────────────
        print(f"  [EVAL] NSGA-III × {n_runs} runs...")
        nsga_costs, nsga_lats, nsga_hvs = [], [], []
        for seed in range(n_runs):
            engine = OrchestrationOptimizationEngine(lpm, pop_size=100, n_gen=n_gen)
            r = engine.run(seed=seed)
            p = engine.select_plan(cost_weight=0.5)
            nsga_costs.append(p['total_cost_usd'])
            nsga_lats.append(p['total_latency_sec'])
            if r['hypervolume'] is not None:
                nsga_hvs.append(r['hypervolume'])
            csv_rows.append([wl_name, 'NSGA-III', seed,
                             p['total_cost_usd'], p['total_latency_sec'],
                             r['hypervolume'] or ''])
            if seed == 0:
                engine.save_pareto_plot(save_path=f"pareto_{wl_name.lower()}.png")

        # ── B2 Weighted-Sum GA: n_runs independent seeds ──────────────
        print(f"  [EVAL] B2 Weighted-Sum GA × {n_runs} runs...")
        b2_costs, b2_lats = [], []
        for seed in range(n_runs):
            b2 = WeightedSumBaseline(lpm, cost_weight=0.5, pop_size=100, n_gen=n_gen)
            r2 = b2.run(seed=seed)
            b2_costs.append(r2['cost'])
            b2_lats.append(r2['latency'])
            csv_rows.append([wl_name, 'B2-WeightedSumGA', seed,
                             r2['cost'], r2['latency'], ''])

        # ── B1 All-Small, B3 All-Large (deterministic) ────────────────
        static_results = {}
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
            static_results[bl_label] = {'cost': total_cost, 'latency': makespan}
            csv_rows.append([wl_name, bl_label, 0, total_cost, makespan, ''])

        # ── Print statistics table ─────────────────────────────────────
        print(f"\n  {'Strategy':<22} {'Cost mean±std (USD)':>24} {'Latency mean±std (s)':>22}")
        print(f"  {'─'*22} {'─'*24} {'─'*22}")
        print(f"  {'NSGA-III Balanced':<22}"
              f"  ${np.mean(nsga_costs):.5f} ± {np.std(nsga_costs):.5f}"
              f"  {np.mean(nsga_lats):.1f} ± {np.std(nsga_lats):.1f}s")
        if nsga_hvs:
            print(f"    HV: {np.mean(nsga_hvs):.6f} ± {np.std(nsga_hvs):.6f}")
        print(f"  {'B2 Weighted-Sum GA':<22}"
              f"  ${np.mean(b2_costs):.5f} ± {np.std(b2_costs):.5f}"
              f"  {np.mean(b2_lats):.1f} ± {np.std(b2_lats):.1f}s")
        for bl_label, v in static_results.items():
            print(f"  {bl_label:<22}  ${v['cost']:>10.5f} (det.)   {v['latency']:>9.1f}s (det.)")

    # ── Save all raw runs to CSV ───────────────────────────────────────
    csv_path = "evaluation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['workload', 'strategy', 'run', 'cost_usd', 'latency_sec', 'hypervolume'])
        w.writerows(csv_rows)
    print(f"\n  [EVAL] Raw results saved → {csv_path}")

    # ── Deploy balanced Medium plan ────────────────────────────────────
    print(f"\n[EVAL] Deploying Medium balanced plan ({('DRY-RUN' if dry_run else 'LIVE')})...")
    lpm_med = LogicalPipelineManager(PIPELINE_YAML)
    lpm_med.build_dag()
    lpm_med.scale_workload(1.0)
    engine_med = OrchestrationOptimizationEngine(lpm_med, pop_size=100, n_gen=n_gen)
    engine_med.run(seed=0)
    plan_med = engine_med.select_plan(cost_weight=0.5)
    KubeVirtAdapter(dry_run=dry_run).deploy_pipeline(plan_med, lpm_med.get_task_order())


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
