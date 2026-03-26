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


def run_evaluation(dry_run: bool = True, n_gen: int = 100):
    """
    Runs the full comparative evaluation (thesis Section 3.7.2):
      - NSGA-III Balanced
      - NSGA-III Cost-Optimised
      - NSGA-III Latency-Optimised
      - Baseline: Static cheap (always small VMI)
      - Baseline: Static large (always large VMI)

    Prints a comparison table suitable for the Results chapter.
    """
    import numpy as np

    print("\n" + "═"*60)
    print("  EVALUATION MODE — Baseline Comparison")
    print("═"*60)

    # Run optimiser once, then select three different plans
    print("\n[EVAL] Building DAG...")
    lpm = LogicalPipelineManager(PIPELINE_YAML)
    lpm.build_dag()

    print("[EVAL] Running NSGA-III (single run, 3 plan variants)...")
    engine = OrchestrationOptimizationEngine(lpm, pop_size=100, n_gen=n_gen)
    engine.run()

    strategies = [
        ("NSGA-III Balanced (0.5)",     engine.select_plan(cost_weight=0.5)),
        ("NSGA-III Cost-opt (0.9)",     engine.select_plan(cost_weight=0.9)),
        ("NSGA-III Latency-opt (0.1)",  engine.select_plan(cost_weight=0.1)),
    ]

    # Baseline: all-small and all-large (static heuristics)
    from optimizer import _task_latency, _task_cost, VMI_NAMES
    for strat_name, vmi_name in [("Baseline B1: All-Small", "small"),
                                  ("Baseline B3: All-Large", "large")]:
        vmi_spec   = lpm.vmi_catalogue[vmi_name]
        total_cost = 0.0
        task_lats  = {}
        for tid in lpm.get_task_order():
            node = lpm.dag.nodes[tid]
            lat  = _task_latency(node['data_gb'], node['mode'],
                                 vmi_spec['cpu'], vmi_spec['mem_gb'])
            cst  = _task_cost(lpm.vmi_catalogue, vmi_name, lat)
            task_lats[tid] = lat
            total_cost    += cst
        paths    = lpm.get_all_paths()
        makespan = max(sum(task_lats[t] for t in p) for p in paths)
        fake_plan = {
            'total_cost_usd'   : total_cost,
            'total_latency_sec': makespan,
            'task_assignments' : {
                tid: {'vmi_type': vmi_name, 'vmi_label': vmi_spec['label'],
                      'cpu': vmi_spec['cpu'], 'mem_gb': vmi_spec['mem_gb'],
                      'mode': lpm.dag.nodes[tid]['mode'],
                      'namespace': lpm.dag.nodes[tid]['namespace'],
                      'image': lpm.dag.nodes[tid]['image']}
                for tid in lpm.get_task_order()
            }
        }
        strategies.append((strat_name, fake_plan))

    # Print comparison table
    print(f"\n  {'Strategy':<30} {'Cost (USD)':>12} {'Latency (s)':>13}")
    print(f"  {'─'*30} {'─'*12} {'─'*13}")
    nsga_cost = strategies[0][1]['total_cost_usd']
    nsga_lat  = strategies[0][1]['total_latency_sec']
    for name, plan in strategies:
        cost = plan['total_cost_usd']
        lat  = plan['total_latency_sec']
        tag  = ""
        if 'NSGA' in name:
            cost_imp = (1 - cost/nsga_cost) * 100 if 'Balanced' not in name else 0
            tag = f"  ← proposed" if 'Balanced' in name else ""
        print(f"  {name:<30} ${cost:>10.5f}  {lat:>11.1f}s {tag}")

    print(f"\n  [EVAL] All results appended to deployment_results.csv")

    # Optionally deploy the balanced plan
    print(f"\n[EVAL] Deploying balanced plan ({('DRY-RUN' if dry_run else 'LIVE')})...")
    adapter = KubeVirtAdapter(dry_run=dry_run)
    adapter.deploy_pipeline(strategies[0][1], lpm.get_task_order())


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
