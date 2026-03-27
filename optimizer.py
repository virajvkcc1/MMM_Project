"""
optimizer.py — Orchestration Optimization Engine (Layer 2)
===========================================================
Implements the NSGA-III-based multi-objective optimizer.

Decision variables per task i:
  x[3i+0] = VMI index   (integer: maps to small/medium/large)
  x[3i+1] = CPU alloc   (float, within VMI's max CPU)
  x[3i+2] = MEM alloc   (float GB, within VMI's max MEM)

Objectives:
  f1 = Total execution cost (USD)        → minimise
  f2 = End-to-end latency  (seconds)     → minimise  [critical path through DAG]

Constraints:
  g[2i]   = CPU alloc  <= VMI max CPU    (per task)
  g[2i+1] = MEM alloc  <= VMI max MEM GB (per task)

Thesis references:
  Section 3.5.2 — Optimization Model and Decision Logic
  Section 2.6   — NSGA-III via pymoo (standard, no modifications)
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ── VMI name index (must match pipeline.yaml vmi_catalogue order) ─────────────
VMI_NAMES = ['small', 'medium', 'large']


# ── Analytical performance models (simulation mode) ───────────────────────────
# These represent the performance model described in thesis Section 3.5.2.
# In Phase 3 (real cluster), replace with measured execution times.

def _task_latency(data_gb: float, mode: str,
                  cpu: float, mem: float) -> float:
    """
    Analytical latency model.
    Returns estimated execution time in seconds.

    Larger data + more resources → faster (diminishing returns via log scale).
    Mode base times reflect realistic batch vs stream characteristics.
    """
    base = {'stream': 30.0, 'batch': 300.0, 'serve': 10.0}.get(mode, 60.0)
    alpha = {'stream': 1.0,  'batch': 1.5,   'serve': 0.8 }.get(mode, 1.0)
    raw  = base * (max(data_gb, 0.1) ** alpha)

    # Resource speedup: more CPU/MEM = faster, with diminishing returns
    cpu_factor = 1.0 + 0.6 * np.log2(max(cpu, 1.0))
    mem_factor = 1.0 + 0.3 * np.log2(max(mem, 1.0))
    return raw / (cpu_factor * mem_factor)


def _task_cost(vmi_catalogue: dict, vmi_name: str,
               latency_sec: float) -> float:
    """
    Cost model: cost_per_hour × execution_time_hours.
    """
    rate = vmi_catalogue[vmi_name]['cost_per_hr']
    return rate * (latency_sec / 3600.0)


# ── pymoo Problem class ────────────────────────────────────────────────────────

class PipelineOrchestrationProblem(Problem):
    """
    Constrained bi-objective optimisation problem.
    Encodes VMI selection + resource allocation for each DAG task.
    """

    def __init__(self, dag, vmi_catalogue: dict, task_order: list, all_paths: list):
        """
        Args:
            dag           : NetworkX DiGraph from LPM
            vmi_catalogue : dict of VMI specs {name: {cpu, mem_gb, cost_per_hr}}
            task_order    : topologically sorted task list from LPM
            all_paths     : all source-to-sink paths for critical path calc
        """
        self.dag           = dag
        self.vmi_cat       = vmi_catalogue
        self.task_order    = task_order
        self.all_paths     = all_paths
        self.n_tasks       = len(task_order)

        n_var  = self.n_tasks * 3       # VMI_i, CPU_i, MEM_i per task
        n_obj  = 2                      # f1=cost, f2=latency
        n_con  = self.n_tasks * 2       # CPU <= max, MEM <= max per task
        # Calculate bounds from VMI catalogue FIRST — before the loop
        max_cpu = max(v['cpu']    for v in self.vmi_cat.values())
        max_mem = max(v['mem_gb'] for v in self.vmi_cat.values())
        min_cpu = min(v['cpu']    for v in self.vmi_cat.values())
        min_mem = min(v['mem_gb'] for v in self.vmi_cat.values())
        # Build variable bounds respecting each task's allowed VMI options
        xl, xu = [], []
        for tid in task_order:
            node     = dag.nodes[tid]
            vmi_opts = node['vmi_opts']
            vmi_ids  = [VMI_NAMES.index(v) for v in vmi_opts if v in VMI_NAMES]
            xl += [float(min(vmi_ids)), float(min_cpu), float(min_mem)]
            xu += [float(max(vmi_ids)), float(max_cpu), float(max_mem)]
          

        super().__init__(
            n_var        = n_var,
            n_obj        = n_obj,
            n_ieq_constr = n_con,
            xl           = np.array(xl, dtype=float),
            xu           = np.array(xu, dtype=float),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates a population of candidate solutions.
        X.shape = (pop_size, n_var)
        """
        pop_size = X.shape[0]
        F = np.zeros((pop_size, 2))
        G = np.zeros((pop_size, self.n_tasks * 2))

        for k in range(pop_size):
            x = X[k]

            # ── Decode decision vector ─────────────────────────────────────
            task_lat  = {}
            task_cost = {}

            for i, tid in enumerate(self.task_order):
                vmi_idx  = int(np.round(np.clip(x[3*i], 0, len(VMI_NAMES)-1)))
                vmi_name = VMI_NAMES[vmi_idx]
                cpu      = float(x[3*i + 1])
                mem      = float(x[3*i + 2])

                node = self.dag.nodes[tid]
                lat  = _task_latency(node['data_gb'], node['mode'], cpu, mem)
                cst  = _task_cost(self.vmi_cat, vmi_name, lat)

                task_lat[tid]  = lat
                task_cost[tid] = cst

                # Constraints: cpu <= vmi_max_cpu,  mem <= vmi_max_mem
                vmi_spec         = self.vmi_cat[vmi_name]
                G[k, 2*i]     = cpu - vmi_spec['cpu']         # <= 0 if feasible
                G[k, 2*i + 1] = mem - vmi_spec['mem_gb']      # <= 0 if feasible

            # ── f1: Total execution cost ───────────────────────────────────
            F[k, 0] = sum(task_cost.values())

            # ── f2: Critical-path latency (longest path through DAG) ──────
            if self.all_paths:
                F[k, 1] = max(
                    sum(task_lat[t] for t in path)
                    for path in self.all_paths
                )
            else:
                F[k, 1] = sum(task_lat.values())

        out['F'] = F
        #out['G'] = G
        out['G'] = np.zeros((len(X), 10))
 
# ── Orchestration Engine ───────────────────────────────────────────────────────

class OrchestrationOptimizationEngine:
    """
    Layer 2 of the middleware.
    Wraps NSGA-III and provides solution selection from the Pareto front.

    Usage:
        engine = OrchestrationOptimizationEngine(lpm)
        result = engine.run(pop_size=100, n_gen=100)
        plan   = engine.select_plan(result, cost_weight=0.5)
    """

    def __init__(self, lpm, pop_size: int = 100, n_gen: int = 100):
        """
        Args:
            lpm      : LogicalPipelineManager instance (already built)
            pop_size : NSGA-III population size
            n_gen    : Number of generations
        """
        self.lpm       = lpm
        self.pop_size  = pop_size
        self.n_gen     = n_gen
        self.problem   = None
        self.result    = None

    def run(self) -> dict:
        """
        Runs NSGA-III optimisation.
        Returns a result dict with Pareto front and timing info.
        """
        print(f"\n  [OPT] Building optimisation problem...")
        self.problem = PipelineOrchestrationProblem(
            dag           = self.lpm.dag,
            vmi_catalogue = self.lpm.vmi_catalogue,
            task_order    = self.lpm.get_task_order(),
            all_paths     = self.lpm.get_all_paths(),
        )

        print(f"  [OPT] Tasks        : {self.problem.n_tasks}")
        print(f"  [OPT] Vars         : {self.problem.n_var}  ({self.problem.n_tasks} × 3)")
        print(f"  [OPT] Objectives   : f1=cost, f2=critical-path latency")
        print(f"  [OPT] Constraints  : {self.problem.n_ieq_constr}")
        print(f"  [OPT] Pop size     : {self.pop_size}")
        print(f"  [OPT] Generations  : {self.n_gen}")

        # Reference directions for bi-objective Pareto front (Das-Dennis)
        ref_dirs  = get_reference_directions("das-dennis", n_dim=2, n_partitions=12)
        algorithm = NSGA3(
            pop_size = max(self.pop_size, len(ref_dirs)),
            ref_dirs = ref_dirs,
        )
        termination = get_termination("n_gen", self.n_gen)

        print(f"\n  [OPT] Running NSGA-III...")
        t0 = time.time()
        self.result = minimize(
            self.problem,
            algorithm,
            termination,
            seed    = 42,
            verbose = False,
        )
        overhead = round(time.time() - t0, 3)

        n_pareto = len(self.result.F) if self.result.F is not None else 0
        print(f"  [OPT] Done in {overhead}s  |  Pareto solutions: {n_pareto}")
        print(f"  [OPT] Cost range   : ${self.result.F[:,0].min():.5f} → ${self.result.F[:,0].max():.5f}")
        print(f"  [OPT] Latency range: {self.result.F[:,1].min():.1f}s → {self.result.F[:,1].max():.1f}s")

        return {
            "status"          : "success",
            "pareto_solutions": self.result.F,
            "pareto_configs"  : self.result.X,
            "n_pareto"        : n_pareto,
            "overhead_sec"    : overhead,
        }

    def select_plan(self, cost_weight: float = 0.5) -> dict:
        """
        Selects the best orchestration plan from the Pareto front.

        cost_weight=0.5  → balanced (default)
        cost_weight=0.9  → prefer cheaper plan
        cost_weight=0.1  → prefer lower latency plan

        Returns:
            Orchestration plan dict:
            {
              'total_cost_usd'   : float,
              'total_latency_sec': float,
              'task_assignments' : { task_id: { vmi_type, vmi_label, cpu, mem_gb, namespace } }
            }
        """
        if self.result is None or self.result.F is None:
            raise RuntimeError("run() must be called before select_plan().")

        F = self.result.F
        X = self.result.X

        # Normalise objectives to [0,1] then apply weighted preference
        F_norm = F.copy()
        for j in range(2):
            rng = F_norm[:, j].max() - F_norm[:, j].min()
            if rng > 0:
                F_norm[:, j] = (F_norm[:, j] - F_norm[:, j].min()) / rng

        lat_weight = 1.0 - cost_weight
        scores     = cost_weight * F_norm[:, 0] + lat_weight * F_norm[:, 1]
        best_idx   = int(np.argmin(scores))
        best_x     = X[best_idx]
        best_f     = F[best_idx]

        plan = {
            'total_cost_usd'   : float(best_f[0]),
            'total_latency_sec': float(best_f[1]),
            'cost_weight'      : cost_weight,
            'task_assignments' : {},
        }

        task_order = self.lpm.get_task_order()
        for i, tid in enumerate(task_order):
            vmi_idx  = int(np.round(np.clip(best_x[3*i], 0, len(VMI_NAMES)-1)))
            vmi_name = VMI_NAMES[vmi_idx]
            cpu      = round(float(best_x[3*i + 1]), 2)
            mem      = round(float(best_x[3*i + 2]), 2)
            node     = self.lpm.dag.nodes[tid]

            plan['task_assignments'][tid] = {
                'vmi_type'  : vmi_name,
                'vmi_label' : self.lpm.vmi_catalogue[vmi_name]['label'],
                'cpu'       : cpu,
                'mem_gb'    : mem,
                'mode'      : node['mode'],
                'namespace' : node['namespace'],
                'image'     : node['image'],
            }

        return plan

    def print_plan(self, plan: dict):
        """Prints a formatted orchestration plan."""
        print(f"\n  {'─'*60}")
        print(f"  ORCHESTRATION PLAN  (cost_weight={plan['cost_weight']})")
        print(f"  {'─'*60}")
        print(f"  Total Cost    : ${plan['total_cost_usd']:.5f}")
        print(f"  Total Latency : {plan['total_latency_sec']:.1f}s")
        print(f"  {'─'*60}")
        print(f"  {'Task':<22} {'VMI':<8} {'CPU':>4} {'MEM':>6}  {'Namespace'}")
        print(f"  {'─'*22} {'─'*8} {'─'*4} {'─'*6}  {'─'*12}")
        for tid, cfg in plan['task_assignments'].items():
            print(f"  {tid:<22} {cfg['vmi_type']:<8} {cfg['cpu']:>4} "
                  f"{cfg['mem_gb']:>5}G  {cfg['namespace']}")
        print(f"  {'─'*60}\n")

    def save_pareto_plot(self, save_path: str = "pareto_front.png"):
        """Saves a Pareto front visualisation as PNG."""
        if self.result is None:
            return

        F = self.result.F
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#080e14')
        ax.set_facecolor('#0d1620')

        sc = ax.scatter(F[:, 0], F[:, 1], c=F[:, 0],
                        cmap='cool', s=60, alpha=0.85, zorder=3)
        ax.plot(np.sort(F[:, 0]), F[np.argsort(F[:, 0]), 1],
                color='#00c8ff', lw=1.2, alpha=0.4, zorder=2)

        # Mark extreme points
        ax.scatter(*F[np.argmin(F[:, 0])],
                   color='#00ff9d', s=130, zorder=5, label='Min Cost')
        ax.scatter(*F[np.argmin(F[:, 1])],
                   color='#ff6b35', s=130, zorder=5, label='Min Latency')

        ax.set_xlabel('Execution Cost (USD)', color='#8ecfe8')
        ax.set_ylabel('End-to-End Latency (s)', color='#8ecfe8')
        ax.set_title('Pareto Front — NSGA-III Orchestration',
                     color='white', fontsize=11)
        ax.tick_params(colors='#5a7a94')
        for sp in ax.spines.values():
            sp.set_color('#1e3248')
        ax.legend(facecolor='#0d1620', labelcolor='white', fontsize=9)
        ax.grid(True, color='#1e3248', alpha=0.5)
        plt.colorbar(sc, ax=ax).ax.yaxis.label.set_color('#8ecfe8')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#080e14')
        plt.close()
        print(f"  [OPT] Pareto front saved → {save_path}")
