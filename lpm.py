"""
lpm.py — Logical Pipeline Manager (Layer 1)
============================================
Reads pipeline.yaml and constructs a validated NetworkX DAG.
Each node carries task metadata needed by the optimizer.

Thesis reference: Section 3.5.1 — Architecture Overview (Layer 1)
"""

import yaml
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class LogicalPipelineManager:
    """
    Parses a pipeline YAML definition and builds a directed acyclic graph.

    Attributes:
        dag          : nx.DiGraph — the pipeline DAG
        vmi_catalogue: dict — VMI type specs from YAML
        namespace_map: dict — mode → K8s namespace mapping
    """

    def __init__(self, yaml_path: str):
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline YAML not found: {yaml_path}")

        with open(path) as f:
            self.config = yaml.safe_load(f)

        self.dag           = nx.DiGraph()
        self.vmi_catalogue = self.config.get('vmi_catalogue', {})
        self.namespace_map = self.config.get('namespaces', {
            'stream': 'speed-layer',
            'batch':  'batch-layer',
            'serve':  'serve-layer',
        })
        self._task_index = {}   # task_id → task dict

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_dag(self) -> nx.DiGraph:
        """
        Constructs and validates the pipeline DAG from config.
        Returns the NetworkX DiGraph.
        """
        # Add nodes
        for task in self.config['tasks']:
            tid = task['id']
            self.dag.add_node(
                tid,
                name       = task['name'],
                mode       = task['mode'],           # 'batch' | 'stream' | 'serve'
                vmi_opts   = task['vmi_options'],    # candidate VMI names
                data_gb    = task.get('data_gb', 1.0),
                image      = task.get('image', 'ubuntu:22.04'),
                namespace  = self.namespace_map.get(task['mode'], 'default'),
            )
            self._task_index[tid] = task

        # Add edges (dependency constraints)
        for dep in self.config['dependencies']:
            src, dst = dep['from'], dep['to']
            if src not in self.dag:
                raise ValueError(f"Dependency source '{src}' not defined in tasks.")
            if dst not in self.dag:
                raise ValueError(f"Dependency target '{dst}' not defined in tasks.")
            self.dag.add_edge(src, dst)

        # Validate acyclicity
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError(
                "Pipeline YAML defines a cycle — invalid DAG. "
                "Check 'dependencies' for circular references."
            )

        return self.dag

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_task_order(self) -> list:
        """Returns task IDs in topological order (dependency-safe execution order)."""
        return list(nx.topological_sort(self.dag))

    def get_all_paths(self) -> list:
        """
        Returns all source-to-sink paths through the DAG.
        Used by the optimizer to compute the critical path latency (f2).
        """
        sources = [n for n in self.dag if self.dag.in_degree(n) == 0]
        sinks   = [n for n in self.dag if self.dag.out_degree(n) == 0]
        paths   = []
        for s in sources:
            for t in sinks:
                paths.extend(nx.all_simple_paths(self.dag, s, t))
        return paths

    def scale_workload(self, factor: float):
        """Scales data_gb for every task by factor. Used for Low/Medium/High workload experiments."""
        for tid in self.dag.nodes:
            self.dag.nodes[tid]['data_gb'] *= factor

    def get_node(self, task_id: str) -> dict:
        """Returns node attribute dict for a given task_id."""
        return self.dag.nodes[task_id]

    def summary(self):
        """Prints a human-readable pipeline summary."""
        print(f"\n{'─'*55}")
        print(f"  Pipeline : {self.config['pipeline']['name']}")
        print(f"  Tasks    : {self.dag.number_of_nodes()}")
        print(f"  Edges    : {self.dag.number_of_edges()}")
        print(f"  Topo order: {self.get_task_order()}")
        print(f"{'─'*55}")
        print(f"  {'Task ID':<22} {'Mode':<8} {'VMI Options':<25} {'Data GB'}")
        print(f"  {'─'*22} {'─'*8} {'─'*25} {'─'*7}")
        for tid in self.get_task_order():
            n = self.dag.nodes[tid]
            print(f"  {tid:<22} {n['mode']:<8} {str(n['vmi_opts']):<25} {n['data_gb']}")
        print(f"\n  Dependencies:")
        for src, dst in self.dag.edges:
            print(f"    {src}  →  {dst}")
        paths = self.get_all_paths()
        print(f"\n  All DAG paths ({len(paths)} total):")
        for p in paths:
            print(f"    {' → '.join(p)}")
        print(f"{'─'*55}\n")

    # ── Visualise ──────────────────────────────────────────────────────────────

    def visualize(self, save_path: str = "dag_visualization.png"):
        """
        Renders the DAG with colour-coded processing modes.
        Saves to a PNG file (safe for headless/server environments).
        """
        mode_colors = {
            'batch':  '#ff6b35',
            'stream': '#00c8ff',
            'serve':  '#00ff9d',
        }
        node_colors = [
            mode_colors.get(self.dag.nodes[n]['mode'], '#888888')
            for n in self.dag.nodes
        ]
        labels = {
            n: f"{n}\n({self.dag.nodes[n]['mode']})"
            for n in self.dag.nodes
        }

        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#080e14')
        ax.set_facecolor('#080e14')

        # Hierarchical layout for cleaner DAG rendering
        try:
            pos = nx.nx_agraph.graphviz_layout(self.dag, prog='dot')
        except Exception:
            pos = nx.spring_layout(self.dag, seed=42, k=2.5)

        nx.draw_networkx(
            self.dag, pos,
            labels      = labels,
            node_color  = node_colors,
            node_size   = 2400,
            font_size   = 8,
            font_color  = '#080e14',
            font_weight = 'bold',
            edge_color  = '#5a7a94',
            arrows      = True,
            arrowstyle  = '->',
            arrowsize   = 20,
            ax          = ax,
        )

        patches = [
            mpatches.Patch(color=c, label=m)
            for m, c in mode_colors.items()
        ]
        ax.legend(
            handles     = patches,
            facecolor   = '#0d1620',
            labelcolor  = 'white',
            loc         = 'upper left',
            fontsize    = 9,
        )
        ax.set_title(
            f"Lambda Architecture DAG — {self.config['pipeline']['name']}",
            color='white', fontsize=11, pad=14,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#080e14')
        plt.close()
        print(f"  [LPM] DAG visualisation saved → {save_path}")
