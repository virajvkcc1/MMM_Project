"""
executor.py — KubeVirt Execution Adapter (Layer 3)
===================================================
Takes an orchestration plan from the optimizer and deploys
VirtualMachineInstance (VMI) objects to the Kubernetes/KubeVirt cluster
in DAG topological order.

Dry-run mode:
    Set DRY_RUN=True (or pass dry_run=True to KubeVirtAdapter)
    to simulate all K8s API calls without touching a real cluster.
    Dry-run produces the same log output and CSV results as a real run,
    making it useful for local testing before Phase 3 cluster deployment.

Thesis reference: Section 3.5.1 — KubeVirt Execution Adapter (Layer 3)
"""

import time
import csv
import json
import yaml
from datetime import datetime
from pathlib import Path

# ── Optional K8s import — skipped gracefully in dry-run mode ──────────────────
try:
    from kubernetes import client, config as k8s_config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("  [EXEC] WARNING: kubernetes package not found — dry-run only.")


# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_CSV = "deployment_results.csv"

VMI_RESOURCE_PROFILES = {
    "small" : {"cpu": "2",  "memory": "4Gi" },
    "medium": {"cpu": "4",  "memory": "8Gi" },
    "large" : {"cpu": "8",  "memory": "16Gi"},
}


class KubeVirtAdapter:
    """
    Layer 3 of the middleware.
    Translates an orchestration plan into K8s VirtualMachineInstance objects
    and applies them to the cluster via the Kubernetes Python API.

    Args:
        dry_run      : If True, simulate all API calls (no real cluster needed).
        kubeconfig   : Path to kubeconfig file. None = use default (~/.kube/config).
        results_csv  : Path to CSV file for logging deployment events.
    """

    def __init__(self, dry_run: bool = False,
                 kubeconfig: str = None,
                 results_csv: str = RESULTS_CSV):
        self.dry_run     = dry_run
        self.results_csv = results_csv
        self._custom_api = None
        self._core_api   = None

        mode_label = "DRY-RUN" if dry_run else "LIVE"
        print(f"  [EXEC] KubeVirtAdapter initialised [{mode_label}]")

        if not dry_run:
            self._connect(kubeconfig)

        # Ensure CSV has a header row on first use
        self._init_csv()

    # ── K8s connection ─────────────────────────────────────────────────────────

    def _connect(self, kubeconfig: str = None):
        if not K8S_AVAILABLE:
            raise RuntimeError(
                "kubernetes package not installed. "
                "Run: pip install kubernetes   or use dry_run=True."
            )
        if kubeconfig:
            k8s_config.load_kube_config(config_file=kubeconfig)
        else:
            k8s_config.load_kube_config()
        self._custom_api = client.CustomObjectsApi()
        self._core_api   = client.CoreV1Api()
        print("  [EXEC] Connected to Kubernetes cluster.")

    # ── CSV logging ────────────────────────────────────────────────────────────

    def _init_csv(self):
        p = Path(self.results_csv)
        if not p.exists():
            with open(p, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "timestamp", "event", "task_id", "vmi_name",
                    "vmi_type", "namespace", "cpu", "mem_gb",
                    "duration_sec", "status", "notes",
                ])

    def _log(self, event: str, task_id: str, vmi_name: str, vmi_type: str,
             namespace: str, cpu: float, mem_gb: float,
             duration: float, status: str, notes: str = ""):
        with open(self.results_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), event, task_id, vmi_name,
                vmi_type, namespace, cpu, mem_gb,
                round(duration, 4), status, notes,
            ])

    # ── Ensure namespace exists ────────────────────────────────────────────────

    def _ensure_namespace(self, namespace: str):
        if self.dry_run:
            print(f"  [EXEC]   [DRY-RUN] Namespace '{namespace}' — skipped create")
            return

        try:
            self._core_api.read_namespace(namespace)
        except Exception:
            ns_body = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=namespace)
            )
            self._core_api.create_namespace(ns_body)
            print(f"  [EXEC]   Created namespace '{namespace}'")

    # ── Build VMI manifest ─────────────────────────────────────────────────────

    def _build_vmi_manifest(self, task_id: str, cfg: dict) -> dict:
        """
        Builds a KubeVirt VirtualMachineInstance manifest from a plan entry.
        Mirrors the structure of test-vmi.yaml with optimized resource values.
        """
        vmi_name = f"task-{task_id.replace('_', '-')}"
        profile  = VMI_RESOURCE_PROFILES.get(cfg['vmi_type'], VMI_RESOURCE_PROFILES['medium'])

        # Use optimizer-chosen values if within VMI profile, else cap to profile
        cpu_req = f"{min(int(cfg['cpu']), int(profile['cpu']))}"
        mem_req = f"{min(round(cfg['mem_gb']), int(profile['memory'].replace('Gi','')))}Gi"

        return {
            "apiVersion": "kubevirt.io/v1",
            "kind":       "VirtualMachineInstance",
            "metadata": {
                "name":      vmi_name,
                "namespace": cfg['namespace'],
                "labels": {
                    "middleware/task-id"   : task_id,
                    "middleware/vmi-type"  : cfg['vmi_type'],
                    "middleware/mode"      : cfg['mode'],
                    "middleware/pipeline"  : "lambda-sentiment",
                },
                "annotations": {
                    "kubevirt.io/allow-pod-bridge-network-live-migration": "true",
                    "middleware/cost-weight": str(cfg.get('cost_weight', 0.5)),
                },
            },
            "spec": {
                "evictionStrategy": "LiveMigrate",
                "domain": {
                    "resources": {
                        "requests": {
                            "cpu":    cpu_req,
                            "memory": mem_req,
                        }
                    },
                    "devices": {
                        "disks": [
                            {"name": "containerdisk", "disk": {"bus": "virtio"}},
                        ],
                        "interfaces": [
                            {"name": "default", "masquerade": {}},
                        ],
                    },
                },
                "networks": [{"name": "default", "pod": {}}],
                "volumes": [{
                    "name": "containerdisk",
                    "containerDisk": {"image": cfg['image']},
                }],
            },
        }

    # ── Deploy single VMI ──────────────────────────────────────────────────────

    def deploy_vmi(self, task_id: str, cfg: dict) -> tuple:
        """
        Deploys one VMI for a pipeline task.

        Args:
            task_id : task identifier
            cfg     : task assignment dict from the orchestration plan

        Returns:
            (success: bool, details: dict)
        """
        t0       = time.time()
        manifest = self._build_vmi_manifest(task_id, cfg)
        vmi_name = manifest['metadata']['name']
        ns       = cfg['namespace']

        print(f"  [EXEC]   task={task_id:<22} vmi={vmi_name:<28} "
              f"type={cfg['vmi_type']:<8} ns={ns}")

        if self.dry_run:
            # ── DRY-RUN: simulate API call ─────────────────────────────────
            time.sleep(0.05)   # simulate small network latency
            duration = round(time.time() - t0, 4)
            details  = {
                "vmi_name"    : vmi_name,
                "namespace"   : ns,
                "dry_run"     : True,
                "manifest"    : manifest,
                "duration_sec": duration,
                "timestamp"   : datetime.now().isoformat(),
            }
            self._log("deploy", task_id, vmi_name, cfg['vmi_type'],
                      ns, cfg['cpu'], cfg['mem_gb'], duration, "DRY_RUN")
            print(f"  [EXEC]   ✓ [DRY-RUN] Would deploy '{vmi_name}' in '{ns}'")
            return True, details

        # ── LIVE: real K8s API call ────────────────────────────────────────
        try:
            self._ensure_namespace(ns)
            self._custom_api.create_namespaced_custom_object(
                group    = "kubevirt.io",
                version  = "v1",
                namespace= ns,
                plural   = "virtualmachineinstances",
                body     = manifest,
            )
            duration = round(time.time() - t0, 4)
            self._log("deploy", task_id, vmi_name, cfg['vmi_type'],
                      ns, cfg['cpu'], cfg['mem_gb'], duration, "SUCCESS")
            print(f"  [EXEC]   ✓ Deployed '{vmi_name}' in '{ns}' ({duration}s)")
            return True, {
                "vmi_name"    : vmi_name,
                "namespace"   : ns,
                "duration_sec": duration,
                "timestamp"   : datetime.now().isoformat(),
            }

        except Exception as e:
            duration = round(time.time() - t0, 4)
            self._log("deploy", task_id, vmi_name, cfg['vmi_type'],
                      ns, cfg['cpu'], cfg['mem_gb'], duration, "ERROR", str(e))
            print(f"  [EXEC]   ✗ FAILED '{vmi_name}': {e}")
            return False, {"error": str(e)}

    # ── Deploy full pipeline ───────────────────────────────────────────────────

    def deploy_pipeline(self, plan: dict, task_order: list) -> dict:
        """
        Deploys all pipeline VMIs in DAG topological order.

        Respects task dependencies — parent tasks are deployed before children,
        matching DAG precedence constraints (thesis Section 3.5.2).

        Args:
            plan       : orchestration plan from OrchestrationOptimizationEngine
            task_order : topologically sorted task list from LPM

        Returns:
            Deployment summary dict with per-task results.
        """
        assignments = plan['task_assignments']
        t_total     = time.time()

        print(f"\n  [EXEC] {'─'*55}")
        print(f"  [EXEC] Starting pipeline deployment")
        print(f"  [EXEC] Mode     : {'DRY-RUN' if self.dry_run else 'LIVE'}")
        print(f"  [EXEC] Tasks    : {len(task_order)}")
        print(f"  [EXEC] Est.Cost : ${plan['total_cost_usd']:.5f}")
        print(f"  [EXEC] Est.Lat  : {plan['total_latency_sec']:.1f}s")
        print(f"  [EXEC] {'─'*55}")

        results   = {}
        succeeded = 0
        failed    = 0

        # Deploy in topological order — guarantees dependency safety
        for task_id in task_order:
            if task_id not in assignments:
                print(f"  [EXEC]   SKIP '{task_id}' — not in plan")
                continue

            cfg             = assignments[task_id]
            success, detail = self.deploy_vmi(task_id, cfg)
            results[task_id] = {"success": success, "detail": detail}

            if success:
                succeeded += 1
            else:
                failed += 1
                # Stop on failure — downstream tasks depend on this one
                print(f"  [EXEC]   Pipeline halted — '{task_id}' failed.")
                break

        total_duration = round(time.time() - t_total, 3)
        status         = "success" if failed == 0 else "partial_failure"

        print(f"  [EXEC] {'─'*55}")
        print(f"  [EXEC] Deployed : {succeeded}/{len(task_order)} tasks")
        print(f"  [EXEC] Duration : {total_duration}s")
        print(f"  [EXEC] Status   : {status.upper()}")
        print(f"  [EXEC] Results  : {self.results_csv}")
        print(f"  [EXEC] {'─'*55}\n")

        return {
            "status"        : status,
            "succeeded"     : succeeded,
            "failed"        : failed,
            "total_duration": total_duration,
            "task_results"  : results,
            "dry_run"       : self.dry_run,
        }

    # ── Delete pipeline VMIs (cleanup) ────────────────────────────────────────

    def delete_pipeline(self, plan: dict, task_order: list):
        """
        Deletes all VMIs for a pipeline. Used for cleanup after evaluation runs.
        Always operates in reverse topological order (sinks first).
        """
        vmi_name = f"task-{task_id.replace('_', '-')}"
        assignments = plan['task_assignments']
        print(f"\n  [EXEC] Cleaning up pipeline VMIs...")
        for task_id in reversed(task_order):
            if task_id not in assignments:
                continue
            cfg      = assignments[task_id]
            ns       = cfg['namespace']
            vmi_name = f"task-{task_id}"

            if self.dry_run:
                print(f"  [EXEC]   [DRY-RUN] Would delete '{vmi_name}' from '{ns}'")
                continue

            try:
                self._custom_api.delete_namespaced_custom_object(
                    group    = "kubevirt.io",
                    version  = "v1",
                    namespace= ns,
                    plural   = "virtualmachineinstances",
                    name     = vmi_name,
                )
                print(f"  [EXEC]   Deleted '{vmi_name}'")
            except Exception as e:
                print(f"  [EXEC]   Could not delete '{vmi_name}': {e}")

    # ── Save plan to YAML (for inspection / reproducibility) ──────────────────

    def save_plan(self, plan: dict, path: str = "deployment_plan.yaml"):
        with open(path, 'w') as f:
            yaml.dump(plan, f, default_flow_style=False, sort_keys=False)
        print(f"  [EXEC] Plan saved → {path}")
