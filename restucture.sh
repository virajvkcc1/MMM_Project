#!/bin/bash
# =============================================================================
# MMM_Project Repository Restructure Script
# Run this from inside ~/MMM_project on mc-control
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  MMM_Project Repository Restructure"
echo "=============================================="

# ── Step 1: Create directory structure ────────────────────────────────────────
echo ""
echo "[1/6] Creating directory structure..."

mkdir -p src
mkdir -p configs
mkdir -p results
mkdir -p figures/pareto
mkdir -p docs
mkdir -p tests
mkdir -p scripts

echo "  ✓ Directories created"

# ── Step 2: Move source files ──────────────────────────────────────────────────
echo ""
echo "[2/6] Moving source files to src/..."

# Only move if files exist
[ -f main.py ]      && mv main.py      src/main.py      && echo "  ✓ main.py"
[ -f lpm.py ]       && mv lpm.py       src/lpm.py       && echo "  ✓ lpm.py"
[ -f optimizer.py ] && mv optimizer.py src/optimizer.py && echo "  ✓ optimizer.py"
[ -f executor.py ]  && mv executor.py  src/executor.py  && echo "  ✓ executor.py"

# ── Step 3: Move configs ───────────────────────────────────────────────────────
echo ""
echo "[3/6] Moving config files to configs/..."

[ -f pipeline.yaml ]        && mv pipeline.yaml        configs/pipeline.yaml        && echo "  ✓ pipeline.yaml"
[ -f deployment_plan.yaml ] && mv deployment_plan.yaml configs/deployment_plan.yaml && echo "  ✓ deployment_plan.yaml"

# ── Step 4: Move results ───────────────────────────────────────────────────────
echo ""
echo "[4/6] Moving result files to results/..."

[ -f deployment_results.csv ]  && mv deployment_results.csv  results/deployment_results.csv  && echo "  ✓ deployment_results.csv"
[ -f evaluation_results.csv ]  && mv evaluation_results.csv  results/evaluation_results.csv  && echo "  ✓ evaluation_results.csv"
[ -f convergence_results.csv ] && mv convergence_results.csv results/convergence_results.csv && echo "  ✓ convergence_results.csv"

# ── Step 5: Move figures ───────────────────────────────────────────────────────
echo ""
echo "[5/6] Moving figures to figures/..."

[ -f dag_visualization.png ] && mv dag_visualization.png figures/dag_visualization.png && echo "  ✓ dag_visualization.png"
[ -f pareto_front.png ]      && mv pareto_front.png      figures/pareto/pareto_combined.png && echo "  ✓ pareto_front.png"
[ -f pareto_low.png ]        && mv pareto_low.png        figures/pareto/pareto_low.png    && echo "  ✓ pareto_low.png"
[ -f pareto_medium.png ]     && mv pareto_medium.png     figures/pareto/pareto_medium.png && echo "  ✓ pareto_medium.png"
[ -f pareto_high.png ]       && mv pareto_high.png       figures/pareto/pareto_high.png   && echo "  ✓ pareto_high.png"

# Move any remaining PNGs to figures/
for f in *.png; do
  [ -f "$f" ] && mv "$f" figures/ && echo "  ✓ $f"
done

echo "[6/6] Done. Run ls -R to verify structure."
echo "=============================================="
