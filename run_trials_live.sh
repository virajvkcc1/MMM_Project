#!/bin/bash

echo "Clearing old data..."
rm -f deployment_results.csv

for i in $(seq 1 30); do
    echo ""
    echo "=== Run $i/30 ==="

    # Delete previous VMIs before each run
    kubectl delete vmi --all -n speed-layer 2>/dev/null
    kubectl delete vmi --all -n batch-layer 2>/dev/null
    kubectl delete vmi --all -n serve-layer 2>/dev/null

    # Wait for deletion
    sleep 5

    # Live run
    python3 main.py
done

