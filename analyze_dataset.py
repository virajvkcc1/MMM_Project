"""
analyze_dataset.py
==================
Reads Sentiment140 CSV and calculates real data_gb values
for each pipeline task. Updates pipeline.yaml automatically.

Usage:
    python3 analyze_dataset.py
"""

import os
import csv
import yaml

DATASET_PATH = "training.1600000.processed.noemoticon.csv"
PIPELINE_PATH = "pipeline.yaml"

def analyze():
    print("=" * 50)
    print("  Sentiment140 Dataset Analysis")
    print("=" * 50)

    # ── File size ──────────────────────────────────
    size_bytes = os.path.getsize(DATASET_PATH)
    size_mb    = size_bytes / (1024 * 1024)
    size_gb    = size_bytes / (1024 * 1024 * 1024)

    print(f"\n  File     : {DATASET_PATH}")
    print(f"  Size     : {size_mb:.1f} MB ({size_gb:.4f} GB)")

    # ── Row count + label distribution ────────────
    positive = 0
    negative = 0
    neutral  = 0
    total    = 0

    with open(DATASET_PATH, encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            total += 1
            sentiment = row[0].strip()
            if sentiment == '4':
                positive += 1
            elif sentiment == '0':
                negative += 1
            else:
                neutral += 1

    print(f"\n  Rows     : {total:,}")
    print(f"  Positive : {positive:,} ({positive/total*100:.1f}%)")
    print(f"  Negative : {negative:,} ({negative/total*100:.1f}%)")
    print(f"  Neutral  : {neutral:,}  ({neutral/total*100:.1f}%)")

    # ── Calculate data_gb per task ─────────────────
    # stream tasks process 50MB chunks at a time
    stream_chunk_gb = 0.050
    # batch task processes full dataset
    batch_gb        = round(size_gb, 4)
    # serve layer stores results (~40% of input size)
    serve_gb        = round(size_gb * 0.4, 4)

    print(f"\n  ── Recommended data_gb values for pipeline.yaml ──")
    print(f"  tweet_ingest    : {stream_chunk_gb}  (50MB stream chunk)")
    print(f"  stream_classify : {stream_chunk_gb}  (same stream chunk)")
    print(f"  batch_retrain   : {batch_gb}  (full {size_mb:.0f}MB dataset)")
    print(f"  result_store    : {serve_gb}  (~40% of input)")
    print(f"  serving_api     : {stream_chunk_gb}  (API response data)")

    # ── Update pipeline.yaml ───────────────────────
    with open(PIPELINE_PATH) as f:
        config = yaml.safe_load(f)

    data_map = {
        'tweet_ingest'    : stream_chunk_gb,
        'stream_classify' : stream_chunk_gb,
        'batch_retrain'   : batch_gb,
        'result_store'    : serve_gb,
        'serving_api'     : stream_chunk_gb,
    }

    for task in config.get('tasks', []):
        tid = task['id']
        if tid in data_map:
            old = task.get('data_gb', '?')
            task['data_gb'] = data_map[tid]
            print(f"  Updated {tid}: {old} → {data_map[tid]}")

    with open(PIPELINE_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✅ pipeline.yaml updated with real Sentiment140 sizes")
    print(f"  ✅ Re-run: python3 main.py --dry-run")
    print("=" * 50)

if __name__ == "__main__":
    analyze()
