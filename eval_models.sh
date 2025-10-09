#!/usr/bin/env bash
set -euo pipefail

# Run from project root to keep relative asset paths working
cd "$(dirname "$0")"

# Auto-activate venv if present
if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Usage examples:
#   bash eval_models.sh
#   bash eval_models.sh --episodes 10 --max-cycles 1000
#   bash eval_models.sh --models models/dqn_model_*.pt models/dqn_final.pt --episodes 5
#   bash eval_models.sh --render --episodes 1 --max-cycles 600
python3 - "$@" <<'PY'
import os, glob, json, sys
import numpy as np
from test_only import evaluate

def main(argv):
    # Simple CLI passthrough (no extra deps)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None, help='Model paths or globs. Default: models/dqn_model_*.pt and models/dqn_final.pt')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--max-cycles', type=int, default=1000)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.0)
    ap.add_argument('--out', default='evaluation_results.jsonl')
    args = ap.parse_args(argv)

    patterns = args.models or ['models/dqn_model_*.pt', 'models/dqn_final.pt']
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    # dedupe while preserving order
    paths = list(dict.fromkeys(paths))

    if not paths:
        print("No models found. Try: --models models/dqn_model_*.pt models/dqn_final.pt", file=sys.stderr)
        return 2

    print(f"Evaluating {len(paths)} model(s):")
    for p in paths:
        print(f"  - {p}")

    results = []
    # Open JSONL to log all results
    with open(args.out, 'w') as fout:
        for model in paths:
            res = evaluate(
                model_path=model,
                episodes=args.episodes,
                render=args.render,
                max_cycles=args.max_cycles,
                sleep=args.sleep,
                log_interval=0,
                quiet=True  # keep stdout clean; we log structured JSON
            )
            row = {
                "model": model,
                "success_rate": float(res.get("success_rate", 0.0)),
                "avg_reward": float(np.mean(res["rewards"])) if res.get("rewards") else -1e9,
                "avg_len": float(np.mean(res["lengths"])) if res.get("lengths") else 0.0,
                "avg_capture_steps": float(np.mean(res["capture_steps"])) if res.get("capture_steps") else float("inf"),
            }
            results.append(row)
            print(json.dumps(row), file=fout)
            print(f"Done: {model} | success={row['success_rate']*100:.1f}% | avg_reward={row['avg_reward']:.2f} | avg_len={row['avg_len']:.1f}")

    # Pick best: highest success_rate, then highest avg_reward, then lowest avg_capture_steps
    results_sorted = sorted(results, key=lambda r: (-r["success_rate"], -r["avg_reward"], r["avg_capture_steps"]))
    best = results_sorted[0]
    print("\nBest model selected:")
    print(f"  {best['model']}")
    print(f"  success={best['success_rate']*100:.1f}% | avg_reward={best['avg_reward']:.2f} | avg_len={best['avg_len']:.1f} | avg_capture_steps={best['avg_capture_steps'] if np.isfinite(best['avg_capture_steps']) else 'NA'}")

    # Write best to file for downstream use
    with open('best_model.txt', 'w') as f:
        f.write(best['model'] + '\n')

    print("\nSummary written to:")
    print(f"  - JSONL: {os.path.abspath(args.out)}")
    print(f"  - Best:  {os.path.abspath('best_model.txt')}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PY
