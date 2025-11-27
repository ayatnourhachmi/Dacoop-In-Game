# PowerShell version of eval_models.sh
param(
    [string[]]$models = $null,
    [int]$episodes = 5,
    [int]$maxCycles = 1000,
    [switch]$render,
    [double]$sleep = 0.0,
    [string]$out = "evaluation_results.jsonl"
)

# Change to script directory
Set-Location $PSScriptRoot

# Auto-activate venv if present (optional, skip if already activated)
if (Test-Path "venv\Scripts\Activate.ps1") {
    # & "venv\Scripts\Activate.ps1"
}

# Build Python script inline
$pythonScript = @'
import os, glob, json, sys
import numpy as np
sys.path.insert(0, r"C:\Github 2025\Dacoop-In-Game")
from test_only import evaluate

def init_pygame_display():
    import pygame
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.set_mode((800, 600))  # or your env size

def main(models, episodes, max_cycles, render, sleep, out):
    if render:
        init_pygame_display()
    
    patterns = models or ["models1/dqn_model_*.pt", "models1/dqn_final.pt"]
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    paths = list(dict.fromkeys(paths))
    
    if not paths:
        print("No models found. Try: --models models1/dqn_model_*.pt models1/dqn_final.pt", file=sys.stderr)
        return 2

    print(f"Evaluating {len(paths)} model(s):")
    for p in paths:
        print(f"  - {p}")
    
    results = []
    
    with open(out, "w") as fout:
        for model in paths:
            res = evaluate(
                model_path=model,
                episodes=episodes,
                render=render,
                max_cycles=max_cycles,
                sleep=sleep,
                log_interval=0,
                quiet=True
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
    
    results_sorted = sorted(results, key=lambda r: (-r["success_rate"], -r["avg_reward"], r["avg_capture_steps"]))
    best = results_sorted[0]

    print("\nBest model selected:")
    print(f"  {best['model']}")
    print(f"  success={best['success_rate']*100:.1f}% | avg_reward={best['avg_reward']:.2f} | avg_len={best['avg_len']:.1f} | avg_capture_steps={best['avg_capture_steps'] if np.isfinite(best['avg_capture_steps']) else 'NA'}")
    
    with open("best_model.txt", "w") as f:
        f.write(best["model"] + "\n")
    
    print("\nSummary written to:")
    print(f"  - JSONL: {os.path.abspath(out)}")
    print(f"  - Best:  {os.path.abspath("best_model.txt")}")
    
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-cycles", type=int, default=1000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out", default="evaluation_results.jsonl")
    
    args = ap.parse_args()
    sys.exit(main(args.models, args.episodes, args.max_cycles, args.render, args.sleep, args.out))
'@

# Save Python script to temp file
$tempPyFile = [System.IO.Path]::GetTempFileName() + ".py"
$pythonScript | Out-File -FilePath $tempPyFile -Encoding UTF8

try {
    $pythonArgs = @()
    if ($models) {
        $pythonArgs += "--models"
        $pythonArgs += $models
    }
    $pythonArgs += "--episodes", $episodes
    $pythonArgs += "--max-cycles", $maxCycles
    if ($render) { $pythonArgs += "--render" }
    $pythonArgs += "--sleep", $sleep
    $pythonArgs += "--out", $out

    & python $tempPyFile $pythonArgs
}
finally {
    if (Test-Path $tempPyFile) { Remove-Item $tempPyFile }
}
# 