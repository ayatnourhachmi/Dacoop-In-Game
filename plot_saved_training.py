import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def moving_average(x, w):
    if len(x) < w:
        return np.array([])
    return np.convolve(x, np.ones(w)/w, mode='valid')

def main(returns_path, success_path, window=100, out=None):
    returns = np.loadtxt(returns_path)
    success = np.loadtxt(success_path)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(returns, label='Return')
    if len(returns) > 20:
        ma20 = moving_average(returns, min(20, len(returns)))
        if len(ma20):
            plt.plot(range(len(returns)-len(ma20)+1, len(returns)+1), ma20, label='MA20')
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(1,2,2)
    if len(success):
        ma = moving_average(success, min(window, len(success)))
        if len(ma):
            plt.plot(ma, label=f"Success MA (win={min(window, len(success))})")
        else:
            plt.plot(success, label="Success")
    plt.ylim(-0.05, 1.05)
    plt.title("Success Rate Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend()

    plt.tight_layout()
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out)
        print(f"Saved plot to {out}")
    else:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="data/final_returns.txt")
    ap.add_argument("--success", default="data/final_success.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.returns, args.success, args.window, args.out)