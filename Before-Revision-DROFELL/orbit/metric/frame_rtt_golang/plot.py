import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_results(all_rtts: Dict[str, List[float]]):
    plt.figure(figsize=(10, 5), dpi=330)
    for mode, rtts in all_rtts.items():
        plt.plot(rtts, label=f'{mode} (avg={sum(rtts)/len(rtts):.3f} ms)')
    plt.xlabel("Round")
    plt.ylabel("RTT (ms)")
    plt.title("RTT Comparison between UNIX and TCP sockets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "normal.png"))
    print("Saved normal.png")


with open(os.path.join(os.path.dirname(__file__), "out.json"), "r") as f:
    data = json.load(f)
    plot_results(data)
