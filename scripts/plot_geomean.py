import sys
import os
from matplotlib import use as plt_use
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

plt_use('pgf')
plt.style.use(["science", "light"])

from _SPEC_WEIGHTS import SPEC2017_SHORTCODE_WEIGHTS
from _SPEC2017_def_ALL_ import SPEC_MEMINT, SPEC2017_BENCHMARKS, SPEC2017_SHORTCODE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- CONFIGURABLE ---
LOG_DIR = os.path.join(ROOT_DIR, 'results_final', 'nocross')
BENCHMARKS = SPEC_MEMINT
BASELINE = 'stride'
PREFETCHERS = ['stride', 'berti', 'bop', 'caerus']

# --- PARSE IPC ---
def parse_ipc_from_file(filepath):
    in_roi_section = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if "Region of Interest Statistics" in line:
                in_roi_section = True
            elif in_roi_section and line.startswith("CPU 0 cumulative IPC:"):
                parts = line.split()
                try:
                    return float(parts[4]) 
                except (IndexError, ValueError):
                    continue
    return None

# --- GATHER RESULTS ---
ipc_data = defaultdict(dict)  # ipc_data[prefetcher]['benchmark/simpoint'] = ipc

for benchmark in BENCHMARKS:
    # Get the last 3 characters of the benchmark name
    benchmark_suffix = benchmark[-3:]

    for prefetcher in PREFETCHERS:
        path = os.path.join(LOG_DIR, prefetcher, benchmark)
        if not os.path.isdir(path):
            print(f"Missing directory: {path}")
            continue

        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                simpoint = filename.replace('.txt', '')
                filepath = os.path.join(path, filename)
                ipc = parse_ipc_from_file(filepath)
                if ipc is not None:
                    # Prepend the suffix to the simpoint
                    modified_simpoint = f"{benchmark_suffix}.{simpoint}"
                    label = f"{benchmark}/{modified_simpoint}"
                    ipc_data[prefetcher][label] = ipc

# --- COMPUTE WEIGHTED SPEEDUP ---
def weighted_geomean(values, weights):
    log_sum = 0
    for v, w in zip(values, weights):
        if v <= 0:
            return 0.0  # invalid speedup
        log_sum += math.log(v) * w
    return math.exp(log_sum)

geomean_speedups = defaultdict(dict)  # geomean_speedups[prefetcher][benchmark] = weighted_geomean

for benchmark in BENCHMARKS:
    weight_map = SPEC2017_SHORTCODE_WEIGHTS.get(benchmark, {})
    simpoints = list(weight_map.keys())

    for prefetcher in PREFETCHERS:
        speedups = []
        weights = []

        for sp in simpoints:
            label = f"{benchmark}/{sp}"
            base_ipc = ipc_data[BASELINE].get(label)
            test_ipc = ipc_data[prefetcher].get(label)

            if base_ipc and test_ipc:
                speedup = test_ipc / base_ipc
                weight = weight_map[sp]
                speedups.append(speedup)
                weights.append(weight)

        if speedups and weights:
            geo = weighted_geomean(speedups, weights)
        else:
            raise KeyError(
              f"Incomplete data for benchmark: {benchmark}")

        geomean_speedups[prefetcher][benchmark] = geo

plot_prefetchers = [p for p in PREFETCHERS if p != BASELINE]

# Compute overall (equally-weighted) geomean across benchmarks
for prefetcher in plot_prefetchers:
    # Get all benchmark-level geomeans for this prefetcher
    values = [geomean_speedups[prefetcher][bm] for bm in BENCHMARKS]
    if values:
        log_sum = sum(math.log(v) for v in values)
        overall_geo = math.exp(log_sum / len(values))
    else:
        overall_geo = 0.0
    geomean_speedups[prefetcher]["geomean"] = overall_geo

print("\n=== Geomean Speedups Over Baseline ===")
for prefetcher in plot_prefetchers:
    geo = geomean_speedups[prefetcher]["geomean"]
    print(f"{prefetcher}: {geo:.4f}x")

caerus_geo = geomean_speedups['caerus']["geomean"]
bop_geo = geomean_speedups['bop']["geomean"]
berti_geo = geomean_speedups['berti']["geomean"]

if bop_geo > 0:
    caerus_over_bop = caerus_geo / bop_geo
    print(f"\nCaerus over BOP geomean speedup: {caerus_over_bop:.4f}x")
else:
    print("\nWarning: BOP geomean is zero, cannot compute Caerus over BOP speedup.")

if berti_geo > 0:
    caerus_over_berti = caerus_geo / berti_geo
    print(f"Caerus over Berti geomean speedup: {caerus_over_berti:.4f}x")
else:
    print("Warning: Berti geomean is zero, cannot compute Caerus over Berti speedup.")

# --- PLOTTING ---
all_labels = BENCHMARKS + ["geomean"]
display_labels = [bm[-3:] + '.' + bm[:-3] for bm in BENCHMARKS] + ["geomean"]
x = np.arange(len(all_labels))
bar_width = 0.3 / len(plot_prefetchers)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
})

fig, ax = plt.subplots(figsize=(13, 6))

for i, prefetcher in enumerate(plot_prefetchers):
    heights = [geomean_speedups[prefetcher].get(bm, 0.0) for bm in all_labels]
    offsets = x + i * bar_width
    ax.bar(offsets, heights, width=bar_width, label=prefetcher, edgecolor='black', linewidth=0.5)

ax.axhline(1.0, linestyle='--', color='black', linewidth=1, label='baseline')

ax.set_xticks(x + bar_width * (len(plot_prefetchers) - 1) / 2)
ax.set_xticklabels(display_labels, rotation=45, ha='right')
ax.set_ylim(bottom=0.8)
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
ax.set_ylabel("Speedup")
ax.set_xlabel("Benchmark")
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(plot_prefetchers) + 1,
    frameon=True,
    edgecolor='black'
)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
FIGURE_DIR = os.path.join(ROOT_DIR, 'figures_final')
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURE_DIR, 'IPC_CROSSING_MEMINT.pdf'), format='pdf', dpi=300)
# plt.show()
