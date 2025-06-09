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
from _SPEC2017_def_ALL_ import SPEC2017_BENCHMARKS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- CONFIGURABLE ---
LOG_DIR = os.path.join(ROOT_DIR, 'results', 'stride_baseline')
BENCHMARKS = ["gcc602"]
# BENCHMARKS = SPEC2017_BENCHMARKS
PREFETCHERS = ['stride', 'bop', 'berti_stride', 'caerus']
NUM_INSTRUCTIONS = 200_000_000
EPSILON = 1e-6

# --- PARSE MPKI DATA ---
def parse_misses_from_file(filepath):
    prefixes = {
        "L1D": "cpu0_L1D LOAD",
        "L2C": "cpu0_L2C LOAD",
        "LLC": "LLC LOAD"
    }
    misses = {}

    with open(filepath) as f:
        for line in f:
            for level, prefix in prefixes.items():
                if line.startswith(prefix):
                    parts = line.split()
                    try:
                        misses[level] = int(parts[7])
                    except (IndexError, ValueError):
                        print(f"Warning: Failed to parse {level} misses in {filepath}")
    return misses if misses else None

# --- COLLECT MPKI DATA ---
mpki_data = defaultdict(lambda: defaultdict(dict))  # mpki_data[prefetcher][level][label] = mpki

for benchmark in BENCHMARKS:
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
                misses = parse_misses_from_file(filepath)
                if misses:
                    label = f"{benchmark}/{benchmark_suffix}.{simpoint}"
                    for level, miss_count in misses.items():
                        mpki = miss_count / (NUM_INSTRUCTIONS / 1000)
                        mpki_data[prefetcher][level][label] = mpki

# --- COMPUTE GEOMEAN MPKI PER BENCHMARK ---
def weighted_geomean(values, weights):
    log_sum = 0
    for v, w in zip(values, weights):
        log_sum += math.log(max(v, EPSILON)) * w
    return math.exp(log_sum)

geomean_mpki = defaultdict(lambda: defaultdict(dict))  # [prefetcher][level][benchmark] = geomean

for benchmark in BENCHMARKS:
    weight_map = SPEC2017_SHORTCODE_WEIGHTS.get(benchmark, {})
    simpoints = list(weight_map.keys())

    for prefetcher in PREFETCHERS:
        for level in ["L1D", "L2C", "LLC"]:
            mpkis = []
            weights = []

            for sp in simpoints:
                label = f"{benchmark}/{sp}"
                mpki = mpki_data[prefetcher][level].get(label)
                if mpki is not None:
                    mpkis.append(mpki)
                    weights.append(weight_map[sp])

            if mpkis and weights:
                geo = weighted_geomean(mpkis, weights)
            else:
                raise KeyError(f"Incomplete MPKI data for {prefetcher} {benchmark}, {level}")
            geomean_mpki[prefetcher][level][benchmark] = geo

# --- COMPUTE OVERALL GEOMEAN MPKI ---
for prefetcher in PREFETCHERS:
    for level in ["L1D", "L2C", "LLC"]:
        values = [geomean_mpki[prefetcher][level][bm] for bm in BENCHMARKS]
        if values:
            log_sum = sum(math.log(max(v, EPSILON)) for v in values)
            overall_geo = math.exp(log_sum / len(values))
        else:
            overall_geo = 0.0
        geomean_mpki[prefetcher][level]["geomean"] = overall_geo

# --- PLOT SINGLE BAR CHART (X = L1D, L2C, LLC; Bars = Prefetchers) ---
FIGURE_DIR = os.path.join(ROOT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

levels = ["L1D", "L2C", "LLC"]
x = np.arange(len(levels))  # [0, 1, 2]
bar_width = 0.3 / len(PREFETCHERS)

for i, prefetcher in enumerate(PREFETCHERS):
    heights = [geomean_mpki[prefetcher][level]["geomean"] for level in levels]
    offsets = x + i * bar_width
    ax.bar(offsets, heights, width=bar_width, label=prefetcher, edgecolor='black', linewidth=0.5)

ax.set_xticks(x + bar_width * (len(PREFETCHERS) - 1) / 2)
ax.set_xticklabels(levels)
ax.set_ylabel("MPKI")
ax.set_xlabel("Cache Level")
ax.set_ylim(bottom=0)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(PREFETCHERS),
    frameon=True,
    edgecolor='black'
)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'cache_mpki.pdf'), format='pdf', dpi=300)
