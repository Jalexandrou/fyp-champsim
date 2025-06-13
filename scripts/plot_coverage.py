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
PREFETCHERS = ['bop', 'caerus']

# --- PARSE COVERAGE ---
def parse_coverage_from_file(filepath, prefetcher):
    if prefetcher == "berti":
        prefetch_line_prefix = "cpu0_L1D PREFETCH REQUESTED:"
        loadmiss_line_prefix = "cpu0_L1D LOAD"
    else:
        prefetch_line_prefix = "cpu0_L2C PREFETCH REQUESTED:"
        loadmiss_line_prefix = "cpu0_L2C LOAD"

    useful = None
    load_misses = None

    with open(filepath) as f:
        for line in f:
            if line.startswith(prefetch_line_prefix):
                parts = line.split()
                try:
                    useful = int(parts[7])
                except (IndexError, ValueError):
                    print(f"Warning: Failed to parse useful prefetches in {filepath}")
            elif line.startswith(loadmiss_line_prefix):
                parts = line.split()
                try:
                    load_misses = int(parts[7])
                except (IndexError, ValueError):
                    print(f"Warning: Failed to parse load misses in {filepath}")

    if useful is not None and load_misses is not None:
        total = useful + load_misses
        return useful / total if total > 0 else None
    return None

# --- GATHER COVERAGE DATA ---
coverage_data = defaultdict(dict)

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
                cov = parse_coverage_from_file(filepath, prefetcher)
                if cov is not None:
                    modified_simpoint = f"{benchmark_suffix}.{simpoint}"
                    label = f"{benchmark}/{modified_simpoint}"
                    coverage_data[prefetcher][label] = cov

EPSILON = 1e-6

def weighted_geomean(values, weights):
    log_sum = 0
    for v, w in zip(values, weights):
        log_sum += math.log(max(v, EPSILON)) * w
    return math.exp(log_sum)

geomean_coverage = defaultdict(dict)

for benchmark in BENCHMARKS:
    weight_map = SPEC2017_SHORTCODE_WEIGHTS.get(benchmark, {})
    simpoints = list(weight_map.keys())

    for prefetcher in PREFETCHERS:
        covs = []
        weights = []

        for sp in simpoints:
            label = f"{benchmark}/{sp}"
            cov = coverage_data[prefetcher].get(label)
            if cov is not None:
                covs.append(cov)
                weights.append(weight_map[sp])

        if covs and weights:
            geo = weighted_geomean(covs, weights)
        else:
            raise KeyError(f"Incomplete coverage data for benchmark: {benchmark}")

        geomean_coverage[prefetcher][benchmark] = geo

# Compute overall (equally-weighted) geomean
for prefetcher in PREFETCHERS:
    values = [geomean_coverage[prefetcher][bm] for bm in BENCHMARKS]
    if values:
        log_sum = sum(math.log(v) for v in values if v > 0)
        overall_geo = math.exp(log_sum / len(values))
    else:
        overall_geo = 0.0
    geomean_coverage[prefetcher]["geomean"] = overall_geo

# --- PLOTTING ---
all_labels = BENCHMARKS + ["geomean"]
display_labels = [bm[-3:] + '.' + bm[:-3] for bm in BENCHMARKS] + ["geomean"]
x = np.arange(len(all_labels))
bar_width = 0.3 / len(PREFETCHERS)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
})

fig, ax = plt.subplots(figsize=(13, 6))

for i, prefetcher in enumerate(PREFETCHERS):
    heights = [geomean_coverage[prefetcher].get(bm, 0.0) for bm in all_labels]
    offsets = x + i * bar_width
    ax.bar(offsets, heights, width=bar_width, label=prefetcher, edgecolor='black', linewidth=0.5)

ax.set_xticks(x + bar_width * (len(PREFETCHERS) - 1) / 2)
ax.set_xticklabels(display_labels, rotation=45, ha='right')
ax.set_ylim(bottom=0.0, top=1.0)
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
ax.set_ylabel("Coverage")
ax.set_xlabel("Benchmark")
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(PREFETCHERS),
    frameon=True,
    edgecolor='black'
)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
FIGURE_DIR = os.path.join(ROOT_DIR, 'figures_final')
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURE_DIR, 'COVERAGE_MEMINT.pdf'), format='pdf', dpi=300)
# plt.show()
