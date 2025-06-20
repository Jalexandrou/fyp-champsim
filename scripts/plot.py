import sys
import os
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

plt.style.use(["science", "light"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- CONFIGURABLE ---
LOG_DIR = os.path.join(ROOT_DIR, 'results', 'no_prefetching')
BENCHMARKS = ['cactusADM436', 'bwaves603', 'xalancbmk623']
BASELINE = 'no'
PREFETCHERS = ['no', 'berti', 'bop', 'no_caerus']
SIMPOINTS = []  # Will be auto-filled with (benchmark/simpoint)

# --- FUNCTION TO PARSE IPC ---
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

simpoint_labels = []

for benchmark in BENCHMARKS:
    for prefetcher in PREFETCHERS:
        path = os.path.join(LOG_DIR, prefetcher, benchmark)
        if not os.path.isdir(path):
            print(f"Missing directory: {path}")
            continue

        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                simpoint = filename.replace('.txt', '')
                label = f"{simpoint}"
                filepath = os.path.join(path, filename)
                ipc = parse_ipc_from_file(filepath)
                if ipc is not None:
                    ipc_data[prefetcher][label] = ipc
                    if label not in simpoint_labels:
                        simpoint_labels.append(label)

simpoint_labels = sorted(simpoint_labels)

# --- CALCULATE SPEEDUPS ---
speedup_data = defaultdict(list)
for prefetcher in PREFETCHERS:
    for label in simpoint_labels:
        base_ipc = ipc_data[BASELINE].get(label)
        test_ipc = ipc_data[prefetcher].get(label)
        if base_ipc and test_ipc:
            speedup = test_ipc / base_ipc
        else:
            speedup = 0.0
        speedup_data[prefetcher].append(speedup)

# --- PLOTTING ---
plot_prefetchers = [p for p in PREFETCHERS if p != BASELINE]
x = range(len(simpoint_labels))
bar_width = 0.3 / len(plot_prefetchers)

fig, ax = plt.subplots(figsize=(12, 6))

for i, prefetcher in enumerate(plot_prefetchers):
    offsets = [xi + i * bar_width for xi in x]
    ax.bar(offsets, speedup_data[prefetcher], width=bar_width, label=prefetcher, edgecolor='black', linewidth=0.5)

ax.axhline(1.0, linestyle='--', color='black', linewidth=1, label='baseline')

ax.set_xticks([xi + bar_width * (len(plot_prefetchers) - 1) / 2 for xi in x])
ax.set_xticklabels(simpoint_labels, rotation=45, ha='right')
ax.set_ylim(bottom=0.5) 
ax.set_ylabel("Speedup")
ax.set_xlabel("Simpoint")
# ax.set_title("Prefetcher Speedup Across Benchmarks")
ax.legend(
    loc='upper center',            
    bbox_to_anchor=(0.5, 1.15),   
    ncol=len(plot_prefetchers) + 1,  
    frameon=True,                 
    edgecolor='black'             
)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
FIGURE_DIR = os.path.join(ROOT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURE_DIR, 'test.pdf'), format='pdf', dpi=300)
# plt.show()
