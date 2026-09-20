# Stress-Sharing: Decentralized Fault Repair in Modular Spacecraft

![An example gif from the simulation framework](gifs/full_damage_response_dual_star.gif)

Decentralized reconfiguration algorithms for restoring connectivity in modular spacecraft after structural damage. Modules use only local information to coordinate a two-phase repair process — no central planner, no global knowledge.

## About

Structural damage in modular spacecraft disrupts connectivity between modules, potentially isolating subsystems and degrading mission capability. This project implements a fully decentralized two-phase reconfiguration algorithm where fault-adjacent modules autonomously recruit nearby movable modules to fill gaps and restore the structure. The approach relies entirely on local neighbor information — each module only knows its immediate surroundings — yet achieves global connectivity repair through emergent coordination.

## Algorithm Overview

### Phase 1 — Coagulation (First-Responder Distress Token Propagation)

Fault-adjacent modules emit direction tokens pointing toward the damage site. These tokens are unlabelled (no fault ID) and regenerated fresh each iteration.

1. Each module selects the closest received token
2. **Movable modules** consume the token and pivot one hop toward the fault
3. **Non-movable modules** relay the token onward
4. One hop per iteration; tokens propagate outward each round
5. Terminates when connectivity is restored

### Phase 2 — Restructuring (Unlabelled Slot-Filling)

After coagulation restores connectivity, displaced modules attempt to return the structure toward its original shape.

1. Only **non-movers** from Phase 1 generate tokens (they have accurate pre-damage direction knowledge)
2. Only **coag-movers** can respond (they are the displaced modules)
3. Token strategy is selectable: `furthest` (default), `nearest`, `random`
4. Slot-filling is position-based — any module can fill any open slot, not tied to specific IDs

### Concurrent Movement

Both phases support concurrent movement with a fully decentralized 2-hop exclusion protocol. Each iteration has two rounds:

1. **Communication round** — Each candidate module queries its neighbors, which check their own neighbors for the `is_moving` flag. If no neighbor-of-neighbor is moving, the module sets its own `is_moving` flag to announce intent. No graph mutation occurs during this round.
2. **Action round** — All committed modules execute their pivots. Each module re-checks destination occupancy locally on arrival (handling the rare case where two modules >2 hops apart targeted the same cell).

This is fully local — each module only reads its direct neighbors' state. The 2-hop exclusion guarantees that no two concurrent movers share a neighbor, preventing mid-pivot collisions without any central coordination.

### Movability Check

A module is movable only if removing it would not disconnect its local neighborhood. The safety radius is configurable (2, 3, or 4-hop neighborhood check).

### Shape Metric

Structural similarity is measured as:

```
diff(P, Q) = (|P| - |P ∩ Q|) / |P|
```

where P and Q are pairwise distance sets (unlabelled) of the original and current configurations.

## Getting Started

### Requirements

- Python >= 3.12
- Dependencies listed in `pyproject.toml`

### Installation

```bash
uv pip install -e .
```

or

```bash
pip install -r requirements.txt
```

### Quick Start

```python
from src.monte_carlo import run_single_graph_trial

# Generate a 15-module fully-connected structure, inject one fault that
# actually disconnects the active graph (auto-retries seeds up to 500x),
# and run the full decentralized coag + restruct pipeline on a
# GraphSimulator (perfect kinematic pivots, no physics).
result = run_single_graph_trial(
    n_modules=15,
    n_faults=1,
    seed=42,
    fully_connected=True,
    restructuring_method="displacement",
    temperature=0.5,
    use_flood_echo=False,
    max_moves_per_module=10,
    safety_radius=2,
)

print(f"Reconnected: {result.restored}")
print(f"Phase 1 moves: {result.phase1_moves}")
print(f"Phase 2 moves: {result.phase2_moves}")
print(f"Shape difference: {result.shape_difference}")
```

The same call signature drives the PyBullet runner via
``run_single_bullet_trial`` in `run_bullet_monte_carlo.py` — both
runners share the simulator-agnostic trial driver in `src/mc_runner.py`.

## Monte Carlo Simulation

### Basic Usage

```bash
py run_monte_carlo_sweep.py
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--n-min N` | Minimum number of modules (default: 5) |
| `--n-max N` | Maximum number of modules (default: 50) |
| `--n-step N` | Step size between n values (default: 1) |
| `--faults F` | Number of faults per trial (default: 1) |
| `--trials T` | Number of trials per configuration (default: 100) |
| `--seed S` | Random seed for reproducibility (default: 42) |
| `--jobs N` / `-j N` | Parallel jobs; -1 for all cores (default: -1) |
| `--tree` | Use tree-based configuration generation |
| `--chain-like` | Connect to only one adjacent module |
| `--mode-2d` | Use 2D mode (XY plane only) |
| `--dynamic-pct` | Run dynamic percentage fault sweep (10%, 20%, 30%) |
| `--cluster-faults` | Run cluster failure sweep |
| `--dynamic-faults` | Dynamic fault count: f = floor(n/10) |
| `--ablation` | Run token selection strategy ablation (furthest/nearest/random) |
| `--ablation-hops` | Run safety radius ablation (2/3/4-hop) |
| `--resume DIR` | Resume from existing output directory |
| `--no-graphs` | Skip graph generation |
| `--output-dir DIR` | Output directory (default: monte_carlo_results) |

### Example Commands

```bash
# Sweep n=10-100 with 500 trials, tree configs
py run_monte_carlo_sweep.py --n-min 10 --n-max 100 --trials 500 --tree

# Dynamic faults with cluster failure mode
py run_monte_carlo_sweep.py --dynamic-faults --cluster-faults

# Token strategy ablation on chain-like structures
py run_monte_carlo_sweep.py --ablation --chain-like

# Safety radius ablation in 2D
py run_monte_carlo_sweep.py --ablation-hops --mode-2d

# Resume a previous run
py run_monte_carlo_sweep.py --resume monte_carlo_results/20260119_113254
```

### Output Structure

Results are saved to timestamped directories:

```
monte_carlo_results/
└── 20260119_113254/
    ├── config.json                  # Simulation parameters
    ├── sweep_summary.csv            # Aggregated results per (n, f)
    ├── trials.csv                   # Individual trial data
    ├── graphs_combined.png          # All metrics in one figure
    ├── graph_reconnection_rate.png  # Reconnection rate plot
    └── graph_shape_difference.png   # Shape difference plot
```

### Metrics

- **Reconnection rate**: Fraction of trials where Phase 1 restored connectivity
- **Shape difference**: `diff(P, Q) = (|P| - |P∩Q|) / |P|` on pairwise distance sets — measures how well the structure recovered its original shape
- **Phase 1 / Phase 2 moves**: Number of pivot operations in each phase
- **Steps to reconnection**: Number of iterations Phase 1 needed
- **Token transmissions**: Total token relay operations (measures communication cost)

## Configuration Types

| Type | Flag | Description |
|------|------|-------------|
| **Fully connected** | *(default)* | Dense random walk growth, connects to all adjacent modules. Redundant paths make single faults less likely to disconnect. |
| **Tree** | `--tree` | Random spanning tree, no cycles, exactly n-1 edges. Every internal fault disconnects the structure. |
| **Chain-like** | `--chain-like` | Connects to only one random adjacent module. Minimal connectivity, more moves required to restore. |

## Fault Modes

The `--cluster-faults` flag runs a sweep across all spatial fault modes:

- **Random** (default): Faults injected at uniformly random positions
- **Cluster / Localized**: Single contiguous cluster grown via BFS from a random seed module
- **Random clusters**: Multiple small clusters (size 2-5) scattered across the structure

Additional fault scaling options:

- **Dynamic percentage** (`--dynamic-pct`): Sweeps over 10%, 20%, 30% fault rates
- **Dynamic count** (`--dynamic-faults`): f = floor(n/10), scaling with structure size

## Repository Structure

```
├── src/
│   ├── udqdg_system.py          # Core system: modules, edges, pivots, coagulation, restructuring
│   ├── monte_carlo.py           # Simulation framework: trials, aggregation, metrics
│   ├── configurations.py        # Predefined and random structure generators
│   ├── dual_quaternion.py       # Unit dual quaternion for lattice translations
│   └── visualizer.py            # PyVista 3D visualization
├── examples/
│   └── visualize_pivots.py      # Interactive pivot demos and GIF export
├── run_monte_carlo_sweep.py     # CLI for parameter sweeps
├── generate_comparison_plots.py # Multi-config plot generation
├── generate_cluster_fault_gif.py # Cluster fault animation generator
├── gifs/                        # Exported animation GIFs
└── monte_carlo_results/         # Simulation output data
```

## Visualization

Interactive 3D visualization powered by PyVista:

```bash
uv run examples/visualize_pivots.py star
uv run examples/visualize_pivots.py corner
uv run examples/visualize_pivots.py spiral
```

GIF export for pivot sequences:

```bash
uv run examples/visualize_pivots.py export corner gifs/corner_pivot.gif
uv run examples/visualize_pivots.py export spiral gifs/spiral.gif
```

See `gifs/` for pre-generated animation examples.

## License

See [LICENSE](LICENSE) for details.
