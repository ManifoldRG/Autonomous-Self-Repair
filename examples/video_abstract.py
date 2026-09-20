"""
Video-abstract clip generator for the Stress-Sharing T-AES submission.

Produces baseline MP4 clips of the full two-phase repair pipeline
(coagulation + retrace restructuring) at the paper's tuned operating
point, rendered offscreen with PyVista from trajectories recorded via
``mc_runner.run_trial(frame_callback=...)``.

Scenarios:
  1  5-module L-shape, fault at the corner module
  2  10-module tree, 3 random faults
  3  10-module fully-connected, 3 random faults
  4  10-module tree, 3 localized faults at the structure center
  5  10-module fully-connected, 3 localized faults at the center
  6  50-module random structure (FC generator), 5 (10%) random faults

Usage:
  py examples/video_abstract.py --scenario 1
  py examples/video_abstract.py --all
  py examples/video_abstract.py --scenario 6 --rerender   # re-render from cache

Recorded frames are cached to <out-dir>/<name>_frames.json.gz so the
(slow) physics run never has to repeat for a re-render.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.configurations import (
    create_random_configuration,
    create_random_tree_configuration,
)
from src.monte_carlo import (
    FAULT_MODE_LOCALIZED,
    FAULT_MODE_RANDOM,
    _bfs_grow,
    _causes_disconnection,
    select_faulty_modules,
)
from src.udqdg_system import UDQDGSystem


# ---------------------------------------------------------------------------
# Tuned operating point (matches the paper's 500-trial main runs)
# ---------------------------------------------------------------------------

TUNED = dict(
    temperature=1.0,
    pivot_exclusion_radius=4,
    dt=0.1,
    restructuring_method="retrace",
    token_strategy="furthest",
    safety_radius=2,
    use_flood_echo=True,
    token_gen_interval=2.0,
    max_moves_per_module=5,
    use_position_history=True,
    random_baseline=False,
    forward_ap_cost=0.1,
)

# Hold lengths for the synthetic intro, in *recorded* frames. Recorded
# frames are expanded --interp x at render time, so with the defaults
# (interp=3, 30 fps) 15 recorded frames = 1.5 s of video.
INTACT_HOLD = 15
DAMAGE_HOLD = 15


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_lshape() -> Tuple[UDQDGSystem, List[str], int]:
    """5-module L in the z=0 plane; the corner module is the fault."""
    system = UDQDGSystem()
    coords = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0)]
    for i, c in enumerate(coords):
        system.add_module(f"M{i}", np.array(c, dtype=float))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        system.connect_modules(f"M{a}", f"M{b}")
    return system, ["M2"], 0


def build_vicsek13() -> Tuple[UDQDGSystem, List[str], int]:
    """13-module 3D Vicsek fractal (first iteration): a center module with
    six 2-module arms along +/-x, +/-y, +/-z. The center module is the
    fault, disconnecting all six arms."""
    system = UDQDGSystem()
    system.add_module("M0", np.array([0.0, 0.0, 0.0]))
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                  (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    k = 1
    for d in directions:
        d = np.array(d, dtype=float)
        inner, outer = f"M{k}", f"M{k + 1}"
        system.add_module(inner, d)
        system.add_module(outer, 2.0 * d)
        system.connect_modules("M0", inner)
        system.connect_modules(inner, outer)
        k += 2
    return system, ["M0"], 0


def build_line20() -> Tuple[UDQDGSystem, List[str], int]:
    """20-module line along x; M8, M10, M12 are the faults (symmetric
    about the center), leaving an 8-module segment, two stranded
    singletons (M9, M11), and a 7-module segment."""
    system = UDQDGSystem()
    for i in range(20):
        system.add_module(f"M{i}", np.array([float(i), 0.0, 0.0]))
        if i > 0:
            system.connect_modules(f"M{i - 1}", f"M{i}")
    return system, ["M8", "M10", "M12"], 0


def select_center_cluster_faults(system, n_faults: int) -> List[str]:
    """Localized cluster seeded at the module nearest the centroid."""
    positions = {mid: m.position for mid, m in system.modules.items()}
    centroid = np.mean(list(positions.values()), axis=0)
    seed_module = min(
        positions, key=lambda mid: float(np.linalg.norm(positions[mid] - centroid)))
    return _bfs_grow(system, seed_module, n_faults, excluded=set())


def build_random_scenario(
    n_modules: int,
    n_faults: int,
    base_seed: int,
    tree: bool,
    fault_mode: str,
) -> Tuple[UDQDGSystem, List[str], int]:
    """Generate structure + fault set, retrying seeds until damage
    disconnects the structure (same idiom as run_single_bullet_trial).

    ``fault_mode``: "random", "center" (localized cluster pinned to the
    structure centroid), or "localized" (paper's localized mode: BFS
    cluster from a random seed module).
    """
    for attempt in range(500):
        gen_seed = base_seed + attempt * 9973
        if tree:
            system = create_random_tree_configuration(
                n_modules, seed=gen_seed, balanced=False)
        else:
            system = create_random_configuration(
                n_modules, seed=gen_seed, fully_connected=True)

        if fault_mode == "center":
            fault_ids = select_center_cluster_faults(system, n_faults)
        elif fault_mode == "localized":
            fault_ids = select_faulty_modules(
                system, n_faults, gen_seed + 1000, FAULT_MODE_LOCALIZED)
        else:
            fault_ids = select_faulty_modules(
                system, n_faults, gen_seed + 1000, FAULT_MODE_RANDOM)

        if fault_ids and _causes_disconnection(system, fault_ids):
            return system, fault_ids, gen_seed
    raise RuntimeError(
        f"no disconnecting fault set found (n={n_modules}, faults={n_faults})")


SCENARIOS = {
    1: dict(name="scenario1_lshape",
            title="5-module L-shape, corner fault",
            # extra freeze on the nominal structure for narration in the edit
            pre_hold=10.0,
            builder=lambda seed: build_lshape()),
    # Pinned seeds pick demo-worthy instances: repair succeeds AND phase 2
    # performs visible retrace moves (scanned 2026-07-01; PyBullet trials
    # are not bit-deterministic, so a re-sim can differ from the cached run).
    2: dict(name="scenario2_tree10_random",
            title="10-module tree, 3 random faults",
            seed=47,
            builder=lambda seed: build_random_scenario(10, 3, seed, True, "random")),
    3: dict(name="scenario3_fc10_random",
            title="10-module fully connected, 3 random faults",
            seed=57,
            builder=lambda seed: build_random_scenario(10, 3, seed, False, "random")),
    4: dict(name="scenario4_tree10_center",
            title="10-module tree, 3 localized faults (center)",
            seed=49,
            builder=lambda seed: build_random_scenario(10, 3, seed, True, "center")),
    5: dict(name="scenario5_fc10_center",
            title="10-module fully connected, 3 localized faults (center)",
            # 42-45 give instances whose repair fails; 46 repairs
            seed=46,
            builder=lambda seed: build_random_scenario(10, 3, seed, False, "center")),
    6: dict(name="scenario6_tree50",
            title="50-module random tree, 15 (30%) localized faults",
            # 30% localized damage reconnects ~33% of tree instances at n=50;
            # 49 is the first scanned seed that repairs (verified single
            # bonded component) with phase-2 activity.
            # camera on the opposite corner: most of this repair's motion
            # happens on the structure's far side under the default view.
            seed=49, view_dir=(-1.0, -1.0, 0.8),
            builder=lambda seed: build_random_scenario(50, 15, seed, True, "localized")),
    # --- outreach GIF scenarios (2026-07-24) ---
    7: dict(name="scenario7_vicsek13",
            title="13-module Vicsek fractal, center fault",
            builder=lambda seed: build_vicsek13()),
    8: dict(name="scenario8_line20",
            title="20-module line, faults at M8, M10, M12",
            # demo-only: lines are the policy's worst case (only segment
            # ends can move), so the clip runs a larger motion budget than
            # the paper's operating point
            tuned=dict(max_moves_per_module=15),
            builder=lambda seed: build_line20()),
    9: dict(name="scenario9_tree150",
            title="150-module tree, 30% localized faults",
            builder=lambda seed: build_random_scenario(150, 45, seed, True, "center")),
}


# ---------------------------------------------------------------------------
# Simulation + trajectory recording
# ---------------------------------------------------------------------------

def _bond_pairs(bond_matrix: np.ndarray) -> List[List[int]]:
    iu = np.triu(bond_matrix, 1)
    return [[int(i), int(j)] for i, j in zip(*np.nonzero(iu))]


def simulate_scenario(key: int, base_seed: int, sample_every: int) -> Dict:
    """Run one scenario through the tuned MC pipeline, recording frames."""
    from src.agent_policy import ModuleState
    from src.bullet_sim import BulletSimulator
    from src.mc_runner import run_trial, udqdg_to_scenario

    spec = SCENARIOS[key]
    system, fault_ids, gen_seed = spec["builder"](base_seed)
    print(f"[{spec['name']}] structure seed={gen_seed} faults={fault_ids}")

    for fid in fault_ids:
        if system.modules[fid].is_active:
            system.mark_fault(fid)
    scenario = udqdg_to_scenario(system, fault_ids)

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = 20.0
    sim = BulletSimulator(
        scenario.n_total, scenario.pos0, scenario.bonded0,
        gui=False, module_shape="sphere")

    frames: List[Dict] = []

    # Synthetic intro: pristine structure, then the faults appear.
    pairs0 = _bond_pairs(scenario.bonded0)
    pos0_list = scenario.pos0.tolist()
    for _ in range(INTACT_HOLD):
        frames.append(dict(t=0.0, phase="intact", pos=pos0_list,
                           bonds=pairs0, moving=[], token=[]))
    for _ in range(DAMAGE_HOLD):
        frames.append(dict(t=0.0, phase="damage", pos=pos0_list,
                           bonds=pairs0, moving=[], token=[]))

    moving_states = (ModuleState.PIVOTING, ModuleState.REVERSING)

    def frame_cb(sim, policy, phase, ticks):
        if ticks % sample_every:
            return
        pos = sim.get_positions()
        pairs = _bond_pairs(sim.get_bond_matrix())
        moving, token = [], []
        for mid, agent in policy.agents.items():
            if agent.state in moving_states:
                moving.append(mid)
            if agent.incoming_tokens:
                token.append(mid)
        frames.append(dict(
            t=float(sim.sim_time), phase=phase,
            pos=np.asarray(pos).tolist(), bonds=pairs,
            moving=moving, token=token))

    try:
        result = run_trial(
            sim=sim,
            scenario=scenario,
            faulty_modules_set=set(fault_ids),
            original_positions=scenario.original_positions,
            trial_id=key,
            n_modules=len(system.modules),
            n_faults=len(fault_ids),
            frame_callback=frame_cb,
            **{**TUNED, **spec.get("tuned", {})},
        )
    finally:
        sim.disconnect()

    print(f"[{spec['name']}] reconnected={result.restored} "
          f"p1_moves={result.phase1_moves} p2_moves={result.phase2_moves} "
          f"frames={len(frames)}")

    return dict(
        name=spec["name"],
        title=spec["title"],
        seed=gen_seed,
        n_total=scenario.n_total,
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
        fault_ids=fault_ids,
        fault_body_idxs=scenario.fault_body_idxs,
        reconnected=bool(result.restored),
        phase1_moves=int(result.phase1_moves),
        phase2_moves=int(result.phase2_moves),
        frames=frames,
    )


# ---------------------------------------------------------------------------
# Idle compression
# ---------------------------------------------------------------------------

def inject_connectivity_wave(frames: List[Dict], fault_idx_set: Set[int],
                             n_total: int, hold: int) -> List[Dict]:
    """After the last coagulation frame, insert a BFS 'connectivity wave':
    the active module nearest the centroid turns green, then each further
    ring of bonded neighbors joins (``hold`` recorded frames per ring)
    until the whole bonded network is green. Runs on the actual bond graph
    of that frame, so the wave covers the full structure only if it truly
    reconnected."""
    idx = max((i for i, f in enumerate(frames)
               if f["phase"] == "coagulation"), default=None)
    if idx is None:
        return frames
    base = frames[idx]
    active = [i for i in range(n_total) if i not in fault_idx_set]
    pos = np.asarray(base["pos"])
    centroid = pos[active].mean(axis=0)
    center = min(active, key=lambda i: float(np.linalg.norm(pos[i] - centroid)))

    adj: Dict[int, List[int]] = {i: [] for i in active}
    for a, b in base["bonds"]:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)

    wave_frames: List[Dict] = []
    green: List[int] = []
    seen = {center}
    frontier = [center]
    while frontier:
        green = green + frontier
        wf = dict(base, phase="connected", wave=list(green),
                  moving=[], token=[])
        wave_frames.extend([wf] * hold)
        nxt: List[int] = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return frames[:idx + 1] + wave_frames + frames[idx + 1:]


def stride_sim_frames(frames: List[Dict], stride: int) -> List[Dict]:
    """Keep every ``stride``-th simulation frame (intro and wave frames
    exempt). With 10 recorded frames per sim-second, playback speed
    relative to real time is fps * stride / (10 * interp)."""
    if stride <= 1:
        return frames
    out, k = [], 0
    for f in frames:
        if f["phase"] in ("intact", "damage", "connected"):
            out.append(f)
            continue
        if k % stride == 0:
            out.append(f)
        k += 1
    return out


def trim_intro(frames: List[Dict], keep: int) -> List[Dict]:
    """Keep at most ``keep`` recorded frames of each synthetic intro phase
    so the intro length stays constant regardless of --interp."""
    out = []
    counts = {"intact": 0, "damage": 0}
    for f in frames:
        ph = f["phase"]
        if ph in counts:
            counts[ph] += 1
            if counts[ph] > keep:
                continue
        out.append(f)
    return out


def compress_idle(frames: List[Dict], max_hold: int) -> List[Dict]:
    """Collapse runs of static frames (no motion, unchanged token set,
    same phase) to at most ``max_hold`` frames. Intro frames are exempt."""
    if not frames:
        return frames
    out = [frames[0]]
    static_run = 0
    for prev, cur in zip(frames, frames[1:]):
        if cur["phase"] in ("intact", "damage", "connected"):
            out.append(cur)
            continue
        delta = np.max(np.abs(
            np.asarray(cur["pos"]) - np.asarray(prev["pos"])))
        static = (delta < 1e-3
                  and not cur["moving"]
                  and set(cur["token"]) == set(prev["token"])
                  and cur["phase"] == prev["phase"])
        static_run = static_run + 1 if static else 0
        if static_run <= max_hold:
            out.append(cur)
    return out


# ---------------------------------------------------------------------------
# PyVista offscreen renderer
# ---------------------------------------------------------------------------

COLORS = {
    "active": "#00D9FF",
    "fault": "#8B0000",
    "moving_p1": "#FF8C00",
    "moving_p2": "#32CD32",
    "token": "#FFD700",
    "edge": "#E0E0E0",
    "background": "black",
}

PHASE_LABEL = {
    "intact": "",
    "damage": "Fault Injection",
    "coagulation": "Coagulation",
    "connected": "Reconnection Success!",
    "restructuring": "Restructuring",
}


def _module_color(idx: int, mid: Optional[str], frame: Dict,
                  fault_idx_set: Set[int]) -> str:
    phase = frame["phase"]
    if idx in fault_idx_set:
        return COLORS["active"] if phase == "intact" else COLORS["fault"]
    if phase == "connected":
        return (COLORS["moving_p2"] if idx in frame["_wave"]
                else COLORS["active"])
    if mid in frame["_moving"]:
        return (COLORS["moving_p2"] if phase == "restructuring"
                else COLORS["moving_p1"])
    if mid in frame["_token"]:
        return COLORS["token"]
    return COLORS["active"]


def _expanded_frames(frames: List[Dict], interp: int):
    """Yield (frame, positions) with ``interp - 1`` linearly interpolated
    position sets inserted between consecutive recorded frames. Discrete
    attributes (phase, colors, bonds, time stamp) come from the earlier
    frame."""
    for a, b in zip(frames, frames[1:]):
        pa = np.asarray(a["pos"])
        pb = np.asarray(b["pos"])
        yield a, pa
        for k in range(1, interp):
            yield a, pa + (pb - pa) * (k / interp)
    last = frames[-1]
    yield last, np.asarray(last["pos"])


def _add_starfield(pl, center: np.ndarray, scene_radius: float) -> None:
    """Procedural starfield on a distant sphere (same recipe as the Crowded
    Orbit Navigation showreel): dense dim stars with magnitude variance plus
    a sprinkling of larger bluish bright stars, unlit so they read as sky."""
    import pyvista as pv
    rng_stars = np.random.default_rng(1)
    r_sky = scene_radius * 40.0
    n_stars = 350000
    star_dirs = rng_stars.normal(size=(n_stars, 3))
    star_dirs /= np.linalg.norm(star_dirs, axis=1, keepdims=True)
    star_poly = pv.PolyData((center + star_dirs * r_sky).astype(np.float32))
    star_poly["brightness"] = (
        rng_stars.uniform(0.25, 1.0, size=n_stars) ** 2).astype(np.float32)
    pl.add_mesh(
        star_poly, scalars="brightness", cmap="gray", clim=[0.0, 1.0],
        render_points_as_spheres=True, point_size=2.6,
        lighting=False, show_scalar_bar=False, reset_camera=False)
    n_bright = 4000
    bright_dirs = rng_stars.normal(size=(n_bright, 3))
    bright_dirs /= np.linalg.norm(bright_dirs, axis=1, keepdims=True)
    bright_poly = pv.PolyData(
        (center + bright_dirs * r_sky).astype(np.float32))
    pl.add_mesh(
        bright_poly, color=(0.9, 0.92, 1.0),
        render_points_as_spheres=True, point_size=5.5,
        lighting=False, reset_camera=False)


def render_clip(data: Dict, mp4_path: str, fps: int,
                width: int, height: int, show_text: bool = True,
                interp: int = 3,
                view_dir: Tuple[float, float, float] = (1.0, 1.0, 0.8),
                ) -> None:
    import pyvista as pv

    frames = data["frames"]
    n_total = data["n_total"]
    fault_idx_set = set(data["fault_body_idxs"])
    idx_to_mid = {v: k for k, v in data["body_indices"].items()}
    for f in frames:
        f["_moving"] = set(f["moving"])
        f["_token"] = set(f["token"])
        f["_wave"] = set(f.get("wave", []))

    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background(COLORS["background"])
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    pl.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
    pl.add_light(pv.Light(position=(-5, -5, 5), intensity=0.4))

    # Fixed isometric camera framed on the union of every frame position.
    all_pos = np.concatenate([np.asarray(f["pos"]) for f in frames], axis=0)
    center = 0.5 * (all_pos.min(axis=0) + all_pos.max(axis=0))
    radius = max(float(np.linalg.norm(all_pos - center, axis=1).max()), 2.0)
    vdir = np.asarray(view_dir, dtype=float)
    vdir /= np.linalg.norm(vdir)
    pl.camera.position = tuple(center + vdir * radius * 5.0)
    pl.camera.focal_point = tuple(center)
    pl.camera.up = (0.0, 0.0, 1.0)

    _add_starfield(pl, center, radius)

    # One sphere actor per body; positions updated per frame.
    sphere = pv.Sphere(radius=0.5, theta_resolution=48, phi_resolution=48)
    actors = []
    for i in range(n_total):
        actor = pl.add_mesh(
            sphere, color=COLORS["active"], smooth_shading=True,
            specular=0.5, specular_power=20,
            name=f"module_{i}", reset_camera=False)
        actors.append(actor)

    header_actor = None
    time_actor = None
    # vtkCornerAnnotation text positions: 0 = lower left, 3 = upper right,
    # 7 = upper edge (centered). On narrow (side-by-side panel) renders the
    # centered header collides with the upper-left title, so move it right.
    header_pos, header_idx = (
        ("upper_edge", 7) if width >= 1200 else ("upper_right", 3))
    if show_text:
        pl.add_text(
            data["title"], position="upper_left", font_size=13, color="white")
        header_actor = pl.add_text(
            "", position=header_pos, font_size=18, color="white")
        time_actor = pl.add_text(
            "", position="lower_left", font_size=13, color="white")

    pl.open_movie(mp4_path, framerate=fps, quality=8)

    interp = max(1, int(interp))
    n_out = (len(frames) - 1) * interp + 1
    for k, (frame, pos) in enumerate(_expanded_frames(frames, interp)):
        for i, actor in enumerate(actors):
            actor.SetPosition(*pos[i])
            actor.GetProperty().SetColor(pv.Color(
                _module_color(i, idx_to_mid.get(i), frame, fault_idx_set)
            ).float_rgb)

        if frame["bonds"]:
            pts = pos
            cells = np.hstack([[2, i, j] for i, j in frame["bonds"]])
            bond_poly = pv.PolyData(pts)
            bond_poly.lines = cells
            pl.add_mesh(bond_poly, color=COLORS["edge"], line_width=3,
                        name="bonds", reset_camera=False)
        else:
            pl.remove_actor("bonds")

        if header_actor is not None:
            header_actor.SetText(
                header_idx, PHASE_LABEL.get(frame["phase"], frame["phase"]))
            time_actor.SetText(0, f"t = {frame['t']:6.1f} s")

        pl.write_frame()
        if k % 500 == 0:
            print(f"  rendered {k}/{n_out} frames", flush=True)

    pl.close()
    print(f"wrote {mp4_path} ({n_out} frames, "
          f"{n_out / fps:.1f} s at {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scenario", type=int, nargs="*", default=None,
                    choices=sorted(SCENARIOS),
                    help="scenario number(s) to produce")
    ap.add_argument("--all", action="store_true", help="produce all 6")
    ap.add_argument("--seed", type=int, default=42, help="base seed")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088,
                    help="default 1088 (divisible by the codec macro block; "
                         "crop to 1080 in the editor if needed)")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="record every Nth tick (tick = 0.1 sim-s)")
    ap.add_argument("--interp", type=int, default=6,
                    help="interpolated frames per recorded frame at render "
                         "time. playback speed = fps * stride / (10 * "
                         "interp); defaults (60 fps, interp 6) = real-time "
                         "at 60 fps smoothness")
    ap.add_argument("--out-dir", type=str, default="videos")
    ap.add_argument("--rerender", action="store_true",
                    help="skip simulation; render from cached frames")
    ap.add_argument("--sim-only", action="store_true",
                    help="simulate and cache frames, skip rendering")
    ap.add_argument("--no-phase2", action="store_true",
                    help="cut the restructuring phase; the clip ends on the "
                         "completed connectivity wave")
    ap.add_argument("--no-compress", action="store_true",
                    help="disable idle-frame compression")
    ap.add_argument("--max-hold", type=float, default=0.5,
                    help="max seconds of static hold kept when compressing")
    ap.add_argument("--no-text", action="store_true",
                    help="omit title/phase overlays")
    args = ap.parse_args()

    keys = sorted(SCENARIOS) if args.all else (args.scenario or [])
    if not keys:
        ap.error("pass --scenario N [N ...] or --all")

    os.makedirs(args.out_dir, exist_ok=True)

    for key in keys:
        spec = SCENARIOS[key]
        cache = os.path.join(args.out_dir, f"{spec['name']}_frames.json.gz")

        if args.rerender and os.path.exists(cache):
            with gzip.open(cache, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"[{spec['name']}] loaded {len(data['frames'])} cached frames")
        else:
            # a per-scenario pinned seed (demo-worthy instance) wins over --seed
            data = simulate_scenario(
                key, spec.get("seed", args.seed), args.sample_every)
            with gzip.open(cache, "wt", encoding="utf-8") as fh:
                json.dump(data, fh)
            print(f"[{spec['name']}] cached frames -> {cache}")

        if args.sim_only:
            continue

        # per-scenario stride/interp overrides (pinned playback speeds)
        interp = spec.get("interp", args.interp)
        frames = data["frames"]
        if data.get("reconnected"):
            # Inject the wave on the RAW frames (before striding): the wave
            # must use the true last coagulation frame's bond graph — the
            # strided timeline can end coagulation a few ticks early, before
            # the final reconnecting bond exists.
            # ~0.5 s of video per BFS ring, in recorded frames.
            wave_hold = max(1, round(0.5 * args.fps / max(interp, 1)))
            frames = inject_connectivity_wave(
                frames, set(data["fault_body_idxs"]), data["n_total"],
                wave_hold)
        if args.no_phase2:
            frames = [f for f in frames if f["phase"] != "restructuring"]
        frames = stride_sim_frames(frames, spec.get("stride", 1))
        # 1.5 s of intro per phase regardless of interpolation factor
        intro_keep = max(1, round(1.5 * args.fps / max(interp, 1)))
        frames = trim_intro(frames, intro_keep)
        pre_hold = float(spec.get("pre_hold", 0.0))
        if pre_hold > 0:
            # freeze-frame on the nominal structure before anything happens
            n_extra = round(pre_hold * args.fps / max(interp, 1))
            frames = [frames[0]] * n_extra + frames
        data = dict(data, frames=frames)
        if not args.no_compress:
            # max_hold is in *recorded* frames (each becomes interp output frames)
            max_hold = max(1, int(args.max_hold * args.fps / max(interp, 1)))
            kept = compress_idle(frames, max_hold=max_hold)
            print(f"[{spec['name']}] idle compression: "
                  f"{len(frames)} -> {len(kept)} frames")
            data = dict(data, frames=kept)

        mp4 = os.path.join(args.out_dir, f"{spec['name']}.mp4")
        render_clip(data, mp4, args.fps, args.width, args.height,
                    show_text=not args.no_text, interp=interp,
                    view_dir=spec.get("view_dir", (1.0, 1.0, 0.8)))


if __name__ == "__main__":
    main()
