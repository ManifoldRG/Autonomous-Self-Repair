"""
2D kinematic repair animation GIF (outreach asset).

Runs the stress-sharing coagulation policy on a planar 9-module L-shape
(two 4-module arms + corner; the corner is the fault) with the kinematic
GraphSimulator (no physics, no momentum), pivots restricted to the z=0
plane, and renders a flat white-background matplotlib animation ending on
the green connectivity wave + "Reconnection Success!".

Usage:
  py examples/flat_repair_gif.py              # simulate + render
  py examples/flat_repair_gif.py --rerender   # re-render from cached frames
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from video_abstract import (  # noqa: E402
    DAMAGE_HOLD,
    INTACT_HOLD,
    PHASE_LABEL,
    TUNED,
    _bond_pairs,
    _expanded_frames,
    compress_idle,
    inject_connectivity_wave,
    trim_intro,
)

from src.configurations import create_random_tree_configuration  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    FAULT_MODE_RANDOM,
    _causes_disconnection,
    select_faulty_modules,
)
from src.udqdg_system import UDQDGSystem  # noqa: E402

# Flat-diagram palette on white (video palette, darkened where needed)
FLAT_COLORS = {
    "active": "#00A8CC",
    "fault": "#8B0000",
    "moving_p1": "#FF8C00",
    "moving_p2": "#32CD32",
    "token": "#FFD700",
    "wave": "#32CD32",
    "edge_line": "#555555",
    "disc_edge": "#222222",
    "text": "#111111",
}


def build_lshape9(seed: int = 0) -> tuple:
    """9-module planar L: corner M0 at the origin, 4-module arm along +y,
    4-module arm along +x. The corner is the fault."""
    system = UDQDGSystem()
    coords = [(0, 0, 0)]
    coords += [(0, k, 0) for k in range(1, 5)]
    coords += [(k, 0, 0) for k in range(1, 5)]
    for i, c in enumerate(coords):
        system.add_module(f"M{i}", np.array(c, dtype=float))
    for k in range(1, 5):  # arm A: M0-M1-M2-M3-M4 up +y
        system.connect_modules(f"M{k - 1}" if k > 1 else "M0", f"M{k}")
    system.connect_modules("M0", "M5")  # arm B: M0-M5-M6-M7-M8 along +x
    for k in range(6, 9):
        system.connect_modules(f"M{k - 1}", f"M{k}")
    return system, ["M0"]


def build_tree50(seed: int = 42) -> tuple:
    """50-module stringy planar tree with 10 scattered random faults.

    Unlike create_random_tree_configuration (whose frontier growth packs
    into a blob with tree topology), every new module here is placed on a
    cell adjacent to EXACTLY ONE existing module, so the structure grows
    dendritically and reads as a tree with many leaves. Fault sets are
    redrawn until the damage disconnects the structure.
    """
    import random as _random

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for attempt in range(500):
        gen_seed = seed + attempt * 9973
        rng = _random.Random(gen_seed)
        cells = {(0, 0)}
        order = [((0, 0), None)]
        while len(cells) < 50:
            cands = []
            for c in cells:
                for d in dirs:
                    nb = (c[0] + d[0], c[1] + d[1])
                    if nb in cells:
                        continue
                    n_occ = sum(
                        (nb[0] + e[0], nb[1] + e[1]) in cells for e in dirs)
                    if n_occ == 1:
                        cands.append((nb, c))
            if not cands:
                break
            nb, parent = rng.choice(cands)
            cells.add(nb)
            order.append((nb, parent))
        if len(cells) < 50:
            continue

        system = UDQDGSystem()
        mid_of = {}
        for i, (cell, parent) in enumerate(order):
            mid = f"M{i}"
            mid_of[cell] = mid
            system.add_module(
                mid, np.array([float(cell[0]), float(cell[1]), 0.0]))
            if parent is not None:
                system.connect_modules(mid_of[parent], mid)

        fault_ids = select_faulty_modules(
            system, 10, gen_seed + 1000, FAULT_MODE_RANDOM)
        if fault_ids and _causes_disconnection(system, fault_ids):
            return system, fault_ids
    raise RuntimeError("no disconnecting fault set found")


SCENARIOS = {
    "lshape9": dict(
        name="flat_lshape9",
        title="9-module L-shape, corner fault (2D, kinematic)",
        builder=build_lshape9),
    "tree50": dict(
        name="flat_tree50",
        title="50-module tree, 10 random faults (2D, kinematic)",
        builder=build_tree50),
}


def simulate(spec: Dict, seed: int = 0, attempt: int = 0) -> Dict:
    import random

    from src import agent_policy
    from src.agent_policy import ModuleState
    from src.graph_sim import GraphSimulator
    from src.mc_runner import run_trial, udqdg_to_scenario

    agent_policy.PLANAR_Z_LOCK = True

    system, fault_ids = spec["builder"](seed)
    # the structure/fault builders call random.seed(seed) GLOBALLY, which
    # would make every attempt replay identically on the deterministic
    # graph sim; reseed per attempt so policy tie-breaks vary
    random.seed(seed * 100003 + attempt * 7919 + 1)
    for fid in fault_ids:
        system.mark_fault(fid)
    scenario = udqdg_to_scenario(system, fault_ids)

    sim = GraphSimulator(
        scenario.n_total, scenario.pos0, scenario.bonded0,
        module_shape="sphere")
    # smooth pivot arcs: policy passes duration 12.0 -> 2.0 s -> 20 ticks
    sim.PIVOT_DURATION_SCALE = 1.0 / 6.0
    # laterals roll the physical two-arc path (axis -> contact -> handoff)
    sim.TWO_ARC_LATERAL = True

    frames: List[Dict] = []
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
            trial_id=0,
            n_modules=len(system.modules),
            n_faults=len(fault_ids),
            frame_callback=frame_cb,
            **TUNED,
        )
    finally:
        sim.disconnect()

    coag = [f for f in frames if f["phase"] == "coagulation"]
    max_z = max((abs(p[2]) for f in coag for p in f["pos"]), default=0.0)
    min_d = 1e9
    for f in coag:
        p = np.asarray(f["pos"])
        dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(dist, 1e9)
        min_d = min(min_d, float(dist.min()))
    print(f"[{spec['name']}] reconnected={result.restored} "
          f"p1_moves={result.phase1_moves} frames={len(frames)} "
          f"max|z|={max_z:.2e} min_pair_dist={min_d:.3f}")

    return dict(
        name=spec["name"],
        title=spec["title"],
        min_pair_dist=min_d,
        n_coag_frames=len(coag),
        n_total=scenario.n_total,
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
        fault_ids=fault_ids,
        fault_body_idxs=scenario.fault_body_idxs,
        reconnected=bool(result.restored),
        phase1_moves=int(result.phase1_moves),
        frames=frames,
    )


def _frame_color(idx: int, mid, frame: Dict, fault_idx_set) -> str:
    phase = frame["phase"]
    if idx in fault_idx_set:
        return FLAT_COLORS["active"] if phase == "intact" \
            else FLAT_COLORS["fault"]
    if phase == "connected":
        return (FLAT_COLORS["wave"] if idx in frame["_wave"]
                else FLAT_COLORS["active"])
    if mid in frame["_moving"]:
        return FLAT_COLORS["moving_p1"]
    if mid in frame["_token"]:
        return FLAT_COLORS["token"]
    return FLAT_COLORS["active"]


def render(data: Dict, gif_path: str, fps: int, interp: int,
           width_px: int) -> None:
    import imageio.v3 as iio

    frames = data["frames"]
    fault_idx_set = set(data["fault_body_idxs"])
    idx_to_mid = {v: k for k, v in data["body_indices"].items()}
    for f in frames:
        f["_moving"] = set(f["moving"])
        f["_token"] = set(f["token"])
        f["_wave"] = set(f.get("wave", []))

    all_pos = np.concatenate(
        [np.asarray(f["pos"]) for f in frames], axis=0)
    pad = 1.2
    x0, x1 = all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad
    y0, y1 = all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad
    span_x, span_y = x1 - x0, y1 - y0
    height_px = int(round(width_px * span_y / span_x / 2)) * 2

    dpi = 100
    fig = plt.figure(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    images = []
    n_out = (len(frames) - 1) * interp + 1
    for k, (frame, pos) in enumerate(_expanded_frames(frames, interp)):
        ax.clear()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        for i, j in frame["bonds"]:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                    color=FLAT_COLORS["edge_line"], linewidth=2.2,
                    zorder=1, solid_capstyle="round")
        for i in range(data["n_total"]):
            ax.add_patch(mpatches.Circle(
                (pos[i][0], pos[i][1]), 0.5,
                facecolor=_frame_color(
                    i, idx_to_mid.get(i), frame, fault_idx_set),
                edgecolor=FLAT_COLORS["disc_edge"], linewidth=1.2,
                zorder=2))

        label = PHASE_LABEL.get(frame["phase"], frame["phase"])
        ax.text(0.5, 0.97, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=15, fontweight="bold",
                color=FLAT_COLORS["text"])
        ax.text(0.01, 0.02, data["title"], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                color=FLAT_COLORS["text"])
        ax.text(0.99, 0.02, f"t = {frame['t']:5.1f} s",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=FLAT_COLORS["text"])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(buf.copy())
        if k % 200 == 0:
            print(f"  rendered {k}/{n_out} frames", flush=True)

    plt.close(fig)
    iio.imwrite(gif_path, images, duration=1000 / fps, loop=0)
    print(f"wrote {gif_path} ({len(images)} frames, "
          f"{len(images) / fps:.1f} s at {fps} fps)")

    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"
    iio.imwrite(mp4_path, images, fps=fps, codec="h264")
    print(f"wrote {mp4_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scenario", type=str, default="lshape9",
                    choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attempts", type=int, default=8,
                    help="sim attempts; the shortest clean reconnecting "
                         "run is kept")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--interp", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out-dir", type=str, default="Media/outreach")
    ap.add_argument("--cache-dir", type=str, default="videos")
    ap.add_argument("--rerender", action="store_true",
                    help="skip simulation; render from cached frames")
    ap.add_argument("--max-hold", type=float, default=0.5,
                    help="max seconds of static hold kept when compressing")
    args = ap.parse_args()

    spec = SCENARIOS[args.scenario]
    name = spec["name"]
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.cache_dir, f"{name}_frames.json.gz")

    if args.rerender and os.path.exists(cache):
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"[{name}] loaded {len(data['frames'])} cached frames")
    else:
        best = None
        for k in range(args.attempts):
            d = simulate(spec, args.seed, attempt=k)
            if not d["reconnected"] or d["min_pair_dist"] < 0.9:
                continue
            if best is None or d["n_coag_frames"] < best["n_coag_frames"]:
                best = d
        if best is None:
            sys.exit(f"[{name}] no clean reconnecting run "
                     f"in {args.attempts} attempts")
        data = best
        print(f"[{name}] best run: coag_frames={data['n_coag_frames']} "
              f"moves={data['phase1_moves']} "
              f"min_dist={data['min_pair_dist']:.3f}")
        with gzip.open(cache, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
        print(f"[{name}] cached frames -> {cache}")

    frames = data["frames"]
    if data.get("reconnected"):
        wave_hold = max(1, round(0.5 * args.fps / args.interp))
        frames = inject_connectivity_wave(
            frames, set(data["fault_body_idxs"]), data["n_total"],
            wave_hold)
    frames = [f for f in frames if f["phase"] != "restructuring"]
    intro_keep = max(1, round(1.5 * args.fps / args.interp))
    frames = trim_intro(frames, intro_keep)
    max_hold = max(1, int(args.max_hold * args.fps / args.interp))
    kept = compress_idle(frames, max_hold=max_hold)
    print(f"[{name}] idle compression: {len(frames)} -> {len(kept)} frames")
    data = dict(data, frames=kept)

    gif = os.path.join(args.out_dir, f"asr_{name}_repair.gif")
    render(data, gif, args.fps, args.interp, args.width)


if __name__ == "__main__":
    main()
