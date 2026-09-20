"""
Simulator-agnostic Monte Carlo trial runner.

Hosts the bits of the trial pipeline that don't care whether the
underlying simulator is `BulletSimulator` (physics-driven) or
`GraphSimulator` (perfect-pivot graph kinematics):

- ``Scenario`` dataclass and the ``udqdg_to_scenario`` translator
- ``_MCCoagulation`` / ``_MCRestructuring`` / ``_MCDisplacementRestructuring``
  policy subclasses (token origin stored in the holder's body frame, plus
  a per-instance ``_safety_radius`` override on ``is_movable``)
- Phase termination predicates (``_all_idle_or_capped``, ``_phase1_done``,
  ``_phase2_done``) and the single-phase driver (``_run_phase``)
- ``run_trial(sim, scenario, ...)`` — drives Phase 1 (coagulation) and
  optionally Phase 2 (restructuring) on an already-constructed simulator
  and returns a ``TrialResult``.

The caller (``run_bullet_monte_carlo.py`` or ``src/monte_carlo.py``) is
responsible for generating the structure, selecting faults, building the
scenario, and constructing the simulator. Both runners share everything
downstream of that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

# Suppress per-pivot/per-token loguru chatter from the policy in MC runs.
# Module-level disable runs in every worker process that imports mc_runner
# (joblib/ProcessPoolExecutor workers don't inherit parent loguru state).
# Callers who want the logs can call ``logger.enable("src.agent_policy")``
# after importing mc_runner.
from loguru import logger as _loguru_logger
_loguru_logger.disable("src.agent_policy")
_loguru_logger.disable("src.bullet_sim")
_loguru_logger.disable("src.bullet_bridge")

from .agent_policy import (
    DecentralizedCoagulation,
    DecentralizedRestructuring,
    DisplacementRestructuring,
    RetraceRestructuring,
    ModuleAgent,
    ModuleState,
)
from .monte_carlo import TrialResult, calculate_shape_difference, FAULT_MODE_RANDOM


# ---------------------------------------------------------------------------
# Scenario translation layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """All inputs needed to run a multi-fault simulator-agnostic trial.

    ``fault_ids`` / ``fault_body_idxs`` hold the full set of
    simultaneously-injected faults. The legacy single-fault fields
    (``fault_id``, ``fault_body_idx``, ``fault_adjacent``) carry the
    first fault for back-compat with callers that only handled one
    fault — unused on the simultaneous-injection path.
    """
    n_total: int
    pos0: np.ndarray
    bonded0: np.ndarray
    fault_id: str
    fault_body_idx: int
    fault_adjacent: List[str]
    fault_ids: List[str]
    fault_body_idxs: List[int]
    adjacent_map: Dict[str, int]
    module_ids: List[str]
    body_indices: Dict[str, int]
    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]]
    original_positions: Dict[str, np.ndarray]


def udqdg_to_scenario(system, fault_ids) -> Scenario:
    """Convert a ``UDQDGSystem`` + fault set into the simulator-agnostic
    ``Scenario`` payload.

    Accepts a single fault id (str) for back-compat or a list of fault
    ids for the simultaneous-injection path. All fault modules become
    full bodies in the simulator (mass + collision on for BulletSimulator,
    lattice obstacles for GraphSimulator) and are excluded from the
    policy's active module set.
    """
    if isinstance(fault_ids, str):
        fault_id_list = [fault_ids]
    else:
        fault_id_list = list(fault_ids)
    fault_set = set(fault_id_list)
    primary_fault = fault_id_list[0] if fault_id_list else ""

    all_mids = sorted(system.modules.keys())
    n_total = len(all_mids)
    mid_to_idx: Dict[str, int] = {mid: i for i, mid in enumerate(all_mids)}

    pos0 = np.zeros((n_total, 3))
    for mid, idx in mid_to_idx.items():
        pos0[idx] = system.modules[mid].position

    bonded0 = np.zeros((n_total, n_total), dtype=bool)
    for (a, b) in system.edges:
        ia, ib = mid_to_idx[a], mid_to_idx[b]
        bonded0[ia, ib] = True
        bonded0[ib, ia] = True

    fault_body_idxs = [mid_to_idx[fid] for fid in fault_id_list]
    module_ids = [mid for mid in all_mids if mid not in fault_set]
    body_indices = {mid: mid_to_idx[mid] for mid in module_ids}

    adjacent_map: Dict[str, int] = {}
    for (a, b) in system.edges:
        if a in fault_set and b not in fault_set:
            adjacent_map.setdefault(b, mid_to_idx[a])
        elif b in fault_set and a not in fault_set:
            adjacent_map.setdefault(a, mid_to_idx[b])

    fault_adjacent: List[str] = []
    for (a, b) in system.edges:
        if a == primary_fault and b != primary_fault:
            fault_adjacent.append(b)
        elif b == primary_fault and a != primary_fault:
            if a not in fault_adjacent:
                fault_adjacent.append(a)

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    for mid in module_ids:
        mid_pos = system.modules[mid].position
        neighbors = system.get_neighbors(mid)
        dirs: List[np.ndarray] = []
        for nbr in neighbors:
            nbr_pos = system.modules[nbr].position
            direction = nbr_pos - mid_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction = direction / norm
            dirs.append(direction)
        pre_damage_neighbor_slots[mid] = dirs

    original_positions = {
        mid: system.modules[mid].position.copy()
        for mid in all_mids
    }

    return Scenario(
        n_total=n_total,
        pos0=pos0,
        bonded0=bonded0,
        fault_id=primary_fault,
        fault_body_idx=mid_to_idx[primary_fault] if primary_fault else -1,
        fault_adjacent=fault_adjacent,
        fault_ids=fault_id_list,
        fault_body_idxs=fault_body_idxs,
        adjacent_map=adjacent_map,
        module_ids=module_ids,
        body_indices=body_indices,
        pre_damage_neighbor_slots=pre_damage_neighbor_slots,
        original_positions=original_positions,
    )


# ---------------------------------------------------------------------------
# Policy subclasses
# ---------------------------------------------------------------------------

class _MCCoagulation(DecentralizedCoagulation):
    """Token origin stored in holder's body frame (drift-invariant).

    The default ``_origin_world_for_pick_target`` returns the token's
    world-frame direction unchanged. Storing it in the body frame and
    transforming on read keeps the target position stable under attitude
    drift of the holder.
    """

    _safety_radius: int = 2

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        return super().is_movable(body_idx, self._safety_radius)


class _MCRestructuring(DecentralizedRestructuring):
    """Token origin stored in holder's body frame (drift-invariant)."""

    _safety_radius: int = 2

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        return super().is_movable(body_idx, self._safety_radius)


class _MCDisplacementRestructuring(DisplacementRestructuring):
    """Displacement-guided restructuring for MC trials."""

    _safety_radius: int = 2

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        return super().is_movable(body_idx, self._safety_radius)


class _MCRetraceRestructuring(RetraceRestructuring):
    """Retrace-guided restructuring for MC trials."""

    _safety_radius: int = 2

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        return super().is_movable(body_idx, self._safety_radius)


# ---------------------------------------------------------------------------
# Phase termination
# ---------------------------------------------------------------------------

IDLE_OR_CAPPED_MIN_SIM_TIME = 20.0


def _all_idle_or_capped(policy) -> bool:
    """True if every agent is "done" for the phase.

    With decoupled budgets an agent is "done" when it can no longer do
    useful work. The cases are:
      - both budgets exhausted (action_points<=0 AND forward_points<=0);
      - IDLE with no pending tokens;
      - holding/queuing a token it cannot act on: it has no forward budget
        to relay it AND it cannot move it (no motion budget, or the
        criticality test rejects it). Such a token is dropped next tick.
    Any in-flight state (PIVOTING/REVERSING/WAITING) blocks termination.

    Note: the old predicate keyed solely on action_points (the unified
    budget). Under the split, a fault-adjacent immovable module never
    spends its motion budget yet keeps receiving self-generated tokens,
    so keying on action_points alone made the phase never terminate.

    Guard: suppressed for the first ``IDLE_OR_CAPPED_MIN_SIM_TIME`` sim
    seconds so tokens can propagate before "done" can be declared.
    """
    if policy.sim.sim_time < IDLE_OR_CAPPED_MIN_SIM_TIME:
        return False

    sr = getattr(policy, "_safety_radius", 2)
    for a in policy.agents.values():
        if a.action_points <= 0 and a.forward_points <= 0:
            continue
        if a.state != ModuleState.IDLE:
            return False
        if a.incoming_tokens or a.token is not None:
            # Active only if it can actually act on the token.
            if a.forward_points > 0:
                return False
            if a.action_points > 0 and policy.is_movable(a.body_idx, sr):
                return False
            # else: token will be dropped — agent is effectively done.
    return True


def _phase1_done(policy, sim) -> bool:
    if sim.has_active_pivots():
        return False
    if policy.is_connected():
        return True
    return _all_idle_or_capped(policy)


def _phase2_done(policy, sim) -> bool:
    if sim.has_active_pivots():
        return False
    if _all_idle_or_capped(policy):
        return True
    return False


# ---------------------------------------------------------------------------
# Action-point usage snapshot
# ---------------------------------------------------------------------------

def _ap_usage_snapshot(policy) -> Dict[str, float]:
    """Aggregate per-agent AP spend across all agents in the current phase.

    Returns mean/max of ``ap_spent_pivots`` and ``ap_spent_forwards``, plus
    mean/max of total AP spent. Excludes the (presumed-isolated) zero
    counters from agents that never received a token, since including them
    would dilute the means with structural non-participants.
    """
    if not policy.agents:
        nan = float("nan")
        return dict(
            mean_ap_pivots=nan, max_ap_pivots=nan,
            mean_ap_forwards=nan, max_ap_forwards=nan,
            mean_ap_total=nan, max_ap_total=nan,
        )
    pivs = [a.ap_spent_pivots for a in policy.agents.values()]
    fwds = [a.ap_spent_forwards for a in policy.agents.values()]
    tots = [p + f for p, f in zip(pivs, fwds)]
    return dict(
        mean_ap_pivots=float(np.mean(pivs)),
        max_ap_pivots=float(np.max(pivs)),
        mean_ap_forwards=float(np.mean(fwds)),
        max_ap_forwards=float(np.max(fwds)),
        mean_ap_total=float(np.mean(tots)),
        max_ap_total=float(np.max(tots)),
    )


# ---------------------------------------------------------------------------
# Active-subgraph component count
# ---------------------------------------------------------------------------

def _count_active_components(sim, scenario: Scenario) -> tuple:
    """Count connected components in the active subgraph.

    The active subgraph is the bond graph restricted to non-fault body
    indices — fault bodies remain physically present in the sim but are
    excluded from connectivity counts (matching the policy's connectivity
    semantics). Returns ``(n_components, largest_component_frac)`` where
    ``largest_component_frac = max_component_size / n_active_modules``;
    ``(0, 0.0)`` if no active modules exist.
    """
    fault_idxs = set(scenario.fault_body_idxs)
    active_idxs = [
        scenario.body_indices[mid] for mid in scenario.module_ids
        if scenario.body_indices[mid] not in fault_idxs
    ]
    if not active_idxs:
        return 0, 0.0
    bm = sim.get_bond_matrix()
    active_set = set(active_idxs)
    seen: Set[int] = set()
    largest = 0
    n_comp = 0
    for start in active_idxs:
        if start in seen:
            continue
        n_comp += 1
        stack = [start]
        size = 0
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            size += 1
            for j in np.where(bm[v])[0]:
                jj = int(j)
                if jj in active_set and jj not in seen:
                    stack.append(jj)
        if size > largest:
            largest = size
    return n_comp, largest / len(active_idxs)


# ---------------------------------------------------------------------------
# Phase driver
# ---------------------------------------------------------------------------

def _run_phase(
    sim,
    policy,
    done_fn: Callable[[Any, Any], bool],
    dt: float,
    max_time: float,
    stall_interval: float,
    stall_patience: int,
    phase_label: str = "phase",
    heartbeat_wall_secs: float = 30.0,
    frame_callback: Optional[Callable[[Any, Any, int], None]] = None,
) -> int:
    """Run a simulation phase. Returns tick count.

    Termination: ``done_fn`` only. ``stall_interval`` / ``stall_patience``
    / ``max_time`` are accepted for API/config back-compat but unused —
    the per-module action-point cap bounds total work decentrally.

    Heartbeat: every ``heartbeat_wall_secs`` of wall time, prints a status
    line with current tick count, sim time, successful_moves, and the
    distribution of completed_moves across agents.

    ``frame_callback(sim, policy, ticks)``, when given, is invoked once
    before the loop (ticks=0) and after every tick — used by rendering
    scripts to record trajectories.
    """
    del max_time, stall_interval, stall_patience
    import time as _time
    cap = getattr(policy, "MAX_MOVES_PER_MODULE", 0)
    ticks = 0
    t0 = _time.monotonic()
    last_hb = t0
    if frame_callback is not None:
        frame_callback(sim, policy, ticks)
    while True:
        sim.step(dt)
        policy.tick()
        ticks += 1
        if frame_callback is not None:
            frame_callback(sim, policy, ticks)
        if done_fn(policy, sim):
            break
        now = _time.monotonic()
        if heartbeat_wall_secs > 0 and (now - last_hb) >= heartbeat_wall_secs:
            last_hb = now
            n_capped = sum(
                1 for a in policy.agents.values()
                if cap and a.completed_moves >= cap)
            n_pivoting = sum(
                1 for a in policy.agents.values()
                if a.state == ModuleState.PIVOTING)
            n_idle = sum(
                1 for a in policy.agents.values()
                if a.state == ModuleState.IDLE)
            n_proc = sum(
                1 for a in policy.agents.values()
                if a.state == ModuleState.PROCESSING)
            total_agents = len(policy.agents)
            print(
                f"  [{phase_label}] tick={ticks} sim_t={sim.sim_time:.1f}s "
                f"wall={now - t0:.0f}s "
                f"successful_moves={policy.successful_moves} "
                f"capped={n_capped}/{total_agents} "
                f"pivoting={n_pivoting} idle={n_idle} proc={n_proc}",
                flush=True)

    # Drain any in-flight pivots so the bond-restoration invariant fires
    # for modules that were mid-pivot at termination.
    for piv_idx in list(sim._active_pivots.keys()):
        sim.stop_pivot(piv_idx)

    now = _time.monotonic()
    n_capped = sum(
        1 for a in policy.agents.values()
        if cap and a.completed_moves >= cap)
    n_pivoting = sum(
        1 for a in policy.agents.values()
        if a.state == ModuleState.PIVOTING)
    n_idle = sum(
        1 for a in policy.agents.values()
        if a.state == ModuleState.IDLE)
    n_proc = sum(
        1 for a in policy.agents.values()
        if a.state == ModuleState.PROCESSING)
    total_agents = len(policy.agents)
    print(
        f"  [{phase_label}] tick={ticks} sim_t={sim.sim_time:.1f}s "
        f"wall={now - t0:.0f}s "
        f"successful_moves={policy.successful_moves} "
        f"capped={n_capped}/{total_agents} "
        f"pivoting={n_pivoting} idle={n_idle} proc={n_proc}",
        flush=True)

    return ticks


# ---------------------------------------------------------------------------
# Sim-agnostic trial driver
# ---------------------------------------------------------------------------

def run_trial(
    *,
    sim,
    scenario: Scenario,
    faulty_modules_set: Set[str],
    original_positions: Dict[str, np.ndarray],
    trial_id: int,
    n_modules: int,
    n_faults: int,
    temperature: float,
    pivot_exclusion_radius: int,
    dt: float,
    restructuring_method: str,
    token_strategy: str,
    safety_radius: int,
    use_flood_echo: bool,
    token_gen_interval: float,
    max_moves_per_module: int,
    use_position_history: bool,
    random_baseline: bool = False,
    forward_ap_cost: float = 0.1,
    fault_mode: str = FAULT_MODE_RANDOM,
    diagnostics_callback: Optional[Callable[..., None]] = None,
    frame_callback: Optional[Callable[[Any, Any, str, int], None]] = None,
) -> TrialResult:
    """Drive a single MC trial on an already-constructed simulator.

    Runs Phase 1 (coagulation). If the structure reconnects, runs Phase 2
    (either token-based ``DecentralizedRestructuring`` or
    ``DisplacementRestructuring`` depending on ``restructuring_method``).
    Returns a ``TrialResult`` capturing moves, shape differences, and
    connectivity outcome.

    The caller owns the simulator's lifecycle — this function does NOT
    call ``sim.disconnect()``. Wrap the call in a try/finally on the
    caller side if cleanup is required.
    """
    total_phase1_moves = 0
    total_phase2_moves = 0
    total_phase1_ticks = 0
    restored = True
    post_phase1_positions: Optional[Dict[str, np.ndarray]] = None
    post_phase2_positions: Optional[Dict[str, np.ndarray]] = None

    coag = _MCCoagulation(
        sim,
        fault_id=scenario.fault_ids[0] if scenario.fault_ids else "",
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
    )
    coag.PIVOT_EXCLUSION_RADIUS = pivot_exclusion_radius
    coag._safety_radius = safety_radius
    coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
    coag.TEMPERATURE = temperature
    coag.RANDOM_BASELINE = bool(random_baseline)
    coag.TOKEN_GEN_INTERVAL = float(token_gen_interval)
    coag.INITIAL_ACTION_POINTS = int(max_moves_per_module)
    coag.MAX_MOVES_PER_MODULE = int(max_moves_per_module)
    _fwd_pts = int(max_moves_per_module) * coag.FORWARD_POINTS_MULTIPLIER
    for _agent in coag.agents.values():
        _agent.action_points = int(max_moves_per_module)
        _agent.forward_points = _fwd_pts
    coag.USE_FLOOD_ECHO = use_flood_echo
    coag.USE_POSITION_HISTORY = bool(use_position_history)
    coag.FORWARD_AP_COST = forward_ap_cost
    coag.set_multi_fault_adjacent(
        fault_ids=scenario.fault_ids,
        fault_body_idxs=scenario.fault_body_idxs,
        adjacent_map=scenario.adjacent_map,
    )

    n_comp_pd, largest_frac_pd = _count_active_components(sim, scenario)

    phase1_ticks = _run_phase(
        sim, coag, _phase1_done, dt,
        max_time=0.0, stall_interval=0.0, stall_patience=0,
        phase_label=f"coag[t{trial_id}/n{n_modules}/f{n_faults}]",
        frame_callback=(
            (lambda s, p, t: frame_callback(s, p, "coagulation", t))
            if frame_callback is not None else None))

    phase1_connected = coag.is_connected()
    total_phase1_moves = coag.total_moves
    total_phase1_ticks = phase1_ticks
    restored = phase1_connected

    pos_snap = sim.get_positions()
    post_phase1_positions = {
        mid: pos_snap[scenario.body_indices[mid]].copy()
        for mid in scenario.module_ids
    }
    n_comp_p1, largest_frac_p1 = _count_active_components(sim, scenario)
    ap_p1 = _ap_usage_snapshot(coag)

    n_comp_p2: Optional[int] = None
    largest_frac_p2: Optional[float] = None
    ap_p2: Optional[Dict[str, float]] = None

    restruct = None
    if phase1_connected:
        coag_moved: Set[str] = {m["module"] for m in coag.move_log}
        if restructuring_method == "retrace":
            restruct = _MCRetraceRestructuring(
                sim=sim,
                module_ids=scenario.module_ids,
                body_indices=scenario.body_indices,
                coag_moved=coag_moved,
                original_positions=scenario.original_positions,
                pivot_history=coag.pivot_history,
            )
        elif restructuring_method == "displacement":
            restruct = _MCDisplacementRestructuring(
                sim=sim,
                module_ids=scenario.module_ids,
                body_indices=scenario.body_indices,
                coag_moved=coag_moved,
                original_positions=scenario.original_positions,
            )
        else:
            restruct = _MCRestructuring(
                sim=sim,
                module_ids=scenario.module_ids,
                body_indices=scenario.body_indices,
                coag_moved=coag_moved,
                pre_damage_neighbor_slots=scenario.pre_damage_neighbor_slots,
                token_strategy=token_strategy,
            )
        restruct.PIVOT_EXCLUSION_RADIUS = pivot_exclusion_radius
        restruct._safety_radius = safety_radius
        restruct.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
        restruct.USE_FLOOD_ECHO = use_flood_echo
        restruct.USE_POSITION_HISTORY = bool(use_position_history)
        restruct.FORWARD_AP_COST = forward_ap_cost
        restruct.INITIAL_ACTION_POINTS = int(max_moves_per_module)
        restruct.MAX_MOVES_PER_MODULE = int(max_moves_per_module)
        _fwd_pts_r = (int(max_moves_per_module)
                      * restruct.FORWARD_POINTS_MULTIPLIER)
        for _agent in restruct.agents.values():
            _agent.action_points = int(max_moves_per_module)
            _agent.forward_points = _fwd_pts_r
        restruct.generate_initial_tokens()

        _run_phase(
            sim, restruct, _phase2_done, dt,
            max_time=0.0, stall_interval=0.0, stall_patience=0,
            phase_label=f"restruct[t{trial_id}/n{n_modules}/f{n_faults}]",
            frame_callback=(
                (lambda s, p, t: frame_callback(s, p, "restructuring", t))
                if frame_callback is not None else None))

        total_phase2_moves = restruct.total_moves

        pos_snap2 = sim.get_positions()
        post_phase2_positions = {
            mid: pos_snap2[scenario.body_indices[mid]].copy()
            for mid in scenario.module_ids
        }
        n_comp_p2, largest_frac_p2 = _count_active_components(sim, scenario)
        ap_p2 = _ap_usage_snapshot(restruct)

    if diagnostics_callback is not None:
        diagnostics_callback(
            sim=sim,
            coag=coag,
            restruct=restruct,
            scenario=scenario,
            phase1_connected=phase1_connected,
            total_phase1_moves=total_phase1_moves,
            total_phase2_moves=total_phase2_moves,
            total_phase1_ticks=total_phase1_ticks,
        )

    if restored and post_phase1_positions is not None:
        shape_diff_phase1 = calculate_shape_difference(
            original_positions, post_phase1_positions, faulty_modules_set)
        final_positions = (
            post_phase2_positions if post_phase2_positions is not None
            else post_phase1_positions)
        shape_diff = calculate_shape_difference(
            original_positions, final_positions, faulty_modules_set)
    else:
        shape_diff = None
        shape_diff_phase1 = None

    return TrialResult(
        trial_id=trial_id,
        n_modules=n_modules,
        n_faults=n_faults,
        seed=0,  # caller overwrites with structure-gen seed before returning
        restored=restored,
        phase1_moves=total_phase1_moves,
        phase2_moves=total_phase2_moves,
        shape_difference=shape_diff,
        shape_difference_phase1=shape_diff_phase1,
        phase1_iterations=total_phase1_ticks,
        total_moves=total_phase1_moves + total_phase2_moves,
        token_transmissions=0,
        fault_mode=fault_mode,
        n_components_post_damage=n_comp_pd,
        largest_component_frac_post_damage=largest_frac_pd,
        n_components_post_phase1=n_comp_p1,
        largest_component_frac_post_phase1=largest_frac_p1,
        n_components_post_phase2=n_comp_p2,
        largest_component_frac_post_phase2=largest_frac_p2,
        ap_phase1_mean_pivots=ap_p1["mean_ap_pivots"],
        ap_phase1_max_pivots=ap_p1["max_ap_pivots"],
        ap_phase1_mean_forwards=ap_p1["mean_ap_forwards"],
        ap_phase1_max_forwards=ap_p1["max_ap_forwards"],
        ap_phase1_mean_total=ap_p1["mean_ap_total"],
        ap_phase1_max_total=ap_p1["max_ap_total"],
        ap_phase2_mean_pivots=ap_p2["mean_ap_pivots"] if ap_p2 else None,
        ap_phase2_max_pivots=ap_p2["max_ap_pivots"] if ap_p2 else None,
        ap_phase2_mean_forwards=ap_p2["mean_ap_forwards"] if ap_p2 else None,
        ap_phase2_max_forwards=ap_p2["max_ap_forwards"] if ap_p2 else None,
        ap_phase2_mean_total=ap_p2["mean_ap_total"] if ap_p2 else None,
        ap_phase2_max_total=ap_p2["max_ap_total"] if ap_p2 else None,
    )
