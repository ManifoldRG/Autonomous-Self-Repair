"""
Decentralized, asynchronous agent policies for modular spacecraft.

Each module is an independent agent that:
- Discovers neighbors by physical proximity (bonds in PyBullet)
- Receives/propagates tokens (direction vectors)
- Decides locally whether to move based on its token
- Picks a target lattice position aligned with its token
- Physically pivots to that position via torque
- Waits for pivot convergence (see ``BulletSimulator.is_pivot_complete``)
  before acting again

Token forwarding has a 1-second processing delay.
"""

import random
import numpy as np
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger


def bond_graph_neighbors(bond_matrix: np.ndarray, n: int, body_idx: int) -> List[int]:
    """Neighbors of ``body_idx`` in the undirected graph defined by a boolean bond matrix."""
    return [j for j in range(n) if j != body_idx and bond_matrix[body_idx, j]]


_LATTICE_DELTAS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)

# When True, pick_target rejects any pivot whose target leaves the z=0
# plane (2D kinematic demos). Off by default so MC paths are unaffected;
# set via ``agent_policy.PLANAR_Z_LOCK = True``.
PLANAR_Z_LOCK = False

# Minimum center-to-center clearance (in module diameters ~1.0) sampled
# along the pivot arc for the planar sweep check. Bonded contact is 1.0.
_PLANAR_SWEEP_CLEARANCE = 0.95


def _planar_arc_clear(pos, my_idx: int, axis_idx: int,
                      target_world: np.ndarray, n_total: int,
                      long_way: bool = False) -> bool:
    """True if the in-plane pivot arc from ``pos[my_idx]`` to
    ``target_world`` about ``pos[axis_idx]`` stays clear of every other
    module. Mirrors GraphSimulator.step: a rotation of the start offset
    about the axis, plus the radius lerp toward the target that the
    end-of-pivot snap performs (relevant for lateral handoffs).
    ``long_way`` tests the complementary arc (rolling around the free
    side, 2*pi - theta in the opposite direction).
    """
    a = pos[axis_idx]
    r0 = pos[my_idx] - a
    rt = np.asarray(target_world, dtype=float) - a
    n0 = float(np.linalg.norm(r0))
    nt = float(np.linalg.norm(rt))
    if n0 < 1e-9 or nt < 1e-9:
        return True
    u0, ut = r0 / n0, rt / nt
    dot = float(np.clip(np.dot(u0, ut), -1.0, 1.0))
    ang = float(np.arccos(dot))
    sgn = 1.0 if (u0[0] * ut[1] - u0[1] * ut[0]) >= 0.0 else -1.0
    if long_way:
        ang, sgn = 2.0 * np.pi - ang, -sgn
    samples = max(9, int(np.ceil(ang / (np.pi / 16))))
    for k in range(1, samples):
        t = k / (samples - 1)
        th = sgn * ang * t
        c, s = np.cos(th), np.sin(th)
        dir_v = np.array(
            [c * u0[0] - s * u0[1], s * u0[0] + c * u0[1], 0.0])
        for rad in (n0, n0 + (nt - n0) * t):
            p = a + rad * dir_v
            for j in range(n_total):
                if j == my_idx or j == axis_idx:
                    continue
                if float(np.linalg.norm(p - pos[j])) \
                        < _PLANAR_SWEEP_CLEARANCE:
                    return False
    return True


def _planar_pivot_admissible(pos, my_idx: int, axis_idx: int,
                             target_world, n_total: int) -> bool:
    """A planar corner pivot is admissible if either arc direction is clear."""
    return (_planar_arc_clear(pos, my_idx, axis_idx, target_world, n_total)
            or _planar_arc_clear(pos, my_idx, axis_idx, target_world,
                                 n_total, long_way=True))


def _planar_lateral_clear(pos, my_idx: int, axis_idx: int, handoff_idx: int,
                          target_world, n_total: int,
                          nom: float = 1.0) -> bool:
    """Sweep check for the physically rolled lateral path: an arc about
    the axis until contact with the handoff module, then an arc about the
    handoff to the target (matches GraphSimulator.TWO_ARC_LATERAL)."""
    a = pos[axis_idx]
    h = pos[handoff_idx]
    r0 = pos[my_idx] - a
    rh = h - a
    n0, nh = float(np.linalg.norm(r0)), float(np.linalg.norm(rh))
    if n0 < 1e-9 or nh < 1e-9:
        return True
    cos_sep = float(np.clip(np.dot(r0, rh) / (n0 * nh), -1.0, 1.0))
    sep0 = float(np.arccos(cos_sep))
    cos_c = (n0 * n0 + nh * nh - nom * nom) / (2.0 * n0 * nh)
    if not -1.0 <= cos_c <= 1.0:
        return True
    sep_c = float(np.arccos(cos_c))
    d1 = max(0.0, sep0 - sep_c)
    sgn1 = 1.0 if (r0[0] * rh[1] - r0[1] * rh[0]) >= 0.0 else -1.0

    def _arc_pts(anchor, r_start, sgn, dtheta):
        samples = max(5, int(np.ceil(dtheta / (np.pi / 16))))
        for k in range(1, samples + 1):
            th = sgn * dtheta * (k / samples)
            c, s = np.cos(th), np.sin(th)
            yield anchor + np.array(
                [c * r_start[0] - s * r_start[1],
                 s * r_start[0] + c * r_start[1], 0.0])

    def _clear(p, skip):
        for j in range(n_total):
            if j == my_idx or j in skip:
                continue
            if float(np.linalg.norm(p - pos[j])) < _PLANAR_SWEEP_CLEARANCE:
                return False
        return True

    p_contact = None
    for p in _arc_pts(a, r0, sgn1, d1):
        p_contact = p
        if not _clear(p, skip=(axis_idx, handoff_idx)):
            return False
    if p_contact is None:
        p_contact = pos[my_idx]
    rc = p_contact - h
    rt = np.asarray(target_world, dtype=float) - h
    nc, nt = float(np.linalg.norm(rc)), float(np.linalg.norm(rt))
    if nc < 1e-9 or nt < 1e-9:
        return True
    cos2 = float(np.clip(np.dot(rc, rt) / (nc * nt), -1.0, 1.0))
    d2 = float(np.arccos(cos2))
    sgn2 = 1.0 if (rc[0] * rt[1] - rc[1] * rt[0]) >= 0.0 else -1.0
    for p in _arc_pts(h, rc, sgn2, d2):
        if not _clear(p, skip=(handoff_idx,)):
            return False
    return True

# Eligibility: angle between pivot–axis arm and axis→target leg (same vectors as
# pick_target).  Perpendicular lattice steps should give cos≈0 for corners.
# Old threshold 0.99 only rejected within ~8° of 0°/180° (~171° still passed).
_CORNER_PIVOT_MAX_ABS_COS = 0.35  # ~90° family: reject |cos| above this
_LATERAL_PIVOT_MAX_ABS_COS = 0.92  # reject near-colinear (within ~23° of 0°/180°)


def _lateral_axis_handoff_step_cardinal_for_axis(
    sim,
    axis_idx: int,
    delta_axis_to_handoff_world: np.ndarray,
) -> bool:
    """True if axis→handoff COM vector has nominal length and is one lattice
    axis in the *axis* body frame.

    Lateral targets are built in the axis lattice frame (same as ``target_cell``
    rounding). Using the *pivot* frame here wrongly rejected laterals after a
    corner reoriented the pivot while the axis–handoff bond stayed lattice-aligned.
    """
    nom = float(sim.NOMINAL_DIST)
    d = np.asarray(delta_axis_to_handoff_world, dtype=float).reshape(3)
    if np.linalg.norm(d) < 0.5 * nom:
        return False
    if abs(np.linalg.norm(d) - nom) > 0.15 * nom:
        return False
    R = sim.body_rotation_matrix(axis_idx)
    d_loc = R.T @ d
    for dir6 in sim.CONNECTOR_DIRS:
        if np.linalg.norm(d_loc - nom * dir6) < 0.12 * nom:
            return True
    return False


def _cube_lateral_swept_clear(
    occupied: set,
    my_cell: Tuple[int, int, int],
    target_cell: Tuple[int, int, int],
    arm_axis_local: np.ndarray,
    R_axis: np.ndarray,
    nom: float,
) -> bool:
    """For cube modules, a 90° lever around the shared edge sweeps a quarter-disk
    that intrudes into the two cells "above" the start and target cells, where
    "above" is the arm direction (axis→pivot) extended by one lattice step.

    Reject the candidate if either of those cells is occupied.
    """
    arm_loc = R_axis.T @ arm_axis_local / nom
    arm_unit = np.round(arm_loc).astype(int)
    if int(np.sum(np.abs(arm_unit))) != 1:
        # Non-cardinal arm in axis frame: skip the test (be permissive — caller
        # may still reject via other checks).
        return True
    above_pivot = tuple((np.array(my_cell, dtype=int) + arm_unit).tolist())
    above_target = tuple((np.array(target_cell, dtype=int) + arm_unit).tolist())
    if above_pivot in occupied or above_target in occupied:
        return False
    return True


def _lattice_delta_perpendicular_to_arm(
    delta: Tuple[int, int, int],
    arm_world: np.ndarray,
    R_axis: np.ndarray,
) -> bool:
    """True if world step ``nom * R_axis @ delta`` is ⟂ ``arm_world`` (pivot−axis).

    Corner / lateral lattice moves are expressed in the *axis* body frame;
    only the four cardinal steps perpendicular to the current attachment in
    world frame are eligible (90° family).
    """
    an = np.linalg.norm(arm_world)
    if an < 1e-12:
        return False
    step_w = R_axis @ np.array(delta, dtype=float)
    return abs(np.dot(step_w, arm_world)) / an < 0.1


class ModuleState(Enum):
    IDLE = auto()           # Waiting, no token
    HAS_TOKEN = auto()      # Has a token, deciding what to do
    PROCESSING = auto()     # Processing delay before forwarding token
    PIVOTING = auto()       # Physically moving to target
    REVERSING = auto()      # Reversing to pre-pivot position after collision/timeout
    WAITING = auto()        # Post-reversal random cooldown before resuming IDLE


@dataclass
class Token:
    """Direction vector pointing toward fault (phase 1) or empty slot (phase 2)."""
    direction: np.ndarray   # relative vector
    source_id: str          # who generated/forwarded this token


@dataclass
class DiscoverMessage:
    """Flood phase message for distributed connected-component discovery."""
    flood_id: int
    initiator_mid: str
    sender_mid: str


@dataclass
class EchoMessage:
    """Echo phase message carrying aggregated component IDs back to the root."""
    flood_id: int
    initiator_mid: str
    sender_mid: str
    component_ids: Set[str]


@dataclass
class FloodParticipation:
    """Per-agent state for participating in one flood/echo round."""
    flood_id: int
    parent_mid: Optional[str]
    children: Set[str] = field(default_factory=set)
    pending_echoes: Set[str] = field(default_factory=set)
    aggregated_component: Set[str] = field(default_factory=set)


@dataclass
class ModuleAgent:
    """
    Independent agent for one module.

    Each agent maintains its own local view and acts asynchronously.
    """
    module_id: str
    body_idx: int                           # PyBullet body index
    state: ModuleState = ModuleState.IDLE
    token: Optional[Token] = None           # current held token
    incoming_tokens: List[Token] = field(default_factory=list)
    target_pos: Optional[np.ndarray] = None  # cached world goal (for logs / legacy)
    target_pos_local: Optional[np.ndarray] = None  # goal COM offset in lattice_ref frame
    lattice_ref_body_idx: Optional[int] = None     # frozen for maneuver (usually first axis)
    attract_body_idx: Optional[int] = None
    attract_connector: Optional[int] = None
    pivot_axis_idx: Optional[int] = None    # which neighbor we're torquing against
    pivot_type: Optional[str] = None        # "corner" or "lateral"
    handoff_idx: Optional[int] = None       # body idx of handoff module (lateral only)
    handoff_done: bool = False              # whether lateral handoff has occurred
    process_ready_time: float = 0.0         # sim time when processing delay ends
    is_faulty: bool = False
    position_history: Set[Tuple] = field(default_factory=set)  # anti-oscillation
    pre_pivot_pos_local: Optional[np.ndarray] = None   # lattice pos before pivot (for reversal)
    pre_pivot_lattice_ref: Optional[int] = None        # ref body for pre_pivot_pos_local
    pending_retry: Optional[Tuple] = None              # (target_local, axis, type, handoff)
    wait_until: float = 0.0                            # sim time when WAITING expires
    token_hold_until: float = 0.0                        # sim time: stay IDLE before acting on token
    moving_token_received_tick: int = -999               # tick when last "moving" token was received
    # Two decoupled budgets, reset per phase by the runner:
    #   action_points  — MOTION budget. One spent per pivot (success,
    #     retarget, or giveup). Init = INITIAL_ACTION_POINTS.
    #   forward_points — MESSAGING budget. One spent per token forward.
    #     Init = INITIAL_ACTION_POINTS * FORWARD_POINTS_MULTIPLIER (100x),
    #     so communication does not starve motion (the prior shared-budget
    #     design let forwarding consume the entire cap at scale).
    # An agent engages a token while EITHER budget remains; it moves only
    # while action_points > 0 and forwards only while forward_points > 0.
    action_points: int = 10
    forward_points: int = 1000
    completed_moves: int = 0                             # diagnostic: # of successful pivots (NOT a cap)
    # Usage accounting (diagnostic). ``ap_spent_pivots`` counts motion
    # points spent (1 per pivot/retarget/giveup); ``ap_spent_forwards``
    # counts messaging points spent (1 per forward). Reset between phases.
    ap_spent_pivots: float = 0.0
    ap_spent_forwards: float = 0.0
    # Legacy field, no longer used as a cap (the action_points budget is
    # the unified cap mechanism). Kept for back-compat with dump readers.
    consecutive_pick_failures: int = 0
    # Cells (in current axis's lattice frame) already attempted by retarget
    # in the current maneuver chain. Used to avoid retargeting to the same
    # failed cell forever. Cleared whenever the agent returns to IDLE.
    retarget_blacklist: Set[Tuple[int, int, int]] = field(default_factory=set)
    # Count of retarget attempts in the current maneuver chain (reset on
    # IDLE). Caps total retargets per primary pivot to MAX_RETARGETS_PER_PIVOT.
    retarget_count: int = 0
    reversal_handoff_idx: Optional[int] = None           # original axis to bond-switch to during lateral reversal
    discover_inbox: List["DiscoverMessage"] = field(default_factory=list)
    echo_inbox: List["EchoMessage"] = field(default_factory=list)
    flood_participations: Dict[int, "FloodParticipation"] = field(default_factory=dict)


def _token_origin_world_pos(
    token: Token,
    pos: np.ndarray,
    agents: Dict[str, ModuleAgent],
    *,
    fault_id: Optional[str] = None,
    fault_body_idx: Optional[int] = None,
    fidx_to_fid: Optional[Dict[int, str]] = None,
) -> Optional[np.ndarray]:
    """World position of the module (or fault body) that emitted the token."""
    if fidx_to_fid:
        fid_to_fidx = {v: k for k, v in fidx_to_fid.items()}
        if token.source_id in fid_to_fidx:
            idx = fid_to_fidx[token.source_id]
            if 0 <= idx < len(pos):
                return pos[idx].copy()
            return None
    if fault_id is not None and token.source_id == fault_id:
        if fault_body_idx is not None and 0 <= fault_body_idx < len(pos):
            return pos[fault_body_idx].copy()
        return None
    if token.source_id in agents:
        idx = agents[token.source_id].body_idx
        if 0 <= idx < len(pos):
            return pos[idx].copy()
    return None


# (target_pos_local, axis_body_idx, pivot_type, handoff_idx or None for corner)
PivotPick = Tuple[np.ndarray, int, str, Optional[int]]


class DecentralizedCoagulation:
    """
    Async decentralized coagulation (phase 1).

    Drives an event loop: advance physics, then let each idle agent
    check its state and act. Agents discover neighbors from the
    physics world, not from a graph data structure.
    """

    TOKEN_PROCESS_DELAY = 1.0   # seconds before forwarding a token
    BOND_THRESHOLD = 1.05       # physical distance gate for bond creation

    # Pivot concurrency control.
    #   None  -> global lock: only one module may be PIVOTING at a time
    #            (legacy default, used by the 5-module demo and AsyncSimRunner).
    #   int k -> local  k-hop exclusion: a module may start a pivot iff no body
    #            within k bonded hops is currently PIVOTING.
    PIVOT_EXCLUSION_RADIUS: Optional[int] = None

    # If True, the fault body is a valid pivot axis / lateral handoff / post-
    # pivot proximity-bond neighbor. Default False matches legacy behavior
    # where the fault is treated as an unreliable anchor. Set True when the
    # fault is a real, full-mass, collision-on body (e.g. line-fault demo) so
    # modules can "swing around" it and close the loop.
    ALLOW_FAULT_AS_PIVOT_NEIGHBOR: bool = True

    # Probability (per tick) of accepting a random safe move when no
    # distance-reducing move exists.  Set > 0 to break symmetric deadlocks
    # (e.g. tie-fighter degeneracy on a center-fault line).
    TEMPERATURE: float = 0.5  # per-tick prob of random exploration move

    # Fully-random motion baseline. When True, coagulation skips distress-
    # directed (greedy) pivot selection entirely: any movable token-holder
    # takes a random admissible (connectivity-safe) pivot. The fault-token
    # flood, criticality gate, and exclusion radius are unchanged, so this is
    # an undirected-motion control for the directed policy.
    RANDOM_BASELINE: bool = False

    # Per-module MOTION budget at phase start (1 point per pivot). Tuned
    # default 5: empirically agents use ~1.2-1.5 motion points regardless
    # of cap, so 5 never binds and keeps phases short. Forwards draw from
    # the separate forward_points pool, not this one.
    INITIAL_ACTION_POINTS: int = 5
    # Back-compat alias for the older CLI/config name.
    MAX_MOVES_PER_MODULE: int = 5
    # Messaging budget = FORWARD_POINTS_MULTIPLIER × motion budget. Token
    # forwards draw from this separate pool so communication never starves
    # the motion budget (see ModuleAgent.forward_points). Kept modest so
    # the phase still terminates promptly once motion is exhausted —
    # too-large values let agents churn forwards long after they can move.
    FORWARD_POINTS_MULTIPLIER: int = 10

    # If True, pick_target rejects (axis, target_cell) pairs the agent has
    # already visited via a successful pivot this phase. Prevents
    # oscillation between previously-visited cells. Set False to allow
    # revisits — useful for diagnosing whether history dedup is what's
    # blocking the policy at a particular cell.
    USE_POSITION_HISTORY: bool = True

    # Hard cap on retarget attempts within a single primary pivot chain.
    # Combined with the retarget_blacklist (no cell retried twice), this
    # bounds the work spent on any one stuck pivot. After this many
    # retargets without success, the agent gives up the maneuver entirely
    # — stop_pivot, IDLE.
    MAX_RETARGETS_PER_PIVOT: int = 10

    # Flood/echo distributed component-discovery protocol. When True
    # (default), fault-adjacent modules withhold tokens until a flood/echo
    # round completes and stop emitting once their local component grows
    # ("fault resolved"). When False, mirrors the non-bullet MC: tokens are
    # emitted every TOKEN_GEN_INTERVAL unconditionally; coag termination is
    # purely global is_connected(). Useful for diagnosing whether the
    # flood/echo gating slows or destabilizes multi-fault scenarios.
    USE_FLOOD_ECHO: bool = True

    FORWARD_AP_COST: float = 0.1

    def __init__(self, sim, fault_id: str, module_ids: List[str],
                 body_indices: Dict[str, int]):
        """
        Args:
            sim: BulletSimulator instance
            fault_id: ID of the faulty module (removed from world)
            module_ids: list of active (non-faulty) module IDs
            body_indices: {module_id: body_idx} mapping
        """
        self.sim = sim
        self.fault_id = fault_id

        # Create an agent for each active module
        self.agents: Dict[str, ModuleAgent] = {}
        for mid in module_ids:
            self.agents[mid] = ModuleAgent(
                module_id=mid,
                body_idx=body_indices[mid],
                action_points=self.INITIAL_ACTION_POINTS,
                forward_points=(self.INITIAL_ACTION_POINTS
                                * self.FORWARD_POINTS_MULTIPLIER),
            )

        # Reverse mapping: body_idx -> module_id
        self._idx_to_mid: Dict[int, str] = {v: k for k, v in body_indices.items()}

        # Fault-adjacent agents and their directions
        self._fault_adjacent: Set[str] = set()
        self._fault_directions: Dict[str, np.ndarray] = {}
        self._last_token_gen_time: float = -999.0
        self.TOKEN_GEN_INTERVAL = 10.0  # re-emit tokens every 10 s (tuned)

        # Statistics
        self.total_moves = 0
        self.successful_moves = 0
        self.move_log: List[Dict] = []
        self._tick_count: int = 0
        self.reversal_count: int = 0
        # Per-module retrace history: ordered list of the pivots each module
        # successfully completed, recorded as (axis_module_id,
        # source_pos_in_axis_frame, pivot_type). Phase-2 retrace replays these
        # in reverse to undo coagulation locally. Each entry is fully local:
        # the axis is a neighbor identity and the source position is expressed
        # in that axis's body frame, so it reconstructs correctly against the
        # axis's *current* pose without any absolute/global measurement.
        self.pivot_history: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}

        # Flood/echo protocol state
        self._flood_counter: int = 0
        self._known_components: Dict[str, FrozenSet[str]] = {}
        self._fault_resolved_mids: Set[str] = set()
        self._pending_floods: Dict[str, int] = {}
        self.FLOOD_ECHO_INTERVAL: float = 5.0
        self._last_flood_time: Dict[str, float] = {}
        self._awaiting_first_flood: Set[str] = set()

    def _decision_graph_neighbors(self, body_idx: int) -> List[int]:
        """Bond neighbors for topology checks: frozen bond matrix at tick start, else live."""
        bm = getattr(self, "_decision_bond_matrix", None)
        if bm is None:
            bm = self.sim.get_bond_matrix()
        return bond_graph_neighbors(bm, self.sim.N, body_idx)

    def _bfs_reachable(self, start_idx: int, max_depth: int,
                       excluded: Set[int]) -> Set[int]:
        """BFS on the decision-time bond graph (snapshot during tick, else live)."""
        visited = {start_idx}
        frontier = {start_idx}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for nbr in self._decision_graph_neighbors(node):
                    if nbr not in excluded and nbr not in visited:
                        next_frontier.add(nbr)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(start_idx)
        return visited


    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        """
        Two-graph movability test.

        **Leaf check** uses the full physical graph (fault included) so that
        fault-adjacent modules are not mistaken for leaves.

        **Articulation-point check** uses the active-only subgraph (fault
        excluded from traversal) so that paths through the passive fault
        body are not counted as valid alternative routes.
        """
        fault_idxs = self._get_fault_idxs()

        all_neighbors = self._decision_graph_neighbors(body_idx)
        if len(all_neighbors) == 0:
            return False
        if len(all_neighbors) == 1:
            return True

        active_neighbors = [n for n in all_neighbors if n not in fault_idxs]
        if len(active_neighbors) == 0:
            return False

        for v in active_neighbors:
            excluded = {body_idx, v} | fault_idxs
            reachable_without_v = set()
            for w in active_neighbors:
                if w == v:
                    continue
                reachable_without_v |= self._bfs_reachable(
                    w, safety_radius - 1, excluded)
            v_neighbors = set(self._decision_graph_neighbors(v)) - fault_idxs
            if not (reachable_without_v & v_neighbors):
                return False

        return True

    def set_fault_adjacent(self, adjacent_mids: List[str],
                           fault_body_idx: int):
        """Register fault-adjacent modules and the fault body index."""
        self._fault_adjacent = set(adjacent_mids)
        self._fault_body_idx = fault_body_idx
        self._fault_body_idxs: Set[int] = {fault_body_idx}
        self._fault_ids: Set[str] = {self.fault_id}
        self._adj_to_fault_idx: Dict[str, int] = {
            mid: fault_body_idx for mid in adjacent_mids}
        self._fidx_to_fid: Dict[int, str] = {fault_body_idx: self.fault_id}
        self._seed_flood_echo_state(adjacent_mids)

    def set_multi_fault_adjacent(
        self,
        fault_ids: List[str],
        fault_body_idxs: List[int],
        adjacent_map: Dict[str, int],
    ):
        """Register multiple faults and their adjacent modules.

        Args:
            fault_ids: IDs of all faulty modules.
            fault_body_idxs: body indices of all faulty modules.
            adjacent_map: {active_module_id: fault_body_idx} mapping each
                fault-adjacent module to the fault body it neighbours.
        """
        self._fault_ids = set(fault_ids)
        self._fault_body_idxs = set(fault_body_idxs)
        self._fault_body_idx = fault_body_idxs[0] if fault_body_idxs else None
        self._fault_adjacent = set(adjacent_map.keys())
        self._adj_to_fault_idx = dict(adjacent_map)
        self._fidx_to_fid = dict(zip(fault_body_idxs, fault_ids))
        self._seed_flood_echo_state(list(adjacent_map.keys()))

    def _seed_flood_echo_state(self, adjacent_mids: List[str]):
        """Initialize flood/echo protocol for newly registered fault-adjacent modules.

        Preserves resolved status for modules that were already resolved from
        a prior call (e.g. stall-recovery re-registration in bullet_bridge).
        """
        if not self.USE_FLOOD_ECHO:
            # Skip token-gating: emit tokens immediately from every fault-
            # adjacent module on the next _generate_fault_tokens tick.
            return
        for mid in adjacent_mids:
            if mid in self._fault_resolved_mids:
                continue
            self._awaiting_first_flood.add(mid)
            self._last_flood_time[mid] = -999.0

    def _get_fault_idxs(self) -> Set[int]:
        """Return the set of all fault body indices."""
        idxs = getattr(self, "_fault_body_idxs", None)
        if idxs is not None:
            return idxs
        single = getattr(self, "_fault_body_idx", None)
        if single is not None:
            return {single}
        return set()

    def _generate_fault_tokens(self):
        """Fault-adjacent modules emit tokens every TOKEN_GEN_INTERVAL seconds.

        Token direction = displacement from agent to fault in the emitter's
        body frame (drift-invariant).  Recomputed each time.

        Multi-fault: each adjacent module points at its specific fault body
        via ``_adj_to_fault_idx``.
        """
        if self.sim.sim_time - self._last_token_gen_time < self.TOKEN_GEN_INTERVAL:
            return
        self._last_token_gen_time = self.sim.sim_time

        pos = self.sim.get_positions()
        adj_map = getattr(self, "_adj_to_fault_idx", None)

        for mid in self._fault_adjacent:
            if mid in self._fault_resolved_mids:
                continue
            if mid in self._awaiting_first_flood:
                continue
            if mid not in self.agents:
                continue
            agent = self.agents[mid]
            if agent.state != ModuleState.IDLE:
                continue
            # Only seed a token if the module can still propagate it.
            # A fault-adjacent module almost never makes a productive
            # self-move (it is already next to the fault); its role is to
            # SEED the wavefront for movers elsewhere, which requires
            # forwarding. Once its forward budget is spent, generating
            # self-tokens just creates non-terminating churn (the seed can
            # neither move nor relay). Gating here bounds total tokens to
            # ~ sum of forward budgets, guaranteeing the phase terminates.
            if agent.forward_points <= 0:
                continue
            if adj_map is not None and mid in adj_map:
                f_idx = adj_map[mid]
            else:
                f_idx = self._fault_body_idx
            fault_pos = pos[f_idx]
            R = self.sim.body_rotation_matrix(agent.body_idx)
            direction = R.T @ (fault_pos - pos[agent.body_idx])
            fidx_to_fid = getattr(self, "_fidx_to_fid", {})
            source = fidx_to_fid.get(f_idx, self.fault_id)
            agent.incoming_tokens.append(
                Token(direction=direction.copy(), source_id=source))

    # ------------------------------------------------------------------
    # Flood / Echo distributed component-discovery protocol
    # ------------------------------------------------------------------

    def _initiate_flood(self, mid: str):
        """Start a new flood/echo round from fault-adjacent module *mid*."""
        if mid not in self.agents:
            return
        agent = self.agents[mid]

        self._flood_counter += 1
        fid = self._flood_counter
        self._pending_floods[mid] = fid

        bond_matrix = self.sim.get_bond_matrix()
        fault_idxs = self._get_fault_idxs()
        neighbors = [
            j for j in range(self.sim.N)
            if j != agent.body_idx
            and bond_matrix[agent.body_idx, j]
            and j not in fault_idxs
        ]

        children: Set[str] = set()
        for j in neighbors:
            n_mid = self._idx_to_mid.get(j)
            if n_mid and n_mid in self.agents:
                self.agents[n_mid].discover_inbox.append(
                    DiscoverMessage(flood_id=fid, initiator_mid=mid,
                                    sender_mid=mid))
                children.add(n_mid)

        participation = FloodParticipation(
            flood_id=fid,
            parent_mid=None,
            children=set(children),
            pending_echoes=set(children),
            aggregated_component={mid},
        )
        agent.flood_participations[fid] = participation

        if not children:
            self._on_flood_complete(mid, frozenset({mid}))

        self._last_flood_time[mid] = self.sim.sim_time

    def _maybe_initiate_floods(self):
        """Initiate flood/echo rounds for fault-adjacent modules on heartbeat."""
        for mid in self._fault_adjacent:
            if mid in self._fault_resolved_mids:
                continue
            if mid in self._pending_floods:
                continue
            last = self._last_flood_time.get(mid, -999.0)
            if self.sim.sim_time - last < self.FLOOD_ECHO_INTERVAL:
                continue
            self._initiate_flood(mid)

    def _process_discover_messages(self):
        """Deliver one hop of the flood wave: each agent processes its discover inbox."""
        fault_idxs = self._get_fault_idxs()
        bond_matrix = self.sim.get_bond_matrix()

        for mid, agent in self.agents.items():
            if not agent.discover_inbox:
                continue
            messages = list(agent.discover_inbox)
            agent.discover_inbox.clear()

            for msg in messages:
                if msg.flood_id in agent.flood_participations:
                    continue

                neighbors = [
                    j for j in range(self.sim.N)
                    if j != agent.body_idx
                    and bond_matrix[agent.body_idx, j]
                    and j not in fault_idxs
                ]
                children: Set[str] = set()
                for j in neighbors:
                    n_mid = self._idx_to_mid.get(j)
                    if n_mid and n_mid in self.agents and n_mid != msg.sender_mid:
                        if msg.flood_id not in self.agents[n_mid].flood_participations:
                            self.agents[n_mid].discover_inbox.append(
                                DiscoverMessage(
                                    flood_id=msg.flood_id,
                                    initiator_mid=msg.initiator_mid,
                                    sender_mid=mid,
                                ))
                            children.add(n_mid)

                participation = FloodParticipation(
                    flood_id=msg.flood_id,
                    parent_mid=msg.sender_mid,
                    children=set(children),
                    pending_echoes=set(children),
                    aggregated_component={mid},
                )
                agent.flood_participations[msg.flood_id] = participation

                if not children:
                    parent_agent = self.agents.get(msg.sender_mid)
                    if parent_agent is not None:
                        parent_agent.echo_inbox.append(
                            EchoMessage(
                                flood_id=msg.flood_id,
                                initiator_mid=msg.initiator_mid,
                                sender_mid=mid,
                                component_ids={mid},
                            ))

    def _process_echo_messages(self):
        """Deliver one hop of the echo wave: aggregate component IDs toward root."""
        for mid, agent in self.agents.items():
            if not agent.echo_inbox:
                continue
            messages = list(agent.echo_inbox)
            agent.echo_inbox.clear()

            for msg in messages:
                part = agent.flood_participations.get(msg.flood_id)
                if part is None:
                    continue

                part.aggregated_component |= msg.component_ids
                part.pending_echoes.discard(msg.sender_mid)

                if not part.pending_echoes:
                    if part.parent_mid is None:
                        self._on_flood_complete(
                            mid, frozenset(part.aggregated_component))
                    else:
                        parent_agent = self.agents.get(part.parent_mid)
                        if parent_agent is not None:
                            parent_agent.echo_inbox.append(
                                EchoMessage(
                                    flood_id=msg.flood_id,
                                    initiator_mid=msg.initiator_mid,
                                    sender_mid=mid,
                                    component_ids=set(part.aggregated_component),
                                ))

    def _on_flood_complete(self, mid: str, component: FrozenSet[str]):
        """Handle completion of a flood/echo round for fault-adjacent module *mid*."""
        self._pending_floods.pop(mid, None)

        if mid in self._awaiting_first_flood:
            self._known_components[mid] = component
            self._awaiting_first_flood.discard(mid)
            logger.debug("Flood/echo baseline for {}: {} modules", mid, len(component))
            return

        prev = self._known_components.get(mid, frozenset())
        if component > prev:
            self._fault_resolved_mids.add(mid)
            logger.info(
                "Flood/echo: {} component grew ({} -> {}), fault resolved",
                mid, len(prev), len(component))
        else:
            self._known_components[mid] = component

    def _cleanup_stale_flood_participations(self):
        """Remove flood participation records for completed floods."""
        active_flood_ids: Set[int] = set(self._pending_floods.values())
        for agent in self.agents.values():
            stale = [fid for fid in agent.flood_participations
                     if fid not in active_flood_ids]
            for fid in stale:
                del agent.flood_participations[fid]

    def get_physical_neighbors(self, body_idx: int) -> List[int]:
        """Discover bonded neighbors from PyBullet bond state."""
        neighbors = []
        bond_matrix = self.sim.get_bond_matrix()
        for j in range(self.sim.N):
            if j != body_idx and bond_matrix[body_idx, j]:
                neighbors.append(j)
        return neighbors

    def get_nearby_lattice_positions(self, body_idx: int) -> List[np.ndarray]:
        """
        Get unoccupied lattice positions near this module's neighbors.

        Looks at neighbors and neighbors' neighbors to find lattice
        positions that are not currently occupied.
        """
        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        neighbors = self.get_physical_neighbors(body_idx)

        # Collect all occupied positions (snapped to lattice)
        occupied = set()
        for i in range(self.sim.N):
            lp = tuple(np.round(pos[i]).astype(int))
            occupied.add(lp)

        # Candidate positions: lattice positions adjacent to my neighbors
        # (and neighbors' neighbors) that are unoccupied
        candidate_set = set()
        check_indices = set(neighbors)
        for n_idx in neighbors:
            for nn_idx in self.get_physical_neighbors(n_idx):
                check_indices.add(nn_idx)

        for idx in check_indices:
            nbr_pos = np.round(pos[idx]).astype(int)
            # 6-connected lattice neighbors
            for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                candidate = tuple(nbr_pos + np.array(delta))
                if candidate not in occupied:
                    candidate_set.add(candidate)

        # Also include positions adjacent to myself
        my_lattice = np.round(my_pos).astype(int)
        for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            candidate = tuple(my_lattice + np.array(delta))
            if candidate not in occupied:
                candidate_set.add(candidate)

        return [np.array(c, dtype=float) for c in candidate_set]

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        """World point used to score pivot candidates (minimize goal COM distance).

        Default: emitter COM from ``_token_origin_world_pos`` (fault body or
        ``source_id`` module). Subclasses may override for e.g. a virtual goal
        carried in assembly body frame (see five-module pivot agent demo).
        """
        if agent.token is None:
            return None
        return _token_origin_world_pos(
            agent.token,
            pos,
            self.agents,
            fault_id=self.fault_id,
            fault_body_idx=getattr(self, "_fault_body_idx", None),
            fidx_to_fid=getattr(self, "_fidx_to_fid", None),
        )

    def _on_no_pick_target(self, agent: ModuleAgent) -> None:
        """Hook when a movable agent with a token has no ``pick_target``.

        Default: no-op. Subclasses may e.g. advance a turn counter before the
        agent enters ``PROCESSING`` and forwards the token.
        """

    def _pick_target_occupancy_skip_body_indices(self) -> Set[int]:
        """Bodies omitted when building the occupied lattice set in ``pick_target``.

        Default: none. A ghost goal body (collision-off, same cell as the nominal
        empty slot) should be listed so pivots **into** that cell are not
        rejected as ``target_cell in occupied``.
        """
        return set()

    def _ghost_goal_world_pos_for_axial_corner(
            self, pos: np.ndarray) -> Optional[np.ndarray]:
        """If not None, allow a **colinear** (spine) corner step into that cell.

        Standard corners require a lattice step perpendicular to ``pivot−axis``;
        the last move onto a ghost goal at the nominal chain tip is parallel to
        that arm and is otherwise never enumerated.
        """
        return None

    def pick_target(
            self,
            agent: ModuleAgent,
            *,
            require_closer_to_origin: bool = False,
            random_choice: bool = False,
    ) -> Optional[PivotPick]:
        """
        Enumerate eligible **corner** and **lateral** pivots, then pick the
        candidate whose goal COM is **closest** to the scoring origin from
        ``_origin_world_for_pick_target`` (default: token emitter COM). Pivot
        type and handoff index are fixed at selection time.

        If ``require_closer_to_origin`` is True, drop any candidate whose goal COM is not **strictly closer** to the origin than the pivoting module's
        current COM (so the agent stops when no pivot improves distance).

        If ``random_choice`` is True, return a uniformly random candidate from
        the eligible set instead of the best-scoring one. Used by the
        temperature-based deadlock breaker to explore non-greedy moves.

        **Corner** (for each graph neighbor ``axis_idx`` of the pivot):

        - Step ``nom * delta`` in the **axis** body frame; the world step
          ``R_axis @ delta`` must be perpendicular to ``arm = pivot − axis``
          (``_lattice_delta_perpendicular_to_arm``).
        - Target cell (axis lattice coords) must not be occupied and must not
          be the pivot's current cell; ``(axis_idx, target_cell)`` must not
          appear in ``agent.position_history``.
        - ``|cos ∠(arm, target − axis)| ≤ _CORNER_PIVOT_MAX_ABS_COS`` (0.35) so
          the corner is **~90°**, not ~0° or ~180° (the old 0.99 bound allowed
          ~171°).

        **Not** applied: rejecting a target that lies **on the chain backbone**
        through ``axis`` (e.g. one more step past M2 colinear with M1–M2 when
        the pivot is still **sideways** to that line). Such a cell is still a
        valid **90°** corner from a sideways arm, so it competes with other
        perpendiculars; only an explicit **spine** filter would forbid it.

        **Lateral:** handoff neighbor ``handoff_idx`` with axis–handoff step
        cardinal in the pivot frame; ``target = pos[handoff] + (pivot − axis)``;
        same occupancy/history checks and ``|cos| ≤ _LATERAL_PIVOT_MAX_ABS_COS``.

        **Tie-break:** higher score first, then lower ``axis_idx``, then lower
        ``handoff_idx`` (see ``scored.sort``).
        """
        if agent.token is None:
            return None

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        token_dir = agent.token.direction
        if np.linalg.norm(token_dir) < 1e-8:
            return None

        origin = self._origin_world_for_pick_target(agent, pos)
        if origin is None:
            return None

        neighbors = self._decision_graph_neighbors(agent.body_idx)
        if not neighbors:
            return None

        fault_idxs = self._get_fault_idxs()
        # score = negative distance to token origin (maximize = minimize dist)
        scored: List[Tuple[float, np.ndarray, int, str, Optional[int]]] = []
        nom = float(self.sim.NOMINAL_DIST)
        module_shape = getattr(self.sim, "_module_shape", "sphere")

        for axis_idx in neighbors:
            if (fault_idxs and axis_idx in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            R = self.sim.body_rotation_matrix(axis_idx)
            p_ref = pos[axis_idx]
            occupied = set()
            skip_occ = self._pick_target_occupancy_skip_body_indices()
            for i in range(self.sim.N):
                if i in skip_occ:
                    continue
                u = R.T @ (pos[i] - p_ref) / nom
                occupied.add(tuple(np.round(u).astype(int)))
            my_cell = tuple(np.round(R.T @ (my_pos - p_ref) / nom).astype(int))

            arm_axis_to_pivot = my_pos - pos[axis_idx]

            gpos_ax = self._ghost_goal_world_pos_for_axial_corner(pos)
            ghost_goal_cell: Optional[Tuple[int, int, int]] = None
            if gpos_ax is not None:
                ghost_goal_cell = tuple(
                    np.round(R.T @ (gpos_ax - p_ref) / nom).astype(int))

            # Cube modules cannot do sphere-style corner pivots: a 90° lever
            # around the contact edge lands the cube at a cell *diagonal* to
            # axis (= the lateral-pivot destination), not at a perpendicular
            # neighbor of axis. Skip the corner enumeration entirely for cubes.
            corner_enabled = (module_shape != "cube")

            # ── Corner: 4 sites on axis — steps ⟂ (pivot − axis) in ref frame ──
            for delta in _LATTICE_DELTAS:
                if not corner_enabled:
                    break
                perp = _lattice_delta_perpendicular_to_arm(
                    delta, arm_axis_to_pivot, R)
                dw = nom * np.array(delta, dtype=float)
                target_world = p_ref + R @ dw
                if PLANAR_Z_LOCK:
                    if abs(target_world[2]) > 1e-6:
                        continue
                    if not _planar_pivot_admissible(
                            pos, agent.body_idx, axis_idx, target_world,
                            len(pos)):
                        continue
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                axial_into_ghost = (
                    ghost_goal_cell is not None
                    and target_cell == ghost_goal_cell)
                if not perp and not axial_into_ghost:
                    continue
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (self.USE_POSITION_HISTORY
                        and (axis_idx, target_cell) in agent.position_history):
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_tar_n = np.linalg.norm(r_target)
                if r_tar_n < 1e-12:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * r_tar_n + 1e-12)
                if axial_into_ghost:
                    if gpos_ax is None:
                        continue
                    d_tgt = float(np.linalg.norm(target_world - gpos_ax))
                    d_my = float(np.linalg.norm(my_pos - gpos_ax))
                    if d_tgt >= d_my - 1e-9:
                        continue
                else:
                    if abs(cos_angle) > _CORNER_PIVOT_MAX_ABS_COS:
                        continue
                dist_to_origin = float(np.linalg.norm(target_world - origin))
                score = -dist_to_origin
                if PLANAR_Z_LOCK:
                    # planar demos: break score ties randomly so the
                    # deterministic kinematic sim samples different paths
                    # across attempts
                    score += random.uniform(0.0, 1e-3)
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "corner", None))

            # ── Lateral: copy axis–pivot offset to handoff (one φ from handoff COM) ──
            # Rule: a lateral pivot is rejected ONLY when going from a
            # non-fault axis to a fault handoff. Allowed cases:
            #   non-fault → non-fault, fault → non-fault, fault → fault.
            # Disallowed: non-fault → fault.
            axis_is_fault = bool(fault_idxs and axis_idx in fault_idxs)
            axis_neighbors = self._decision_graph_neighbors(axis_idx)
            for handoff_idx in axis_neighbors:
                if handoff_idx == agent.body_idx:
                    continue
                handoff_is_fault = bool(
                    fault_idxs and handoff_idx in fault_idxs)
                if handoff_is_fault and not axis_is_fault:
                    continue
                d_ah = pos[handoff_idx] - pos[axis_idx]
                if not _lateral_axis_handoff_step_cardinal_for_axis(
                        self.sim, axis_idx, d_ah):
                    continue
                target_world = pos[handoff_idx] + (my_pos - pos[axis_idx])
                if PLANAR_Z_LOCK:
                    if abs(target_world[2]) > 1e-6:
                        continue
                    if not _planar_lateral_clear(
                            pos, agent.body_idx, axis_idx, handoff_idx,
                            target_world, len(pos)):
                        continue
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (self.USE_POSITION_HISTORY
                        and (axis_idx, target_cell) in agent.position_history):
                    continue
                if np.linalg.norm(target_world - pos[axis_idx]) < 1.05:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_vec_n = np.linalg.norm(r_vec)
                r_tar_n = np.linalg.norm(r_target)
                if r_vec_n < 1e-8 or r_tar_n < 1e-8:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (r_vec_n * r_tar_n)
                if abs(cos_angle) > _LATERAL_PIVOT_MAX_ABS_COS:
                    continue
                if module_shape == "cube":
                    if not _cube_lateral_swept_clear(
                            occupied, my_cell, target_cell,
                            arm_axis_to_pivot, R, nom):
                        continue
                dist_to_origin = float(np.linalg.norm(target_world - origin))
                score = -dist_to_origin
                if PLANAR_Z_LOCK:
                    # planar demos: break score ties randomly so the
                    # deterministic kinematic sim samples different paths
                    # across attempts
                    score += random.uniform(0.0, 1e-3)
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "lateral", handoff_idx))

        if not scored:
            return None

        if require_closer_to_origin:
            current_d = float(np.linalg.norm(my_pos - origin))
            eps = 1e-9 * max(1.0, nom)
            scored = [s for s in scored if (-s[0]) < current_d - eps]
            if not scored:
                return None

        if random_choice:
            pick = random.choice(scored)
        else:
            scored.sort(key=lambda x: (-x[0], x[2], x[4] if x[4] is not None else -1))
            pick = scored[0]
        return pick[1].copy(), pick[2], pick[3], pick[4]

    def _propagate_moving_tokens(self):
        """Flood-fill moving tokens from PIVOTING/REVERSING agents."""
        if self.PIVOT_EXCLUSION_RADIUS is None:
            return
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        bm = self._decision_bond_matrix.copy()
        for a in self.agents.values():
            if a.state in _moving and a.pivot_axis_idx is not None:
                bm[a.body_idx, a.pivot_axis_idx] = True
                bm[a.pivot_axis_idx, a.body_idx] = True

        seeds: set = set()
        for a in self.agents.values():
            if a.state in _moving:
                a.moving_token_received_tick = self._tick_count
                seeds.add(a.body_idx)

        frontier = set(seeds)
        for _hop in range(self.PIVOT_EXCLUSION_RADIUS):
            nxt: set = set()
            for i in frontier:
                for j in range(self.sim.N):
                    if bm[i, j] and j not in seeds:
                        mid_j = self._idx_to_mid.get(j)
                        if mid_j and mid_j in self.agents:
                            self.agents[mid_j].moving_token_received_tick = self._tick_count
                        seeds.add(j)
                        nxt.add(j)
            frontier = nxt
            if not frontier:
                break

    def tick(self) -> bool:
        """
        Run one decision cycle for all agents.

        Called after each physics time slice. Each agent checks its state
        and acts if ready. Returns True if any agent is still active
        (pivoting or has tokens to process).
        """
        self._tick_count += 1

        # Flood/echo protocol: process messages, then check heartbeats.
        # Skipped entirely when USE_FLOOD_ECHO=False (matches non-bullet MC).
        if self.USE_FLOOD_ECHO:
            self._process_discover_messages()
            self._process_echo_messages()
            self._maybe_initiate_floods()
            if self._tick_count % 500 == 0:
                self._cleanup_stale_flood_participations()

        # Fault-adjacent modules emit tokens (skips resolved / awaiting-baseline)
        self._generate_fault_tokens()

        # One consistent bond graph for all move decisions this tick.
        # Includes the fault body so fault-adjacent modules are non-leaves.
        self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

        self._propagate_moving_tokens()

        any_active = False
        agent_items = list(self.agents.items())
        random.shuffle(agent_items)

        for mid, agent in agent_items:
            if agent.is_faulty:
                continue

            if agent.state == ModuleState.IDLE:
                # Skip entirely only when BOTH budgets are spent — a
                # motion-exhausted agent can still relay tokens, so it must
                # fall through to the token-consumption logic below.
                if agent.action_points <= 0 and agent.forward_points <= 0:
                    continue
                neighbors = self._decision_graph_neighbors(agent.body_idx)
                bm = self._decision_bond_matrix
                pos = self.sim.get_positions()
                triangle_found = False
                # Triangle cleanup is a corrective pivot (motion), so it is
                # gated on motion budget AND is_movable: the cleanup
                # unconditionally removes a bond and corner-pivots the agent,
                # which can sever the agent's other neighbors when the agent
                # is an articulator. Skip cleanup for non-movable / motion-
                # exhausted agents.
                if agent.action_points > 0 and self.is_movable(
                        agent.body_idx,
                        getattr(self, "_safety_radius", 2)):
                    for ni_idx in range(len(neighbors)):
                        for nj_idx in range(ni_idx + 1, len(neighbors)):
                            a, b = neighbors[ni_idx], neighbors[nj_idx]
                            if bm[a, b]:
                                d_a = float(np.linalg.norm(pos[agent.body_idx] - pos[a]))
                                d_b = float(np.linalg.norm(pos[agent.body_idx] - pos[b]))
                                if d_a <= d_b:
                                    close, far = a, b
                                else:
                                    close, far = b, a
                                self.sim.remove_bond(agent.body_idx, far)
                                self._decision_bond_matrix = self.sim.get_bond_matrix().copy()
                                self._start_corrective_pivot(agent, close)
                                self._propagate_moving_tokens()
                                triangle_found = True
                                any_active = True
                                break
                        if triangle_found:
                            break
                if triangle_found:
                    continue

            if agent.state == ModuleState.PIVOTING:
                # Lateral handoff: when the module reaches the handoff
                # module mid-pivot, bond to it and disconnect from axis.
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        # Bond-level handoff
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        # Physics constraint switch: restart pivot
                        # around the handoff module toward the target
                        # Lateral handoff: agent has already swapped the
                        # axis bond to the handoff partner above; tell the
                        # sim NOT to restore a rigid M↔original-axis bond
                        # (would create a stretched duplicate).
                        self.sim.stop_pivot(agent.body_idx,
                                            restore_axis_bond=False)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} lateral handoff: bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                # Check if pivot converged
                if self.sim.is_pivot_complete(agent.body_idx):
                    collided = self.sim.pivot_collided(agent.body_idx)
                    timed_out = self.sim.pivot_timed_out(agent.body_idx)

                    if collided or timed_out:
                        # Failure: retarget to the nearest empty lattice cell
                        # WITHOUT calling stop_pivot. The rolling-sphere
                        # coupling persists — the module stays tethered to
                        # its axis throughout the retry chain. If no empty
                        # cell exists around the axis, give up cleanly
                        # (stop_pivot + IDLE).
                        if self._retarget_to_nearest_empty_cell(agent):
                            agent.action_points -= 1
                            agent.ap_spent_pivots += 1.0
                            self.reversal_count = (
                                getattr(self, "reversal_count", 0) + 1)
                            any_active = True
                            logger.info(
                                "Module {} pivot {} — retargeting to "
                                "nearest empty cell",
                                mid,
                                "collided" if collided else "timed out")
                        else:
                            agent.action_points -= 1
                            agent.ap_spent_pivots += 1.0
                            self.sim.stop_pivot(agent.body_idx)
                            self._reconnect_bonds(agent.body_idx)
                            self._decision_bond_matrix = (
                                self.sim.get_bond_matrix().copy())
                            agent.state = ModuleState.IDLE
                            agent.target_pos = None
                            agent.target_pos_local = None
                            agent.lattice_ref_body_idx = None
                            agent.attract_body_idx = None
                            agent.attract_connector = None
                            agent.pivot_axis_idx = None
                            agent.pivot_type = None
                            agent.handoff_idx = None
                            agent.handoff_done = False
                            agent.token = None
                            agent.retarget_blacklist.clear()
                            agent.retarget_count = 0
                            logger.info(
                                "Module {} pivot failed (no empty cell or "
                                "retarget cap hit) — giving up", mid)
                    else:
                        # Success: tear down rolling coupling, install rigid
                        # bond at the new lattice cell.
                        self.sim.stop_pivot(agent.body_idx)
                        if (agent.lattice_ref_body_idx is not None
                                and agent.target_pos_local is not None):
                            nom = float(self.sim.NOMINAL_DIST)
                            cell = tuple(
                                np.round(
                                    agent.target_pos_local / nom).astype(int))
                            agent.position_history.add(
                                (agent.lattice_ref_body_idx, cell))
                        self._reconnect_bonds(agent.body_idx)
                        self._decision_bond_matrix = (
                            self.sim.get_bond_matrix().copy())
                        # Record the retrace step BEFORE the per-pivot fields
                        # are reset below. Source position is stored in the
                        # axis's body frame (local + frame-robust).
                        if (agent.pre_pivot_lattice_ref is not None
                                and agent.pre_pivot_pos_local is not None):
                            _ax_mid = self._idx_to_mid.get(
                                agent.pre_pivot_lattice_ref)
                            if _ax_mid is not None:
                                self.pivot_history.setdefault(mid, []).append(
                                    (_ax_mid,
                                     agent.pre_pivot_pos_local.copy(),
                                     agent.pivot_type))
                        agent.state = ModuleState.IDLE
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.token = None
                        agent.retarget_blacklist.clear()
                        agent.retarget_count = 0
                        agent.completed_moves += 1
                        agent.action_points -= 1
                        agent.ap_spent_pivots += 1.0
                        self.successful_moves += 1
                        logger.info(
                            "Module {} pivot complete (ap left: {})",
                            mid, agent.action_points)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.REVERSING:
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        # Lateral handoff: agent has already swapped the
                        # axis bond to the handoff partner above; tell the
                        # sim NOT to restore a rigid M↔original-axis bond
                        # (would create a stretched duplicate).
                        self.sim.stop_pivot(agent.body_idx,
                                            restore_axis_bond=False)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} reversal handoff: bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    rev_collided = self.sim.pivot_collided(agent.body_idx)
                    rev_timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    self.sim.stop_pivot(agent.body_idx)
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if rev_collided or rev_timed_out:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None
                        self._emergency_snap_to_lattice(agent)
                        any_active = True
                        logger.info(
                            "Module {} reversal failed — snapping to "
                            "lattice", mid)
                    else:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None

                        cooldown = random.uniform(0, 10)
                        agent.wait_until = self.sim.sim_time + cooldown
                        agent.state = ModuleState.WAITING
                        any_active = True
                        logger.info(
                            "Module {} reversal complete — waiting "
                            "{:.1f}s", mid, cooldown)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.WAITING:
                if self.sim.sim_time >= agent.wait_until:
                    agent.state = ModuleState.IDLE
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.PROCESSING:
                # Wait for processing delay
                if self.sim.sim_time >= agent.process_ready_time:
                    if agent.forward_points > 0:
                        self._forward_token(agent)
                        agent.forward_points -= 1
                        agent.ap_spent_forwards += 1.0
                    agent.state = ModuleState.IDLE
                    agent.token = None
                else:
                    any_active = True
                continue

            # IDLE state: engage a token while EITHER budget remains —
            # motion (to pivot) or messaging (to relay). Only when both are
            # exhausted does the agent drop tokens and go quiet.
            if agent.incoming_tokens and (
                    agent.action_points > 0 or agent.forward_points > 0):
                best = min(agent.incoming_tokens,
                           key=lambda t: np.linalg.norm(t.direction))
                agent.token = best
                agent.incoming_tokens.clear()
                agent.token_hold_until = self.sim.sim_time + 1.0
                agent.state = ModuleState.HAS_TOKEN
            elif agent.incoming_tokens:
                # Both budgets exhausted — drop tokens so the agent no
                # longer cycles through HAS_TOKEN/PROCESSING.
                agent.incoming_tokens.clear()

            if agent.state == ModuleState.HAS_TOKEN:
                if self.sim.sim_time < agent.token_hold_until:
                    any_active = True
                    continue
                if (not self.is_movable(agent.body_idx)
                        or agent.action_points <= 0):
                    # Not safe to move, or motion budget exhausted --
                    # forward the token instead (uses the messaging budget).
                    agent.state = ModuleState.PROCESSING
                    agent.process_ready_time = (self.sim.sim_time
                                                + self.TOKEN_PROCESS_DELAY)
                    any_active = True
                    continue

                # Pivot concurrency check.
                _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
                if self.PIVOT_EXCLUSION_RADIUS is None:
                    if any(a.state in _moving
                           for a in self.agents.values()):
                        any_active = True
                        continue
                else:
                    if (self._tick_count - agent.moving_token_received_tick
                            < self.PIVOT_EXCLUSION_RADIUS):
                        any_active = True
                        continue

                if self.RANDOM_BASELINE:
                    # Undirected baseline: skip distress-directed selection;
                    # take a random admissible (connectivity-safe) pivot.
                    result = self.pick_target(
                        agent, require_closer_to_origin=False,
                        random_choice=True)
                else:
                    result = self.pick_target(
                        agent, require_closer_to_origin=True)
                    if (result is None and self.TEMPERATURE > 0
                            and random.random() < self.TEMPERATURE):
                        result = self.pick_target(
                            agent, require_closer_to_origin=False,
                            random_choice=True)

                if result is not None:
                    target_pos_local, axis_idx, pivot_type, handoff_idx = result
                    self._start_pivot(
                        agent, target_pos_local, axis_idx,
                        pivot_type, handoff_idx)
                    self._propagate_moving_tokens()
                    any_active = True
                    continue

                # No target found — agent will forward this token (and
                # spend an action point) once the PROCESSING delay expires.
                self._on_no_pick_target(agent)
                agent.state = ModuleState.PROCESSING
                agent.process_ready_time = (self.sim.sim_time
                                            + self.TOKEN_PROCESS_DELAY)
                any_active = True
                continue

        # Also check if any tokens or flood/echo messages are still in flight
        for agent in self.agents.values():
            if (agent.state != ModuleState.IDLE or
                    agent.incoming_tokens or
                    agent.token is not None or
                    agent.discover_inbox or
                    agent.echo_inbox):
                any_active = True
                break

        if not any_active and self._pending_floods:
            any_active = True

        self._decision_bond_matrix = None
        return any_active

    def is_connected(self) -> bool:
        """Check if all active (non-faulty) modules form a single connected component.

        Excludes the fault body — it's physically present but doesn't
        count for algorithmic connectivity.
        """
        active_indices = set(a.body_idx for a in self.agents.values())
        if not active_indices:
            return True
        bond_matrix = self.sim.get_bond_matrix()
        start = next(iter(active_indices))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for j in active_indices:
                if j not in visited and bond_matrix[node, j]:
                    visited.add(j)
                    queue.append(j)
        return visited == active_indices

    def _start_pivot(
        self,
        agent: ModuleAgent,
        target_pos_local: np.ndarray,
        axis_idx: int,
        pivot_type: str,
        handoff_idx: Optional[int],
    ):
        """Start a physical pivot for an agent.

        *pivot_type* and *handoff_idx* must match the maneuver chosen in
        ``pick_target`` (``handoff_idx`` is ``None`` for corner pivots).
        """
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        lattice_ref_body_idx = axis_idx

        R_ax = self.sim.body_rotation_matrix(axis_idx)
        agent.pre_pivot_pos_local = (R_ax.T @ (my_pos - axis_pos)).copy()
        agent.pre_pivot_lattice_ref = axis_idx

        target_pos_local = np.asarray(target_pos_local, dtype=float).reshape(3)
        target_world = self.sim.target_world_from_local(
            lattice_ref_body_idx, target_pos_local)

        if pivot_type == "lateral" and handoff_idx is not None:
            attract_body = handoff_idx
            attract_conn = self.sim.lateral_handoff_attract_connector(
                axis_idx, agent.body_idx, handoff_idx)
        else:
            attract_body = axis_idx
            attract_conn = self.sim.nearest_connector(
                axis_idx, target_world - pos[axis_idx])

        # Remove bonds to non-axis neighbors regardless of pivot type
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n in neighbors:
            if n != axis_idx:
                self.sim.remove_bond(agent.body_idx, n)

        if pivot_type == "corner":
            # Spring mode keeps COM–COM bond to axis; legacy uses point-to-point only.
            if not self.sim.USE_SPRING_BONDS:
                self.sim.remove_bond(agent.body_idx, axis_idx)
        # Lateral: keep axis bond for now; handoff will swap it mid-pivot

        # Compute rotation parameters
        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)

        r_target = target_world - axis_pos
        cos_angle = np.clip(
            np.dot(r_vec, r_target) / (np.linalg.norm(r_vec) *
                                        np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_angle)

        # Planar demos: if the short arc would sweep through another
        # module but the complementary arc is clear, roll the long way
        # around the free side instead.
        if PLANAR_Z_LOCK and pivot_type == "corner":
            if (not _planar_arc_clear(
                    pos, agent.body_idx, axis_idx, target_world, len(pos))
                    and _planar_arc_clear(
                        pos, agent.body_idx, axis_idx, target_world,
                        len(pos), long_way=True)):
                rot_axis = -rot_axis
                angle = 2.0 * np.pi - angle

        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle, kp, kd,
            duration=12.0,
            lattice_ref_body_idx=lattice_ref_body_idx,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body,
            attract_connector=attract_conn,
            pivot_type=pivot_type)

        agent.state = ModuleState.PIVOTING
        agent.target_pos = target_world.copy()
        agent.target_pos_local = target_pos_local.copy()
        agent.lattice_ref_body_idx = lattice_ref_body_idx
        agent.attract_body_idx = attract_body
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = pivot_type
        agent.handoff_idx = handoff_idx
        agent.handoff_done = False
        self.total_moves += 1

        self.move_log.append({
            "module": agent.module_id,
            "from": my_pos.tolist(),
            "to": target_world.tolist(),
            "axis": self._idx_to_mid.get(axis_idx, f"idx_{axis_idx}"),
            "sim_time": self.sim.sim_time,
            "pivot_type": pivot_type,
        })

        logger.info("Module {} starting {} pivot to {} (axis={}{})",
                     agent.module_id, pivot_type, target_world,
                     self._idx_to_mid.get(axis_idx, axis_idx),
                     f", handoff={self._idx_to_mid.get(handoff_idx, handoff_idx)}"
                     if handoff_idx is not None else "")

    def _retarget_to_nearest_empty_cell(self, agent: ModuleAgent) -> bool:
        """Retarget the active pivot to the closest empty cardinal-adjacent cell.

        Called on pivot failure (timeout/collision) INSTEAD of stop_pivot +
        reversal. The rolling-sphere coupling stays alive — only the target
        and rotation parameters change. ``sim.start_pivot`` is idempotent: it
        detects that this module already has an active pivot and updates the
        PivotState fields in place. The module never goes bondless.

        Returns True if a valid empty cell was found and the pivot retargeted;
        False if no such cell exists (every cardinal slot around the current
        axis is occupied), in which case the caller should give up gracefully.

        Excludes cells occupied by ANY body (including fault bodies, which sit
        at their original lattice positions). Faults are obstacles, not
        targets.
        """
        if agent.body_idx not in self.sim._active_pivots:
            return False
        ps = self.sim._active_pivots[agent.body_idx]
        axis_idx = ps.axis_idx
        if not (0 <= axis_idx < self.sim.N):
            return False

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        nom = float(self.sim.NOMINAL_DIST)

        # Hard cap: bail out if too many retargets in this maneuver chain.
        if agent.retarget_count >= self.MAX_RETARGETS_PER_PIVOT:
            return False

        # Cells (in axis-local lattice frame) occupied by any body. This
        # naturally excludes cells already on a fault body because the
        # fault body still sits at its lattice position and gets enumerated
        # here. Corner pivots adjacent to faults are allowed — only the
        # cell occupied BY a fault is rejected.
        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - axis_pos) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        best_target_local = None
        best_cell: Optional[Tuple[int, int, int]] = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            cell = (int(delta[0]), int(delta[1]), int(delta[2]))
            if cell in occupied:
                continue
            # Skip cells already tried by retarget in this chain.
            if cell in agent.retarget_blacklist:
                continue
            dw = nom * np.array(delta, dtype=float)
            target_world = axis_pos + R @ dw
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - axis_pos)
                best_cell = cell

        if best_target_local is None:
            return False

        # Record this cell as tried + bump retarget counter.
        if best_cell is not None:
            agent.retarget_blacklist.add(best_cell)
        agent.retarget_count += 1

        target_world = self.sim.target_world_from_local(
            axis_idx, best_target_local)
        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
        r_target = target_world - axis_pos
        cos_a = np.clip(
            np.dot(r_vec, r_target) / (
                np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_a)
        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)
        attract_conn = self.sim.nearest_connector(
            axis_idx, target_world - axis_pos)

        # sim.start_pivot's transition path updates PivotState in place
        # because pivot_idx is already in _active_pivots — no constraint or
        # bond manipulation, no gap in the rolling coupling.
        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle, kp, kd,
            duration=12.0,
            lattice_ref_body_idx=axis_idx,
            target_pos_local=best_target_local,
            attract_body_idx=axis_idx,
            attract_connector=attract_conn,
            pivot_type="corner")

        agent.target_pos = target_world.copy()
        agent.target_pos_local = np.asarray(best_target_local).copy()
        agent.lattice_ref_body_idx = axis_idx
        agent.attract_body_idx = axis_idx
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = "corner"
        agent.handoff_idx = None
        agent.handoff_done = False
        return True

    def _start_reversal(self, agent: ModuleAgent):
        """Pivot the module back to its pre-pivot lattice position.

        For lateral pivots that already completed a handoff, the reversal
        is itself a lateral pivot in reverse: roll back on the current
        axis (handoff module) toward the original axis, bond-switch, then
        roll on the original axis back to pre_pivot_pos_local.
        """
        self.reversal_count = getattr(self, "reversal_count", 0) + 1
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]

        neighbors = self.get_physical_neighbors(agent.body_idx)
        if not neighbors:
            # Find the nearest body in the sim. Only bond if within
            # BOND_THRESHOLD (essentially at lattice spacing). Otherwise
            # refuse to create a wildly-stretched rest-pose bond — leave
            # the module orphaned and abort the reversal. Articulation
            # checks elsewhere will see the truth.
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            nearest_dist = float(np.linalg.norm(
                pos[nearest] - pos[agent.body_idx]))
            if nearest_dist < self.BOND_THRESHOLD:
                self.sim.create_bond(agent.body_idx, nearest)
                neighbors = [nearest]
            else:
                agent.state = ModuleState.IDLE
                agent.token = None
                agent.pending_retry = None
                return

        ref = agent.pre_pivot_lattice_ref
        target_local = agent.pre_pivot_pos_local
        if target_local is None:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None
            return

        original_axis = ref if ref is not None else None
        is_lateral_reversal = (
            agent.pivot_type == "lateral"
            and agent.handoff_done
            and original_axis is not None
            and original_axis != agent.pivot_axis_idx
        )

        best_ax = min(neighbors,
                      key=lambda n: np.linalg.norm(pos[n] - my_pos))
        if ref is None:
            ref = best_ax
        target_world = self.sim.target_world_from_local(ref, target_local)
        axis_pos = pos[best_ax]

        for n in neighbors:
            if n != best_ax:
                self.sim.remove_bond(agent.body_idx, n)

        if is_lateral_reversal:
            attract_conn = self.sim.lateral_handoff_attract_connector(
                best_ax, agent.body_idx, original_axis)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=original_axis,
                attract_connector=attract_conn,
                pivot_type="lateral",
                repel_idx=-1)

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = original_axis
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "lateral"
            agent.handoff_idx = original_axis
            agent.handoff_done = False
            agent.reversal_handoff_idx = original_axis
        else:
            attract_conn = self.sim.nearest_connector(
                best_ax, target_world - axis_pos)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=best_ax,
                attract_connector=attract_conn,
                pivot_type="corner")

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = best_ax
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "corner"
            agent.handoff_idx = None
            agent.handoff_done = False
            agent.reversal_handoff_idx = None

    def _emergency_snap_to_lattice(self, agent: ModuleAgent):
        """Ensure module is bonded and pivot to the nearest empty lattice site.

        Called when a reversal pivot fails (timeout/collision).  Guarantees
        the module keeps at least one bond and attempts a corner pivot to
        the closest unoccupied lattice cell so it ends up in a valid
        position.
        """
        neighbors = self.get_physical_neighbors(agent.body_idx)
        pos = self.sim.get_positions()

        if not neighbors:
            # Only bond if the nearest body is within BOND_THRESHOLD. If
            # nothing is close enough, refuse to create a stretched rest-
            # pose bond and abort the snap; the module stays orphaned.
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            nearest_dist = float(np.linalg.norm(
                pos[nearest] - pos[agent.body_idx]))
            if nearest_dist < self.BOND_THRESHOLD:
                self.sim.create_bond(agent.body_idx, nearest)
                neighbors = [nearest]
            else:
                agent.state = ModuleState.IDLE
                agent.token = None
                agent.pending_retry = None
                return

        axis_idx = min(neighbors,
                       key=lambda n: np.linalg.norm(
                           pos[n] - pos[agent.body_idx]))

        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)
        my_pos = pos[agent.body_idx]

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is not None:
            self._start_pivot(agent, best_target_local, axis_idx, "corner",
                              None)
            logger.info("Module {} emergency snap pivot around {}",
                        agent.module_id,
                        self._idx_to_mid.get(axis_idx, axis_idx))
        else:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None

    def _start_corrective_pivot(self, agent: ModuleAgent, axis_idx: int):
        """Corner pivot to nearest empty lattice site on *axis_idx* to fix a triangular bond."""
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        arm = my_pos - p_ref
        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            if not _lattice_delta_perpendicular_to_arm(delta, arm, R):
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is None:
            return

        self._start_pivot(agent, best_target_local, axis_idx, "corner", None)
        logger.info("Module {} corrective pivot (triangle fix) around {}",
                    agent.module_id, self._idx_to_mid.get(axis_idx, axis_idx))

    def _forward_token(self, agent: ModuleAgent):
        """Forward agent's token to all bonded neighbors."""
        if agent.token is None:
            return
        pos = self.sim.get_positions()
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid is None or n_mid not in self.agents:
                continue
            n_agent = self.agents[n_mid]
            if n_agent.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
                continue
            my_pos = pos[agent.body_idx]
            nbr_pos = pos[n_idx]
            R_me = self.sim.body_rotation_matrix(agent.body_idx)
            R_nbr = self.sim.body_rotation_matrix(n_idx)
            world_dir = R_me @ agent.token.direction + (my_pos - nbr_pos)
            propagated_dir = R_nbr.T @ world_dir
            n_agent.incoming_tokens.append(
                Token(direction=propagated_dir, source_id=agent.module_id))

    def _reconnect_bonds(self, body_idx: int):
        """Reconnect bonds dictated by the policy after a pivot.

        Only creates bonds to the pivot axis module and to modules that
        occupy lattice-adjacent positions of the target, provided they
        are within the distance threshold.  No blind proximity scan.
        """
        agent = None
        for a in self.agents.values():
            if a.body_idx == body_idx:
                agent = a
                break
        if agent is None:
            return

        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        if (agent.lattice_ref_body_idx is not None
                and agent.target_pos_local is not None):
            target = self.sim.target_world_from_local(
                agent.lattice_ref_body_idx, agent.target_pos_local)
        else:
            target = agent.target_pos if agent.target_pos is not None else my_pos

        # Determine which body indices the policy says we should bond to:
        # 1) the pivot axis module
        # 2) any module at a ±1 lattice step from goal in *lattice ref* frame
        expected_neighbors: Set[int] = set()
        if agent.pivot_axis_idx is not None:
            expected_neighbors.add(agent.pivot_axis_idx)

        if agent.lattice_ref_body_idx is not None:
            ref = agent.lattice_ref_body_idx
            R = self.sim.body_rotation_matrix(ref)
            p_ref = pos[ref]
            nom = float(self.sim.NOMINAL_DIST)
            goal_cell = np.round(
                agent.target_pos_local / nom).astype(int)  # type: ignore[union-attr]
            for delta in _LATTICE_DELTAS:
                nbr_cell = (int(goal_cell[0] + delta[0]),
                            int(goal_cell[1] + delta[1]),
                            int(goal_cell[2] + delta[2]))
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    j_cell = tuple(np.round(
                        R.T @ (pos[j] - p_ref) / nom).astype(int))
                    if j_cell == nbr_cell:
                        expected_neighbors.add(j)
        else:
            target_key = tuple(np.round(target).astype(int))
            for delta in _LATTICE_DELTAS:
                nbr_key = (target_key[0] + delta[0],
                           target_key[1] + delta[1],
                           target_key[2] + delta[2])
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    if tuple(np.round(pos[j]).astype(int)) == nbr_key:
                        expected_neighbors.add(j)

        # Create bonds only for policy-approved neighbors within threshold
        for j in expected_neighbors:
            dist = np.linalg.norm(my_pos - pos[j])
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

        # Proximity: fixed joints to every other module within bond distance
        # (e.g. after rolling), excluding passive fault bodies.
        fault_idxs = self._get_fault_idxs()
        for j in range(self.sim.N):
            if j == body_idx:
                continue
            if (fault_idxs and j in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            dist = float(np.linalg.norm(my_pos - pos[j]))
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

    def _has_pivoting_neighbor_of_neighbor(self, body_idx: int) -> bool:
        """Check if any neighbor's neighbor is currently pivoting (2-hop exclusion)."""
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        neighbors = self.get_physical_neighbors(body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid and n_mid in self.agents:
                if self.agents[n_mid].state in _moving:
                    return True
            nn_indices = self.get_physical_neighbors(n_idx)
            for nn_idx in nn_indices:
                nn_mid = self._idx_to_mid.get(nn_idx)
                if nn_mid and nn_mid in self.agents:
                    if self.agents[nn_mid].state in _moving:
                        return True
        return False


class DecentralizedRestructuring:
    """
    Async decentralized restructuring (phase 2).

    Same architecture as coagulation but tokens point to empty slots.
    Only modules that moved during coagulation are candidates to move.
    """

    TOKEN_PROCESS_DELAY = 1.0
    BOND_THRESHOLD = 1.05

    # See DecentralizedCoagulation.PIVOT_EXCLUSION_RADIUS for semantics.
    PIVOT_EXCLUSION_RADIUS: Optional[int] = None

    # See DecentralizedCoagulation.ALLOW_FAULT_AS_PIVOT_NEIGHBOR.
    ALLOW_FAULT_AS_PIVOT_NEIGHBOR: bool = True

    # See DecentralizedCoagulation.INITIAL_ACTION_POINTS — same semantics
    # in phase 2. Displacement-based restructuring usually retraces within
    # this budget; token-based restructuring respects the same cap.
    INITIAL_ACTION_POINTS: int = 5
    MAX_MOVES_PER_MODULE: int = 5  # back-compat alias
    FORWARD_POINTS_MULTIPLIER: int = 10
    MAX_RETARGETS_PER_PIVOT: int = 10
    USE_POSITION_HISTORY: bool = True

    FORWARD_AP_COST: float = 0.1

    @staticmethod
    def _fault_body_index(sim, agents: Dict[str, ModuleAgent]) -> Optional[int]:
        have = {a.body_idx for a in agents.values()}
        rest = [i for i in range(sim.N) if i not in have]
        return rest[0] if len(rest) == 1 else None

    @staticmethod
    def _fault_body_indices(sim, agents: Dict[str, ModuleAgent]) -> Set[int]:
        have = {a.body_idx for a in agents.values()}
        return {i for i in range(sim.N) if i not in have}

    def _get_fault_idxs(self) -> Set[int]:
        """Return the set of all fault body indices."""
        idxs = getattr(self, "_fault_body_idxs", None)
        if idxs is not None:
            return idxs
        single = getattr(self, "_fault_body_idx", None)
        if single is not None:
            return {single}
        return self._fault_body_indices(self.sim, self.agents)

    def __init__(self, sim, module_ids: List[str],
                 body_indices: Dict[str, int],
                 coag_moved: Set[str],
                 pre_damage_neighbor_slots: Dict[str, List[np.ndarray]],
                 token_strategy: str = "furthest"):
        """
        Args:
            sim: BulletSimulator
            module_ids: active module IDs
            body_indices: {mid: body_idx}
            coag_moved: set of module IDs that moved during coagulation
            pre_damage_neighbor_slots: {mid: [relative_direction_vectors]}
                for modules that did NOT move (their slot knowledge is accurate)
            token_strategy: "furthest", "nearest", or "random"
        """
        self.sim = sim
        self.coag_moved = coag_moved
        self.pre_damage_neighbor_slots = pre_damage_neighbor_slots
        self.token_strategy = token_strategy

        self.agents: Dict[str, ModuleAgent] = {}
        for mid in module_ids:
            self.agents[mid] = ModuleAgent(
                module_id=mid,
                body_idx=body_indices[mid],
                action_points=self.INITIAL_ACTION_POINTS,
                forward_points=(self.INITIAL_ACTION_POINTS
                                * self.FORWARD_POINTS_MULTIPLIER),
            )

        self._idx_to_mid: Dict[int, str] = {v: k for k, v in body_indices.items()}
        self._fault_body_idx: Optional[int] = self._fault_body_index(
            sim, self.agents)
        self._fault_body_idxs: Set[int] = self._fault_body_indices(
            sim, self.agents)
        self.total_moves = 0
        self.successful_moves = 0
        self.move_log: List[Dict] = []
        self._tick_count: int = 0
        self.reversal_count: int = 0

    def generate_initial_tokens(self):
        """Non-movers emit tokens for their empty neighbor slots."""
        pos = self.sim.get_positions()
        for mid, slots in self.pre_damage_neighbor_slots.items():
            if mid in self.coag_moved:
                continue  # only non-movers generate
            if mid not in self.agents:
                continue
            agent = self.agents[mid]
            R = self.sim.body_rotation_matrix(agent.body_idx)
            for slot_dir in slots:
                slot_pos = pos[agent.body_idx] + R @ slot_dir
                occupied = False
                for j in range(self.sim.N):
                    if np.linalg.norm(pos[j] - slot_pos) < 0.5:
                        occupied = True
                        break
                if not occupied:
                    agent.incoming_tokens.append(
                        Token(direction=slot_dir.copy(), source_id=mid))

    # The rest follows the same pattern as DecentralizedCoagulation
    # but with different candidate filtering and token selection

    def get_physical_neighbors(self, body_idx: int) -> List[int]:
        neighbors = []
        bond_matrix = self.sim.get_bond_matrix()
        for j in range(self.sim.N):
            if j != body_idx and bond_matrix[body_idx, j]:
                neighbors.append(j)
        return neighbors

    def _decision_graph_neighbors(self, body_idx: int) -> List[int]:
        bm = getattr(self, "_decision_bond_matrix", None)
        if bm is None:
            bm = self.sim.get_bond_matrix()
        return bond_graph_neighbors(bm, self.sim.N, body_idx)

    def _bfs_reachable(self, start_idx: int, max_depth: int,
                       excluded: Set[int]) -> Set[int]:
        visited = {start_idx}
        frontier = {start_idx}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for nbr in self._decision_graph_neighbors(node):
                    if nbr not in excluded and nbr not in visited:
                        next_frontier.add(nbr)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(start_idx)
        return visited

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        """Two-graph movability test (see DecentralizedCoagulation.is_movable)."""
        fault_idxs = self._get_fault_idxs()

        all_neighbors = self._decision_graph_neighbors(body_idx)
        if len(all_neighbors) == 0:
            return False
        if len(all_neighbors) == 1:
            return True

        active_neighbors = [n for n in all_neighbors if n not in fault_idxs]
        if len(active_neighbors) == 0:
            return False

        for v in active_neighbors:
            excluded = {body_idx, v} | fault_idxs
            reachable_without_v = set()
            for w in active_neighbors:
                if w == v:
                    continue
                reachable_without_v |= self._bfs_reachable(
                    w, safety_radius - 1, excluded)
            v_neighbors = set(self._decision_graph_neighbors(v)) - fault_idxs
            if not (reachable_without_v & v_neighbors):
                return False
        return True

    def get_nearby_lattice_positions(self, body_idx: int) -> List[np.ndarray]:
        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        neighbors = self.get_physical_neighbors(body_idx)
        occupied = set()
        for i in range(self.sim.N):
            occupied.add(tuple(np.round(pos[i]).astype(int)))

        candidate_set = set()
        check_indices = set(neighbors)
        for n_idx in neighbors:
            for nn_idx in self.get_physical_neighbors(n_idx):
                check_indices.add(nn_idx)

        for idx in check_indices:
            nbr_pos = np.round(pos[idx]).astype(int)
            for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                candidate = tuple(nbr_pos + np.array(delta))
                if candidate not in occupied:
                    candidate_set.add(candidate)

        my_lattice = np.round(my_pos).astype(int)
        for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            candidate = tuple(my_lattice + np.array(delta))
            if candidate not in occupied:
                candidate_set.add(candidate)

        return [np.array(c, dtype=float) for c in candidate_set]

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        return _token_origin_world_pos(
            agent.token, pos, self.agents,
            fault_id=None, fault_body_idx=None)

    def _on_no_pick_target(self, agent: ModuleAgent) -> None:
        """See ``DecentralizedCoagulation._on_no_pick_target``."""

    def _pick_target_occupancy_skip_body_indices(self) -> Set[int]:
        """See ``DecentralizedCoagulation._pick_target_occupancy_skip_body_indices``."""
        return set()

    def _ghost_goal_world_pos_for_axial_corner(
            self, pos: np.ndarray) -> Optional[np.ndarray]:
        """See ``DecentralizedCoagulation._ghost_goal_world_pos_for_axial_corner``."""
        return None

    def pick_target(
            self,
            agent: ModuleAgent,
            *,
            require_closer_to_origin: bool = False,
    ) -> Optional[PivotPick]:
        """Same corner / lateral enumeration and scoring as
        ``DecentralizedCoagulation.pick_target`` (see that docstring)."""
        if agent.token is None:
            return None
        if agent.module_id not in self.coag_moved:
            return None

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        token_dir = agent.token.direction
        if np.linalg.norm(token_dir) < 1e-8:
            return None

        origin = self._origin_world_for_pick_target(agent, pos)
        if origin is None:
            return None

        neighbors = self._decision_graph_neighbors(agent.body_idx)
        if not neighbors:
            return None

        fault_idxs = self._get_fault_idxs()
        scored: List[Tuple[float, np.ndarray, int, str, Optional[int]]] = []
        nom = float(self.sim.NOMINAL_DIST)
        module_shape = getattr(self.sim, "_module_shape", "sphere")

        for axis_idx in neighbors:
            if (fault_idxs and axis_idx in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            R = self.sim.body_rotation_matrix(axis_idx)
            p_ref = pos[axis_idx]
            occupied = set()
            skip_occ = self._pick_target_occupancy_skip_body_indices()
            for i in range(self.sim.N):
                if i in skip_occ:
                    continue
                u = R.T @ (pos[i] - p_ref) / nom
                occupied.add(tuple(np.round(u).astype(int)))
            my_cell = tuple(np.round(R.T @ (my_pos - p_ref) / nom).astype(int))

            arm_axis_to_pivot = my_pos - pos[axis_idx]

            gpos_ax = self._ghost_goal_world_pos_for_axial_corner(pos)
            ghost_goal_cell: Optional[Tuple[int, int, int]] = None
            if gpos_ax is not None:
                ghost_goal_cell = tuple(
                    np.round(R.T @ (gpos_ax - p_ref) / nom).astype(int))

            corner_enabled = (module_shape != "cube")

            for delta in _LATTICE_DELTAS:
                if not corner_enabled:
                    break
                perp = _lattice_delta_perpendicular_to_arm(
                    delta, arm_axis_to_pivot, R)
                dw = nom * np.array(delta, dtype=float)
                target_world = p_ref + R @ dw
                if PLANAR_Z_LOCK:
                    if abs(target_world[2]) > 1e-6:
                        continue
                    if not _planar_pivot_admissible(
                            pos, agent.body_idx, axis_idx, target_world,
                            len(pos)):
                        continue
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                axial_into_ghost = (
                    ghost_goal_cell is not None
                    and target_cell == ghost_goal_cell)
                if not perp and not axial_into_ghost:
                    continue
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (self.USE_POSITION_HISTORY
                        and (axis_idx, target_cell) in agent.position_history):
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_tar_n = np.linalg.norm(r_target)
                if r_tar_n < 1e-12:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * r_tar_n + 1e-12)
                if axial_into_ghost:
                    if gpos_ax is None:
                        continue
                    d_tgt = float(np.linalg.norm(target_world - gpos_ax))
                    d_my = float(np.linalg.norm(my_pos - gpos_ax))
                    if d_tgt >= d_my - 1e-9:
                        continue
                else:
                    if abs(cos_angle) > _CORNER_PIVOT_MAX_ABS_COS:
                        continue
                score = -float(np.linalg.norm(target_world - origin))
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "corner", None))

            # Lateral pivots: reject only if HANDOFF is a fault (the
            # second-leg axis). The initial axis can be a fault — modules
            # may lateral-pivot off a fault.
            axis_neighbors = self._decision_graph_neighbors(axis_idx)
            for handoff_idx in axis_neighbors:
                if handoff_idx == agent.body_idx:
                    continue
                if (fault_idxs and handoff_idx in fault_idxs):
                    continue
                d_ah = pos[handoff_idx] - pos[axis_idx]
                if not _lateral_axis_handoff_step_cardinal_for_axis(
                        self.sim, axis_idx, d_ah):
                    continue
                target_world = pos[handoff_idx] + (my_pos - pos[axis_idx])
                if PLANAR_Z_LOCK:
                    if abs(target_world[2]) > 1e-6:
                        continue
                    if not _planar_lateral_clear(
                            pos, agent.body_idx, axis_idx, handoff_idx,
                            target_world, len(pos)):
                        continue
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (self.USE_POSITION_HISTORY
                        and (axis_idx, target_cell) in agent.position_history):
                    continue
                if np.linalg.norm(target_world - pos[axis_idx]) < 1.05:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_vec_n = np.linalg.norm(r_vec)
                r_tar_n = np.linalg.norm(r_target)
                if r_vec_n < 1e-8 or r_tar_n < 1e-8:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (r_vec_n * r_tar_n)
                if abs(cos_angle) > _LATERAL_PIVOT_MAX_ABS_COS:
                    continue
                if module_shape == "cube":
                    if not _cube_lateral_swept_clear(
                            occupied, my_cell, target_cell,
                            arm_axis_to_pivot, R, nom):
                        continue
                score = -float(np.linalg.norm(target_world - origin))
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "lateral", handoff_idx))

        if not scored:
            return None

        if require_closer_to_origin:
            current_d = float(np.linalg.norm(my_pos - origin))
            eps = 1e-9 * max(1.0, nom)
            scored = [s for s in scored if (-s[0]) < current_d - eps]
            if not scored:
                return None

        scored.sort(key=lambda x: (-x[0], x[2], x[4] if x[4] is not None else -1))
        best = scored[0]
        return best[1].copy(), best[2], best[3], best[4]

    def _propagate_moving_tokens(self):
        """Flood-fill moving tokens from PIVOTING/REVERSING agents."""
        if self.PIVOT_EXCLUSION_RADIUS is None:
            return
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        bm = self._decision_bond_matrix.copy()
        for a in self.agents.values():
            if a.state in _moving and a.pivot_axis_idx is not None:
                bm[a.body_idx, a.pivot_axis_idx] = True
                bm[a.pivot_axis_idx, a.body_idx] = True

        seeds: set = set()
        for a in self.agents.values():
            if a.state in _moving:
                a.moving_token_received_tick = self._tick_count
                seeds.add(a.body_idx)

        frontier = set(seeds)
        for _hop in range(self.PIVOT_EXCLUSION_RADIUS):
            nxt: set = set()
            for i in frontier:
                for j in range(self.sim.N):
                    if bm[i, j] and j not in seeds:
                        mid_j = self._idx_to_mid.get(j)
                        if mid_j and mid_j in self.agents:
                            self.agents[mid_j].moving_token_received_tick = self._tick_count
                        seeds.add(j)
                        nxt.add(j)
            frontier = nxt
            if not frontier:
                break

    def tick(self) -> bool:
        self._tick_count += 1
        self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

        self._propagate_moving_tokens()

        any_active = False
        agent_items = list(self.agents.items())
        random.shuffle(agent_items)

        for mid, agent in agent_items:
            if agent.is_faulty:
                continue

            if agent.state == ModuleState.IDLE:
                # Skip entirely only when BOTH budgets are spent — a
                # motion-exhausted agent can still relay tokens, so it must
                # fall through to the token-consumption logic below.
                if agent.action_points <= 0 and agent.forward_points <= 0:
                    continue
                neighbors = self._decision_graph_neighbors(agent.body_idx)
                bm = self._decision_bond_matrix
                pos = self.sim.get_positions()
                triangle_found = False
                # Triangle cleanup is a corrective pivot (motion), so it is
                # gated on motion budget AND is_movable: the cleanup
                # unconditionally removes a bond and corner-pivots the agent,
                # which can sever the agent's other neighbors when the agent
                # is an articulator. Skip cleanup for non-movable / motion-
                # exhausted agents.
                if agent.action_points > 0 and self.is_movable(
                        agent.body_idx,
                        getattr(self, "_safety_radius", 2)):
                    for ni_idx in range(len(neighbors)):
                        for nj_idx in range(ni_idx + 1, len(neighbors)):
                            a, b = neighbors[ni_idx], neighbors[nj_idx]
                            if bm[a, b]:
                                d_a = float(np.linalg.norm(pos[agent.body_idx] - pos[a]))
                                d_b = float(np.linalg.norm(pos[agent.body_idx] - pos[b]))
                                if d_a <= d_b:
                                    close, far = a, b
                                else:
                                    close, far = b, a
                                self.sim.remove_bond(agent.body_idx, far)
                                self._decision_bond_matrix = self.sim.get_bond_matrix().copy()
                                self._start_corrective_pivot(agent, close)
                                self._propagate_moving_tokens()
                                triangle_found = True
                                any_active = True
                                break
                        if triangle_found:
                            break
                if triangle_found:
                    continue

            if agent.state == ModuleState.PIVOTING:
                # Lateral handoff check
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        # Lateral handoff: agent has already swapped the
                        # axis bond to the handoff partner above; tell the
                        # sim NOT to restore a rigid M↔original-axis bond
                        # (would create a stretched duplicate).
                        self.sim.stop_pivot(agent.body_idx,
                                            restore_axis_bond=False)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} lateral handoff (phase 2): "
                            "bonded to {}, pivot switched",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    collided = self.sim.pivot_collided(agent.body_idx)
                    timed_out = self.sim.pivot_timed_out(agent.body_idx)

                    if collided or timed_out:
                        # Failure: retarget without releasing the rolling
                        # tether. See coag copy for rationale.
                        if self._retarget_to_nearest_empty_cell(agent):
                            agent.action_points -= 1
                            agent.ap_spent_pivots += 1.0
                            self.reversal_count = (
                                getattr(self, "reversal_count", 0) + 1)
                            any_active = True
                            logger.info(
                                "Module {} pivot {} (phase 2) — retargeting",
                                mid,
                                "collided" if collided else "timed out")
                        else:
                            agent.action_points -= 1
                            agent.ap_spent_pivots += 1.0
                            self.sim.stop_pivot(agent.body_idx)
                            self._reconnect_bonds(agent.body_idx)
                            self._decision_bond_matrix = (
                                self.sim.get_bond_matrix().copy())
                            agent.state = ModuleState.IDLE
                            agent.target_pos = None
                            agent.target_pos_local = None
                            agent.lattice_ref_body_idx = None
                            agent.attract_body_idx = None
                            agent.attract_connector = None
                            agent.pivot_axis_idx = None
                            agent.pivot_type = None
                            agent.handoff_idx = None
                            agent.handoff_done = False
                            agent.token = None
                            agent.retarget_blacklist.clear()
                            agent.retarget_count = 0
                            logger.info(
                                "Module {} pivot failed (phase 2 — no empty "
                                "cell or retarget cap hit) — giving up", mid)
                    else:
                        self.sim.stop_pivot(agent.body_idx)
                        if (agent.lattice_ref_body_idx is not None
                                and agent.target_pos_local is not None):
                            nom = float(self.sim.NOMINAL_DIST)
                            cell = tuple(
                                np.round(
                                    agent.target_pos_local / nom).astype(int))
                            agent.position_history.add(
                                (agent.lattice_ref_body_idx, cell))
                        self._reconnect_bonds(agent.body_idx)
                        self._decision_bond_matrix = (
                            self.sim.get_bond_matrix().copy())
                        agent.state = ModuleState.IDLE
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.token = None
                        agent.retarget_blacklist.clear()
                        agent.retarget_count = 0
                        agent.completed_moves += 1
                        agent.action_points -= 1
                        agent.ap_spent_pivots += 1.0
                        self.successful_moves += 1
                        logger.info(
                            "Module {} pivot complete (phase 2, ap left: {})",
                            mid, agent.action_points)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.REVERSING:
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        # Lateral handoff: agent has already swapped the
                        # axis bond to the handoff partner above; tell the
                        # sim NOT to restore a rigid M↔original-axis bond
                        # (would create a stretched duplicate).
                        self.sim.stop_pivot(agent.body_idx,
                                            restore_axis_bond=False)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} reversal handoff (phase 2): bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    rev_collided = self.sim.pivot_collided(agent.body_idx)
                    rev_timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    self.sim.stop_pivot(agent.body_idx)
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if rev_collided or rev_timed_out:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None
                        self._emergency_snap_to_lattice(agent)
                        any_active = True
                        logger.info(
                            "Module {} reversal failed (phase 2) — "
                            "snapping to lattice", mid)
                    else:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None

                        cooldown = random.uniform(0, 10)
                        agent.wait_until = self.sim.sim_time + cooldown
                        agent.state = ModuleState.WAITING
                        any_active = True
                        logger.info(
                            "Module {} reversal complete (phase 2) "
                            "— waiting {:.1f}s", mid, cooldown)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.WAITING:
                if self.sim.sim_time >= agent.wait_until:
                    agent.state = ModuleState.IDLE
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.PROCESSING:
                if self.sim.sim_time >= agent.process_ready_time:
                    if agent.forward_points > 0:
                        self._forward_token(agent)
                        agent.forward_points -= 1
                        agent.ap_spent_forwards += 1.0
                    agent.state = ModuleState.IDLE
                    agent.token = None
                else:
                    any_active = True
                continue

            if agent.incoming_tokens and (
                    agent.action_points > 0 or agent.forward_points > 0):
                if self.token_strategy == "nearest":
                    best = min(agent.incoming_tokens,
                               key=lambda t: np.linalg.norm(t.direction))
                elif self.token_strategy == "random":
                    best = random.choice(agent.incoming_tokens)
                else:
                    best = max(agent.incoming_tokens,
                               key=lambda t: np.linalg.norm(t.direction))
                agent.token = best
                agent.incoming_tokens.clear()
                agent.token_hold_until = self.sim.sim_time + 1.0
                agent.state = ModuleState.HAS_TOKEN
            elif agent.incoming_tokens:
                # Both budgets exhausted — drop incoming tokens.
                agent.incoming_tokens.clear()

            if agent.state == ModuleState.HAS_TOKEN:
                if self.sim.sim_time < agent.token_hold_until:
                    any_active = True
                    continue
                _d = getattr(self, "_p2diag", None)
                if _d is not None:
                    _d["engaged_ids"].add(agent.module_id)
                if (not self.is_movable(agent.body_idx)
                        or agent.action_points <= 0):
                    if _d is not None:
                        if not self.is_movable(agent.body_idx):
                            _d["blocked_not_movable_ids"].add(agent.module_id)
                        else:
                            _d["blocked_no_budget_ids"].add(agent.module_id)
                    agent.state = ModuleState.PROCESSING
                    agent.process_ready_time = (self.sim.sim_time
                                                + self.TOKEN_PROCESS_DELAY)
                    any_active = True
                    continue

                # Pivot concurrency check.
                _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
                if self.PIVOT_EXCLUSION_RADIUS is None:
                    if any(a.state in _moving
                           for a in self.agents.values()):
                        any_active = True
                        continue
                else:
                    if (self._tick_count - agent.moving_token_received_tick
                            < self.PIVOT_EXCLUSION_RADIUS):
                        any_active = True
                        continue

                result = self.pick_target(agent, require_closer_to_origin=True)
                if result is not None:
                    target_pos_local, axis_idx, pivot_type, handoff_idx = result
                    self._start_pivot(
                        agent, target_pos_local, axis_idx,
                        pivot_type, handoff_idx)
                    self._propagate_moving_tokens()
                    if _d is not None:
                        _d["moved_ids"].add(agent.module_id)
                    any_active = True
                    continue

                if _d is not None:
                    _d["no_target_ids"].add(agent.module_id)
                self._on_no_pick_target(agent)
                agent.state = ModuleState.PROCESSING
                agent.process_ready_time = (self.sim.sim_time
                                            + self.TOKEN_PROCESS_DELAY)
                any_active = True

        for agent in self.agents.values():
            if (agent.state != ModuleState.IDLE or
                    agent.incoming_tokens or agent.token is not None):
                any_active = True
                break

        self._decision_bond_matrix = None
        return any_active

    def _start_pivot(
        self,
        agent: ModuleAgent,
        target_pos_local: np.ndarray,
        axis_idx: int,
        pivot_type: str,
        handoff_idx: Optional[int],
    ):
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        lattice_ref_body_idx = axis_idx

        R_ax = self.sim.body_rotation_matrix(axis_idx)
        agent.pre_pivot_pos_local = (R_ax.T @ (my_pos - axis_pos)).copy()
        agent.pre_pivot_lattice_ref = axis_idx

        target_pos_local = np.asarray(target_pos_local, dtype=float).reshape(3)
        target_world = self.sim.target_world_from_local(
            lattice_ref_body_idx, target_pos_local)

        if pivot_type == "lateral" and handoff_idx is not None:
            attract_body = handoff_idx
            attract_conn = self.sim.lateral_handoff_attract_connector(
                axis_idx, agent.body_idx, handoff_idx)
        else:
            attract_body = axis_idx
            attract_conn = self.sim.nearest_connector(
                axis_idx, target_world - pos[axis_idx])

        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n in neighbors:
            if n != axis_idx:
                self.sim.remove_bond(agent.body_idx, n)

        if pivot_type == "corner":
            if not self.sim.USE_SPRING_BONDS:
                self.sim.remove_bond(agent.body_idx, axis_idx)

        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
        r_target = target_world - axis_pos
        cos_angle = np.clip(
            np.dot(r_vec, r_target) / (np.linalg.norm(r_vec) *
                                        np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_angle)
        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle,
            kp, kd, duration=12.0,
            lattice_ref_body_idx=lattice_ref_body_idx,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body,
            attract_connector=attract_conn,
            pivot_type=pivot_type)

        agent.state = ModuleState.PIVOTING
        agent.target_pos = target_world.copy()
        agent.target_pos_local = target_pos_local.copy()
        agent.lattice_ref_body_idx = lattice_ref_body_idx
        agent.attract_body_idx = attract_body
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = pivot_type
        agent.handoff_idx = handoff_idx
        agent.handoff_done = False
        self.total_moves += 1

        self.move_log.append({
            "module": agent.module_id,
            "from": my_pos.tolist(),
            "to": target_world.tolist(),
            "axis": self._idx_to_mid.get(axis_idx, f"idx_{axis_idx}"),
            "sim_time": self.sim.sim_time,
            "pivot_type": pivot_type,
        })
        logger.info("Module {} starting {} pivot to {} (phase 2{})",
                     agent.module_id, pivot_type, target_world,
                     f", handoff={self._idx_to_mid.get(handoff_idx, handoff_idx)}"
                     if handoff_idx is not None else "")

    def _retarget_to_nearest_empty_cell(self, agent: ModuleAgent) -> bool:
        """See DecentralizedCoagulation._retarget_to_nearest_empty_cell."""
        if agent.body_idx not in self.sim._active_pivots:
            return False
        ps = self.sim._active_pivots[agent.body_idx]
        axis_idx = ps.axis_idx
        if not (0 <= axis_idx < self.sim.N):
            return False

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        nom = float(self.sim.NOMINAL_DIST)

        if agent.retarget_count >= self.MAX_RETARGETS_PER_PIVOT:
            return False

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - axis_pos) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        fault_idxs = self._get_fault_idxs()
        fault_positions = [pos[fidx] for fidx in fault_idxs
                           if 0 <= fidx < self.sim.N]

        def _is_fault_adjacent(world_pt) -> bool:
            for fp in fault_positions:
                if float(np.linalg.norm(world_pt - fp)) < nom * 1.5:
                    return True
            return False

        best_target_local = None
        best_cell = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            cell = (int(delta[0]), int(delta[1]), int(delta[2]))
            if cell in occupied:
                continue
            if cell in agent.retarget_blacklist:
                continue
            dw = nom * np.array(delta, dtype=float)
            target_world = axis_pos + R @ dw
            if _is_fault_adjacent(target_world):
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - axis_pos)
                best_cell = cell

        if best_target_local is None:
            return False

        if best_cell is not None:
            agent.retarget_blacklist.add(best_cell)
        agent.retarget_count += 1

        target_world = self.sim.target_world_from_local(
            axis_idx, best_target_local)
        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
        r_target = target_world - axis_pos
        cos_a = np.clip(
            np.dot(r_vec, r_target) / (
                np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_a)
        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)
        attract_conn = self.sim.nearest_connector(
            axis_idx, target_world - axis_pos)

        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle, kp, kd,
            duration=12.0,
            lattice_ref_body_idx=axis_idx,
            target_pos_local=best_target_local,
            attract_body_idx=axis_idx,
            attract_connector=attract_conn,
            pivot_type="corner")

        agent.target_pos = target_world.copy()
        agent.target_pos_local = np.asarray(best_target_local).copy()
        agent.lattice_ref_body_idx = axis_idx
        agent.attract_body_idx = axis_idx
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = "corner"
        agent.handoff_idx = None
        agent.handoff_done = False
        return True

    def _start_reversal(self, agent: ModuleAgent):
        """Pivot the module back to its pre-pivot lattice position (phase 2).

        See DecentralizedCoagulation._start_reversal for lateral reversal logic.
        """
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]

        neighbors = self.get_physical_neighbors(agent.body_idx)
        if not neighbors:
            # Find the nearest body in the sim. Only bond if within
            # BOND_THRESHOLD (essentially at lattice spacing). Otherwise
            # refuse to create a wildly-stretched rest-pose bond — leave
            # the module orphaned and abort the reversal. Articulation
            # checks elsewhere will see the truth.
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            nearest_dist = float(np.linalg.norm(
                pos[nearest] - pos[agent.body_idx]))
            if nearest_dist < self.BOND_THRESHOLD:
                self.sim.create_bond(agent.body_idx, nearest)
                neighbors = [nearest]
            else:
                agent.state = ModuleState.IDLE
                agent.token = None
                agent.pending_retry = None
                return

        ref = agent.pre_pivot_lattice_ref
        target_local = agent.pre_pivot_pos_local
        if target_local is None:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None
            return

        original_axis = ref if ref is not None else None
        is_lateral_reversal = (
            agent.pivot_type == "lateral"
            and agent.handoff_done
            and original_axis is not None
            and original_axis != agent.pivot_axis_idx
        )

        best_ax = min(neighbors,
                      key=lambda n: np.linalg.norm(pos[n] - my_pos))
        if ref is None:
            ref = best_ax
        target_world = self.sim.target_world_from_local(ref, target_local)
        axis_pos = pos[best_ax]

        for n in neighbors:
            if n != best_ax:
                self.sim.remove_bond(agent.body_idx, n)

        if is_lateral_reversal:
            attract_conn = self.sim.lateral_handoff_attract_connector(
                best_ax, agent.body_idx, original_axis)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=original_axis,
                attract_connector=attract_conn,
                pivot_type="lateral",
                repel_idx=-1)

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = original_axis
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "lateral"
            agent.handoff_idx = original_axis
            agent.handoff_done = False
            agent.reversal_handoff_idx = original_axis
        else:
            attract_conn = self.sim.nearest_connector(
                best_ax, target_world - axis_pos)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=best_ax,
                attract_connector=attract_conn,
                pivot_type="corner")

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = best_ax
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "corner"
            agent.handoff_idx = None
            agent.handoff_done = False
            agent.reversal_handoff_idx = None

    def _emergency_snap_to_lattice(self, agent: ModuleAgent):
        """Ensure module is bonded and pivot to the nearest empty lattice site.

        Called when a reversal pivot fails (timeout/collision) during phase 2.
        Guarantees the module keeps at least one bond and attempts a corner
        pivot to the closest unoccupied lattice cell.
        """
        neighbors = self.get_physical_neighbors(agent.body_idx)
        pos = self.sim.get_positions()

        if not neighbors:
            # Only bond if the nearest body is within BOND_THRESHOLD. If
            # nothing is close enough, refuse to create a stretched rest-
            # pose bond and abort the snap; the module stays orphaned.
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            nearest_dist = float(np.linalg.norm(
                pos[nearest] - pos[agent.body_idx]))
            if nearest_dist < self.BOND_THRESHOLD:
                self.sim.create_bond(agent.body_idx, nearest)
                neighbors = [nearest]
            else:
                agent.state = ModuleState.IDLE
                agent.token = None
                agent.pending_retry = None
                return

        axis_idx = min(neighbors,
                       key=lambda n: np.linalg.norm(
                           pos[n] - pos[agent.body_idx]))

        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)
        my_pos = pos[agent.body_idx]

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is not None:
            self._start_pivot(agent, best_target_local, axis_idx, "corner",
                              None)
            logger.info("Module {} emergency snap pivot (phase 2) around {}",
                        agent.module_id,
                        self._idx_to_mid.get(axis_idx, axis_idx))
        else:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None

    def _start_corrective_pivot(self, agent: ModuleAgent, axis_idx: int):
        """Corner pivot to nearest empty lattice site on *axis_idx* to fix a triangular bond."""
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        arm = my_pos - p_ref
        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            if not _lattice_delta_perpendicular_to_arm(delta, arm, R):
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is None:
            return

        self._start_pivot(agent, best_target_local, axis_idx, "corner", None)
        logger.info("Module {} corrective pivot (triangle fix, phase 2) around {}",
                    agent.module_id, self._idx_to_mid.get(axis_idx, axis_idx))

    def _forward_token(self, agent):
        if agent.token is None:
            return
        pos = self.sim.get_positions()
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid is None or n_mid not in self.agents:
                continue
            n_agent = self.agents[n_mid]
            if n_agent.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
                continue
            my_pos = pos[agent.body_idx]
            nbr_pos = pos[n_idx]
            R_me = self.sim.body_rotation_matrix(agent.body_idx)
            R_nbr = self.sim.body_rotation_matrix(n_idx)
            world_dir = R_me @ agent.token.direction + (my_pos - nbr_pos)
            propagated_dir = R_nbr.T @ world_dir
            n_agent.incoming_tokens.append(
                Token(direction=propagated_dir, source_id=agent.module_id))

    def _reconnect_bonds(self, body_idx):
        """Reconnect bonds dictated by the policy after a pivot.

        Only creates bonds to the pivot axis module and to modules that
        occupy lattice-adjacent positions of the target, provided they
        are within the distance threshold.  No blind proximity scan.
        """
        agent = None
        for a in self.agents.values():
            if a.body_idx == body_idx:
                agent = a
                break
        if agent is None:
            return

        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        if (agent.lattice_ref_body_idx is not None
                and agent.target_pos_local is not None):
            target = self.sim.target_world_from_local(
                agent.lattice_ref_body_idx, agent.target_pos_local)
        else:
            target = agent.target_pos if agent.target_pos is not None else my_pos

        expected_neighbors: Set[int] = set()
        if agent.pivot_axis_idx is not None:
            expected_neighbors.add(agent.pivot_axis_idx)

        if agent.lattice_ref_body_idx is not None:
            ref = agent.lattice_ref_body_idx
            R = self.sim.body_rotation_matrix(ref)
            p_ref = pos[ref]
            nom = float(self.sim.NOMINAL_DIST)
            goal_cell = np.round(
                agent.target_pos_local / nom).astype(int)  # type: ignore[union-attr]
            for delta in _LATTICE_DELTAS:
                nbr_cell = (int(goal_cell[0] + delta[0]),
                            int(goal_cell[1] + delta[1]),
                            int(goal_cell[2] + delta[2]))
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    j_cell = tuple(np.round(
                        R.T @ (pos[j] - p_ref) / nom).astype(int))
                    if j_cell == nbr_cell:
                        expected_neighbors.add(j)
        else:
            target_key = tuple(np.round(target).astype(int))
            for delta in _LATTICE_DELTAS:
                nbr_key = (target_key[0] + delta[0],
                           target_key[1] + delta[1],
                           target_key[2] + delta[2])
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    if tuple(np.round(pos[j]).astype(int)) == nbr_key:
                        expected_neighbors.add(j)

        for j in expected_neighbors:
            dist = np.linalg.norm(my_pos - pos[j])
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

        fault_idxs = self._get_fault_idxs()
        for j in range(self.sim.N):
            if j == body_idx:
                continue
            if (fault_idxs and j in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            dist = float(np.linalg.norm(my_pos - pos[j]))
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

    def _has_pivoting_neighbor_of_neighbor(self, body_idx):
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        neighbors = self.get_physical_neighbors(body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid and n_mid in self.agents:
                if self.agents[n_mid].state in _moving:
                    return True
            for nn_idx in self.get_physical_neighbors(n_idx):
                nn_mid = self._idx_to_mid.get(nn_idx)
                if nn_mid and nn_mid in self.agents:
                    if self.agents[nn_mid].state in _moving:
                        return True
        return False


class DisplacementRestructuring(DecentralizedRestructuring):
    """Displacement-guided restructuring (phase 2 alternative).

    Instead of propagating rendezvous tokens from non-movers' empty slots,
    each displaced module computes its displacement from its pre-damage
    position and uses that as a synthetic token direction.  No inter-module
    token propagation is needed -- each agent independently knows where it
    should return to.

    Modules are driven by largest-displacement-first priority (the greedy
    ordering from the graph-based ``restructuring_displacement``), adapted
    to the async PyBullet tick loop.
    """

    DISPLACEMENT_THRESHOLD = 0.5

    def __init__(self, sim, module_ids: List[str],
                 body_indices: Dict[str, int],
                 coag_moved: Set[str],
                 original_positions: Dict[str, np.ndarray],
                 pre_damage_neighbor_slots: Optional[Dict[str, List[np.ndarray]]] = None):
        super().__init__(
            sim=sim,
            module_ids=module_ids,
            body_indices=body_indices,
            coag_moved=coag_moved,
            pre_damage_neighbor_slots=pre_damage_neighbor_slots or {},
            token_strategy="furthest",
        )
        self.original_positions = original_positions
        # Phase-2 diagnostics (why does restructuring make so few moves?).
        # Counters are integers; the *_ids are sets of module ids so repeated
        # per-tick visits are idempotent. Read by _write_trial_diagnostics.
        self._p2diag = {
            "coag_moved": len(coag_moved),
            "tokens_injected": 0,
            "inject_skip_below_threshold": 0,
            "inject_skip_not_idle": 0,
            "inject_skip_no_orig": 0,
            "engaged_ids": set(),          # reached HAS_TOKEN past the hold
            "blocked_not_movable_ids": set(),
            "blocked_no_budget_ids": set(),
            "no_target_ids": set(),        # held token, pick_target found nothing
            "moved_ids": set(),            # executed >=1 pivot
        }

    def generate_initial_tokens(self):
        """Inject displacement-based tokens for all displaced coag-movers."""
        self._inject_displacement_tokens()

    def _inject_displacement_tokens(self):
        """For each displaced coag-mover, set a synthetic token pointing home."""
        pos = self.sim.get_positions()
        d = self._p2diag
        for mid in self.coag_moved:
            if mid not in self.agents:
                continue
            if mid not in self.original_positions:
                d["inject_skip_no_orig"] += 1
                continue
            agent = self.agents[mid]
            if agent.state != ModuleState.IDLE:
                d["inject_skip_not_idle"] += 1
                continue
            disp = self.original_positions[mid] - pos[agent.body_idx]
            dist = float(np.linalg.norm(disp))
            if dist < self.DISPLACEMENT_THRESHOLD:
                d["inject_skip_below_threshold"] += 1
                continue
            R = self.sim.body_rotation_matrix(agent.body_idx)
            direction = R.T @ disp
            agent.incoming_tokens.append(
                Token(direction=direction.copy(), source_id=mid))
            d["tokens_injected"] += 1

    def _forward_token(self, agent: ModuleAgent):
        """Displacement tokens are per-module; forwarding is meaningless. Drop."""
        pass

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        """Scoring origin = the module's original (pre-damage) position."""
        if agent.module_id in self.original_positions:
            return self.original_positions[agent.module_id].copy()
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class RetraceRestructuring(DisplacementRestructuring):
    """Phase-2 restructuring by locally retracing coagulation pivots.

    Each module replays the pivots it executed during coagulation in reverse
    order, attempting to undo each one. A step's target is the source cell
    of that pivot, reconstructed relative to the current pose of the
    axis neighbor it pivoted on: the axis is a neighbor identity and the
    source position is stored in the axis's body frame, so the target is
    well-defined without any absolute/global measurement and remains correct
    even if the surrounding structure has been rigidly repositioned. When the
    original axis is still bonded the reverse pivot is exact; when the local
    geometry no longer admits it, the inherited ``pick_target`` rule selects
    the admissible pivot that gets closest to the source cell --- an
    intermediate, still-improved pose. The objective is therefore purely
    relative (rotation/translation/reflection invariant), matching the
    Gromov--Wasserstein shape metric, and fully decentralized: each agent uses
    only its own pivot history and current neighborhood.

    Steps whose recorded axis is no longer an active module (e.g. it was a
    fault used as a passive pivot reference) cannot be anchored locally and
    are skipped; the module retraces the remainder.
    """

    def __init__(self, sim, module_ids, body_indices, coag_moved,
                 original_positions, pivot_history,
                 pre_damage_neighbor_slots=None):
        super().__init__(
            sim=sim, module_ids=module_ids, body_indices=body_indices,
            coag_moved=coag_moved, original_positions=original_positions,
            pre_damage_neighbor_slots=pre_damage_neighbor_slots)
        # module-id -> body-index for active modules (base class only keeps the
        # reverse map). Used to anchor retrace targets to the axis body.
        self._mid_to_idx: Dict[str, int] = dict(body_indices)
        # Reversed per-module step lists (most-recent pivot undone first).
        self._retrace: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}
        self._retrace_idx: Dict[str, int] = {}
        self._cur_step: Dict[str, Tuple[str, np.ndarray, str]] = {}
        for mid, hist in (pivot_history or {}).items():
            if mid in self.agents and hist:
                self._retrace[mid] = list(reversed(hist))
                self._retrace_idx[mid] = 0

    def _set_retrace_budgets(self):
        """Give each retracing agent enough motion budget to undo every logged
        pivot. Retracing is a directed rewind, not the exploratory search that
        phase 1 caps as a regularizer, so the tight motion cap does not apply
        (design decision). Called from generate_initial_tokens so it runs
        AFTER any external per-agent budget reset (e.g. in the MC runner)."""
        for mid, steps in self._retrace.items():
            ag = self.agents[mid]
            ag.action_points = max(int(self.INITIAL_ACTION_POINTS), len(steps))

    def generate_initial_tokens(self):
        # No displacement-to-home injection; retrace steps are fed one at a
        # time by _refill_retrace_tokens() at the start of each tick.
        self._set_retrace_budgets()
        self._refill_retrace_tokens()

    def tick(self) -> bool:
        self._refill_retrace_tokens()
        return super().tick()

    def _retrace_target_world(self, step, pos: np.ndarray) -> Optional[np.ndarray]:
        """World position of a step's source cell, anchored to the axis's
        current pose. Returns None if the axis is not an active module."""
        ax_mid, pre_local, _ptype = step
        ax_idx = self._mid_to_idx.get(ax_mid)
        if ax_idx is None:
            return None
        R_ax = self.sim.body_rotation_matrix(ax_idx)
        return pos[ax_idx] + R_ax @ pre_local

    def _refill_retrace_tokens(self):
        """For each idle agent with no pending token, advance to its next
        anchorable retrace step and inject a token toward that step's source
        cell. Steps whose axis has vanished are skipped."""
        pos = self.sim.get_positions()
        for mid, steps in self._retrace.items():
            ag = self.agents.get(mid)
            if ag is None or ag.is_faulty:
                continue
            if not (ag.state == ModuleState.IDLE
                    and not ag.incoming_tokens
                    and ag.token is None
                    and ag.action_points > 0):
                continue
            idx = self._retrace_idx.get(mid, 0)
            target = None
            while idx < len(steps):
                target = self._retrace_target_world(steps[idx], pos)
                if target is not None:
                    break
                idx += 1  # axis gone; skip this step
            if idx >= len(steps) or target is None:
                self._retrace_idx[mid] = len(steps)
                self._cur_step.pop(mid, None)
                continue
            self._cur_step[mid] = steps[idx]
            self._retrace_idx[mid] = idx + 1
            R = self.sim.body_rotation_matrix(ag.body_idx)
            direction = R.T @ (target - pos[ag.body_idx])
            ag.incoming_tokens.append(
                Token(direction=direction.copy(), source_id=mid))

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        """Scoring origin = the source cell of the agent's current retrace
        step, anchored to the axis's current pose (purely relative)."""
        step = self._cur_step.get(agent.module_id)
        if step is not None:
            tgt = self._retrace_target_world(step, pos)
            if tgt is not None:
                return tgt
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction
