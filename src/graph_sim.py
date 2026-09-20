"""
Graph-only simulator that mirrors `BulletSimulator`'s API surface used by
`src/agent_policy.py`. Pivots are deterministic kinematic arcs about the
policy-supplied axis, interpolated over the policy's pivot duration so the
mid-pivot lateral-handoff branch fires naturally. No physics, no
collisions, no PD overshoot. `pivot_collided` and `pivot_timed_out` always
return False — pivots are considered perfect.

The decentralized agent policies (`DecentralizedCoagulation`,
`DecentralizedRestructuring`, `DisplacementRestructuring`) consume this
simulator interchangeably with `BulletSimulator`, so feature parity with
the PyBullet path is automatic.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class PivotState:
    """Active pivot tracking for one body in `GraphSimulator`."""
    axis_idx: int
    rot_axis: np.ndarray
    target_angle: float
    duration: float
    t: float
    start_pos: np.ndarray
    target_pos: np.ndarray
    start_orient: np.ndarray
    target_orient: np.ndarray
    lattice_ref_body_idx: int
    target_pos_local: np.ndarray
    attract_body_idx: int
    attract_connector: int
    pivot_type: str = "corner"
    repel_idx: Optional[int] = None
    completed: bool = False
    timed_out: bool = False
    collided: bool = False
    # Optional piecewise rolling path (TWO_ARC_LATERAL): list of
    # (anchor_pos, r_start, rot_axis_unit, dtheta) segments traversed in
    # order; interpolation walks them by cumulative angle.
    segments: Optional[list] = None


def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for unit-norm ``axis`` and ``angle`` (radians)."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


class GraphSimulator:
    """Drop-in replacement for `BulletSimulator` in the decentralized agent
    policies. Pivots are deterministic kinematic arcs with no physics.
    """

    NOMINAL_DIST = 1.0
    MODULE_RADIUS = 0.5
    USE_SPRING_BONDS = False
    USE_PIVOT_PD = False
    USE_ROLLING_SPHERE_PIVOT = True
    USE_CRYSTAL_ATTITUDE_SNAP_AFTER_ROLLING_PIVOT = False
    MAX_PIVOT_TIME = 20.0
    BOND_RESTORE_DIST_THRESHOLD = 1.05

    CONNECTOR_DIRS = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)

    # Multiplier on the policy-supplied pivot duration. Graph sim has no
    # physics, so pivot duration only governs how many ticks the agent
    # spends in PIVOTING (for mid-flight lateral handoff and pivot-
    # exclusion lockout). Policy passes duration=12.0; scale=1/60
    # collapses that to 0.2 s = 2 ticks at dt=0.1, the minimum that
    # preserves a clean mid-flight handoff window.
    PIVOT_DURATION_SCALE = 1.0 / 60.0

    # When True, lateral handoffs interpolate along the physically rolled
    # two-arc path (around the axis until contact with the handoff module,
    # then around the handoff to the target) instead of the single-arc +
    # end-snap approximation. Off by default (MC parity); animation demos
    # can enable it per class/instance.
    TWO_ARC_LATERAL = False

    def __init__(self, N: int, pos0: np.ndarray, bonded0: np.ndarray,
                 vel0: Optional[np.ndarray] = None, gui: bool = False,
                 *, module_shape: str = "sphere",
                 **_ignored: Any):
        del vel0, gui, _ignored
        self.N = int(N)
        if module_shape not in ("sphere", "cube"):
            raise ValueError(
                f"module_shape must be 'sphere' or 'cube', got {module_shape!r}")
        self._module_shape = module_shape

        self._positions = np.asarray(pos0, dtype=float).copy()
        if self._positions.shape != (self.N, 3):
            raise ValueError(
                f"pos0 shape {self._positions.shape} != ({self.N}, 3)")

        self._orientations = np.tile(np.eye(3), (self.N, 1, 1)).astype(float)

        bonded0 = np.asarray(bonded0, dtype=bool)
        if bonded0.shape != (self.N, self.N):
            raise ValueError(
                f"bonded0 shape {bonded0.shape} != ({self.N}, {self.N})")
        self._bonds: Set[Tuple[int, int]] = set()
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if bonded0[i, j]:
                    self._bonds.add((i, j))

        self._active_pivots: Dict[int, PivotState] = {}
        self._sim_time = 0.0

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def step(self, dt: float):
        """Advance sim time and interpolate active pivots along their arcs.

        Position interpolates as a rotation about ``rot_axis`` anchored at
        the axis body's current world position. Orientation interpolates
        by applying the same partial rotation to the start orientation.
        When ``t >= duration``, the body snaps exactly to the target and
        the pivot is marked complete (``is_pivot_complete`` returns True).
        """
        dt = float(dt)
        self._sim_time += dt

        for pivot_idx, ps in list(self._active_pivots.items()):
            if ps.completed:
                continue
            ps.t += dt
            if ps.t >= ps.duration:
                self._positions[pivot_idx] = ps.target_pos.copy()
                self._orientations[pivot_idx] = ps.target_orient.copy()
                ps.completed = True
                continue
            frac = ps.t / max(ps.duration, 1e-9)
            if ps.segments:
                total = sum(seg[3] for seg in ps.segments)
                s = total * frac
                p = ps.start_pos
                for anchor, r_start, seg_axis, dtheta in ps.segments:
                    step_ang = min(s, dtheta)
                    R_seg = _rotation_matrix_axis_angle(seg_axis, step_ang)
                    p = anchor + R_seg @ r_start
                    s -= step_ang
                    if s <= 1e-12:
                        break
                self._positions[pivot_idx] = p
                R_partial = _rotation_matrix_axis_angle(
                    ps.rot_axis, ps.target_angle * frac)
                self._orientations[pivot_idx] = R_partial @ ps.start_orient
                continue
            theta = ps.target_angle * frac
            R_partial = _rotation_matrix_axis_angle(ps.rot_axis, theta)
            r0 = ps.start_pos - self._positions[ps.axis_idx]
            self._positions[pivot_idx] = (
                self._positions[ps.axis_idx] + R_partial @ r0)
            self._orientations[pivot_idx] = R_partial @ ps.start_orient

    def get_positions(self) -> np.ndarray:
        return self._positions

    def get_bond_matrix(self) -> np.ndarray:
        bonded = np.zeros((self.N, self.N), dtype=bool)
        for (i, j) in self._bonds:
            bonded[i, j] = True
            bonded[j, i] = True
        return bonded

    def body_rotation_matrix(self, body_idx: int) -> np.ndarray:
        return self._orientations[body_idx].copy()

    def target_world_from_local(self, ref_idx: int,
                                local_offset: np.ndarray) -> np.ndarray:
        local = np.asarray(local_offset, dtype=float).reshape(3)
        R = self.body_rotation_matrix(ref_idx)
        return self._positions[ref_idx] + R @ local

    def create_bond(self, i: int, j: int):
        if i == j:
            return
        lo, hi = (i, j) if i < j else (j, i)
        self._bonds.add((int(lo), int(hi)))

    def remove_bond(self, i: int, j: int):
        if i == j:
            return
        lo, hi = (i, j) if i < j else (j, i)
        self._bonds.discard((int(lo), int(hi)))

    def start_pivot(self, pivot_idx: int, axis_idx: int,
                    rot_axis: np.ndarray, target_angle: float,
                    kp: float, kd: float, duration: float,
                    lattice_ref_body_idx: int,
                    target_pos_local: np.ndarray,
                    attract_body_idx: int,
                    attract_connector: int,
                    pivot_type: str = "corner",
                    repel_idx: Optional[int] = None,
                    **_ignored: Any):
        """Begin (or retarget) a pivot.

        The destination is resolved once at start time via
        ``target_world_from_local(lattice_ref_body_idx, target_pos_local)``
        (matches `BulletSimulator`'s pivot semantics — the axis body is
        stationary during the maneuver). ``rot_axis`` / ``target_angle``
        parameterize the arc. ``kp``/``kd`` are accepted for API parity but
        unused. Calling ``start_pivot`` on a body already in
        ``_active_pivots`` is a retarget: the new state replaces the old
        with ``t=0`` and ``start_pos`` captured from the body's current
        (possibly mid-arc) position.
        """
        del kp, kd, _ignored
        target_pos = self.target_world_from_local(
            lattice_ref_body_idx, target_pos_local)
        target_angle_f = float(target_angle)
        rot_axis_arr = np.asarray(rot_axis, dtype=float).reshape(3).copy()
        R_total = _rotation_matrix_axis_angle(rot_axis_arr, target_angle_f)
        start_orient = self._orientations[pivot_idx].copy()
        target_orient = R_total @ start_orient

        dur = float(duration) * self.PIVOT_DURATION_SCALE
        if dur < 1e-6:
            dur = 1e-6

        ps = PivotState(
            axis_idx=int(axis_idx),
            rot_axis=rot_axis_arr,
            target_angle=target_angle_f,
            duration=dur,
            t=0.0,
            start_pos=self._positions[pivot_idx].copy(),
            target_pos=target_pos.copy(),
            start_orient=start_orient,
            target_orient=target_orient,
            lattice_ref_body_idx=int(lattice_ref_body_idx),
            target_pos_local=np.asarray(
                target_pos_local, dtype=float).reshape(3).copy(),
            attract_body_idx=int(attract_body_idx),
            attract_connector=int(attract_connector),
            pivot_type=str(pivot_type),
            repel_idx=int(repel_idx) if repel_idx is not None else None,
        )
        if (self.TWO_ARC_LATERAL and ps.pivot_type == "lateral"
                and 0 <= ps.attract_body_idx < self.N):
            ps.segments = self._lateral_two_arc_segments(pivot_idx, ps)
        self._active_pivots[int(pivot_idx)] = ps

    def _lateral_two_arc_segments(self, pivot_idx: int, ps: PivotState):
        """Rolled lateral path: arc about the axis body until contact with
        the handoff body, then arc about the handoff to the target. Returns
        a list of (anchor_pos, r_start, rot_axis_unit, dtheta) segments, or
        None when the geometry degenerates (falls back to the single arc).
        """
        a = self._positions[ps.axis_idx].copy()
        h = self._positions[ps.attract_body_idx].copy()
        p0 = ps.start_pos
        tgt = ps.target_pos
        nom = float(self.NOMINAL_DIST)
        r0 = p0 - a
        rh = h - a
        n0, nh = float(np.linalg.norm(r0)), float(np.linalg.norm(rh))
        if n0 < 1e-9 or nh < 1e-9:
            return None
        axis1 = np.cross(r0, rh)
        if float(np.linalg.norm(axis1)) < 1e-9:
            return None
        axis1 /= np.linalg.norm(axis1)
        cos_sep = float(np.clip(np.dot(r0, rh) / (n0 * nh), -1.0, 1.0))
        sep0 = float(np.arccos(cos_sep))
        # contact separation: |p - h| = nom on circles of radius n0, nh
        cos_c = (n0 * n0 + nh * nh - nom * nom) / (2.0 * n0 * nh)
        if not -1.0 <= cos_c <= 1.0:
            return None
        sep_c = float(np.arccos(cos_c))
        d1 = max(0.0, sep0 - sep_c)
        p_contact = a + _rotation_matrix_axis_angle(axis1, d1) @ r0
        rc = p_contact - h
        rt = tgt - h
        nc, nt = float(np.linalg.norm(rc)), float(np.linalg.norm(rt))
        if nc < 1e-9 or nt < 1e-9:
            return None
        axis2 = np.cross(rc, rt)
        if float(np.linalg.norm(axis2)) < 1e-9:
            return None
        axis2 /= np.linalg.norm(axis2)
        cos2 = float(np.clip(np.dot(rc, rt) / (nc * nt), -1.0, 1.0))
        d2 = float(np.arccos(cos2))
        return [(a, r0, axis1, d1), (h, rc, axis2, d2)]

    def stop_pivot(self, pivot_idx: int, restore_axis_bond: bool = True):
        """End an active pivot.

        Mirrors `BulletSimulator.stop_pivot` semantics: if the pivot
        already completed, snap the body exactly to its target. If
        ``restore_axis_bond`` and the body now sits within
        ``BOND_RESTORE_DIST_THRESHOLD`` of the axis, recreate the
        pivot↔axis bond.
        """
        ps = self._active_pivots.pop(int(pivot_idx), None)
        if ps is None:
            return

        if ps.completed:
            self._positions[pivot_idx] = ps.target_pos.copy()
            self._orientations[pivot_idx] = ps.target_orient.copy()

        if restore_axis_bond and 0 <= ps.axis_idx < self.N:
            d = float(np.linalg.norm(
                self._positions[pivot_idx] - self._positions[ps.axis_idx]))
            if d <= self.BOND_RESTORE_DIST_THRESHOLD:
                self.create_bond(int(pivot_idx), int(ps.axis_idx))

    def is_pivot_complete(self, pivot_idx: int) -> bool:
        ps = self._active_pivots.get(int(pivot_idx))
        if ps is None:
            return True
        return bool(ps.completed)

    def pivot_collided(self, pivot_idx: int) -> bool:
        del pivot_idx
        return False

    def pivot_timed_out(self, pivot_idx: int) -> bool:
        del pivot_idx
        return False

    def has_active_pivots(self) -> bool:
        return len(self._active_pivots) > 0

    def get_connector_world_pos(
            self, body_idx: int, connector_idx: int) -> np.ndarray:
        rot = self.body_rotation_matrix(body_idx)
        world_dir = rot @ self.CONNECTOR_DIRS[connector_idx]
        return self._positions[body_idx] + self.MODULE_RADIUS * world_dir

    def nearest_connector(
            self, body_idx: int, world_direction: np.ndarray) -> int:
        d = np.asarray(world_direction, dtype=float).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return 0
        d_hat = d / n
        rot = self.body_rotation_matrix(body_idx)
        dots = (rot @ self.CONNECTOR_DIRS.T).T @ d_hat
        return int(np.argmax(dots))

    def lateral_handoff_attract_connector(
            self, axis_idx: int, pivot_idx: int, handoff_idx: int) -> int:
        d_bond = self._positions[pivot_idx] - self._positions[axis_idx]
        c_axis = self.nearest_connector(axis_idx, d_bond)
        p_attach = self.get_connector_world_pos(axis_idx, c_axis)
        nom = float(self.NOMINAL_DIST)
        best_j = 0
        best_err = float("inf")
        for j in range(6):
            err = abs(
                float(np.linalg.norm(
                    self.get_connector_world_pos(handoff_idx, j) - p_attach))
                - nom)
            if err < best_err:
                best_err = err
                best_j = j
        return best_j

    def handoff_contact_distance(self) -> float:
        return 1.15 if self._module_shape == "cube" else 1.05

    @classmethod
    def compute_pd_gains(cls, r_vec, duration):
        del r_vec, duration
        return 0.0, 0.0

    @staticmethod
    def get_rotation_axis(pivot_pos, axis_pos, target_pos):
        v1 = pivot_pos - axis_pos
        v2 = target_pos - axis_pos
        axis = np.cross(v1, v2)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            return axis / norm
        ref = v1 if np.linalg.norm(v1) > 1e-8 else v2
        if np.linalg.norm(ref) > 1e-8:
            perp = (np.array([1.0, 0.0, 0.0])
                    if abs(ref[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
            axis = np.cross(ref, perp)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                return axis / norm
        return np.array([0.0, 0.0, 1.0])

    def get_pivot_diagnostic_log(self) -> List[Dict[str, Any]]:
        return []

    def get_auto_bond_stats(self) -> Dict[str, int]:
        return {}

    def get_attitude_drift_log(self) -> List[Tuple[Any, ...]]:
        return []

    def disconnect(self):
        return
