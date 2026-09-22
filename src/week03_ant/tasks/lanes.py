"""Treadmill terrain lanes and MDP terms for the v5 rough-terrain Ant task.

Earlier terrain tasks placed every Ant on a finite 8 x 8 m tile grid. A trained
Ant runs 3--10 m/s, so within one 16 s episode it left its tile, entered harder
rows and finally ran off the end of the world: faster walking was scored as a
failure. v5 instead gives each (terrain family, difficulty level) pair its own
lane along +x that starts and ends with a flat portal tile. When an Ant reaches
the centre of the exit portal it is moved back by exactly one lane period to the
centre of the entrance portal, so it can walk for the whole episode on terrain of
a single known family and level.

Portals are flat 0.1 m height fields like the terrain tiles, so every seam an Ant
crosses joins tiles with matching edge vertices. The world border is never
crossed: its long edge has vertices only at its corners, and at 10 m/s a foot
entering a tile from the border received a large vertical impulse that
overturned the robot.

The flat family does not use mesh tiles at all. PhysX contacts differ between the
course's infinite ground plane and triangle meshes (the course flat policy runs
10.5 m/s on the plane but about 4 m/s on a flat 0.1 m height field, and policies
tuned on either surface fall more often on the other). Flat "lanes" therefore run
on a real PhysX ground plane placed ``plane_height`` below the terrain, the same
surface model as the course task.

The observation and action interfaces are the course ``Isaac-Ant-v0`` ones
(60D / 8D). Only three quantities are made lane-aware:

* the target direction follows the lane centre line instead of ``(1000, 0, 0)``,
* torso height is measured above the local ground instead of the world origin,
* progress is the displacement towards the lane target, unaffected by wraps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import interpolate

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards import power_consumption

from week03_ant.lane_math import (
    body_height_deficit,
    cap_terrain_progress,
    lane_boundary_masks,
    lane_family_level,
    lane_odometer,
    lane_target_direction,
    lane_wrap_mask,
    lane_wrap_shift,
    swing_clearance_cost,
    weighted_lane_layout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# Horizontal reach of an Ant foot from the torso centre (hip 0.57 m + ankle
# segment at its flattest joint limit). Portals must keep every foot on flat ground.
ANT_FOOT_REACH = 1.1
# Furthest a torso can move past the exit-portal centre in one 1/60 s step (at 12 m/s).
MAX_WRAP_OVERSHOOT = 0.2


##
# Terrain generation
##


@height_field_to_mesh
def symmetric_noise_terrain(difficulty: float, cfg: HfSymmetricNoiseTerrainCfg) -> np.ndarray:
    """Zero-mean bumpy ground whose amplitude grows with the difficulty.

    Isaac Lab's ``random_uniform_terrain`` ignores the difficulty value, so every
    row of the earlier "rough" family had the same noise. Here the bump amplitude
    is interpolated from ``amplitude_range`` and heights are bilinearly
    interpolated between control points spaced ``downsampled_scale`` apart.
    """
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    coarse_width = max(2, int(round(cfg.size[0] / cfg.downsampled_scale)) + 1)
    coarse_length = max(2, int(round(cfg.size[1] / cfg.downsampled_scale)) + 1)
    coarse = np.random.uniform(-amplitude, amplitude, size=(coarse_width, coarse_length))
    spline = interpolate.RectBivariateSpline(
        np.linspace(0.0, 1.0, coarse_width), np.linspace(0.0, 1.0, coarse_length), coarse, kx=1, ky=1
    )
    heights = spline(np.linspace(0.0, 1.0, width_pixels), np.linspace(0.0, 1.0, length_pixels))
    return np.rint(heights / cfg.vertical_scale).astype(np.int16)


@configclass
class HfSymmetricNoiseTerrainCfg(HfTerrainBaseCfg):
    """Configuration for :func:`symmetric_noise_terrain`."""

    function = symmetric_noise_terrain

    amplitude_range: tuple[float, float] = MISSING
    """Bump amplitude (m) at difficulty 0 and 1; heights are sampled from ``[-a, a]``."""

    downsampled_scale: float = 0.25
    """Spacing (m) of the random control points."""


class LaneTerrainGenerator(TerrainGenerator):
    """Terrain generator whose columns are homogeneous (family, level) lanes.

    Column ``c`` holds family ``c // num_levels`` at difficulty level
    ``c % num_levels``. Row 0 and the last row are flat portal tiles; every row in
    between repeats the (family, level) pair with a different random seed, cycling
    through the family's sub-terrain variants (for example an upward and an
    inverted pyramid).
    """

    cfg: LaneTerrainGeneratorCfg

    def __init__(self, cfg: LaneTerrainGeneratorCfg, device: str = "cpu"):
        expected_cols = len(cfg.lane_families) * len(cfg.lane_difficulties)
        if cfg.num_cols != expected_cols:
            raise ValueError(f"lane terrain needs num_cols={expected_cols}, got {cfg.num_cols}")
        if not cfg.curriculum:
            raise ValueError("lane terrain requires curriculum=True so columns are laid out in order")
        names = {name for variants in cfg.lane_families.values() for name in variants} | {cfg.portal_sub_terrain}
        unknown = names - set(cfg.sub_terrains)
        if unknown:
            raise ValueError(f"lane layout references unknown sub-terrains: {sorted(unknown)}")
        if cfg.num_rows < 3:
            raise ValueError("a lane needs an entrance portal, at least one terrain row and an exit portal")
        if 0.5 * cfg.size[0] < ANT_FOOT_REACH + MAX_WRAP_OVERSHOOT:
            raise ValueError("portal tiles are too short to keep every foot on flat ground during a wrap")
        super().__init__(cfg, device)

    def _generate_curriculum_terrains(self):
        base_seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        families = list(self.cfg.lane_families.values())
        num_levels = len(self.cfg.lane_difficulties)
        rng_state = np.random.get_state()
        try:
            for col in range(self.cfg.num_cols):
                family, level = divmod(col, num_levels)
                difficulty = float(self.cfg.lane_difficulties[level])
                variants = families[family]
                for row in range(self.cfg.num_rows):
                    if row == 0 or row == self.cfg.num_rows - 1:
                        sub_cfg = self.cfg.sub_terrains[self.cfg.portal_sub_terrain]
                    else:
                        sub_cfg = self.cfg.sub_terrains[variants[(row - 1) % len(variants)]]
                    # A per-tile seed gives every tile its own cache entry and makes the
                    # height-field functions (which draw from numpy's global RNG) reproducible.
                    tile_seed = base_seed * 10007 + col * 131 + row
                    np.random.seed(tile_seed)
                    self.cfg.seed = tile_seed
                    mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                    self._add_sub_terrain(mesh, origin, row, col, sub_cfg)
        finally:
            self.cfg.seed = base_seed
            np.random.set_state(rng_state)


@configclass
class LaneTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration of the treadmill lane layout."""

    class_type: type = LaneTerrainGenerator

    lane_families: dict[str, list[str]] = MISSING
    """Family name -> sub-terrain names cycled along the lane (one per tile)."""

    lane_difficulties: list[float] = MISSING
    """Difficulty value passed to the sub-terrains of each level."""

    portal_sub_terrain: str = MISSING
    """Flat sub-terrain used for the first (entrance) and last (exit) row of a lane."""

    plane_family: str | None = None
    """Name of an extra family that runs on a PhysX ground plane instead of the mesh.

    Its lanes follow the mesh lanes in the lane index (one per difficulty level, all
    identical). The scene must contain a ground plane at :attr:`plane_height`.
    """

    plane_height: float = -50.0
    """Height (m) of that ground plane, far enough below the terrain to never touch it."""

    target_lookahead: float = 12.0
    """Distance (m) ahead of the torso of the lane target used for heading and progress."""


##
# Runtime lane state
##


class LaneState:
    """Per-environment lane geometry, wrap bookkeeping and ground-height queries."""

    def __init__(self, env: ManagerBasedEnv):
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        if not isinstance(generator, LaneTerrainGeneratorCfg) or terrain.terrain_origins is None:
            raise TypeError("lane MDP terms require a LaneTerrainGeneratorCfg terrain")
        device = env.device
        origins = terrain.terrain_origins
        self.num_envs = env.num_envs
        self.family_names = list(generator.lane_families)
        self.difficulties = [float(value) for value in generator.lane_difficulties]
        self.num_levels = len(self.difficulties)
        self.lookahead = float(generator.target_lookahead)
        # Portal tile centres of the mesh lanes: spawn/wrap destination and wrap trigger.
        x_entrance = origins[0, :, 0].clone()
        x_exit = origins[-1, :, 0].clone()
        center_y = origins[0, :, 1].clone()
        ground_z = torch.zeros_like(x_entrance)
        self.num_mesh_lanes = x_entrance.numel()
        self.plane_height = float(generator.plane_height)
        if generator.plane_family is not None:
            # Plane lanes are unbounded: no wrap, flat ground at the plane height.
            self.family_names.append(generator.plane_family)
            extra = torch.zeros(self.num_levels, device=device)
            x_entrance = torch.cat((x_entrance, extra))
            x_exit = torch.cat((x_exit, extra + float("inf")))
            center_y = torch.cat((center_y, extra))
            ground_z = torch.cat((ground_z, extra + self.plane_height))
        self.x_entrance, self.x_exit, self.center_y, self.ground_z = x_entrance, x_exit, center_y, ground_z
        self.shift = torch.where(torch.isfinite(x_exit), lane_wrap_shift(x_entrance, x_exit), torch.zeros_like(x_exit))
        self.lane = terrain.terrain_types.clone().to(torch.long)
        self.wrap_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.spawn_x = torch.zeros(self.num_envs, device=device)
        self.last_distance = torch.zeros(self.num_envs, device=device)
        self.tile_length = float(generator.size[0])
        self.terrain_tiles = generator.num_rows - 2
        self.lane_half_width = float(generator.size[1]) / 2.0
        self.containment_foot_margin = ANT_FOOT_REACH
        self.world_half_size = (
            generator.num_rows * self.tile_length / 2.0 + generator.border_width,
            generator.num_cols * self.lane_half_width + generator.border_width,
        )
        self.out_of_lane = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.world_exit = torch.zeros_like(self.out_of_lane)
        self.last_out_of_lane = torch.zeros_like(self.out_of_lane)
        self.last_world_exit = torch.zeros_like(self.out_of_lane)
        self._mesh = _load_ground_mesh(terrain.cfg.prim_path, device)
        # 3 x 3 ground samples over a 0.5 m square under the torso sphere.
        grid = torch.linspace(-0.25, 0.25, 3, device=device)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        self._patch = torch.stack((gx.flatten(), gy.flatten()), dim=-1)

    def assign(self, lanes: torch.Tensor) -> None:
        """Move environments to the given lane indices (applied at their next reset)."""
        if bool((lanes < 0).any()) or bool((lanes >= self.x_entrance.numel()).any()):
            raise ValueError("lane index out of range")
        self.lane[:] = lanes.to(self.lane.device, torch.long)

    def family_level(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        lanes = self.lane if env_ids is None else self.lane[env_ids]
        return lane_family_level(lanes, self.num_levels)

    def target_direction(self, pos_w: torch.Tensor) -> torch.Tensor:
        return lane_target_direction(pos_w[:, :2], self.center_y[self.lane], self.lookahead)

    def odometer(self, x: torch.Tensor) -> torch.Tensor:
        return lane_odometer(x, self.wrap_count, self.shift[self.lane])

    def track_boundaries(self, pos_w: torch.Tensor) -> None:
        """Record conservative whole-footprint departures, even if the robot returns."""
        out_of_lane, world_exit = lane_boundary_masks(
            pos_w[:, :2], self.center_y[self.lane], self.lane_half_width - self.containment_foot_margin,
            tuple(size - self.containment_foot_margin for size in self.world_half_size),
            self.lane < self.num_mesh_lanes,
        )
        self.out_of_lane |= out_of_lane
        self.world_exit |= world_exit

    def ground_height(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Mean ground height below each torso (all environments, in lane order).

        Mesh lanes are ray-cast; missing hits count as a deep drop. Plane lanes use the
        ground-plane height directly.
        """
        count = pos_w.shape[0]
        samples = pos_w[:, None, :2] + self._patch[None]
        starts = torch.cat((samples, torch.full_like(samples[..., :1], 20.0)), dim=-1).reshape(-1, 3)
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1.0
        hits = raycast_mesh(starts.contiguous(), directions.contiguous(), self._mesh, max_dist=40.0)[0]
        heights = hits[:, 2].reshape(count, -1)
        heights = torch.where(torch.isfinite(heights), heights, torch.full_like(heights, -20.0))
        on_plane = self.lane >= self.num_mesh_lanes
        return torch.where(on_plane, torch.full_like(heights[:, 0], self.plane_height), heights.mean(dim=-1))


def _load_ground_mesh(prim_path: str, device: str):
    from pxr import UsdGeom
    import omni.usd

    mesh_prim = sim_utils.get_first_matching_child_prim(prim_path, lambda prim: prim.GetTypeName() == "Mesh")
    if mesh_prim is None or not mesh_prim.IsValid():
        raise RuntimeError(f"no terrain mesh below {prim_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    transform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
    points = points @ transform[:3, :3].T + transform[:3, 3]
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    return convert_to_warp_mesh(points, indices, device=device)


def get_lane_state(env: ManagerBasedEnv) -> LaneState:
    """Return the lane state of ``env``, creating it on first use."""
    state = getattr(env, "_week03_lane_state", None)
    if state is None:
        state = LaneState(env)
        env._week03_lane_state = state
    return state


def _resolve_ids(env: ManagerBasedEnv, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


##
# Events
##


def lane_assign(env: ManagerBasedEnv, env_ids, family_levels: list[tuple[str, int]]):
    """Startup event: place environment ``i`` on ``family_levels[i % len]``."""
    state = get_lane_state(env)
    lanes = []
    for index in range(env.num_envs):
        family, level = family_levels[index % len(family_levels)]
        lanes.append(state.family_names.index(family) * state.num_levels + int(level))
    state.assign(torch.tensor(lanes, device=env.device))


def lane_assign_weighted(env: ManagerBasedEnv, env_ids, family_weights: dict[str, float]):
    """Startup event: split environments over families by weight, cycling levels."""
    state = get_lane_state(env)
    weights = torch.tensor([float(family_weights.get(name, 0.0)) for name in state.family_names])
    lanes = weighted_lane_layout(env.num_envs, weights, state.num_levels)
    state.assign(lanes.to(env.device))


def lane_reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the Ant at the centre of its lane's flat entrance portal.

    Before overwriting the pose, the forward distance of the finished episode is
    stored in ``LaneState.last_distance`` for evaluation scripts.
    """
    state = get_lane_state(env)
    asset = env.scene[asset_cfg.name]
    ids = _resolve_ids(env, env_ids)
    lanes = state.lane[ids]
    state.track_boundaries(asset.data.root_pos_w)
    state.last_out_of_lane[ids] = state.out_of_lane[ids]
    state.last_world_exit[ids] = state.world_exit[ids]
    final_x = asset.data.root_pos_w[ids, 0]
    state.last_distance[ids] = final_x + state.wrap_count[ids] * state.shift[lanes] - state.spawn_x[ids]

    root_states = asset.data.default_root_state[ids].clone()
    ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")], device=asset.device
    )
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(ids), 6), device=asset.device)
    positions = torch.stack(
        (
            state.x_entrance[lanes] + samples[:, 0],
            state.center_y[lanes] + samples[:, 1],
            state.ground_z[lanes] + root_states[:, 2] + samples[:, 2],
        ),
        dim=-1,
    )
    delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], delta)
    ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    velocities = root_states[:, 7:13] + math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(ids), 6), device=asset.device
    )
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=ids)
    state.spawn_x[ids] = positions[:, 0]
    state.wrap_count[ids] = 0
    state.out_of_lane[ids] = False
    state.world_exit[ids] = False


def lane_wrap(env: ManagerBasedEnv, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Interval event run every step: move Ants past the exit portal centre to the entrance."""
    state = get_lane_state(env)
    asset = env.scene[asset_cfg.name]
    state.track_boundaries(asset.data.root_pos_w)
    crossed = lane_wrap_mask(asset.data.root_pos_w[:, 0], state.x_exit[state.lane])
    if not bool(crossed.any()):
        return
    ids = crossed.nonzero().flatten()
    pose = asset.data.root_link_pose_w[ids].clone()
    pose[:, 0] -= state.shift[state.lane[ids]]
    asset.write_root_link_pose_to_sim(pose, env_ids=ids)
    state.wrap_count[ids] += 1


##
# Observations (drop-in replacements for the course terms)
##


def lane_base_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Torso height above the local ground (equals world z on the course plane)."""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w
    return (pos[:, 2] - get_lane_state(env).ground_height(pos)).unsqueeze(-1)


def lane_angle_to_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Course ``base_angle_to_target`` with the lane target."""
    asset = env.scene[asset_cfg.name]
    direction = get_lane_state(env).target_direction(asset.data.root_pos_w)
    walk_target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    angle = walk_target_angle - yaw
    return torch.atan2(torch.sin(angle), torch.cos(angle)).unsqueeze(-1)


def lane_heading_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Course ``base_heading_proj`` with the lane target."""
    asset = env.scene[asset_cfg.name]
    direction = get_lane_state(env).target_direction(asset.data.root_pos_w)
    heading = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    return (heading[:, :2] * direction).sum(dim=-1, keepdim=True)


##
# Rewards
##


class lane_progress_reward(ManagerTermBase):
    """Course progress reward (m/s towards the target) measured on the lane odometer."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset = env.scene[asset_cfg.name]
        self._prev_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._prev_xy[:] = self._current_xy(torch.arange(env.num_envs, device=env.device))

    def _current_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
        state = get_lane_state(self._env)
        pos = self._asset.data.root_pos_w[env_ids]
        odo = pos[:, 0] + state.wrap_count[env_ids] * state.shift[state.lane[env_ids]]
        return torch.stack((odo, pos[:, 1]), dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = _resolve_ids(self._env, env_ids)
        self._prev_xy[ids] = self._current_xy(ids)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        terrain_speed_limit: float | None = None,
    ) -> torch.Tensor:
        state = get_lane_state(env)
        current = self._current_xy(torch.arange(env.num_envs, device=env.device))
        direction = state.target_direction(self._asset.data.root_pos_w)
        progress = ((current - self._prev_xy) * direction).sum(dim=-1) / env.step_dt
        self._prev_xy[:] = current
        if terrain_speed_limit is not None:
            progress = cap_terrain_progress(progress, state.lane >= state.num_mesh_lanes, terrain_speed_limit)
        return progress


def lane_move_to_target_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Course ``move_to_target_bonus`` with the lane target."""
    heading_proj = lane_heading_proj(env, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, torch.ones_like(heading_proj), heading_proj / threshold)


def lane_centering_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Discourage mesh-lane drift; leave the course ground-plane reward untouched."""
    state = get_lane_state(env)
    error = (env.scene["robot"].data.root_pos_w[:, 1] - state.center_y[state.lane]).abs()
    return (error - 0.5).clamp(min=0.0).square() * (state.lane < state.num_mesh_lanes)


def lane_stall_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remove the incentive to stand safely before an obstacle after the first second."""
    state = get_lane_state(env)
    slow = (1.0 - env.scene["robot"].data.root_lin_vel_w[:, 0]).clamp(min=0.0)
    active = (state.lane < state.num_mesh_lanes) & (env.episode_length_buf * env.step_dt > 1.0)
    return slow * active


def stone_body_height_penalty(
    env: ManagerBasedRLEnv, target_height: float = 0.5, tolerance: float = 0.2,
) -> torch.Tensor:
    """Training-only cost for resting the torso on stones while feet sit in gaps.

    v5 stones are centred on world z=0 with at most 5 cm top variation. Using
    nominal top height here is intentional: the observation's mean ground probe
    includes gap bottoms and otherwise makes a stuck, low torso look elevated.
    This privileged reward never changes the policy's course-compatible inputs.
    """
    state = get_lane_state(env)
    family, _ = state.family_level()
    on_stones = family == state.family_names.index("stepping_stones")
    height = env.scene["robot"].data.root_pos_w[:, 2]
    return body_height_deficit(height, target_height, tolerance) * on_stones


class stone_swing_clearance_penalty(ManagerTermBase):
    """Discourage a forward-swinging distal foot from dragging through stone gaps.

    The course Ant's ankle link origin is not its foot tip. Loaded USD collision
    geometry has a 0.565685 m capsule with radius 0.08 and distal centre offsets
    (+.4,+.4), (-.4,+.4), (-.4,-.4), (+.4,-.4) in the ordered ankle link frames.
    Tip velocity includes omega cross offset about the *link* origin, not COM.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene["robot"]
        names = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
        self._body_ids, found = self._asset.find_bodies(names, preserve_order=True)
        if found != names:
            raise ValueError(f"stone swing reward requires the course Ant feet in order, got {found}")
        self._tip_offsets = torch.tensor(
            [[0.4, 0.4, 0.0], [-0.4, 0.4, 0.0], [-0.4, -0.4, 0.0], [0.4, -0.4, 0.0]],
            device=env.device,
        ).expand(env.num_envs, -1, -1)

    def __call__(
        self, env: ManagerBasedRLEnv, target_clearance: float = 0.1, trap_weight: float = 0.0,
    ) -> torch.Tensor:
        state = get_lane_state(env)
        family, _ = state.family_level()
        on_stones = family == state.family_names.index("stepping_stones")
        data = self._asset.data
        ids = self._body_ids
        offset = math_utils.quat_apply(data.body_link_quat_w[:, ids], self._tip_offsets)
        tip = data.body_link_pos_w[:, ids] + offset
        tip_velocity = data.body_link_lin_vel_w[:, ids] + torch.cross(
            data.body_link_ang_vel_w[:, ids], offset, dim=-1,
        )
        relative_velocity = tip_velocity - data.root_link_lin_vel_w[:, None]
        direction = state.target_direction(data.root_pos_w)
        swing_speed = (relative_velocity[..., :2] * direction[:, None]).sum(dim=-1)
        clearance = tip[..., 2] - 0.08  # distal capsule bottom above nominal stone tops (z=0)
        return swing_clearance_cost(clearance, swing_speed, target_clearance, trap_weight) * on_stones


class lane_power_consumption(power_consumption):
    """Allow more effort to step over terrain while preserving the plane objective."""

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        energy = super().__call__(env, gear_ratio, asset_cfg)
        state = get_lane_state(env)
        return torch.where(state.lane < state.num_mesh_lanes, energy * 0.5, energy)


def lane_out_of_bounds(env: ManagerBasedRLEnv, margin: float = 0.5) -> torch.Tensor:
    """Training-only early reset before the torso leaves its assigned mesh lane."""
    state = get_lane_state(env)
    error = (env.scene["robot"].data.root_pos_w[:, 1] - state.center_y[state.lane]).abs()
    return (state.lane < state.num_mesh_lanes) & (error > state.lane_half_width - margin)


##
# Terminations
##


def lane_torso_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Course fall test (torso below ``minimum_height``) against the local ground."""
    return lane_base_height(env, asset_cfg).squeeze(-1) < minimum_height
