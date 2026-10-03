"""v28: teammate and own factors, alone and combined, scored like the course demo.

Every v28 policy keeps the course interface: the 60D Isaac-Ant-v0 observation, 8D joint
effort action (scale 7.5) and the [400, 200, 100] ELU actor-critic.  Two changes are common
to all v28 tasks because both teammates found them necessary on uneven ground:

* ``base_height`` is the torso height above the ground directly below it (one downward ray),
  not world z.  On the flat plane the two values are identical.
* the fall termination uses the same ground-relative height (< 0.31 m).

The factors under test are the training terrain (flat, Stick's rough/wave/slope mix, Lim's
+-10 cm boxes, my v5 rough-lane families, or mixtures), my Robust42 randomization (friction, mass, COM, reset perturbation,
pushes and observation noise) and, for continuation runs, Stick's recovery reward.  PPO entropy
(Lim) is an agent-side override.  The Play task used for every score keeps the original seven
reward terms, so a training-only reward can never change the scoring rule.

Terrain recipes follow the teammates' published configurations:
Stick-0/isaac-ant-rough-terrain@3cc718a (``ant_env_cfg.py``, ``recovery_mdp.py``) and
LimDaeKyung/IsaacLab_RS@8d9eed1 (``ant_rough_env_cfg.py``, ``ant_rough2_env_cfg.py``,
``ant_eval_env_cfg.py``), both BSD-3-Clause.  Stick's continuous 720 m field is adapted to the
same 160 m terrain strips used by Lim so that every v28 task shares one terrain layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg, RewardsCfg
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_cfg import RobustEventCfg
from .lanes import HfSymmetricNoiseTerrainCfg

HEIGHT_RAY = "height_ray"
TARGET_POS = (1000.0, 0.0, 0.0)

TRAIN_TERRAIN_SEED = 42
"""Generator seed of every training terrain (fixed across training seeds, like Lim's)."""

SELECTION_TERRAIN_SEED = 2028
"""Generator seed of the evaluation strips used to rank recipes."""

CONFIRMATION_TERRAIN_SEED = 2029
"""Fresh evaluation strips used only to re-score the finalists after selection."""

STRIP_TILE = 10.0
STRIP_ROWS = 16
STRIP_COLS = 20
"""160 m of terrain ahead of every robot: 16 rows of 10 m tiles along +x, 20 lanes in y."""


##
# Ground-relative height (Stick and Lim)
##


def base_height_above_ground(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg(HEIGHT_RAY)) -> torch.Tensor:
    """Torso height above the terrain directly below it, measured with one downward ray."""
    sensor = env.scene.sensors[sensor_cfg.name]
    height = env.scene["robot"].data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[:, :1, 2]
    # A missed ray (outside the terrain) falls back to a nominal standing height.
    return torch.nan_to_num(height, nan=0.5, posinf=0.5, neginf=0.5).clamp(-1.0, 2.0)


def height_above_ground_below_minimum(
    env, minimum_height: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg(HEIGHT_RAY)
) -> torch.Tensor:
    """Fall termination relative to the ground below the torso (identical to world z on a plane)."""
    return base_height_above_ground(env, sensor_cfg)[:, 0] < minimum_height


def height_ray_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def use_relative_height(env_cfg: AntEnvCfg) -> None:
    """Install the ray, the relative height observation and the relative fall termination."""
    env_cfg.scene.height_ray = height_ray_cfg()
    # The ray caster finds its prims on the USD stage, which fabric-only clones do not populate.
    env_cfg.scene.clone_in_fabric = False
    env_cfg.observations.policy.base_height = ObsTerm(
        func=base_height_above_ground, params={"sensor_cfg": SceneEntityCfg(HEIGHT_RAY)}
    )
    env_cfg.terminations.torso_height = DoneTerm(
        func=height_above_ground_below_minimum,
        params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg(HEIGHT_RAY)},
    )


##
# Stick's recovery reward (training only)
##


def capped_target_speed(env, speed_cap: float, target_pos: tuple[float, float, float]) -> torch.Tensor:
    """Planar speed toward the target, clipped to +-speed_cap (Stick's progress replacement)."""
    robot = env.scene["robot"].data
    direction = robot.root_pos_w.new_tensor(target_pos[:2]) - robot.root_pos_w[:, :2]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    speed = (robot.root_lin_vel_w[:, :2] * direction).sum(dim=-1)
    return speed.clamp(min=-speed_cap, max=speed_cap)


def low_clearance_risk(
    env, safe_height: float, fall_height: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg(HEIGHT_RAY)
) -> torch.Tensor:
    """Quadratic warning between the safe torso clearance and the fall threshold."""
    height = base_height_above_ground(env, sensor_cfg)[:, 0]
    return ((safe_height - height) / (safe_height - fall_height)).clamp(0.0, 1.0).square()


def tilt_risk(env, safe_up: float, critical_up: float) -> torch.Tensor:
    """Penalize large tilt with a dead band for normal gait and slopes."""
    up = -env.scene["robot"].data.projected_gravity_b[:, 2]
    return ((safe_up - up) / (safe_up - critical_up)).clamp(0.0, 1.0).square()


def fall_event(env) -> torch.Tensor:
    """One-off fall cost: RewardManager multiplies by step_dt, so divide it out here.

    Time-outs are not ``terminated``; a real fall on the last step still counts.
    """
    return env.termination_manager.terminated.float() / env.step_dt


@configclass
class StickRecoveryRewardsCfg(RewardsCfg):
    """Stick's reward revision: the original terms plus fall and stability costs."""

    progress = RewTerm(
        func=capped_target_speed, weight=1.0, params={"speed_cap": 4.5, "target_pos": TARGET_POS}
    )
    fall = RewTerm(func=fall_event, weight=-10.0)
    clearance_risk = RewTerm(
        func=low_clearance_risk,
        weight=-2.0,
        params={"safe_height": 0.48, "fall_height": 0.31, "sensor_cfg": SceneEntityCfg(HEIGHT_RAY)},
    )
    tilt_risk = RewTerm(func=tilt_risk, weight=-2.0, params={"safe_up": 0.93, "critical_up": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    body_sway = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)


##
# My Robust42 randomization
##


ROBUST42_NOISE = {
    "base_height": 0.01,
    "base_lin_vel": 0.05,
    "base_ang_vel": 0.10,
    "base_yaw_roll": 0.02,
    "base_angle_to_target": 0.02,
    "base_up_proj": 0.01,
    "base_heading_proj": 0.01,
    "joint_pos_norm": 0.01,
    "joint_vel_rel": 0.10,
    "feet_body_forces": 0.10,
    "actions": 0.01,
}
"""Bounded uniform observation noise of the submitted Robust42 (``RobustObservationCfg``)."""


def use_robust42_randomization(env_cfg: AntEnvCfg) -> None:
    """Robust42 events (friction, mass, COM, reset perturbation, pushes) and observation noise."""
    env_cfg.events = RobustEventCfg()
    policy = env_cfg.observations.policy
    policy.enable_corruption = True
    for name, magnitude in ROBUST42_NOISE.items():
        getattr(policy, name).noise = Unoise(n_min=-magnitude, n_max=magnitude)


##
# Terrains
##


def _random_uniform(lo: float, hi: float, *, downsampled: float | None = None, step: float = 0.01):
    kwargs = {} if downsampled is None else {"downsampled_scale": downsampled}
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(lo, hi), noise_step=step, border_width=0.25, **kwargs
    )


def _wave(amplitude_range: tuple[float, float], num_waves: int):
    return terrain_gen.HfWaveTerrainCfg(
        proportion=1.0, amplitude_range=amplitude_range, num_waves=num_waves, border_width=0.25
    )


def _slope(slope_range: tuple[float, float], inverted: bool = False):
    cls = terrain_gen.HfInvertedPyramidSlopedTerrainCfg if inverted else terrain_gen.HfPyramidSlopedTerrainCfg
    return cls(proportion=1.0, slope_range=slope_range, platform_width=2.0, border_width=0.25)


def _stairs(step: float, inverted: bool = False):
    cls = terrain_gen.MeshInvertedPyramidStairsTerrainCfg if inverted else terrain_gen.MeshPyramidStairsTerrainCfg
    return cls(proportion=1.0, step_height_range=(step, step), step_width=0.3, platform_width=3.0, border_width=1.0)


def _boxes(height: float, grid_width: float = 0.45):
    # Every grid cell is shifted uniformly in [-height, height].
    return terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.0, grid_width=grid_width, grid_height_range=(height, height), platform_width=2.0
    )


def _with_proportion(cfg, proportion: float):
    cfg = copy.deepcopy(cfg)
    cfg.proportion = proportion
    return cfg


def stick_terrains() -> dict:
    """Stick's training mix: random rough +-5 cm, waves 5-15 cm, slopes 0.05-0.15 up and down.

    Stick used 6 m tiles with four waves; seven waves on a 10 m tile keep his ~1.5 m wavelength.
    """
    return {
        "stick_random_rough": _with_proportion(_random_uniform(-0.05, 0.05, downsampled=0.25), 0.5),
        "stick_waves": _with_proportion(_wave((0.05, 0.15), num_waves=7), 0.3),
        "stick_slope": _with_proportion(_slope((0.05, 0.15)), 0.1),
        "stick_slope_inv": _with_proportion(_slope((0.05, 0.15), inverted=True), 0.1),
    }


def lim_terrains() -> dict:
    """Lim's final training terrain ("Oracle"): random boxes of +-10 cm only."""
    return {"lim_boxes": _boxes(0.10)}


def stick_lim_terrains() -> dict:
    """Both teammates' training terrains, half boxes and half Stick's mix."""
    mixed = {name: _with_proportion(cfg, cfg.proportion * 0.5) for name, cfg in stick_terrains().items()}
    mixed["lim_boxes"] = _with_proportion(_boxes(0.10), 0.5)
    return mixed


def mine_terrains() -> dict:
    """My v5 rough-lane families on random tiles: bumps, slopes and stairs up/down, waves, obstacles,
    stepping stones and flat ground (Week03-Ant-Rough-Lanes-v5).

    v5 placed them in homogeneous lanes behind flat entrance portals on 8 m tiles.  Here every tile
    keeps a 2 m central start platform for obstacles/stones and obstacle density is scaled to 10 m tiles.
    """
    p = 1.0 / 9.0
    return {
        "mine_flat": terrain_gen.MeshPlaneTerrainCfg(proportion=p),
        "mine_rough": HfSymmetricNoiseTerrainCfg(
            proportion=p, amplitude_range=(0.02, 0.12), downsampled_scale=0.25, border_width=0.25
        ),
        "mine_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=p, slope_range=(0.05, 0.45), platform_width=1.0, border_width=0.25
        ),
        "mine_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=p, slope_range=(0.05, 0.45), platform_width=1.0, border_width=0.25
        ),
        "mine_stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=p, step_height_range=(0.03, 0.15), step_width=0.5, platform_width=1.0, border_width=0.25
        ),
        "mine_stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=p, step_height_range=(0.03, 0.15), step_width=0.5, platform_width=1.0, border_width=0.25
        ),
        "mine_waves": terrain_gen.HfWaveTerrainCfg(
            proportion=p, amplitude_range=(0.04, 0.16), num_waves=4, border_width=0.25
        ),
        "mine_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=p,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0),
            obstacle_height_range=(0.04, 0.22),
            num_obstacles=47,
            platform_width=2.0,
            border_width=0.25,
        ),
        "mine_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=p,
            stone_height_max=0.05,
            stone_width_range=(0.5, 1.0),
            stone_distance_range=(0.05, 0.30),
            holes_depth=-0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
    }


def all_terrains() -> dict:
    """One third each: Stick's mix, Lim's boxes and my v5 families."""
    mixed = {}
    for builder in (stick_terrains, lim_terrains, mine_terrains):
        parts = builder()
        total = sum(cfg.proportion for cfg in parts.values())
        for name, cfg in parts.items():
            mixed[name] = _with_proportion(cfg, cfg.proportion / total / 3.0)
    return mixed


TRAIN_TERRAINS = {
    "flat": None,
    "stick": stick_terrains,
    "lim": lim_terrains,
    "sticklim": stick_lim_terrains,
    "mine": mine_terrains,
    "all": all_terrains,
}


class SeededTerrainGenerator(TerrainGenerator):
    """Terrain generator whose whole geometry follows ``cfg.seed``, built on the simulation GPU.

    Isaac Lab draws tile types and difficulties from ``cfg.seed``, but the sub-terrain functions use
    NumPy's global generator (bumps, obstacles, stones) or torch on the *current* CUDA device (random
    box heights), i.e. they follow the environment seed and default to GPU 0.  Here both generators are
    seeded from the terrain seed inside a forked RNG scope (restored afterwards), and the current CUDA
    device is the simulation device, so no other GPU is touched.
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        torch_device = torch.device(device)
        cuda_index = None
        if torch_device.type == "cuda":
            cuda_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=[] if cuda_index is None else [cuda_index]):
            if cuda_index is not None:
                torch.cuda.set_device(cuda_index)
            if cfg.seed is not None:
                np.random.seed(cfg.seed)
                torch.random.default_generator.manual_seed(cfg.seed)
                if cuda_index is not None:
                    torch.cuda.manual_seed(cfg.seed)
            try:
                super().__init__(cfg, device)
            finally:
                np.random.set_state(numpy_state)


def strip_generator(sub_terrains: dict, *, seed: int, difficulty_range=(0.0, 1.0)) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        class_type=SeededTerrainGenerator,
        seed=seed,
        size=(STRIP_TILE, STRIP_TILE),
        border_width=20.0,
        num_rows=STRIP_ROWS,
        num_cols=STRIP_COLS,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        difficulty_range=difficulty_range,
        sub_terrains=copy.deepcopy(sub_terrains),
    )


def apply_terrain(
    env_cfg: AntEnvCfg,
    sub_terrains: dict | None,
    *,
    seed: int,
    difficulty_range=(0.0, 1.0),
    friction: float = 1.0,
    combine_mode: str = "average",
) -> None:
    """Replace the ground with a plane (``None``) or a 160 m strip of the given sub-terrains.

    Every robot starts on the first row, spread over the 20 lanes, so its whole episode is spent on
    the requested terrain.  Only the ground material is changed for friction conditions; the robot
    keeps the default material, as in the stock task.
    """
    terrain_cfg = env_cfg.scene.terrain
    terrain_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=combine_mode,
        restitution_combine_mode="average",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )
    if sub_terrains is None:
        terrain_cfg.terrain_type = "plane"
        terrain_cfg.terrain_generator = None
        return
    terrain_cfg.terrain_type = "generator"
    terrain_cfg.terrain_generator = strip_generator(sub_terrains, seed=seed, difficulty_range=difficulty_range)
    terrain_cfg.max_init_terrain_level = 0
    terrain_cfg.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.33, 0.36))


##
# Demo evaluation suite (fixed before any v28 training result was observed)
##


@dataclass(frozen=True)
class EvalCondition:
    name: str
    category: str
    builder: object | None
    friction: float = 1.0
    combine_mode: str = "average"
    difficulty_range: tuple[float, float] = (1.0, 1.0)
    description: str = ""

    def sub_terrains(self) -> dict | None:
        if self.builder is None:
            return None
        built = self.builder()
        return built if isinstance(built, dict) else {self.name: built}


def _isaaclab_rough() -> dict:
    return copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains)


def _obstacles():
    return terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=1.0,
        obstacle_height_mode="choice",
        obstacle_width_range=(0.4, 1.2),
        obstacle_height_range=(0.10, 0.10),
        num_obstacles=40,
        platform_width=2.0,
        border_width=0.25,
    )


def _rails():
    return terrain_gen.MeshRailsTerrainCfg(
        proportion=1.0, rail_thickness_range=(0.2, 0.2), rail_height_range=(0.08, 0.08), platform_width=2.0
    )


def _gaps():
    return terrain_gen.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.2, 0.2), platform_width=3.0)


def _pits():
    return terrain_gen.MeshPitTerrainCfg(proportion=1.0, pit_depth_range=(0.15, 0.15), platform_width=3.0)


def _stones():
    return terrain_gen.HfSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_height_max=0.03,
        stone_width_range=(0.8, 1.2),
        stone_distance_range=(0.05, 0.10),
        holes_depth=-0.3,
        platform_width=2.0,
        border_width=0.25,
    )


def _cylinders():
    obj = terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=40, height=0.10, radius=0.15)
    return terrain_gen.MeshRepeatedCylindersTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=obj, object_params_end=obj
    )


def _cones():
    obj = terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=30, height=0.12, radius=0.30)
    return terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=obj, object_params_end=obj
    )


def _tilted_blocks():
    obj = terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=40, height=0.08, size=(0.4, 0.4), max_yx_angle=15.0
    )
    return terrain_gen.MeshRepeatedBoxesTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=obj, object_params_end=obj
    )


def _platform():
    return terrain_gen.MeshBoxTerrainCfg(proportion=1.0, box_height_range=(0.10, 0.10), platform_width=5.0)


EVAL_CONDITIONS: tuple[EvalCondition, ...] = (
    # flat and friction
    EvalCondition("flat", "flat_friction", None, description="stock plane, friction 1.0"),
    EvalCondition("flat_mu05", "flat_friction", None, friction=0.5, description="plane, ground 0.5 average"),
    EvalCondition("flat_mu01", "flat_friction", None, friction=0.1, description="plane, ground 0.1 average"),
    EvalCondition(
        "flat_mu02_mult", "flat_friction", None, friction=0.2, combine_mode="multiply",
        description="plane, effective friction 0.2 (multiply)",
    ),
    # boxes (the lecture's example unseen terrain is random boxes)
    EvalCondition("boxes_5", "boxes", lambda: _boxes(0.05), description="random boxes +-5 cm"),
    EvalCondition("boxes_10", "boxes", lambda: _boxes(0.10), description="random boxes +-10 cm"),
    EvalCondition("boxes_15", "boxes", lambda: _boxes(0.15), description="random boxes +-15 cm"),
    EvalCondition("boxes_fine_10", "boxes", lambda: _boxes(0.10, 0.30), description="0.3 m boxes +-10 cm"),
    EvalCondition(
        "boxes_10_mu02", "boxes", lambda: _boxes(0.10), friction=0.2, description="boxes +-10 cm, ground 0.2 average"
    ),
    EvalCondition(
        "boxes_10_mu02_mult", "boxes", lambda: _boxes(0.10), friction=0.2, combine_mode="multiply",
        description="boxes +-10 cm, effective friction 0.2 (multiply)",
    ),
    # rough and waves
    EvalCondition("rough_5", "rough_wave", lambda: _random_uniform(0.0, 0.05), description="uniform 0-5 cm"),
    EvalCondition("rough_10", "rough_wave", lambda: _random_uniform(0.0, 0.10), description="uniform 0-10 cm"),
    EvalCondition(
        "stick_rough", "rough_wave", stick_terrains, difficulty_range=(0.0, 1.0),
        description="Stick's training mix (rough +-5 cm, waves 5-15 cm, slopes 0.05-0.15)",
    ),
    EvalCondition("wave_15", "rough_wave", lambda: _wave((0.15, 0.15), 4), description="waves 15 cm, 2.5 m period"),
    # slopes and stairs
    EvalCondition("slope_20", "slope_stairs", lambda: _slope((0.2, 0.2)), description="pyramid slope 0.2"),
    EvalCondition(
        "slope_inv_20", "slope_stairs", lambda: _slope((0.2, 0.2), inverted=True), description="inverted slope 0.2"
    ),
    EvalCondition("stairs_10", "slope_stairs", lambda: _stairs(0.10), description="pyramid stairs 10 cm"),
    EvalCondition(
        "stairs_inv_10", "slope_stairs", lambda: _stairs(0.10, inverted=True), description="inverted stairs 10 cm"
    ),
    EvalCondition(
        "isaaclab_rough", "slope_stairs", _isaaclab_rough, difficulty_range=(0.0, 1.0),
        description="Isaac Lab ROUGH_TERRAINS_CFG mix, random difficulty",
    ),
    # discrete obstacles
    EvalCondition("obstacles_10", "obstacles", _obstacles, description="discrete obstacles 10 cm"),
    EvalCondition("rails_8", "obstacles", _rails, description="8 cm rails"),
    EvalCondition("cylinders_10", "obstacles", _cylinders, description="40 cylinders 10 cm"),
    EvalCondition("cones_12", "obstacles", _cones, description="30 cones 12 cm"),
    EvalCondition("tilted_blocks_8", "obstacles", _tilted_blocks, description="40 tilted blocks 8 cm"),
    EvalCondition("platform_10", "obstacles", _platform, description="10 cm platform"),
    # gaps and holes
    EvalCondition("gaps_20", "gaps_holes", _gaps, description="20 cm gap ring"),
    EvalCondition("pits_15", "gaps_holes", _pits, description="start in a 15 cm pit"),
    EvalCondition("stones", "gaps_holes", _stones, description="stepping stones, 30 cm holes"),
)

EVAL_BY_NAME = {condition.name: condition for condition in EVAL_CONDITIONS}

HOME_CONDITIONS = ("flat", "boxes_10", "stick_rough")
"""Exact training terrain of some recipe (flat, Lim's boxes, Stick's mix); excluded in a robustness view."""


def apply_eval_condition(env_cfg: AntEnvCfg, name: str, terrain_seed: int) -> EvalCondition:
    condition = EVAL_BY_NAME[name]
    apply_terrain(
        env_cfg,
        condition.sub_terrains(),
        seed=terrain_seed,
        difficulty_range=condition.difficulty_range,
        friction=condition.friction,
        combine_mode=condition.combine_mode,
    )
    return condition


##
# Environment configurations
##


@configclass
class ComboV28PlayEnvCfg(AntEnvCfg):
    """Scoring/submission task: original rewards and events, ground-relative height, swappable terrain.

    The default ground is the +-10 cm random boxes, similar to the lecture's example unseen terrain.
    Replace ``scene.terrain`` (prim path ``/World/ground``) to evaluate another terrain.
    """

    def __post_init__(self):
        super().__post_init__()
        use_relative_height(self)
        apply_eval_condition(self, "boxes_10", SELECTION_TERRAIN_SEED)


TRAIN_DR = ("d0", "d1")
TRAIN_REWARDS = ("stock", "recovery")


def configure_train(env_cfg: AntEnvCfg, terrain: str, dr: str, reward: str) -> None:
    if terrain not in TRAIN_TERRAINS or dr not in TRAIN_DR or reward not in TRAIN_REWARDS:
        raise ValueError(f"unknown v28 training variant: {terrain}/{dr}/{reward}")
    use_relative_height(env_cfg)
    builder = TRAIN_TERRAINS[terrain]
    apply_terrain(
        env_cfg,
        None if builder is None else builder(),
        seed=TRAIN_TERRAIN_SEED,
        difficulty_range=(1.0, 1.0) if terrain == "lim" else (0.0, 1.0),
    )
    if dr == "d1":
        use_robust42_randomization(env_cfg)
    else:
        env_cfg.events = EventCfg()
    if reward == "recovery":
        env_cfg.rewards = StickRecoveryRewardsCfg()


def train_cfg_name(terrain: str, dr: str, reward: str) -> str:
    return f"ComboV28{terrain.capitalize()}{dr.upper()}{reward.capitalize()}EnvCfg"


def _make_train_cfg(terrain: str, dr: str, reward: str):
    def __post_init__(self):
        AntEnvCfg.__post_init__(self)
        configure_train(self, terrain, dr, reward)

    name = train_cfg_name(terrain, dr, reward)
    cls = configclass(type(name, (AntEnvCfg,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls
    return name


TRAIN_VARIANTS = tuple(
    (terrain, dr, reward) for terrain in TRAIN_TERRAINS for dr in TRAIN_DR for reward in TRAIN_REWARDS
)
for _variant in TRAIN_VARIANTS:
    _make_train_cfg(*_variant)
