"""Ant environment variants used for the robustness experiment and demo.

The course and v0--v2 terrain variants preserve the 60-dimensional observation
and 8-dimensional action spaces. The v3 extreme terrain variant appends a
54-dimensional local height scan (114D total) while retaining the same action
interface. Terrain variants report base height relative to the local terrain
origin so elevated patches keep the same semantic observation as the original
ground plane.
"""

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfWaveTerrainCfg,
    MeshBoxTerrainCfg,
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import (
    ActionsCfg,
    AntEnvCfg,
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp


MULTI_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(6.0, 6.0),
    border_width=8.0,
    num_rows=8,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.005, 0.055),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "slope": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.02, 0.16),
            platform_width=1.5,
            border_width=0.25,
        ),
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.015, 0.065),
            step_width=0.5,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
    },
)

DEMO_TERRAIN_GENERATOR_CFG = MULTI_TERRAIN_GENERATOR_CFG.replace(
    seed=43,
    num_rows=1,
    difficulty_range=(0.25, 0.25),
)

MILD_TERRAIN_GENERATOR_CFG = MULTI_TERRAIN_GENERATOR_CFG.replace(
    seed=44,
    difficulty_range=(0.0, 0.45),
)

# A deliberately broader distribution for the second robustness pass.  In addition to the
# original flat/rough/slope/stairs set, this includes undulating ground, discrete blocks, and
# stepping stones with shallow gaps.  The ranges stay within a scale the Ant can physically
# negotiate while exposing different contact patterns during the same rollout.
COMPLEX_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=45,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=8.0,
    num_rows=8,
    num_cols=7,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0 / 7.0),
        "rough": HfRandomUniformTerrainCfg(
            proportion=1.0 / 7.0,
            noise_range=(0.01, 0.09),
            noise_step=0.01,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "slope": HfPyramidSlopedTerrainCfg(
            proportion=1.0 / 7.0,
            slope_range=(0.05, 0.28),
            platform_width=1.2,
            border_width=0.25,
        ),
        "stairs": HfPyramidStairsTerrainCfg(
            proportion=1.0 / 7.0,
            step_height_range=(0.025, 0.08),
            step_width=0.5,
            platform_width=1.2,
            border_width=0.25,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=1.0 / 7.0,
            amplitude_range=(0.025, 0.10),
            num_waves=4,
            border_width=0.25,
        ),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 7.0,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.35, 0.75),
            obstacle_height_range=(0.06, 0.20),
            num_obstacles=18,
            platform_width=1.5,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=1.0 / 7.0,
            stone_height_max=0.12,
            stone_width_range=(0.55, 0.90),
            stone_distance_range=(0.15, 0.35),
            holes_depth=-0.20,
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)

COMPLEX_MILD_TERRAIN_GENERATOR_CFG = COMPLEX_TERRAIN_GENERATOR_CFG.replace(
    seed=46,
    difficulty_range=(0.0, 0.35),
)

# Extreme v3 terrain set.  The three mesh terrains (pit, gap, and stacked boxes) create
# discontinuous contact and elevation changes that height-field roughness alone cannot expose.
# Difficulty is still curriculum-controlled so the policy sees recoverable examples before the
# largest gaps and steps.
EXTREME_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=47,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=8.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.02, 0.14),
            noise_step=0.01,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "steep_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.10, 0.42),
            platform_width=1.0,
            border_width=0.25,
        ),
        "deep_stairs": HfPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.14),
            step_width=0.45,
            platform_width=1.0,
            border_width=0.25,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.06, 0.16),
            num_waves=5,
            border_width=0.25,
        ),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.40, 0.90),
            obstacle_height_range=(0.10, 0.28),
            num_obstacles=22,
            platform_width=1.4,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.18,
            stone_width_range=(0.45, 0.80),
            stone_distance_range=(0.25, 0.50),
            holes_depth=-0.35,
            platform_width=1.4,
            border_width=0.25,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.20, 0.55),
            platform_width=1.4,
            double_pit=True,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.25, 0.65),
            platform_width=1.4,
        ),
        "boxes": MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.18, 0.45),
            platform_width=1.8,
            double_box=True,
        ),
    },
)

EXTREME_MILD_TERRAIN_GENERATOR_CFG = EXTREME_TERRAIN_GENERATOR_CFG.replace(
    seed=48,
    difficulty_range=(0.0, 0.25),
)

MULTI_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MULTI_TERRAIN_GENERATOR_CFG,
    max_init_terrain_level=7,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    visual_material=None,
    debug_vis=False,
)

MILD_TERRAIN_IMPORTER_CFG = MULTI_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=MILD_TERRAIN_GENERATOR_CFG,
)

COMPLEX_TERRAIN_IMPORTER_CFG = MULTI_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=COMPLEX_TERRAIN_GENERATOR_CFG,
)

COMPLEX_MILD_TERRAIN_IMPORTER_CFG = COMPLEX_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=COMPLEX_MILD_TERRAIN_GENERATOR_CFG,
)

EXTREME_TERRAIN_IMPORTER_CFG = MULTI_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=EXTREME_TERRAIN_GENERATOR_CFG,
)

EXTREME_MILD_TERRAIN_IMPORTER_CFG = EXTREME_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=EXTREME_MILD_TERRAIN_GENERATOR_CFG,
    max_init_terrain_level=2,
)

# Intermediate hand-off stage.  The generator still contains the complete extreme range, but
# resets initially sample only the first five rows.  This avoids destroying the warm-start policy
# when discontinuous pits/gaps are introduced, while exposing more than the mild warm-up range.
EXTREME_APPROACH_TERRAIN_IMPORTER_CFG = EXTREME_TERRAIN_IMPORTER_CFG.replace(max_init_terrain_level=4)

# Final ramp stage: only the last generator row (the most destructive geometry) is held out
# during this pass.  It gives the policy a stable bridge from the approach stage to full reset
# coverage instead of switching all environments to row 7 at once.
EXTREME_RAMP_TERRAIN_IMPORTER_CFG = EXTREME_TERRAIN_IMPORTER_CFG.replace(max_init_terrain_level=6)

# v4 keeps the exact v3 geometry and observation width, but uses a forward-biased
# scanner and a reset curriculum that reacts to how far the ant actually travelled.
# Keeping the generator unchanged makes v3/v4 metrics directly comparable while the
# policy gets a longer useful preview of the discontinuous pit/gap/box transitions.
EXTREME_RECOVERY_TERRAIN_IMPORTER_CFG = EXTREME_TERRAIN_IMPORTER_CFG
EXTREME_RECOVERY_MILD_TERRAIN_IMPORTER_CFG = EXTREME_MILD_TERRAIN_IMPORTER_CFG
EXTREME_RECOVERY_APPROACH_TERRAIN_IMPORTER_CFG = EXTREME_APPROACH_TERRAIN_IMPORTER_CFG
EXTREME_RECOVERY_RAMP_TERRAIN_IMPORTER_CFG = EXTREME_RAMP_TERRAIN_IMPORTER_CFG

# Focused geometry stage: keep every family represented, but allocate more columns
# to the two discontinuities that dominate failures in the v3 evaluation.  This is
# a training-only distribution; the v4 evaluation task remains the balanced ten-
# family generator above.
EXTREME_RECOVERY_FOCUS_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=49,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=8.0,
    num_rows=8,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.05),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.05,
            noise_range=(0.02, 0.14),
            noise_step=0.01,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "steep_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.10, 0.42),
            platform_width=1.0,
            border_width=0.25,
        ),
        "deep_stairs": HfPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.14),
            step_width=0.45,
            platform_width=1.0,
            border_width=0.25,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.06, 0.16),
            num_waves=5,
            border_width=0.25,
        ),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.40, 0.90),
            obstacle_height_range=(0.10, 0.28),
            num_obstacles=22,
            platform_width=1.4,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.10,
            stone_height_max=0.18,
            stone_width_range=(0.45, 0.80),
            stone_distance_range=(0.25, 0.50),
            holes_depth=-0.35,
            platform_width=1.4,
            border_width=0.25,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.25,
            pit_depth_range=(0.20, 0.55),
            platform_width=1.4,
            double_pit=True,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.25, 0.65),
            platform_width=1.4,
        ),
        "boxes": MeshBoxTerrainCfg(
            proportion=0.10,
            box_height_range=(0.18, 0.45),
            platform_width=1.8,
            double_box=True,
        ),
    },
)
EXTREME_RECOVERY_FOCUS_TERRAIN_IMPORTER_CFG = EXTREME_TERRAIN_IMPORTER_CFG.replace(
    terrain_generator=EXTREME_RECOVERY_FOCUS_TERRAIN_GENERATOR_CFG,
    max_init_terrain_level=2,
)


def base_height_above_terrain_origin(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Return torso height relative to the environment's terrain origin."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def root_height_below_terrain_origin(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terminate falls using local terrain height rather than world height."""
    asset = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return relative_height < minimum_height


def extreme_terrain_levels(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_level: int | None = None,
):
    """Move successful rollouts up the terrain curriculum and failed ones down.

    The classic Ant task has no velocity command manager, so the upstream velocity
    curriculum cannot be reused directly.  This term uses forward displacement from
    the local terrain origin instead.  It is intentionally evaluated only after the
    first simulator step so the initial random terrain level is not changed by the
    bootstrap reset.
    """
    terrain = env.scene.terrain
    if terrain.terrain_origins is None or env.common_step_counter == 0:
        return {"mean_level": float(terrain.terrain_levels.float().mean().item())}

    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return {"mean_level": float(terrain.terrain_levels.float().mean().item())}

    asset = env.scene[asset_cfg.name]
    forward_distance = asset.data.root_pos_w[ids, 0] - env.scene.env_origins[ids, 0]
    terrain_size_x = float(terrain.cfg.terrain_generator.size[0])
    # A successful traversal reaches the next terrain row; a short failed rollout
    # is returned to an easier row.  The hysteresis prevents oscillation near the
    # boundary and lets a policy practice the same hard family repeatedly.
    move_up = forward_distance > terrain_size_x * 0.45
    move_down = (forward_distance < terrain_size_x * 0.18) & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)
    if max_level is not None:
        max_level = min(int(max_level), terrain.max_terrain_level - 1)
        capped_levels = torch.clamp(terrain.terrain_levels[ids], max=max_level)
        terrain.terrain_levels[ids] = capped_levels
        terrain.env_origins[ids] = terrain.terrain_origins[capped_levels, terrain.terrain_types[ids]]
    return {
        "mean_level": float(terrain.terrain_levels.float().mean().item()),
        "reset_mean_forward_distance": float(forward_distance.mean().item()),
    }


def pit_clearance_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward torso clearance after climbing out of the generated pit.

    The term is active only for the pit terrain family and is bounded so it cannot
    dominate forward progress.  It uses the same local terrain origin as the
    terrain-relative height observation, rather than a world-height shortcut.
    """
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    if generator is None or "pit" not in generator.sub_terrains:
        return torch.zeros(env.num_envs, device=env.device)
    pit_index = list(generator.sub_terrains).index("pit")
    pit_mask = terrain.terrain_types == pit_index
    asset = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    clearance = torch.clamp(relative_height - 0.45, min=0.0, max=0.65)
    return torch.where(pit_mask, clearance, torch.zeros_like(clearance))


@configclass
class MultiTerrainSceneCfg(MySceneCfg):
    """Four reproducible terrain families arranged by generator column."""

    terrain = MULTI_TERRAIN_IMPORTER_CFG


@configclass
class MultiTerrainDemoSceneCfg(MultiTerrainSceneCfg):
    """Use the easiest row so four demo envs show one terrain family each."""

    terrain = MULTI_TERRAIN_IMPORTER_CFG.replace(
        terrain_generator=DEMO_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=0,
        debug_vis=True,
    )


@configclass
class MildTerrainSceneCfg(MultiTerrainSceneCfg):
    """Training distribution concentrated on the moderate demo range."""

    terrain = MILD_TERRAIN_IMPORTER_CFG


@configclass
class ComplexTerrainSceneCfg(MySceneCfg):
    """Seven-family terrain distribution used by the complex v2 task."""

    terrain = COMPLEX_TERRAIN_IMPORTER_CFG


@configclass
class ComplexMildTerrainSceneCfg(ComplexTerrainSceneCfg):
    """Lower-difficulty warm-up distribution for the complex terrain curriculum."""

    terrain = COMPLEX_MILD_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeTerrainSceneCfg(MySceneCfg):
    """Ten-family terrain distribution used by the extreme v3 task."""

    terrain = EXTREME_TERRAIN_IMPORTER_CFG
    # A small body-mounted scan gives the policy short-horizon preview of pits, gaps, and
    # box edges.  The scanner is intentionally kept local (1.6 x 1.0 m) so it remains useful
    # under yaw randomization and does not leak the terrain generator's privileged map.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class ExtremeMildTerrainSceneCfg(ExtremeTerrainSceneCfg):
    """Recoverable warm-up range for the extreme terrain curriculum."""

    terrain = EXTREME_MILD_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeApproachTerrainSceneCfg(ExtremeTerrainSceneCfg):
    """Intermediate reset range before exposing every extreme terrain row."""

    terrain = EXTREME_APPROACH_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeRampTerrainSceneCfg(ExtremeTerrainSceneCfg):
    """Final curriculum ramp covering terrain rows 0--6."""

    terrain = EXTREME_RAMP_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeRecoveryTerrainSceneCfg(ExtremeTerrainSceneCfg):
    """v4 scene with a forward-biased preview of upcoming terrain."""

    terrain = EXTREME_RECOVERY_TERRAIN_IMPORTER_CFG
    # The original v3 scan is centered on the torso.  For forward locomotion this
    # spends half of its rays on terrain that has already been crossed.  Shifting the
    # same 54 rays forward keeps the 114D checkpoint interface compatible while
    # exposing a full body-length preview before a pit, gap, or box edge.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.60, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class ExtremeRecoveryMildTerrainSceneCfg(ExtremeRecoveryTerrainSceneCfg):
    """Recoverable v4 warm-up terrain range."""

    terrain = EXTREME_RECOVERY_MILD_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeRecoveryApproachTerrainSceneCfg(ExtremeRecoveryTerrainSceneCfg):
    """v4 intermediate terrain range covering rows 0--4."""

    terrain = EXTREME_RECOVERY_APPROACH_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeRecoveryRampTerrainSceneCfg(ExtremeRecoveryTerrainSceneCfg):
    """v4 final ramp covering rows 0--6."""

    terrain = EXTREME_RECOVERY_RAMP_TERRAIN_IMPORTER_CFG


@configclass
class ExtremeRecoveryFocusTerrainSceneCfg(ExtremeRecoveryTerrainSceneCfg):
    """Training scene that over-samples pits and gaps without hiding other families."""

    terrain = EXTREME_RECOVERY_FOCUS_TERRAIN_IMPORTER_CFG


@configclass
class FrictionEventCfg(EventCfg):
    """Randomize contact friction while leaving all other baseline terms intact."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.45, 1.35),
            "dynamic_friction_range": (0.35, 1.15),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class RobustObservationCfg(ObservationsCfg):
    """Apply small state-sensor perturbations without changing tensor shape."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            self.base_height.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.base_lin_vel.noise = Unoise(n_min=-0.05, n_max=0.05)
            self.base_ang_vel.noise = Unoise(n_min=-0.10, n_max=0.10)
            self.base_yaw_roll.noise = Unoise(n_min=-0.02, n_max=0.02)
            self.base_angle_to_target.noise = Unoise(n_min=-0.02, n_max=0.02)
            self.base_up_proj.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.base_heading_proj.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_pos_norm.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_vel_rel.noise = Unoise(n_min=-0.10, n_max=0.10)
            self.feet_body_forces.noise = Unoise(n_min=-0.10, n_max=0.10)
            self.actions.noise = Unoise(n_min=-0.01, n_max=0.01)

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerrainObservationCfg(ObservationsCfg):
    """Keep the original 60D observation but make height terrain-relative."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.base_height.func = base_height_above_terrain_origin

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobustTerrainObservationCfg(RobustObservationCfg):
    """Robust observation noise with terrain-relative torso height."""

    @configclass
    class PolicyCfg(RobustObservationCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.base_height.func = base_height_above_terrain_origin

    policy: PolicyCfg = PolicyCfg()


@configclass
class ExtremeTerrainObservationCfg(RobustTerrainObservationCfg):
    """Noisy proprioception plus a clipped local height scan for discontinuities."""

    @configclass
    class PolicyCfg(RobustTerrainObservationCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class ExtremeRecoveryCleanObservationCfg(TerrainObservationCfg):
    """Noise-free 114D observations used to learn terrain geometry first."""

    @configclass
    class PolicyCfg(TerrainObservationCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class ExtremeRecoveryObservationCfg(ExtremeTerrainObservationCfg):
    """Front-scanned observations with the small v3 sensor noise retained."""


@configclass
class TerrainTerminationsCfg(TerminationsCfg):
    """Detect a fall relative to the current terrain patch."""

    torso_height = DoneTerm(
        func=root_height_below_terrain_origin,
        params={"minimum_height": 0.31},
    )


@configclass
class TerrainPostureTerminationsCfg(TerminationsCfg):
    """Detect actual overturning without assuming a flat world-height floor."""

    torso_height = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},
    )


@configclass
class ExtremeRecoveryTerminationsCfg(TerrainPostureTerminationsCfg):
    """Allow transient climbing pitch while still terminating a true overturn."""

    torso_height = DoneTerm(
        func=mdp.bad_orientation,
        # v3 ended a recovery at roughly 69 degrees.  The wider margin allows
        # the torso to pitch over a pit lip or tall box and recover on the next
        # contact, without accepting a fully inverted body.
        params={"limit_angle": 1.42},
    )


@configclass
class ExtremeTerrainCurriculumCfg:
    """Adaptive row curriculum for the v4 training stages."""

    terrain_levels = CurrTerm(func=extreme_terrain_levels, params={"max_level": 7})


@configclass
class ExtremeRecoveryMildCurriculumCfg(ExtremeTerrainCurriculumCfg):
    """Keep the warm-up policy within terrain rows 0--2."""

    terrain_levels = CurrTerm(func=extreme_terrain_levels, params={"max_level": 2})


@configclass
class ExtremeRecoveryApproachCurriculumCfg(ExtremeTerrainCurriculumCfg):
    """Keep the approach policy within terrain rows 0--4."""

    terrain_levels = CurrTerm(func=extreme_terrain_levels, params={"max_level": 4})


@configclass
class ExtremeRecoveryRampCurriculumCfg(ExtremeTerrainCurriculumCfg):
    """Keep the ramp policy within terrain rows 0--6."""

    terrain_levels = CurrTerm(func=extreme_terrain_levels, params={"max_level": 6})


@configclass
class ExtremeRewardsCfg(RewardsCfg):
    """Leave enough reward margin for the high-energy recovery motions in v3."""

    # The v0/v1 energy terms strongly discourage the impulses needed to clear a gap or climb
    # a tall step.  These are still penalties, just small enough that survival and progress can
    # win when a recovery motion is the correct behavior.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.02, params={"gear_ratio": {".*": 15.0}})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.05,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


@configclass
class ExtremeRecoveryRewardsCfg(ExtremeRewardsCfg):
    """Recovery-friendly shaping for discontinuous terrain transitions."""

    # Keep forward progress and survival dominant, but do not charge the impulses
    # that are required for a gap jump or for climbing out of a pit.
    progress = RewTerm(func=mdp.progress_reward, weight=1.25, params={"target_pos": (1000.0, 0.0, 0.0)})
    alive = RewTerm(func=mdp.is_alive, weight=0.65)
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus,
        weight=0.65,
        params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0015)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.015, params={"gear_ratio": {".*": 15.0}})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.025,
        # Keep the denominator in the penalty well-conditioned.  A 0.995
        # threshold would divide by 0.005 and create rare, enormous negative
        # returns during robust fine-tuning.
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


@configclass
class ExtremeRecoveryPitFocusRewardsCfg(ExtremeRecoveryRewardsCfg):
    """Bounded pit-exit shaping for the short discontinuity repair stage."""

    pit_clearance = RewTerm(func=pit_clearance_reward, weight=0.25)


@configclass
class RobustEventCfg(FrictionEventCfg):
    """Randomize dynamics, resets, and intermittent disturbances."""

    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (0.80, 1.20),
            "operation": "scale",
        },
    )

    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.25, 0.25)},
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.25, 0.25),
            },
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}},
    )


@configclass
class ExtremeRecoveryGeometryEventCfg(EventCfg):
    """Reset-only events for the geometry warm-up stages.

    Dynamics and observation randomization are deliberately delayed until the policy
    can solve the terrain transitions.  This prevents a large early failure signal
    from teaching the policy to stop before a discontinuity.
    """


@configclass
class ExtremeRecoveryRobustEventCfg(RobustEventCfg):
    """Late-stage robustness randomization that preserves the learned gait."""

    # A slightly narrower range than the original v3 push keeps recovery motions
    # learnable while still testing friction, mass, COM, reset, and impulse shifts.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={"velocity_range": {"x": (-0.45, 0.45), "y": (-0.45, 0.45)}},
    )


@configclass
class ExtremeRecoveryActionsCfg(ActionsCfg):
    """Permit the high-torque impulse needed to clear a gap or tall step."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=8.5)


@configclass
class LowFrictionEventCfg(EventCfg):
    """Held-out friction below the training randomization range."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.30, 0.30),
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class HeavyEventCfg(EventCfg):
    """Held-out 30% torso payload with an off-center mass."""

    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (1.30, 1.30),
            "operation": "scale",
        },
    )

    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (0.04, 0.04), "y": (-0.03, -0.03), "z": (0.0, 0.0)},
        },
    )


@configclass
class PushEventCfg(EventCfg):
    """Held-out lateral/longitudinal impulses stronger than training pushes."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.80, 0.80), "y": (-0.80, 0.80)}},
    )


@configclass
class BaselineAntEnvCfg(AntEnvCfg):
    """Exact course environment under a project-local task ID."""


@configclass
class FrictionAntEnvCfg(AntEnvCfg):
    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class RobustAntEnvCfg(AntEnvCfg):
    observations: RobustObservationCfg = RobustObservationCfg()
    events: RobustEventCfg = RobustEventCfg()


@configclass
class LowFrictionAntEnvCfg(AntEnvCfg):
    events: LowFrictionEventCfg = LowFrictionEventCfg()


@configclass
class HeavyAntEnvCfg(AntEnvCfg):
    events: HeavyEventCfg = HeavyEventCfg()


@configclass
class PushAntEnvCfg(AntEnvCfg):
    events: PushEventCfg = PushEventCfg()


@configclass
class TerrainAntEnvCfg(AntEnvCfg):
    """Evaluation terrain set with the course policy interface unchanged."""

    scene: MultiTerrainSceneCfg = MultiTerrainSceneCfg(num_envs=4096, env_spacing=6.0, clone_in_fabric=True)
    observations: TerrainObservationCfg = TerrainObservationCfg()
    terminations: TerrainTerminationsCfg = TerrainTerminationsCfg()


@configclass
class TerrainRobustAntEnvCfg(RobustAntEnvCfg):
    """Training terrain set plus the project domain-randomization terms."""

    scene: MultiTerrainSceneCfg = MultiTerrainSceneCfg(num_envs=4096, env_spacing=6.0, clone_in_fabric=True)
    observations: RobustTerrainObservationCfg = RobustTerrainObservationCfg()
    terminations: TerrainTerminationsCfg = TerrainTerminationsCfg()


@configclass
class MildTerrainAntEnvCfg(TerrainAntEnvCfg):
    """Moderate terrain curriculum used only to stabilize the visual demo."""

    scene: MildTerrainSceneCfg = MildTerrainSceneCfg(num_envs=4096, env_spacing=6.0, clone_in_fabric=True)


@configclass
class TerrainDemoAntEnvCfg(TerrainAntEnvCfg):
    """Four-env GUI layout: flat, rough, slope, and stairs from left to right."""

    scene: MultiTerrainDemoSceneCfg = MultiTerrainDemoSceneCfg(num_envs=4, env_spacing=6.0, clone_in_fabric=True)


@configclass
class TerrainPostureAntEnvCfg(TerrainAntEnvCfg):
    """Terrain task whose fall detector is independent of terrain elevation."""

    terminations: TerrainPostureTerminationsCfg = TerrainPostureTerminationsCfg()


@configclass
class MildTerrainPostureAntEnvCfg(MildTerrainAntEnvCfg):
    """Moderate curriculum with elevation-independent fall detection."""

    terminations: TerrainPostureTerminationsCfg = TerrainPostureTerminationsCfg()


@configclass
class ComplexTerrainPostureAntEnvCfg(TerrainPostureAntEnvCfg):
    """Complex terrain evaluation with posture termination and sensor/dynamics randomization."""

    scene: ComplexTerrainSceneCfg = ComplexTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=True,
    )
    observations: RobustTerrainObservationCfg = RobustTerrainObservationCfg()
    events: RobustEventCfg = RobustEventCfg()


@configclass
class ComplexMildTerrainPostureAntEnvCfg(ComplexTerrainPostureAntEnvCfg):
    """Mild complex curriculum used for warm-starting the full v2 terrain task."""

    scene: ComplexMildTerrainSceneCfg = ComplexMildTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=True,
    )


@configclass
class ExtremeTerrainPostureAntEnvCfg(TerrainPostureAntEnvCfg):
    """Extreme terrain task with posture termination and recovery-friendly rewards."""

    scene: ExtremeTerrainSceneCfg = ExtremeTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        # RayCaster reset bookkeeping needs materialized environment prims (Fabric cloning
        # exposes only env_0 to the sensor's parent-prim lookup).
        clone_in_fabric=False,
    )
    observations: ExtremeTerrainObservationCfg = ExtremeTerrainObservationCfg()
    rewards: ExtremeRewardsCfg = ExtremeRewardsCfg()
    events: RobustEventCfg = RobustEventCfg()


@configclass
class ExtremeMildTerrainPostureAntEnvCfg(ExtremeTerrainPostureAntEnvCfg):
    """Mild extreme terrain curriculum used for the v3 warm-up phase."""

    scene: ExtremeMildTerrainSceneCfg = ExtremeMildTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )


@configclass
class ExtremeApproachTerrainPostureAntEnvCfg(ExtremeTerrainPostureAntEnvCfg):
    """Intermediate v3 curriculum stage covering terrain rows 0--4."""

    scene: ExtremeApproachTerrainSceneCfg = ExtremeApproachTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )


@configclass
class ExtremeRampTerrainPostureAntEnvCfg(ExtremeTerrainPostureAntEnvCfg):
    """Final v3 ramp before the full row-7 extreme reset distribution."""

    scene: ExtremeRampTerrainSceneCfg = ExtremeRampTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )


@configclass
class ExtremeRecoveryTerrainPostureAntEnvCfg(ExtremeTerrainPostureAntEnvCfg):
    """v4 full extreme evaluation with forward scan and recovery shaping."""

    scene: ExtremeRecoveryTerrainSceneCfg = ExtremeRecoveryTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )
    observations: ExtremeRecoveryObservationCfg = ExtremeRecoveryObservationCfg()
    actions: ExtremeRecoveryActionsCfg = ExtremeRecoveryActionsCfg()
    rewards: ExtremeRecoveryRewardsCfg = ExtremeRecoveryRewardsCfg()
    terminations: ExtremeRecoveryTerminationsCfg = ExtremeRecoveryTerminationsCfg()
    events: ExtremeRecoveryRobustEventCfg = ExtremeRecoveryRobustEventCfg()


@configclass
class ExtremeRecoveryGeometryTerrainPostureAntEnvCfg(ExtremeRecoveryTerrainPostureAntEnvCfg):
    """Full v4 geometry stage without dynamics randomization."""

    observations: ExtremeRecoveryCleanObservationCfg = ExtremeRecoveryCleanObservationCfg()
    events: ExtremeRecoveryGeometryEventCfg = ExtremeRecoveryGeometryEventCfg()
    curriculum: ExtremeTerrainCurriculumCfg = ExtremeTerrainCurriculumCfg()


@configclass
class ExtremeRecoveryMildTerrainPostureAntEnvCfg(ExtremeRecoveryGeometryTerrainPostureAntEnvCfg):
    """v4 warm-up stage (rows 0--2) used before discontinuity training."""

    scene: ExtremeRecoveryMildTerrainSceneCfg = ExtremeRecoveryMildTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )
    curriculum: ExtremeRecoveryMildCurriculumCfg = ExtremeRecoveryMildCurriculumCfg()


@configclass
class ExtremeRecoveryApproachTerrainPostureAntEnvCfg(ExtremeRecoveryGeometryTerrainPostureAntEnvCfg):
    """v4 approach stage (rows 0--4) introducing the hard families."""

    scene: ExtremeRecoveryApproachTerrainSceneCfg = ExtremeRecoveryApproachTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )
    curriculum: ExtremeRecoveryApproachCurriculumCfg = ExtremeRecoveryApproachCurriculumCfg()


@configclass
class ExtremeRecoveryRampTerrainPostureAntEnvCfg(ExtremeRecoveryGeometryTerrainPostureAntEnvCfg):
    """v4 ramp stage (rows 0--6) before full difficulty."""

    scene: ExtremeRecoveryRampTerrainSceneCfg = ExtremeRecoveryRampTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )
    curriculum: ExtremeRecoveryRampCurriculumCfg = ExtremeRecoveryRampCurriculumCfg()


@configclass
class ExtremeRecoveryFocusTerrainPostureAntEnvCfg(ExtremeRecoveryGeometryTerrainPostureAntEnvCfg):
    """Pit/gap-focused geometry stage used to repair the remaining failure modes."""

    scene: ExtremeRecoveryFocusTerrainSceneCfg = ExtremeRecoveryFocusTerrainSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
        clone_in_fabric=False,
    )
    curriculum: ExtremeTerrainCurriculumCfg = ExtremeTerrainCurriculumCfg()


@configclass
class ExtremeRecoveryPitFocusTerrainPostureAntEnvCfg(ExtremeRecoveryFocusTerrainPostureAntEnvCfg):
    """Pit-focus repair task with bounded torso-clearance shaping."""

    rewards: ExtremeRecoveryPitFocusRewardsCfg = ExtremeRecoveryPitFocusRewardsCfg()


@configclass
class ExtremeRecoveryRobustTrainTerrainPostureAntEnvCfg(ExtremeRecoveryTerrainPostureAntEnvCfg):
    """v4 final robustness fine-tune with adaptive terrain levels."""

    curriculum: ExtremeTerrainCurriculumCfg = ExtremeTerrainCurriculumCfg()
