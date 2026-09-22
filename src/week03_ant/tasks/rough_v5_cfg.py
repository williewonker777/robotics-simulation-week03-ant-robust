"""v5 rough-terrain Ant tasks on treadmill lanes.

The policy interface, action scale, network and reward weights are those of the
course ``Isaac-Ant-v0`` task, so a v5 checkpoint loads and runs unchanged in the
course environment. See :mod:`week03_ant.tasks.lanes` for the lane mechanics.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfWaveTerrainCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import (
    AntEnvCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_cfg import RobustObservationCfg
from .lanes import (
    HfSymmetricNoiseTerrainCfg,
    LaneTerrainGeneratorCfg,
    lane_angle_to_target,
    lane_assign,
    lane_assign_weighted,
    lane_base_height,
    lane_centering_penalty,
    lane_heading_proj,
    lane_move_to_target_bonus,
    lane_out_of_bounds,
    lane_power_consumption,
    lane_progress_reward,
    lane_reset_root_state,
    lane_stall_penalty,
    lane_torso_below_minimum,
    lane_wrap,
    stone_body_height_penalty,
    stone_swing_clearance_penalty,
)

# Mesh families (terrain lanes) followed by the flat family on the course-style PhysX plane.
ROUGH_LANE_MESH_FAMILIES = ("rough", "slope", "stairs", "waves", "obstacles", "stepping_stones")
ROUGH_LANE_FAMILIES = ROUGH_LANE_MESH_FAMILIES + ("flat",)
ROUGH_LANE_DIFFICULTIES = (0.2, 0.4, 0.6, 0.8, 1.0)
FLAT_PLANE_HEIGHT = -50.0

# Course ground friction, shared by the terrain mesh and the flat-family plane.
COURSE_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)

# Each mesh lane is a flat entrance portal, six 8 m tiles of one family and level, and
# a flat exit portal (a 56 m treadmill period between the portal centres). All tiles
# are 0.1 m height fields, so every crossed seam joins matching edge vertices.
# Ranges below are the difficulty-0 and difficulty-1 values; the levels use 0.2--1.0.
ROUGH_LANES_GENERATOR_CFG = LaneTerrainGeneratorCfg(
    seed=51,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=8,
    num_cols=len(ROUGH_LANE_MESH_FAMILIES) * len(ROUGH_LANE_DIFFICULTIES),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=True,
    lane_difficulties=list(ROUGH_LANE_DIFFICULTIES),
    portal_sub_terrain="plain",
    plane_family="flat",
    plane_height=FLAT_PLANE_HEIGHT,
    lane_families={
        "rough": ["rough"],
        "slope": ["slope_up", "slope_down"],
        "stairs": ["stairs_up", "stairs_down"],
        "waves": ["waves"],
        "obstacles": ["obstacles"],
        "stepping_stones": ["stepping_stones"],
    },
    sub_terrains={
        # entrance/exit portals: flat 0.1 m height field
        "plain": HfSymmetricNoiseTerrainCfg(proportion=1.0, amplitude_range=(0.0, 0.0), border_width=0.25),
        # zero-mean bumps: +-4 cm (level 0) to +-12 cm (level 4)
        "rough": HfSymmetricNoiseTerrainCfg(
            proportion=1.0, amplitude_range=(0.02, 0.12), downsampled_scale=0.25, border_width=0.25
        ),
        # pyramids up to 0.45 rise/run (24 deg), crossed uphill then downhill
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.05, 0.45), platform_width=1.0, border_width=0.25
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.05, 0.45), platform_width=1.0, border_width=0.25
        ),
        # 0.5 m treads with 5.4 cm (level 0) to 15 cm (level 4) risers
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.03, 0.15), step_width=0.5, platform_width=1.0, border_width=0.25
        ),
        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.03, 0.15), step_width=0.5, platform_width=1.0, border_width=0.25
        ),
        "waves": HfWaveTerrainCfg(proportion=1.0, amplitude_range=(0.04, 0.16), num_waves=4, border_width=0.25),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0),
            obstacle_height_range=(0.04, 0.22),
            num_obstacles=30,
            platform_width=0.0,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.05,
            stone_width_range=(0.5, 1.0),
            stone_distance_range=(0.05, 0.30),
            holes_depth=-0.30,
            platform_width=0.0,
            border_width=0.25,
        ),
    },
)


@configclass
class RoughLaneSceneCfg(MySceneCfg):
    """Course Ant scene: terrain lanes plus the course ground plane for the flat family."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_LANES_GENERATOR_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=COURSE_GROUND_MATERIAL,
        debug_vis=False,
    )
    # Same spawner and material as the course `terrain_type="plane"` ground, moved below
    # the terrain so only flat-family Ants (spawned at FLAT_PLANE_HEIGHT) touch it.
    flat_plane = AssetBaseCfg(
        prim_path="/World/flatPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=COURSE_GROUND_MATERIAL, size=(2.0e6, 2.0e6)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLAT_PLANE_HEIGHT)),
    )


def _use_lane_terms(policy_cfg) -> None:
    """Swap the three target/height terms for their lane versions (60D layout unchanged)."""
    policy_cfg.base_height.func = lane_base_height
    policy_cfg.base_angle_to_target.func = lane_angle_to_target
    policy_cfg.base_angle_to_target.params = {}
    policy_cfg.base_heading_proj.func = lane_heading_proj
    policy_cfg.base_heading_proj.params = {}


@configclass
class RoughLaneObservationCfg(ObservationsCfg):
    """Noise-free course observations with lane-aware height and target terms."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            _use_lane_terms(self)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RoughLaneTrainObservationCfg(RobustObservationCfg):
    """Robust-study sensor noise with lane-aware terms; height noise covers ground-probe error."""

    @configclass
    class PolicyCfg(RobustObservationCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            _use_lane_terms(self)
            self.base_height.noise = Unoise(n_min=-0.02, n_max=0.02)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RoughLaneRewardsCfg(RewardsCfg):
    """Course reward weights; progress and heading follow the lane target."""

    progress = RewTerm(func=lane_progress_reward, weight=1.0)
    move_to_target = RewTerm(func=lane_move_to_target_bonus, weight=0.5, params={"threshold": 0.8})


@configclass
class RoughLaneTerminationsCfg(TerminationsCfg):
    """Course torso-height fall test against the local ground, plus an overturn test."""

    torso_height = DoneTerm(func=lane_torso_below_minimum, params={"minimum_height": 0.31})
    overturn = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


_LANE_RESET_JOINTS = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
)
_LANE_WRAP = EventTerm(func=lane_wrap, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True)


@configclass
class RoughLaneEventCfg:
    """Robust-study randomization (friction, torso mass/COM, reset state, pushes) on lanes.

    A quarter of the environments run on the flat family so the course flat-ground gait
    keeps being trained; the other families share the rest, levels cycled evenly.
    """

    lane_layout = EventTerm(
        func=lane_assign_weighted,
        mode="startup",
        params={"family_weights": {"flat": 2.0, "rough": 1.0, "slope": 1.0, "stairs": 1.0, "waves": 1.0,
                                   "obstacles": 1.0, "stepping_stones": 1.0}},
    )

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
        func=lane_reset_root_state,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-1.0, 1.0), "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05),
                           "yaw": (-0.25, 0.25)},
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "roll": (-0.15, 0.15),
                               "pitch": (-0.15, 0.15), "yaw": (-0.25, 0.25)},
        },
    )
    reset_robot_joints = _LANE_RESET_JOINTS
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}},
    )
    lane_wrap = _LANE_WRAP


@configclass
class RoughLaneEvalEventCfg:
    """Course-style evaluation resets: no dynamics randomization or pushes.

    Environments are split evenly over the seven families, levels cycled within each.
    """

    lane_layout = EventTerm(
        func=lane_assign_weighted,
        mode="startup",
        params={"family_weights": {family: 1.0 for family in ROUGH_LANE_FAMILIES}},
    )

    reset_base = EventTerm(
        func=lane_reset_root_state,
        mode="reset",
        params={"pose_range": {"y": (-1.0, 1.0)}, "velocity_range": {}},
    )
    reset_robot_joints = _LANE_RESET_JOINTS
    lane_wrap = _LANE_WRAP


@configclass
class RoughLaneDemoEventCfg(RoughLaneEvalEventCfg):
    """One environment per terrain family at a single difficulty level (replaces the layout)."""

    lane_layout = EventTerm(
        func=lane_assign,
        mode="startup",
        params={"family_levels": [(family, 3) for family in ROUGH_LANE_FAMILIES]},
    )


@configclass
class RoughLaneAntEnvCfg(AntEnvCfg):
    """v5 training task: course interface on seven terrain families x five levels."""

    scene: RoughLaneSceneCfg = RoughLaneSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    observations: RoughLaneTrainObservationCfg = RoughLaneTrainObservationCfg()
    rewards: RoughLaneRewardsCfg = RoughLaneRewardsCfg()
    terminations: RoughLaneTerminationsCfg = RoughLaneTerminationsCfg()
    events: RoughLaneEventCfg = RoughLaneEventCfg()


@configclass
class RoughLaneEvalAntEnvCfg(RoughLaneAntEnvCfg):
    """v5 benchmark: noise-free observations and course-style resets on every lane."""

    observations: RoughLaneObservationCfg = RoughLaneObservationCfg()
    events: RoughLaneEvalEventCfg = RoughLaneEvalEventCfg()


@configclass
class RoughLaneDemoAntEnvCfg(RoughLaneEvalAntEnvCfg):
    """Seven-environment GUI layout, one per family at difficulty level 3 (0.8)."""

    scene: RoughLaneSceneCfg = RoughLaneSceneCfg(num_envs=len(ROUGH_LANE_FAMILIES), env_spacing=5.0,
                                                 clone_in_fabric=True)
    events: RoughLaneDemoEventCfg = RoughLaneDemoEventCfg()


@configclass
class RoughLaneSafeRewardsCfg(RoughLaneRewardsCfg):
    """Training-only recovery objective; the fixed v5 benchmark stays unchanged.

    Terrain speed above 3 m/s no longer outweighs stable crossing. Plane progress
    remains uncapped to preserve the course gait. Rewards are integrated over dt,
    so the -600 fall weight is a -10 terminal penalty at the 60 Hz control rate.
    """

    progress = RewTerm(func=lane_progress_reward, weight=1.0, params={"terrain_speed_limit": 3.0})
    fall = RewTerm(func=mdp.is_terminated, weight=-600.0)


@configclass
class RoughLaneSafeAntEnvCfg(RoughLaneAntEnvCfg):
    """Fine-tune the 60D v5 policy for fewer falls without changing evaluation terrain."""

    rewards: RoughLaneSafeRewardsCfg = RoughLaneSafeRewardsCfg()


@configclass
class RoughLaneTraversalRewardsCfg(RoughLaneSafeRewardsCfg):
    """Training-only lane keeping and stepping-stone recovery after safe fine-tuning."""

    fall = RewTerm(func=mdp.is_terminated, weight=-1800.0)
    centering = RewTerm(func=lane_centering_penalty, weight=-0.5)
    stall = RewTerm(func=lane_stall_penalty, weight=-1.0)
    energy = RewTerm(func=lane_power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})


@configclass
class RoughLaneTraversalTerminationsCfg(RoughLaneTerminationsCfg):
    lane_departure = DoneTerm(func=lane_out_of_bounds, params={"margin": 0.5})


@configclass
class RoughLaneTraversalAntEnvCfg(RoughLaneAntEnvCfg):
    """Train actual in-lane traversal; evaluate only with the unchanged v5 Eval task."""

    rewards: RoughLaneTraversalRewardsCfg = RoughLaneTraversalRewardsCfg()
    terminations: RoughLaneTraversalTerminationsCfg = RoughLaneTraversalTerminationsCfg()


@configclass
class RoughLaneRecoveryRewardsCfg(RoughLaneTraversalRewardsCfg):
    """Training-only stone-top posture and foot lift; no new policy observations."""

    stone_body_height = RewTerm(
        func=stone_body_height_penalty, weight=-2.0, params={"target_height": 0.5, "tolerance": 0.2},
    )
    stone_swing_clearance = RewTerm(
        func=stone_swing_clearance_penalty, weight=-0.5, params={"target_clearance": 0.1, "trap_weight": 0.0},
    )


@configclass
class RoughLaneRecoveryAntEnvCfg(RoughLaneTraversalAntEnvCfg):
    """Recovery curriculum task; pit depth overrides belong only in training runs."""

    rewards: RoughLaneRecoveryRewardsCfg = RoughLaneRecoveryRewardsCfg()
