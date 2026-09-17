"""Ant environment variants used for the robustness experiment.

All variants preserve the course task's 60-dimensional observation and
8-dimensional action spaces. Only the physical/reset distributions or sensor
noise change, keeping checkpoints mutually evaluable across every task.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import (
    AntEnvCfg,
    EventCfg,
    ObservationsCfg,
)
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp


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
