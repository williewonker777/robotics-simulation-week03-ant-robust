"""Opt-in v18 train-only scan mask; policy, rewards and physics inherit v16."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from week03_ant.depth_math import encode_height_scan
from week03_ant.foothold_math import GRID_RAYS
from week03_ant.style_math_v18 import teacher_mask_from_scan
from .command_v14_cfg import CommandPolicyCfg, CommandTrainObservationCfg
from .contact_v16_cfg import ContactTrainAntEnvCfg, ContactAntPPORunnerCfg
from .rough_v5_cfg import FLAT_PLANE_HEIGHT
from .agents.prior_v10_cfg import PriorAlgorithmCfg


def depth_teacher_mask(env):
    sensor = env.scene["height_scanner"]
    sensor.update(0., force_recompute=True)
    encoded = encode_height_scan(
        env.scene["robot"].data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
        ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
        plane_height=FLAT_PLANE_HEIGHT,
    )
    if encoded.shape != (env.num_envs, 2 * GRID_RAYS):
        raise ValueError("v18 requires 825-ray height scan")
    heights = -encoded[:, :GRID_RAYS] - .5
    valid = (encoded[:, GRID_RAYS:] > .5) & (encoded[:, :GRID_RAYS].abs() < .99)
    return teacher_mask_from_scan(heights, valid)


@configclass
class StyleGateGroupCfg(ObsGroup):
    teacher_active = ObsTerm(func=depth_teacher_mask)


@configclass
class StyleTrainObservationCfg(CommandTrainObservationCfg):
    style_gate: StyleGateGroupCfg = StyleGateGroupCfg()


@configclass
class StyleTrainAntEnvCfg(ContactTrainAntEnvCfg):
    observations: StyleTrainObservationCfg = StyleTrainObservationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.contact_slip.weight = 0.


@configclass
class StylePolicyCfg(CommandPolicyCfg):
    class_name = "StyleCommandPriorActorCritic"
    style_mode: str = "always"


@configclass
class StyleAlgorithmCfg(PriorAlgorithmCfg):
    class_name = "StylePriorPPO"


@configclass
class StyleAntPPORunnerCfg(ContactAntPPORunnerCfg):
    experiment_name = "week03_ant_style_v18"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = StylePolicyCfg(
        init_noise_std=.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
    algorithm = StyleAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.002,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4, schedule="fixed",
        desired_kl=None, gamma=.995, lam=.95, max_grad_norm=1.,
    )
