"""Reward-only terrain-contact sensors; frozen v14 observations and dynamics."""

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from week03_ant.contact_math import contact_slip_metrics
from week03_ant.footmap_math import FOOT_NAMES, TIP_OFFSETS
from .command_v14_cfg import CommandTrainAntEnvCfg, CommandEvalAntEnvCfg, CommandAntPPORunnerCfg
from .foothold_v9_cfg import FootholdSceneCfg
from .posture_v13_cfg import AdaptivePostureRewardsCfg

TERRAIN_CONTACT_PATH = '/World/ground/terrain/mesh'
FLAT_CONTACT_PATH = '/World/flatPlane/GroundPlane/CollisionPlane'
CONTACT_TARGET_PATHS = (TERRAIN_CONTACT_PATH, FLAT_CONTACT_PATH)
CONTACT_SENSOR_NAMES = tuple('contact_' + name for name in FOOT_NAMES)


def terrain_contact_sensor(foot):
    return ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/' + foot,
        filter_prim_paths_expr=list(CONTACT_TARGET_PATHS), update_period=0.,
        track_air_time=False, debug_vis=False, history_length=0,
    )


def validate_terrain_contact_target():
    """Fail closed if the live generated collision mesh is not the filter target."""
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    for path in CONTACT_TARGET_PATHS:
        prim = stage.GetPrimAtPath(path) if stage is not None else None
        if prim is None or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('contact filter must resolve to a live collision prim: ' + path)


class ContactSlipReward(ManagerTermBase):
    """Fresh filtered forces and link-origin kinematics; no history or observations."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        validate_terrain_contact_target()
        self.robot = env.scene['robot']
        self.ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if names != list(FOOT_NAMES) or len(set(self.ids)) != 4:
            raise ValueError(f'unexpected feet: {names}')
        self.offsets = torch.tensor(TIP_OFFSETS, device=env.device).expand(env.num_envs, -1, -1)
        self.sensors = [env.scene[name] for name in CONTACT_SENSOR_NAMES]
        for foot, sensor in zip(FOOT_NAMES, self.sensors):
            if sensor.body_names != [foot]:
                raise ValueError(f'expected one contact sensor body: {foot}')
            if sensor.cfg.filter_prim_paths_expr != list(CONTACT_TARGET_PATHS):
                raise ValueError('contact sensor must filter exactly the terrain mesh and flat collision plane')
            if sensor.cfg.update_period != 0. or sensor.cfg.track_air_time:
                raise ValueError('contact sensors require update_period=0 and no air-time tracking')

    def geometry(self, env):
        forces = []
        for sensor in self.sensors:
            sensor.update(0., force_recompute=True)
            force = sensor.data.force_matrix_w
            if force is None or force.shape != (env.num_envs, 1, 2, 3):
                raise ValueError('expected terrain-filtered force_matrix_w[N,1,2,3]')
            forces.append(force[:, 0].sum(dim=1))
        data = self.robot.data
        offset = quat_apply(data.body_link_quat_w[:, self.ids], self.offsets)
        velocity = data.body_link_lin_vel_w[:, self.ids] + torch.cross(
            data.body_link_ang_vel_w[:, self.ids], offset, dim=-1)
        return contact_slip_metrics(torch.stack(forces, dim=1), velocity)

    def __call__(self, env):
        return self.geometry(env)['reward']


@configclass
class ContactSceneCfg(FootholdSceneCfg):
    robot = FootholdSceneCfg().robot.replace(
        spawn=FootholdSceneCfg().robot.spawn.replace(activate_contact_sensors=True))
    contact_front_left_foot = terrain_contact_sensor(FOOT_NAMES[0])
    contact_front_right_foot = terrain_contact_sensor(FOOT_NAMES[1])
    contact_left_back_foot = terrain_contact_sensor(FOOT_NAMES[2])
    contact_right_back_foot = terrain_contact_sensor(FOOT_NAMES[3])


@configclass
class ContactRewardsCfg(AdaptivePostureRewardsCfg):
    contact_slip = RewTerm(func=ContactSlipReward, weight=1.)


@configclass
class ContactTrainAntEnvCfg(CommandTrainAntEnvCfg):
    scene: ContactSceneCfg = ContactSceneCfg(num_envs=4096, env_spacing=5., clone_in_fabric=False)
    rewards: ContactRewardsCfg = ContactRewardsCfg()


@configclass
class ContactEvalAntEnvCfg(CommandEvalAntEnvCfg):
    scene: ContactSceneCfg = ContactSceneCfg(num_envs=175, env_spacing=5., clone_in_fabric=False)


@configclass
class ContactAntPPORunnerCfg(CommandAntPPORunnerCfg):
    experiment_name = 'week03_ant_contact_v16'


def validate_contact_config_parity():
    """Check entire env configs after removing only the declared sensor/reward delta."""
    for added, original in ((ContactTrainAntEnvCfg, CommandTrainAntEnvCfg),
                            (ContactEvalAntEnvCfg, CommandEvalAntEnvCfg)):
        candidate, reference = added().to_dict(), original().to_dict()
        scene = candidate['scene']
        for name in CONTACT_SENSOR_NAMES:
            scene.pop(name)
        if scene['robot']['spawn']['activate_contact_sensors'] is not True:
            raise ValueError('v16 requires the asset contact reporter')
        scene['robot']['spawn']['activate_contact_sensors'] = reference['scene']['robot']['spawn']['activate_contact_sensors']
        if added is ContactTrainAntEnvCfg:
            candidate['rewards'].pop('contact_slip')
        if candidate != reference:
            raise ValueError(f'v16 undeclared environment configuration difference: {added.__name__}')
    return True
