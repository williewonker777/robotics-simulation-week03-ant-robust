"""RSL-RL configuration shared by every experiment for a fair comparison."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class Week03AntPPORunnerCfg(AntPPORunnerCfg):
    """Unmodified course PPO budget with a project-specific log namespace."""

    experiment_name = "week03_ant"
