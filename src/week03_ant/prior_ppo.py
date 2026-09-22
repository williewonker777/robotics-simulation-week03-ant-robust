# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Version-pinned PPO 3.0.1 update adaptation for the v10 teacher loss.

The base class owns rollout collection, storage, returns, and optimizer setup.
Only its single-GPU, feedforward, fixed-schedule update loop is reproduced here;
the installed source is pinned by the experiment harness to SHA-256
``deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c``.
"""

from __future__ import annotations

import math
import hashlib
import importlib.metadata
import inspect

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from .prior_policy import PRIOR_COEFFICIENTS, PriorActorCritic


REFERENCE_STD = 0.2
PINNED_PPO_VERSION = "3.0.1"
PINNED_PPO_SHA256 = "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c"


def _validate_installed_ppo():
    version = importlib.metadata.version("rsl-rl-lib")
    source = inspect.getsourcefile(PPO)
    digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
    if version != PINNED_PPO_VERSION or digest != PINNED_PPO_SHA256:
        raise RuntimeError("PriorPPO requires the pinned rsl-rl-lib 3.0.1 PPO source")


def fixed_gaussian_prior_loss(student_mean, teacher_mean, reference_std=REFERENCE_STD):
    """Mean per-action KL for equal fixed-sigma Gaussian means."""
    if not math.isfinite(reference_std) or reference_std <= 0:
        raise ValueError("reference std must be positive and finite")
    if student_mean.shape != teacher_mean.shape:
        raise ValueError("student and teacher means must have equal shapes")
    return ((student_mean - teacher_mean.detach()).square() / (2.0 * reference_std**2)).mean()


class PriorPPO(PPO):
    """PPO with the predeclared frozen-teacher auxiliary mean loss."""

    def __init__(self, policy, teacher_coef=0.0, **kwargs):
        if teacher_coef not in (0.0, 0.02):
            raise ValueError("teacher_coef must be exactly 0.0 or 0.02")
        if kwargs.get("schedule", "adaptive") != "fixed":
            raise ValueError("PriorPPO supports only the fixed learning-rate schedule")
        if kwargs.get("desired_kl", 0.01) is not None:
            raise ValueError("PriorPPO requires desired_kl=None")
        if kwargs.get("rnd_cfg") is not None:
            raise ValueError("PriorPPO does not support RND")
        if kwargs.get("symmetry_cfg") is not None:
            raise ValueError("PriorPPO does not support symmetry")
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("PriorPPO does not support multi-GPU training")
        if not isinstance(policy, PriorActorCritic):
            raise TypeError("PriorPPO requires PriorActorCritic")
        if policy.is_recurrent:
            raise ValueError("PriorPPO supports only feedforward policies")
        _validate_installed_ppo()
        super().__init__(policy, **kwargs)
        self.teacher_coef = float(teacher_coef)

    def _validate_mode(self):
        expected = PRIOR_COEFFICIENTS[self.policy.prior_mode]
        if self.teacher_coef != expected:
            raise ValueError("teacher_coef does not match the loaded policy prior mode")

    def update(self):
        self._validate_mode()
        if self.policy.is_recurrent:
            raise ValueError("PriorPPO supports only feedforward policies")
        if self.rnd is not None or self.symmetry is not None or self.is_multi_gpu:
            raise ValueError("unsupported PriorPPO feature enabled")
        if self.schedule != "fixed":
            raise ValueError("PriorPPO supports only the fixed learning-rate schedule")
        if self.desired_kl is not None:
            raise ValueError("PriorPPO requires desired_kl=None")
        if not torch.equal(
            self.policy.prior_teacher_coef.cpu(), torch.tensor(self.teacher_coef, dtype=torch.float32)
        ):
            raise ValueError("teacher_coef differs from the checkpoint coefficient")
        if not torch.equal(
            self.policy.prior_reference_std.cpu(), torch.tensor(REFERENCE_STD, dtype=torch.float32)
        ):
            raise ValueError("checkpoint reference std differs from PriorPPO")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_prior_loss = 0.0
        mean_prior_mse = 0.0
        mean_weighted_prior = 0.0
        mean_large_action_fraction = 0.0
        mean_student_rms = 0.0
        mean_teacher_rms = 0.0
        mean_prior_action_difference_rms = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            del old_mu_batch, old_sigma_batch
            original_batch_size = obs_batch.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            teacher_mean = self.policy.teacher_mean(obs_batch)
            prior_mse = (mu_batch - teacher_mean).square().mean()
            prior_loss = fixed_gaussian_prior_loss(mu_batch, teacher_mean)
            weighted_prior = self.teacher_coef * prior_loss
            # Preserve the exact installed PPO graph for the free-arm parity test.
            if self.teacher_coef != 0.0:
                loss = loss + weighted_prior

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_prior_loss += prior_loss.item()
            mean_prior_mse += prior_mse.item()
            mean_weighted_prior += weighted_prior.item()
            mean_large_action_fraction += (mu_batch.detach().abs() > 1.0).float().mean().item()
            mean_student_rms += mu_batch.detach().square().mean().sqrt().item()
            mean_teacher_rms += teacher_mean.square().mean().sqrt().item()
            mean_prior_action_difference_rms += prior_mse.detach().sqrt().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "prior_loss": mean_prior_loss / num_updates,
            "prior_mse": mean_prior_mse / num_updates,
            "weighted_prior": mean_weighted_prior / num_updates,
            "action_mean_abs_gt1_fraction": mean_large_action_fraction / num_updates,
            "student_action_mean_rms": mean_student_rms / num_updates,
            "teacher_action_mean_rms": mean_teacher_rms / num_updates,
            "prior_action_difference_rms": mean_prior_action_difference_rms / num_updates,
        }
