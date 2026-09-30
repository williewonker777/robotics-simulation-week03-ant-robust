"""Stateless terrain-contact-conditioned tip-speed proxy, not friction force."""

import torch

CONTACT_FORCE_THRESHOLD = 2.0
TIP_SPEED_SCALE = 1.0


def contact_slip_metrics(force, tip_velocity):
    """Inputs [N,4,3] world-frame filtered normal force and kinematic tip velocity.

    Invalid rows abstain and remain distinguishable from valid airborne rows.
    The caller/RewardManager alone applies weight and simulation timestep.
    """
    if force.ndim != 3 or force.shape[1:] != (4, 3) or tip_velocity.shape != force.shape:
        raise ValueError('expected force and tip_velocity[N,4,3]')
    if not force.is_floating_point() or not tip_velocity.is_floating_point():
        raise ValueError('contact inputs must be floating point')
    if force.dtype != tip_velocity.dtype or force.device != tip_velocity.device:
        raise ValueError('contact inputs must share dtype and device')
    valid = torch.isfinite(force).all(dim=(1, 2)) & torch.isfinite(tip_velocity).all(dim=(1, 2))
    force = torch.where(valid[:, None, None], force, 0.)
    velocity = torch.where(valid[:, None, None], tip_velocity, 0.)
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1)
    # Finite but extreme inputs can overflow their norms: abstain rather than
    # silently report an infinite diagnostic or saturated invalid observation.
    valid = valid & torch.isfinite(force_norm).all(-1) & torch.isfinite(speed).all(-1)
    force_norm = torch.where(valid[:, None], force_norm, 0.)
    speed = torch.where(valid[:, None], speed, 0.)
    contact = (force_norm > CONTACT_FORCE_THRESHOLD) & valid[:, None]
    count = contact.sum(-1)
    cost = (contact * (speed / TIP_SPEED_SCALE).clamp(max=1.).square()).mean(-1)
    mean_speed = (torch.where(contact, speed, 0.) / count.clamp(min=1)[:, None]).sum(-1)
    return dict(reward=-cost, bounded_cost=cost, valid=valid, contact=contact,
                contact_fraction=count.to(force.dtype) / 4., no_contact=valid & (count == 0),
                contacted_tip_speed=mean_speed, force_norm=force_norm, tip_speed=speed)
