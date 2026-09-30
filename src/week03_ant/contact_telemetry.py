"""Passive pre-action contact-slip diagnostics over each first episode only."""

import math

import torch


SUM_FIELDS = ("contact_fraction", "contacted_tip_speed", "bounded_cost")


class ContactTelemetry:
    def __init__(self, num_envs, device):
        self.active_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.valid_steps = torch.zeros_like(self.active_steps)
        self.no_contact_steps = torch.zeros_like(self.active_steps)
        self.contact_foot_samples = torch.zeros_like(self.active_steps)
        self.speed_capped_foot_samples = torch.zeros_like(self.active_steps)
        self.sums = {name + "_sum": torch.zeros(num_envs, dtype=torch.float64, device=device)
                     for name in SUM_FIELDS}

    @torch.inference_mode()
    def before_step(self, active, metrics, sample_valid=None):
        if active.shape != self.active_steps.shape or active.dtype != torch.bool:
            raise ValueError("expected boolean first-episode active mask")
        valid = metrics["valid"]
        contact = metrics["contact"]
        tip_speed = metrics["tip_speed"]
        if (valid.shape != active.shape or valid.dtype != torch.bool
                or contact.shape != (len(active), 4) or contact.dtype != torch.bool
                or tip_speed.shape != contact.shape):
            raise ValueError("invalid contact metric shape/type")
        if sample_valid is None:
            sample_valid = torch.ones_like(active)
        if sample_valid.shape != active.shape or sample_valid.dtype != torch.bool:
            raise ValueError("invalid post-reset/wrap sample mask")
        selected = active & valid & sample_valid
        if not torch.isfinite(tip_speed[selected]).all():
            raise ValueError("nonfinite valid foot-tip speed")
        for name in SUM_FIELDS:
            value = metrics[name]
            if value.shape != active.shape or not torch.isfinite(value[selected]).all():
                raise ValueError(f"invalid finite contact metric: {name}")
            if name in ("contact_fraction", "bounded_cost") and ((value[selected] < 0).any() or (value[selected] > 1).any()):
                raise ValueError(f"out-of-bounds contact metric: {name}")
            if name == "contacted_tip_speed" and (value[selected] < 0).any():
                raise ValueError("negative contacted foot speed")
        self.active_steps += active.long()
        self.valid_steps += selected.long()
        self.no_contact_steps += (selected & ~contact.any(-1)).long()
        self.contact_foot_samples += torch.where(selected, contact.sum(-1), 0)
        self.speed_capped_foot_samples += torch.where(selected, (contact & (tip_speed >= 1.)).sum(-1), 0)
        for name in SUM_FIELDS:
            value = metrics[name]
            self.sums[name + "_sum"] += torch.where(selected, value, 0.).double()

    def to_dict(self, episode_lengths):
        if not torch.equal(self.active_steps, episode_lengths):
            raise ValueError("contact duration differs from first episodes")
        return dict(pre_action_first_episode_only=True, reward_unweighted_not_dt_integrated=True,
                    active_steps=self.active_steps.cpu().tolist(),
                    valid_steps=self.valid_steps.cpu().tolist(),
                    no_contact_steps=self.no_contact_steps.cpu().tolist(),
                    contact_foot_samples=self.contact_foot_samples.cpu().tolist(),
                    speed_capped_foot_samples=self.speed_capped_foot_samples.cpu().tolist(),
                    **{name: value.cpu().tolist() for name, value in self.sums.items()})


def audit_contact(data, rows):
    telemetry = data.get("contact_telemetry")
    keys = {"pre_action_first_episode_only", "reward_unweighted_not_dt_integrated",
            "active_steps", "valid_steps", "no_contact_steps", "contact_foot_samples", "speed_capped_foot_samples",
            *(name + "_sum" for name in SUM_FIELDS)}
    if not isinstance(telemetry, dict) or set(telemetry) != keys:
        raise ValueError("unexpected contact telemetry schema")
    if telemetry["pre_action_first_episode_only"] is not True or telemetry["reward_unweighted_not_dt_integrated"] is not True:
        raise ValueError("contact sampling convention changed")
    arrays = {key: value for key, value in telemetry.items() if isinstance(value, list)}
    if set(arrays) != keys - {"pre_action_first_episode_only", "reward_unweighted_not_dt_integrated"}:
        raise ValueError("contact arrays missing")
    if any(len(values) != len(rows) for values in arrays.values()):
        raise ValueError("contact telemetry length mismatch")
    for i, row in enumerate(rows):
        values = {key: array[i] for key, array in arrays.items()}
        active, valid, no_foot_contact, feet, capped = (values[key] for key in
            ("active_steps", "valid_steps", "no_contact_steps", "contact_foot_samples", "speed_capped_foot_samples"))
        if (any(type(value) is not int for value in (active, valid, no_foot_contact, feet, capped))
                or active != row["steps"] or not 0 <= no_foot_contact <= valid <= active
                or not 0 <= capped <= feet <= 4 * valid):
            raise ValueError("contact count/episode mismatch")
        for name in SUM_FIELDS:
            value = values[name + "_sum"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("nonfinite contact sum")
            ceiling = math.inf if name == "contacted_tip_speed" else valid
            if value < -1.e-7 or value > ceiling + 1.e-7 or (valid == 0 and value != 0):
                raise ValueError("contact sum outside bounds")
        if abs(values["contact_fraction_sum"] * 4 - feet) > 1.e-4 * max(1, valid):
            raise ValueError("contact fraction/foot count mismatch")
        row["contact"] = values


def aggregate_contact(rows):
    names = ("active_steps", "valid_steps", "no_contact_steps", "contact_foot_samples", "speed_capped_foot_samples",
             *(name + "_sum" for name in SUM_FIELDS))
    sums = {name: sum(row["contact"][name] for row in rows) for name in names}
    valid, active = sums["valid_steps"], sums["active_steps"]
    return dict(sums=sums,
                means={name: sums[name + "_sum"] / valid if valid else None for name in SUM_FIELDS},
                valid_coverage=valid / active if active else None,
                invalid_steps=active - valid,
                no_contact_fraction=sums["no_contact_steps"] / valid if valid else None,
                contacted_foot_fraction=sums["contact_foot_samples"] / (4 * valid) if valid else None,
                contacted_speed_cap_fraction=sums["speed_capped_foot_samples"] / sums["contact_foot_samples"]
                if sums["contact_foot_samples"] else None)
