"""Passive, same-first-episode 16s/64s window tracking; no simulator imports."""
from __future__ import annotations

import json

import torch

from .horizon import HorizonEpisodeTracker


WINDOWS = (16, 64)


def immutable_snapshot(value):
    """Detach nested JSON-safe evidence without retaining live telemetry lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def snapshot_passivity(before, after):
    """Fail closed if taking a snapshot changed any declared physical/RNG state."""
    if not before or before != after:
        raise ValueError('window snapshot changed physical, policy, history or RNG state')
    return {'before': immutable_snapshot(before), 'after': immutable_snapshot(after), 'unchanged': True}


class PairedHorizonTracker:
    """One real 64s lifetime with an optional scoring-only 16s cutoff.

    Only ``full`` controls physical first-episode activity. The prefix never
    modifies inputs, returns a reset mask to the environment, or samples RNG.
    Real terminal evidence wins when done and virtual cutoff coincide.
    """

    def __init__(self, num_envs, device, *, instrumented=True):
        if type(instrumented) is not bool:
            raise ValueError('instrumented must be boolean')
        self.instrumented = instrumented
        self.trackers = {seconds: HorizonEpisodeTracker(
            num_envs, device, dt=1 / 60, max_steps=seconds * 60,
            snapshot_seconds=8 if seconds == 16 else 16,
            one_tile_threshold=13.1, all_tiles_threshold=53.1)
            for seconds in (WINDOWS if instrumented else (64,))}
        self.full = self.trackers[64]
        self.physical_done = {seconds: torch.zeros_like(tracker.finished) for seconds, tracker in self.trackers.items()}
        self.physical_timeout = {seconds: torch.zeros_like(tracker.finished) for seconds, tracker in self.trackers.items()}
        self.virtual_censored = {seconds: torch.zeros_like(tracker.finished) for seconds, tracker in self.trackers.items()}
        self.captured = set()
        self.last_step = 0

    def update(self, *, step, rewards, dones, reset_terminated, current_distance, last_distance,
               current_out_of_lane, last_out_of_lane, current_world_exit, last_world_exit):
        if type(step) is not int or step != self.last_step + 1 or self.full.complete:
            raise ValueError('paired window steps must be sequential while first episodes remain')
        done = self.full._vector(dones, dtype=torch.bool)
        terminated = self.full._vector(reset_terminated, dtype=torch.bool)
        if bool((terminated & ~done).any()):
            raise ValueError('physical termination without physical done')
        for seconds in (64, 16) if self.instrumented else (64,):
            tracker = self.trackers[seconds]
            if tracker.complete:
                continue
            active = ~tracker.finished
            physical = active & done
            virtual = active & ~done & (step == tracker.max_steps) if seconds == 16 else torch.zeros_like(done)
            self.physical_done[seconds] |= physical
            self.physical_timeout[seconds] |= physical & ~terminated
            self.virtual_censored[seconds] |= virtual
            # All tensors are fresh selections; never write the caller's terminal buffers.
            tracker.update(step=step, rewards=rewards, dones=done | virtual,
                reset_terminated=terminated,
                current_distance=current_distance,
                last_distance=torch.where(virtual, current_distance, last_distance),
                current_out_of_lane=current_out_of_lane,
                last_out_of_lane=torch.where(virtual, current_out_of_lane, last_out_of_lane),
                current_world_exit=current_world_exit,
                last_world_exit=torch.where(virtual, current_world_exit, last_world_exit))
        self.last_step = step

    def ready_windows(self):
        return tuple(seconds for seconds, tracker in self.trackers.items()
                     if seconds not in self.captured and tracker.complete)

    def window_data(self, seconds):
        if seconds not in self.trackers or seconds in self.captured or not self.trackers[seconds].complete:
            raise ValueError('window unavailable, incomplete or already captured')
        tracker = self.trackers[seconds]
        data = tracker.to_dict()
        data['distance_at_snapshot_m'] = data.pop('distance_at_16s')
        data['episode_full_horizon_survival'] = data.pop('episode_full_64s_survival')
        data['aggregates']['full_horizon_survivals'] = data['aggregates'].pop('full_64s_survivals')
        data.update(window_seconds=seconds, captured_step=self.last_step,
            capture_reason='window_cutoff' if self.last_step == seconds * 60 else 'all_first_episodes_finished',
            episode_physical_done=self.physical_done[seconds].cpu().tolist(),
            episode_physical_timeout=self.physical_timeout[seconds].cpu().tolist(),
            episode_window_censored=self.virtual_censored[seconds].cpu().tolist())
        return immutable_snapshot(data)

    def mark_captured(self, seconds):
        if seconds not in self.ready_windows():
            raise ValueError('cannot mark an unavailable window captured')
        self.captured.add(seconds)
