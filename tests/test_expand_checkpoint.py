import hashlib
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "expand_checkpoint.py"


def _make_checkpoint(path: Path) -> dict:
    generator = torch.Generator().manual_seed(123)
    state = {
        # Small exactly representable values make bit-for-bit action equality a
        # meaningful assertion rather than a matrix-kernel rounding accident.
        "actor.0.weight": torch.randint(-2, 3, (5, 60), generator=generator).float(),
        "actor.0.bias": torch.randint(-2, 3, (5,), generator=generator).float(),
        "actor.2.weight": torch.randint(-2, 3, (8, 5), generator=generator).float(),
        "actor.2.bias": torch.randint(-2, 3, (8,), generator=generator).float(),
        "critic.0.weight": torch.randint(-2, 3, (7, 60), generator=generator).float(),
        "critic.0.bias": torch.randint(-2, 3, (7,), generator=generator).float(),
        "std": torch.full((8,), 0.7),
    }
    checkpoint = {
        "model_state_dict": state,
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(9), "exp_avg": torch.ones(1)}},
            "param_groups": [{"lr": 0.01, "params": [0]}],
        },
        "iter": 77,
        "infos": {"source_tag": "synthetic"},
    }
    torch.save(checkpoint, path)
    return checkpoint


def _expand(tmp_path: Path) -> tuple[dict, dict, Path, str]:
    source = tmp_path / "source.pt"
    destination = tmp_path / "expanded.pt"
    original = _make_checkpoint(source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(destination),
            "--extra-dims",
            "286",
            "--std",
            "0.2",
            "--learning-rate",
            "0.0001",
            "--reset-iteration",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expanded = torch.load(destination, map_location="cpu", weights_only=False)
    return original, expanded, source, source_digest


def _actor(state: dict[str, torch.Tensor], observations: torch.Tensor) -> torch.Tensor:
    hidden = torch.tanh(F.linear(observations, state["actor.0.weight"], state["actor.0.bias"]))
    return F.linear(hidden, state["actor.2.weight"], state["actor.2.bias"])


def test_expansion_appends_286_zero_columns_without_changing_actor_parameters(tmp_path):
    original, expanded, _, _ = _expand(tmp_path)
    before = original["model_state_dict"]
    after = expanded["model_state_dict"]

    assert after["actor.0.weight"].shape == (5, 346)
    torch.testing.assert_close(after["actor.0.weight"][:, :60], before["actor.0.weight"])
    torch.testing.assert_close(after["actor.0.weight"][:, 60:], torch.zeros(5, 286))
    assert after["critic.0.weight"].shape == (7, 346)
    torch.testing.assert_close(after["critic.0.weight"][:, :60], before["critic.0.weight"])
    torch.testing.assert_close(after["critic.0.weight"][:, 60:], torch.zeros(7, 286))
    for key in ("actor.0.bias", "actor.2.weight", "actor.2.bias"):
        torch.testing.assert_close(after[key], before[key])


def test_expanded_actor_ignores_arbitrary_appended_features_at_initialization(tmp_path):
    original, expanded, _, _ = _expand(tmp_path)
    observations = torch.randint(
        -2, 3, (11, 60), generator=torch.Generator().manual_seed(321)
    ).float()
    depth_features = torch.randn(11, 286, generator=torch.Generator().manual_seed(456)) * 100.0

    expected = _actor(original["model_state_dict"], observations)
    actual = _actor(expanded["model_state_dict"], torch.cat((observations, depth_features), dim=1))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_expansion_resets_optimizer_noise_and_learning_rate(tmp_path):
    _, expanded, _, _ = _expand(tmp_path)

    assert expanded["optimizer_state_dict"]["state"] == {}
    assert expanded["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.0001
    torch.testing.assert_close(expanded["model_state_dict"]["std"], torch.full((8,), 0.2))


def test_v6_expansion_resets_iteration_and_records_source_iteration(tmp_path):
    _, expanded, _, _ = _expand(tmp_path)

    assert expanded["iter"] == 0
    assert expanded["infos"]["source_iteration"] == 77


def test_expansion_does_not_modify_source_checkpoint(tmp_path):
    _, expanded, source, digest_before = _expand(tmp_path)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    assert expanded["infos"]["expanded_observation_dims"] == 286
    assert expanded["infos"]["source_tag"] == "synthetic"


def test_legacy_defaults_preserve_54d_expansion_iteration_and_learning_rate(tmp_path):
    source = tmp_path / "legacy_source.pt"
    destination = tmp_path / "legacy_expanded.pt"
    _make_checkpoint(source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(destination)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expanded = torch.load(destination, map_location="cpu", weights_only=False)

    assert expanded["model_state_dict"]["actor.0.weight"].shape[1] == 114
    assert expanded["model_state_dict"]["critic.0.weight"].shape[1] == 114
    assert expanded["iter"] == 77
    assert "source_iteration" not in expanded["infos"]
    assert expanded["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.0002
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
