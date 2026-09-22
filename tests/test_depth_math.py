import pytest
import torch

from week03_ant.depth_math import encode_height_scan


def test_height_scan_returns_bounded_finite_values_and_valid_bits():
    encoded = encode_height_scan(
        torch.tensor([0.5]),
        torch.tensor([[0.5, 0.0, -1.0, float("nan")]]),
        plane_height=-50.0,
    )

    assert encoded.shape == (1, 8)
    assert torch.isfinite(encoded).all()
    assert torch.all(encoded[:, :4].abs() <= 1.0)
    assert encoded[0, 4:].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_height_scan_rejects_mesh_hits_above_origin_or_beyond_max_range():
    encoded = encode_height_scan(
        torch.tensor([0.0]),
        torch.tensor([[2.1, -2.01]]),
        ray_offset_z=2.0,
        max_distance=4.0,
        plane_height=-50.0,
    )

    torch.testing.assert_close(encoded[0, :2], torch.ones(2))
    torch.testing.assert_close(encoded[0, 2:], torch.zeros(2))


def test_height_scan_uses_physically_reachable_infinite_plane():
    encoded = encode_height_scan(
        torch.tensor([-49.5]),
        torch.tensor([[float("inf")]]),
        plane_height=-50.0,
    )

    torch.testing.assert_close(encoded, torch.tensor([[0.0, 1.0]]))


def test_height_scan_prefers_nearer_mesh_surface_over_plane():
    encoded = encode_height_scan(
        torch.tensor([0.5]),
        torch.tensor([[0.25]]),
        plane_height=0.0,
    )

    # The mesh produces -0.25 while selecting the lower plane would produce 0.
    torch.testing.assert_close(encoded, torch.tensor([[-0.25, 1.0]]))


def test_height_scan_marks_missing_mesh_and_unreachable_plane_invalid():
    encoded = encode_height_scan(
        torch.tensor([10.0]),
        torch.tensor([[float("nan"), float("inf")]]),
        plane_height=0.0,
    )

    torch.testing.assert_close(encoded, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_height_scan_is_invariant_to_global_vertical_translation():
    root = torch.tensor([0.5, 2.0])
    hits = torch.tensor([[0.2, -0.4], [1.8, float("nan")]])
    original = encode_height_scan(root, hits, plane_height=0.0)
    shifted = encode_height_scan(root + 17.0, hits + 17.0, plane_height=17.0)

    torch.testing.assert_close(shifted, original)


def test_height_scan_preserves_spatial_height_distinctions_between_rays():
    encoded = encode_height_scan(
        torch.tensor([0.5]),
        torch.tensor([[0.5, 0.25, 0.0, -0.5]]),
        plane_height=-50.0,
    )

    torch.testing.assert_close(encoded[0, :4], torch.tensor([-0.5, -0.25, 0.0, 0.5]))
    torch.testing.assert_close(encoded[0, 4:], torch.ones(4))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("ray_offset_z", float("nan")),
        ("max_distance", 0.0),
        ("max_distance", float("inf")),
        ("plane_height", float("nan")),
        ("height_offset", float("inf")),
        ("height_clip", 0.0),
        ("height_clip", -1.0),
    ],
)
def test_height_scan_rejects_invalid_constants(keyword, value):
    with pytest.raises(ValueError):
        encode_height_scan(torch.zeros(1), torch.zeros(1, 1), **{keyword: value})


@pytest.mark.parametrize(
    ("root", "hits"),
    [
        (torch.zeros(1, 1), torch.zeros(1, 1)),
        (torch.zeros(1), torch.zeros(1)),
        (torch.zeros(2), torch.zeros(1, 3)),
    ],
)
def test_height_scan_rejects_shape_mismatches(root, hits):
    with pytest.raises(ValueError):
        encode_height_scan(root, hits)
