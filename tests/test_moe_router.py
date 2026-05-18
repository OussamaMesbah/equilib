"""Tests for the MoE topological router (research demo).

The router is explicitly documented as a research demo, not a production
component. These tests check the documented behaviour:
- output is a valid simplex vector
- instantiation emits the UserWarning
- the routing weights are not degenerate (no single expert dominates by ~1.0)
"""

import warnings

import torch
import pytest
from sperner.moe_router import TopologicalMoERouter


def test_router_init_emits_warning():
    """The router emits a UserWarning to discourage production use."""
    with pytest.warns(UserWarning, match="research demo"):
        router = TopologicalMoERouter(num_experts=4, latent_dim=64)
    assert router.num_experts == 4
    assert router.latent_dim == 64
    assert router.device == "cpu"


def test_router_rejects_one_expert():
    """1 expert isn't a routing problem; the underlying solver rejects it."""
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            TopologicalMoERouter(num_experts=1, latent_dim=32)


def _make_router(num_experts: int, latent_dim: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TopologicalMoERouter(num_experts=num_experts,
                                    latent_dim=latent_dim)


def test_forward_route_output_shape():
    router = _make_router(num_experts=3, latent_dim=32)
    hidden = torch.randn(1, 1, 32)
    weights = router.forward_route(hidden, precision=10)
    assert weights.shape == (3, )
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=0.05)
    assert (weights >= 0).all()


def test_forward_route_no_routing_collapse():
    """The README claims 'no routing collapse' — the panchromatic-cell centroid
    has all coordinates > 0 because it's an interior point of the simplex.
    Verify no single expert takes more than 0.99 of the mass."""
    router = _make_router(num_experts=4, latent_dim=32)
    torch.manual_seed(0)
    hidden = torch.randn(1, 1, 32)
    weights = router.forward_route(hidden, precision=20)
    assert weights.max().item() < 0.99, (
        f"router collapsed onto a single expert: {weights}")


def test_forward_route_deterministic_with_seed():
    """Same input with same model state should produce consistent output."""
    router = _make_router(num_experts=3, latent_dim=32)
    torch.manual_seed(42)
    hidden = torch.randn(1, 1, 32)
    torch.manual_seed(99)
    w1 = router.forward_route(hidden, precision=10)
    torch.manual_seed(99)
    w2 = router.forward_route(hidden, precision=10)
    assert torch.allclose(w1, w2)


def test_router_many_experts():
    router = _make_router(num_experts=6, latent_dim=16)
    hidden = torch.randn(1, 1, 16)
    weights = router.forward_route(hidden, precision=8)
    assert weights.shape == (6, )
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=0.05)
