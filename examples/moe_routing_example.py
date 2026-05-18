"""Demo: per-token MoE routing via a Sperner walk.

This is a research demo, not a production routing layer. Per-token Sperner
walks are many orders of magnitude slower than softmax gating; the demo is
kept to show what the resulting routing weights look like.
"""

import warnings

import torch
from sperner import TopologicalMoERouter


def main():
    num_experts = 6
    latent_dim = 128  # small for demo speed

    with warnings.catch_warnings():
        # The router emits a UserWarning on construction — suppress here
        # because we are deliberately running the demo.
        warnings.simplefilter("ignore", UserWarning)
        router = TopologicalMoERouter(
            num_experts=num_experts,
            latent_dim=latent_dim,
            device="cpu",
        )

    hidden = torch.randn(1, 1, latent_dim)

    print(f"Running Sperner walk for {num_experts} experts "
          f"(slow on purpose)...")
    weights = router.forward_route(hidden, precision=20)

    print(f"\nExpert weights (sum={weights.sum():.3f}):")
    for i, w in enumerate(weights.tolist()):
        bar = "#" * int(w * 40)
        print(f"  Expert {i}: {w:.3f}  {bar}")

    print("\nAll experts received a positive allocation — that's a property "
          "of the panchromatic-cell centroid, not a guarantee of routing "
          "quality. See the module docstring for caveats.")


if __name__ == "__main__":
    main()
