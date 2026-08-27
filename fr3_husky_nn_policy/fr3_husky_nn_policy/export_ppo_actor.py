from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def export_actor(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export an RSL-RL MLP actor state dict to a pickle-free NumPy archive."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required only for export. Run this tool in the Isaac/RSL-RL environment."
        ) from error

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("actor_state_dict")
    if state is None:
        raise KeyError("Checkpoint has no actor_state_dict")

    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    if not layer_indices:
        raise ValueError("No actor MLP weights found")
    if any(f"mlp.{index}.bias" not in state for index in layer_indices):
        raise ValueError("Every actor MLP weight must have a matching bias")

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = {
        "format_version": np.asarray(1, dtype=np.int64),
        "activation": np.asarray("elu"),
        "num_layers": np.asarray(len(layer_indices), dtype=np.int64),
        "source_sha256": np.asarray(digest),
        "source_iteration": np.asarray(int(checkpoint.get("iter", -1)), dtype=np.int64),
    }
    for output_index, state_index in enumerate(layer_indices):
        payload[f"weight_{output_index}"] = (
            state[f"mlp.{state_index}.weight"].detach().cpu().numpy().astype(np.float32)
        )
        payload[f"bias_{output_index}"] = (
            state[f"mlp.{state_index}.bias"].detach().cpu().numpy().astype(np.float32)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export an RSL-RL MLP actor to NPZ")
    parser.add_argument("checkpoint", help="Path to model_*.pt")
    parser.add_argument("output", help="Output .npz path")
    args = parser.parse_args()
    path = export_actor(args.checkpoint, args.output)
    print(path)


if __name__ == "__main__":
    main()
