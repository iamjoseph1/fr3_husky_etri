from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyMLPActor:
    """Small deterministic MLP actor loaded from a non-pickle NPZ artifact."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        archive = np.load(self.model_path, allow_pickle=False)
        self.format_version = (
            int(archive["format_version"].item())
            if "format_version" in archive
            else 1
        )
        if self.format_version not in (1, 2):
            raise ValueError(f"Unsupported actor format version: {self.format_version}")
        self.activation = str(archive["activation"].item())
        self.output_activation = (
            str(archive["output_activation"].item())
            if "output_activation" in archive
            else "identity"
        )
        self.num_layers = int(archive["num_layers"].item())
        if self.activation != "elu":
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.output_activation not in ("identity", "tanh"):
            raise ValueError(
                f"Unsupported actor output activation: {self.output_activation}"
            )

        self.weights = []
        self.biases = []
        for index in range(self.num_layers):
            weight = np.asarray(archive[f"weight_{index}"], dtype=np.float32)
            bias = np.asarray(archive[f"bias_{index}"], dtype=np.float32)
            if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                raise ValueError(f"Invalid layer {index} shapes: {weight.shape}, {bias.shape}")
            if index and weight.shape[1] != self.weights[-1].shape[0]:
                raise ValueError(f"Layer {index} input dimension does not match previous output")
            self.weights.append(np.ascontiguousarray(weight))
            self.biases.append(np.ascontiguousarray(bias))

        self.input_dim = int(self.weights[0].shape[1])
        self.output_dim = int(self.weights[-1].shape[0])
        self.source_sha256 = str(archive["source_sha256"].item())

    @staticmethod
    def _elu_in_place(values: np.ndarray) -> np.ndarray:
        negative = values < 0.0
        values[negative] = np.expm1(values[negative])
        return values

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        output = np.asarray(observation, dtype=np.float32)
        if output.shape != (self.input_dim,):
            raise ValueError(
                f"Expected observation shape ({self.input_dim},), got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("Observation contains NaN or Inf")

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            output = weight @ output + bias
            if index + 1 < self.num_layers:
                output = self._elu_in_place(output)
        if self.output_activation == "tanh":
            output = np.tanh(output)
        if not np.all(np.isfinite(output)):
            raise ValueError("Actor output contains NaN or Inf")
        return np.asarray(output, dtype=np.float32)
