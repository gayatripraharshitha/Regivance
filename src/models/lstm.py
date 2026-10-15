
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class LSTMConfig:
    """Configuration for a compact volatility forecasting LSTM."""

    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    random_state: int = 42
    objective: str = "mse_log"

    def __post_init__(self) -> None:
        integer_fields = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }

        for name, value in integer_fields.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if (
            not isinstance(self.random_state, int)
            or isinstance(self.random_state, bool)
        ):
            raise ValueError("random_state must be an integer.")

        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1).")

        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite.")

        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative.")

        if self.objective not in {"mse_log", "qlike"}:
            raise ValueError("objective must be 'mse_log' or 'qlike'.")


class _LSTMNetwork(nn.Module):
    """LSTM encoder with a single scalar regression output."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, lookback, features).
        recurrent_output, _ = self.recurrent(sequences)

        # The last hidden output summarizes the supplied historical window.
        final_output = recurrent_output[:, -1, :]

        # Return one standardized log-volatility prediction per sequence.
        return self.output_layer(final_output).squeeze(-1)


class LSTMVolatilityForecaster:
    """
    Train a compact LSTM using training and validation sequences.

    The model standardizes log-volatility targets using training targets
    only. Its default objective is MSE in that standardized-log space.
    The optional QLIKE objective converts predictions back to log-volatility
    before applying the variance-ratio loss. Input-feature scaling should
    also be fitted on training data by the calling evaluation pipeline.
    """

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()

    @staticmethod
    def _validate_sequences(
        sequences: np.ndarray,
        name: str,
    ) -> np.ndarray:
        try:
            values = np.asarray(sequences, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain numeric sequence data."
            ) from exc

        if values.ndim != 3:
            raise ValueError(
                f"{name} must have shape (samples, lookback, features)."
            )

        if any(size < 1 for size in values.shape):
            raise ValueError(f"{name} must have non-empty dimensions.")

        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or non-finite values.")

        return np.ascontiguousarray(values)

    @staticmethod
    def _validate_targets(
        targets: np.ndarray,
        expected_length: int,
        name: str,
    ) -> np.ndarray:
        try:
            values = np.asarray(targets, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric targets.") from exc

        if len(values) != expected_length:
            raise ValueError(
                f"{name} length must match the number of sequences."
            )

        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or non-finite values.")

        if (values <= 0).any():
            raise ValueError(
                f"{name} must contain strictly positive volatility targets."
            )

        return values

    @staticmethod
    def qlike_loss_from_standardized_log_predictions(
        standardized_log_predictions: torch.Tensor,
        targets: torch.Tensor,
        target_log_mean: float,
        target_log_scale: float,
    ) -> torch.Tensor:
        """Return mean variance-based QLIKE from standardized log forecasts."""
        if not np.isfinite(target_log_mean) or not np.isfinite(target_log_scale):
            raise ValueError("Target log scaling parameters must be finite.")
        if target_log_scale <= 0:
            raise ValueError("Target log scaling parameter must be positive.")
        if not torch.isfinite(standardized_log_predictions).all():
            raise FloatingPointError("QLIKE predictions are non-finite.")
        if not torch.isfinite(targets).all() or (targets <= 0).any():
            raise ValueError("QLIKE targets must be finite and positive.")

        predicted_log = (
            standardized_log_predictions * target_log_scale + target_log_mean
        )
        log_ratio = 2.0 * (torch.log(targets) - predicted_log)
        maximum_log = float(np.log(torch.finfo(log_ratio.dtype).max))

        # Reject numerical overflow instead of clipping a loss or target.
        if not torch.isfinite(log_ratio).all() or (log_ratio > maximum_log).any():
            raise FloatingPointError("QLIKE ratio became numerically unstable.")

        loss = torch.exp(log_ratio) - log_ratio - 1.0
        if not torch.isfinite(loss).all():
            raise FloatingPointError("QLIKE loss became non-finite.")
        return loss.mean()

    def _objective_loss(
        self,
        standardized_log_predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
        original_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured training or validation objective."""
        if self.config.objective == "mse_log":
            return nn.functional.mse_loss(
                standardized_log_predictions, normalized_targets
            )
        return self.qlike_loss_from_standardized_log_predictions(
            standardized_log_predictions,
            original_targets,
            self.target_log_mean_,
            self.target_log_scale_,
        )

    def fit(
        self,
        training_sequences: np.ndarray,
        training_targets: np.ndarray,
        validation_sequences: np.ndarray,
        validation_targets: np.ndarray,
    ) -> LSTMVolatilityForecaster:
        """Fit on training samples and use validation loss for early stopping."""
        x_train = self._validate_sequences(
            training_sequences, "training_sequences"
        )
        x_validation = self._validate_sequences(
            validation_sequences, "validation_sequences"
        )

        if x_train.shape[1:] != x_validation.shape[1:]:
            raise ValueError(
                "Training and validation sequence dimensions must match."
            )

        y_train = self._validate_targets(
            training_targets,
            len(x_train),
            "training_targets",
        )
        y_validation = self._validate_targets(
            validation_targets,
            len(x_validation),
            "validation_targets",
        )

        # Predict log-volatility and normalize it using training labels only.
        log_train = np.log(y_train)
        self.target_log_mean_ = float(log_train.mean())

        target_std = float(log_train.std())
        self.target_log_scale_ = target_std if target_std > 1e-8 else 1.0

        normalized_train = (
            log_train - self.target_log_mean_
        ) / self.target_log_scale_

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(
            normalized_train.astype(np.float32)
        )
        y_train_original_tensor = torch.from_numpy(y_train.astype(np.float32))

        x_validation_tensor = torch.from_numpy(x_validation)

        normalized_validation = (
            np.log(y_validation) - self.target_log_mean_
        ) / self.target_log_scale_

        y_validation_tensor = torch.from_numpy(
            normalized_validation.astype(np.float32)
        )
        y_validation_original_tensor = torch.from_numpy(
            y_validation.astype(np.float32)
        )

        # Explicitly use CPU for compatibility with CPU-only installations.
        torch.manual_seed(self.config.random_state)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.random_state)

        train_loader = DataLoader(
            TensorDataset(
                x_train_tensor,
                y_train_tensor,
                y_train_original_tensor,
            ),
            batch_size=min(self.config.batch_size, len(x_train)),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        self.model_ = _LSTMNetwork(
            input_size=x_train.shape[2],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_validation_loss = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        self.validation_history_ = []

        for epoch in range(1, self.config.max_epochs + 1):
            self.model_.train()

            for batch_x, batch_y, batch_original_y in train_loader:
                optimizer.zero_grad(set_to_none=True)

                predictions = self.model_(batch_x)
                loss = self._objective_loss(
                    predictions,
                    batch_y,
                    batch_original_y,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Training loss became non-finite."
                    )

                loss.backward()

                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in self.model_.parameters()
                ):
                    raise RuntimeError("Training gradients became non-finite.")

                optimizer.step()

            self.model_.eval()

            with torch.no_grad():
                validation_predictions = self.model_(
                    x_validation_tensor
                )
                validation_loss = float(
                    self._objective_loss(
                        validation_predictions,
                        y_validation_tensor,
                        y_validation_original_tensor,
                    ).item()
                )

            if not np.isfinite(validation_loss):
                raise RuntimeError(
                    "Validation loss became non-finite."
                )

            self.validation_history_.append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    name: value.detach().clone()
                    for name, value in self.model_.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        if best_state is None:
            raise RuntimeError("LSTM training did not produce a valid model.")

        # Restore the model from the best validation epoch.
        self.model_.load_state_dict(best_state)
        self.model_.eval()

        self.n_epochs_trained_ = len(self.validation_history_)
        self.best_epoch_ = best_epoch
        self.best_validation_loss_ = best_validation_loss
        self.objective_ = self.config.objective
        self.lookback_ = x_train.shape[1]
        self.n_features_in_ = x_train.shape[2]

        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Return positive volatility forecasts in the original target units."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predict().")

        values = self._validate_sequences(sequences, "sequences")

        if values.shape[1:] != (
            self.lookback_,
            self.n_features_in_,
        ):
            raise ValueError(
                "Prediction sequences must match the fitted lookback "
                "and feature count."
            )

        tensor = torch.from_numpy(values)
        batch_size = 512
        predictions = []

        self.model_.eval()

        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start : start + batch_size]
                normalized_log_predictions = self.model_(batch)
                predictions.append(
                    normalized_log_predictions.cpu().numpy()
                )

        normalized_log = np.concatenate(predictions)

        log_forecasts = (
            normalized_log * self.target_log_scale_
            + self.target_log_mean_
        )

        # Guard against numerical overflow or underflow.
        forecasts = np.exp(np.clip(log_forecasts, -30.0, 10.0))

        if not np.isfinite(forecasts).all() or (forecasts <= 0).any():
            raise RuntimeError("LSTM produced invalid volatility forecasts.")

        return forecasts
