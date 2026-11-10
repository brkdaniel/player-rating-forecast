"""PyTorch multilayer perceptron with optional player embeddings, as a scikit-learn regressor."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class RatingNet(nn.Module):
    """MLP over the tabular features, with optional per-player embeddings.

    player_bias: one learned number per player, added to the output. This is a learned
        "random intercept": its L2 penalty pulls every player towards 0 (the population level),
        the same shrinkage the mixed model in hierarchical.py computes in closed form.
    player_dim: one learned vector per player, fed into the MLP so the player effect can
        interact with the other inputs.

    Index 0 is reserved for players not seen during training; its embedding stays at zero.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden: tuple[int, ...] = (64, 32),
        dropout: tuple[float, ...] = (0.2, 0.1),
        n_players: int = 0,
        player_bias: bool = False,
        player_dim: int = 0,
    ):
        super().__init__()
        self.player_bias = nn.Embedding(n_players + 1, 1, padding_idx=0) if player_bias else None
        self.player_vector = (
            nn.Embedding(n_players + 1, player_dim, padding_idx=0) if player_dim else None
        )
        if self.player_bias is not None:
            nn.init.zeros_(self.player_bias.weight)
        if self.player_vector is not None:
            nn.init.normal_(self.player_vector.weight, std=0.01)
            with torch.no_grad():
                self.player_vector.weight[0].zero_()

        layers: list[nn.Module] = []
        width = n_inputs + player_dim
        for size, rate in zip(hidden, dropout, strict=True):
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(rate)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        if self.player_vector is not None:
            x = torch.cat([x, self.player_vector(player)], dim=1)
        out = self.mlp(x).squeeze(-1)
        if self.player_bias is not None:
            out = out + self.player_bias(player).squeeze(-1)
        return out

    def penalty(self, l2: float, embedding_l2: float) -> torch.Tensor:
        """L2 penalty on the hidden weight matrices (like Keras' kernel_regularizer in v1)
        and on the player embeddings."""
        hidden_layers = [m for m in self.mlp if isinstance(m, nn.Linear)][:-1]
        total = l2 * sum((layer.weight**2).sum() for layer in hidden_layers)
        for embedding in (self.player_bias, self.player_vector):
            if embedding is not None:
                total = total + embedding_l2 * (embedding.weight**2).sum()
        return total


class TorchRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn style ``fit(X, y)`` / ``predict(X)`` around :class:`RatingNet`.

    Training: Adam on mini-batches, loss = MSE + L2 penalty, learning rate halved when the
    validation loss stops improving, early stopping with the best weights restored.

    Early stopping uses the *last* ``validation_fraction`` of the rows, so rows must be in
    chronological order (the most recent matches are held out, never a random sample). With
    ``refit_full=True`` the network is then retrained from scratch on all rows for the best
    number of epochs, replaying the same learning-rate schedule.

    With ``player_index=True`` the last column of ``X`` holds integer player indices
    (see ``preprocessing.PlayerIndexer``).
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: tuple[float, ...] = (0.2, 0.1),
        l2: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 150,
        patience: int = 12,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        min_lr: float = 1e-5,
        validation_fraction: float = 0.15,
        refit_full: bool = True,
        player_index: bool = False,
        player_bias: bool = False,
        player_dim: int = 0,
        embedding_l2: float = 1e-4,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.validation_fraction = validation_fraction
        self.refit_full = refit_full
        self.player_index = player_index
        self.player_bias = player_bias
        self.player_dim = player_dim
        self.embedding_l2 = embedding_l2
        self.seed = seed

    # -- public API ---------------------------------------------------------------------------

    def fit(self, X, y):
        x, players = self._to_tensors(X)
        y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.n_players_ = int(players.max()) if self.player_index else 0

        n_validation = int(round(len(y) * self.validation_fraction))
        if n_validation == 0:
            schedule = [self.lr] * self.max_epochs
            self.model_ = self._train_with_schedule(x, players, y, schedule)
            self.best_epoch_, self.validation_curve_ = self.max_epochs, []
            return self

        cut = len(y) - n_validation
        model, schedule, curve = self._train_with_early_stopping(
            (x[:cut], players[:cut], y[:cut]), (x[cut:], players[cut:], y[cut:])
        )
        self.validation_curve_ = curve
        self.best_epoch_ = int(np.argmin(curve)) + 1
        if self.refit_full:
            model = self._train_with_schedule(x, players, y, schedule[: self.best_epoch_])
        self.model_ = model
        return self

    def predict(self, X) -> np.ndarray:
        x, players = self._to_tensors(X)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(x, players).numpy().astype(np.float64)

    # -- training -----------------------------------------------------------------------------

    def _to_tensors(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        X = np.asarray(X, dtype=np.float32)
        if self.player_index:
            features, players = X[:, :-1], X[:, -1].astype(np.int64)
        else:
            features, players = X, np.zeros(len(X), dtype=np.int64)
        return torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(players)

    def _new_network(self, n_inputs: int, target_mean: float) -> RatingNet:
        torch.manual_seed(self.seed)
        network = RatingNet(
            n_inputs,
            tuple(self.hidden),
            tuple(self.dropout),
            self.n_players_,
            self.player_bias,
            self.player_dim,
        )
        with torch.no_grad():  # start from the average rating instead of 0
            network.mlp[-1].bias.fill_(target_mean)
        return network

    def _train_one_epoch(self, network, optimizer, x, players, y, generator) -> None:
        network.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), self.batch_size):
            batch = order[start : start + self.batch_size]
            optimizer.zero_grad()
            prediction = network(x[batch], players[batch])
            loss = nn.functional.mse_loss(prediction, y[batch])
            loss = loss + network.penalty(self.l2, self.embedding_l2)
            loss.backward()
            optimizer.step()

    @staticmethod
    def _mse(network, x, players, y) -> float:
        network.eval()
        with torch.no_grad():
            return float(nn.functional.mse_loss(network(x, players), y))

    def _train_with_early_stopping(self, train, validation):
        x, players, y = train
        network = self._new_network(x.shape[1], float(y.mean()))
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr
        )
        generator = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, epochs_without_improvement = math.inf, None, 0
        schedule, curve = [], []
        for _ in range(self.max_epochs):
            schedule.append(optimizer.param_groups[0]["lr"])
            self._train_one_epoch(network, optimizer, x, players, y, generator)
            validation_loss = self._mse(network, *validation)
            curve.append(validation_loss)
            scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss, best_state = validation_loss, copy.deepcopy(network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        network.load_state_dict(best_state)
        return network, schedule, curve

    def _train_with_schedule(self, x, players, y, schedule: list[float]) -> RatingNet:
        network = self._new_network(x.shape[1], float(y.mean()))
        optimizer = torch.optim.Adam(network.parameters(), lr=schedule[0])
        generator = torch.Generator().manual_seed(self.seed)
        for lr in schedule:
            for group in optimizer.param_groups:
                group["lr"] = lr
            self._train_one_epoch(network, optimizer, x, players, y, generator)
        return network
