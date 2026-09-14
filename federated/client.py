"""FedTrap Flower client — on-device FL training participant.

Each Raspberry Pi trap runs one instance of this client.
Raw images NEVER leave the device — only model parameter numpy arrays
are transmitted via Flower gRPC.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl  # noqa: E402

from fedtrap.model import get_trainable_params, set_trainable_params  # noqa: E402


class FedTrapClient(fl.client.NumPyClient):
    """Flower NumPy client for FedTrap federated learning."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        local_epochs: int = 1,
        device: str = "cpu",
        lr: float = 0.001,
        method: str = "fedavg",
        mu: float = 0.01,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        self.lr = lr
        self.method = method
        self.mu = mu
        self.criterion = torch.nn.CrossEntropyLoss()

    # ---- Flower API ----

    def get_parameters(self, config: Dict | None = None) -> List[np.ndarray]:
        """Return trainable parameters as list of numpy arrays."""
        return get_trainable_params(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load numpy arrays into model trainable parameters."""
        set_trainable_params(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training round."""
        self.set_parameters(parameters)
        start = time.time()

        if self.method == "fedprox":
            from federated.fedprox import train_fedprox

            global_params = [torch.tensor(p).to(self.device) for p in parameters]
            loss, acc = train_fedprox(
                self.model, self.train_loader, global_params,
                self.local_epochs, self.lr, self.mu, self.device,
            )
        else:
            loss, acc = self._train_local()

        train_time = time.time() - start
        updated_params = self.get_parameters(config)

        # Measure communication
        bytes_sent = sum(p.nbytes for p in updated_params)
        bytes_received = sum(p.nbytes for p in parameters)

        num_examples = len(self.train_loader.dataset)
        metrics = {
            "loss": float(loss),
            "accuracy": float(acc),
            "train_time": train_time,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
        }
        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """Evaluate global model on local validation data."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / max(1, total)
        avg_acc = correct / max(1, total)
        num_examples = len(self.val_loader.dataset)
        return avg_loss, num_examples, {"accuracy": float(avg_acc)}

    # ---- Internal ----

    def _train_local(self) -> Tuple[float, float]:
        """Standard local SGD training."""
        self.model.to(self.device)
        self.model.train()
        optimiser = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.lr,
        )

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimiser.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / max(1, total)
        avg_acc = correct / max(1, total)
        return avg_loss, avg_acc
