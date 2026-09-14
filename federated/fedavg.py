"""FedTrap FedAvg — weighted Federated Averaging.

Provides both:
- ``FedAvgWithMetrics``: Flower Strategy subclass with communication logging
- ``weighted_fedavg``: standalone function for manual aggregation / testing
- ``measure_parameter_bytes``: byte-count utility
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import flwr as fl
    from flwr.common import FitRes, Parameters, Scalar
    from flwr.server.client_proxy import ClientProxy

    _HAS_FLOWER = True
except ImportError:
    _HAS_FLOWER = False


# =====================================================================
# Standalone helpers (no Flower dependency — used by tests + manual FL)
# =====================================================================


def weighted_fedavg(
    client_updates: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """Weighted FedAvg: w_global = Σ(n_i / N) * w_i.

    Args:
        client_updates: list of (parameters, num_samples) per client.
            Each ``parameters`` is a list of numpy arrays (one per layer).

    Returns:
        Aggregated list of numpy arrays.
    """
    # Filter out clients with zero samples
    filtered = [(params, n) for params, n in client_updates if n > 0]
    if not filtered:
        # Edge case: return first client's params unchanged
        return client_updates[0][0]

    total_samples = sum(n for _, n in filtered)
    num_layers = len(filtered[0][0])

    global_params: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(filtered[0][0][layer_idx], dtype=np.float64)
        for params, n in filtered:
            weighted_sum += (n / total_samples) * params[layer_idx].astype(np.float64)
        global_params.append(weighted_sum.astype(filtered[0][0][layer_idx].dtype))

    return global_params


def measure_parameter_bytes(params: List[np.ndarray]) -> int:
    """Total bytes occupied by a list of numpy parameter arrays."""
    return sum(p.nbytes for p in params)


# =====================================================================
# Flower Strategy (requires flwr)
# =====================================================================


if _HAS_FLOWER:

    class FedAvgWithMetrics(fl.server.strategy.FedAvg):
        """FedAvg with per-round communication measurement."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metrics_history: List[Dict] = []
            self.total_bytes_received: int = 0
            self.total_bytes_sent: int = 0

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            if not results:
                return None, {}

            start = time.time()

            # Measure bytes from client fit metrics
            round_recv = sum(
                fr.metrics.get("bytes_sent", 0) for _, fr in results
            )
            round_sent = sum(
                fr.metrics.get("bytes_received", 0) for _, fr in results
            )
            self.total_bytes_received += round_recv
            self.total_bytes_sent += round_sent

            # Standard FedAvg aggregation via parent
            agg_params, agg_metrics = super().aggregate_fit(
                server_round, results, failures
            )

            agg_time = time.time() - start
            sample_counts = [fr.num_examples for _, fr in results]

            round_metrics = {
                "round": server_round,
                "client_sample_counts": sample_counts,
                "round_bytes_received": round_recv,
                "round_bytes_sent": round_sent,
                "total_bytes_received": self.total_bytes_received,
                "total_bytes_sent": self.total_bytes_sent,
                "aggregation_time": agg_time,
            }
            self.metrics_history.append(round_metrics)

            if agg_metrics is None:
                agg_metrics = {}

            return agg_params, agg_metrics

        def save_communication_log(self, path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, indent=2)
