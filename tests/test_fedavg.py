"""Tests for FedAvg aggregation correctness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


class TestFedAvgAggregation:
    """Verify weighted FedAvg produces correct global model."""

    def test_equal_weight_averaging(self):
        """Two clients with equal data → simple average."""
        from federated.fedavg import weighted_fedavg
        params_a = [np.array([1.0, 2.0, 3.0])]
        params_b = [np.array([3.0, 4.0, 5.0])]
        n_a, n_b = 100, 100
        result = weighted_fedavg(
            [(params_a, n_a), (params_b, n_b)]
        )
        expected = [np.array([2.0, 3.0, 4.0])]
        np.testing.assert_allclose(result[0], expected[0])

    def test_unequal_weight_averaging(self):
        """Client A has 3x more data → result skewed toward A."""
        from federated.fedavg import weighted_fedavg
        params_a = [np.array([10.0])]
        params_b = [np.array([0.0])]
        n_a, n_b = 300, 100  # A has 75% of data
        result = weighted_fedavg(
            [(params_a, n_a), (params_b, n_b)]
        )
        # Expected: (300/400)*10 + (100/400)*0 = 7.5
        np.testing.assert_allclose(result[0], [7.5])

    def test_single_client(self):
        """One client → global = local."""
        from federated.fedavg import weighted_fedavg
        params = [np.array([1.0, 2.0])]
        result = weighted_fedavg([(params, 50)])
        np.testing.assert_allclose(result[0], params[0])

    def test_multiple_parameter_tensors(self):
        """Averaging works across multiple parameter arrays."""
        from federated.fedavg import weighted_fedavg
        params_a = [np.array([1.0, 2.0]), np.array([10.0])]
        params_b = [np.array([3.0, 4.0]), np.array([20.0])]
        result = weighted_fedavg(
            [(params_a, 50), (params_b, 50)]
        )
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [15.0])

    def test_zero_samples_handled(self):
        """Edge case: if a client reports 0 samples, skip it."""
        from federated.fedavg import weighted_fedavg
        params_a = [np.array([10.0])]
        params_b = [np.array([0.0])]
        result = weighted_fedavg(
            [(params_a, 100), (params_b, 0)]
        )
        np.testing.assert_allclose(result[0], [10.0])


class TestCommunicationMeasurement:
    """Verify byte-counting for FL communication."""

    def test_measure_bytes(self):
        from federated.fedavg import measure_parameter_bytes
        params = [np.ones((10,), dtype=np.float32)]  # 10 * 4 bytes = 40
        size = measure_parameter_bytes(params)
        assert size == 40

    def test_measure_bytes_multiple(self):
        from federated.fedavg import measure_parameter_bytes
        params = [
            np.ones((10,), dtype=np.float32),    # 40
            np.ones((5, 3), dtype=np.float32),   # 60
        ]
        size = measure_parameter_bytes(params)
        assert size == 100
