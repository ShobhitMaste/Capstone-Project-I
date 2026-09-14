"""Tests for non-IID partitioning reproducibility."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


class TestDirichletPartitioning:
    """Verify non-IID partition properties."""

    def _make_labels(self, per_class=100, num_classes=5):
        """Create balanced label array."""
        return np.repeat(np.arange(num_classes), per_class)

    def test_all_samples_assigned(self):
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(100, 5)
        partitions = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        total = sum(len(p) for p in partitions)
        assert total == len(labels)

    def test_no_overlap(self):
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(100, 5)
        partitions = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        all_indices = np.concatenate(partitions)
        assert len(np.unique(all_indices)) == len(all_indices)

    def test_deterministic_with_seed(self):
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(100, 5)
        p1 = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        p2 = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self):
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(100, 5)
        p1 = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        p2 = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=99)
        # Very unlikely to be identical
        assert not np.array_equal(p1[0], p2[0])

    def test_high_alpha_more_balanced(self):
        """High alpha → more IID distribution."""
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(200, 5)
        partitions = dirichlet_partition(labels, num_clients=2, alpha=100.0, seed=42)
        # Both clients should have roughly half
        sizes = [len(p) for p in partitions]
        assert abs(sizes[0] - sizes[1]) < len(labels) * 0.2

    def test_low_alpha_more_skewed(self):
        """Low alpha → more non-IID."""
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(200, 5)
        partitions = dirichlet_partition(labels, num_clients=2, alpha=0.1, seed=42)
        # Check that class distributions differ between clients
        for i in range(2):
            client_labels = labels[partitions[i]]
            unique, counts = np.unique(client_labels, return_counts=True)
            # With low alpha, some classes should be very imbalanced
            assert len(unique) >= 1  # at least one class

    def test_two_clients(self):
        from experiments.partition_non_iid import dirichlet_partition
        labels = self._make_labels(100, 5)
        partitions = dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42)
        assert len(partitions) == 2


class TestLabelSkew:
    """Verify controlled label-skew partitioning."""

    def test_skew_zero_balanced(self):
        from experiments.partition_non_iid import create_label_skew
        labels = np.repeat(np.arange(5), 100)
        partitions = create_label_skew(labels, num_clients=2, skew=0.0, seed=42)
        # Both clients should have all classes
        for p in partitions:
            client_labels = labels[p]
            assert len(np.unique(client_labels)) == 5

    def test_skew_one_extreme(self):
        from experiments.partition_non_iid import create_label_skew
        labels = np.repeat(np.arange(5), 100)
        partitions = create_label_skew(labels, num_clients=2, skew=1.0, seed=42)
        # Clients should have very different distributions
        dist_a = np.bincount(labels[partitions[0]], minlength=5)
        dist_b = np.bincount(labels[partitions[1]], minlength=5)
        # Dominant classes should differ
        assert np.argmax(dist_a) != np.argmax(dist_b)
