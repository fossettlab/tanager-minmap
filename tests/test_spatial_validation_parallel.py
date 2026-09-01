"""Determinism and filtering coverage for parallel block permutations."""

from __future__ import annotations

import numpy as np
import pytest

from tanager_minmap.spatial_validation import (
    BlockSample,
    complete_blocks,
    sample_blocks,
    whole_block_permutation_test,
)


def _parallel_samples() -> tuple[BlockSample, ...]:
    blocks = complete_blocks((2, 8), 2)
    return sample_blocks(
        np.array([[0.9, 0.9, 0.8, 0.1, 0.2, 0.2, 0.6, 0.6]] * 2),
        np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]] * 2),
        blocks,
    )


def _samples_with_invalid_permutations() -> tuple[BlockSample, ...]:
    blocks = complete_blocks((2, 8), 2)
    return sample_blocks(
        np.array([[0.9, 0.9, 0.8, 0.1, 0.2, 0.2, np.nan, np.nan]] * 2),
        np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]] * 2),
        blocks,
    )


def test_parallel_permutations_match_single_worker_exactly():
    samples = _parallel_samples()

    serial = whole_block_permutation_test(samples, halo_pixels=0, permutations=24, seed=42)
    single_worker = whole_block_permutation_test(
        samples, halo_pixels=0, permutations=24, seed=42, workers=1
    )
    two_workers = whole_block_permutation_test(
        samples, halo_pixels=0, permutations=24, seed=42, workers=2
    )

    assert single_worker == serial
    assert two_workers == single_worker


def test_parallel_permutations_are_repeatably_deterministic():
    samples = _parallel_samples()

    first = whole_block_permutation_test(samples, halo_pixels=0, permutations=24, seed=7, workers=2)
    second = whole_block_permutation_test(
        samples, halo_pixels=0, permutations=24, seed=7, workers=2
    )

    assert second == first


@pytest.mark.parametrize("workers", (0, -1, 1.5, True))
def test_whole_block_permutation_rejects_invalid_worker_counts(workers: object):
    with pytest.raises((TypeError, ValueError), match="workers"):
        whole_block_permutation_test(
            _parallel_samples(), halo_pixels=0, permutations=1, workers=workers
        )


def test_parallel_permutations_preserve_invalid_replicate_filtering():
    samples = _samples_with_invalid_permutations()

    single_worker = whole_block_permutation_test(
        samples, halo_pixels=0, permutations=11, seed=42, workers=1
    )
    two_workers = whole_block_permutation_test(
        samples, halo_pixels=0, permutations=11, seed=42, workers=2
    )

    assert two_workers == single_worker
    assert two_workers.valid_auc_permutations == 11
    assert 0 < two_workers.valid_balanced_accuracy_permutations < 11


def test_parallel_auc_null_retains_rank_blocks_with_failed_threshold_folds():
    blocks = complete_blocks((1, 5), 1)
    samples = sample_blocks(
        np.array([[4.0, 4.0, 1.0, 1.0, 1.0]]),
        np.array([[1.0, 1.0, 0.0, 0.0, 0.0]]),
        blocks,
    )

    serial = whole_block_permutation_test(
        samples, halo_pixels=1, permutations=24, seed=42, workers=1
    )
    parallel = whole_block_permutation_test(
        samples, halo_pixels=1, permutations=24, seed=42, workers=2
    )

    assert parallel == serial
    assert parallel.valid_auc_permutations == 24
    assert parallel.valid_balanced_accuracy_permutations == 0
    assert parallel.balanced_accuracy_p_value is None
