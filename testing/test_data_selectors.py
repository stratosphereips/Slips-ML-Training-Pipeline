import pytest
import numpy as np
import pandas as pd
from src.data_selectors import (
    SequenceMixer,
    RandomBatchesMixer,
    BalancedByLabelMixer,
    BalancedByLabelWithOversamplingMixer,
    batch_len,
    is_dataframe,
    concat_batches,
    slice_batch,
)


class MockLoader:
    """Mock loader that yields batches sequentially."""

    def __init__(self, batches, name="loader"):
        self.batches = batches
        self.name = name
        self.batch_idx = 0

    def reset_epoch(self, batch_size=None):
        self.batch_idx = 0

    def next_batch(self):
        if self.batch_idx >= len(self.batches):
            return None
        batch = self.batches[self.batch_idx]
        self.batch_idx += 1
        return batch

    def next_n(self, n):
        """Return up to n samples."""
        collected = []
        need = n
        while need > 0 and self.batch_idx < len(self.batches):
            batch = self.next_batch()
            if batch is None:
                break
            batch_size = batch_len(batch)
            if batch_size <= need:
                collected.append(batch)
                need -= batch_size
            else:
                # Slice the batch
                if is_dataframe(batch):
                    part = batch.iloc[:need].reset_index(drop=True)
                else:
                    part = batch[:need]
                collected.append(part)
                need = 0

        if not collected:
            return None
        return concat_batches(collected)

    def as_dataframe(self):
        """Return the entire dataset as a DataFrame to mimic real loaders."""
        combined = concat_batches(self.batches)
        if combined is None:
            return pd.DataFrame()
        if is_dataframe(combined):
            return combined.reset_index(drop=True)
        if isinstance(combined, list):
            if not combined:
                return pd.DataFrame()
            first = combined[0]
            if isinstance(first, dict):
                return pd.DataFrame(combined).reset_index(drop=True)
            return pd.DataFrame({"value": combined}).reset_index(drop=True)
        return pd.DataFrame({"value": np.atleast_1d(combined)}).reset_index(drop=True)


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_batches_list():
    """Sample list-based batches."""
    return [
        [{"id": 1, "label": "benign"}, {"id": 2, "label": "benign"}],
        [{"id": 3, "label": "malicious"}, {"id": 4, "label": "benign"}],
        [{"id": 5, "label": "malicious"}],
    ]


@pytest.fixture
def sample_batches_dataframe():
    """Sample DataFrame-based batches."""
    df1 = pd.DataFrame({"id": [1, 2], "label": ["benign", "benign"]})
    df2 = pd.DataFrame({"id": [3, 4], "label": ["malicious", "benign"]})
    df3 = pd.DataFrame({"id": [5], "label": ["malicious"]})
    return [df1, df2, df3]


# ========== Helper Functions Tests ==========
class TestHelperFunctions:
    """Test utility functions."""

    def test_batch_len_with_various_types(self):
        """Test batch_len with list, DataFrame, dict, None."""
        assert batch_len([1, 2, 3]) == 3
        assert batch_len([]) == 0
        assert batch_len(None) == 0
        assert batch_len({"key": "value"}) == 1

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert batch_len(df) == 3

    def test_is_dataframe_detection(self):
        """Test is_dataframe correctly identifies DataFrames."""
        assert is_dataframe(pd.DataFrame())
        assert not is_dataframe([1, 2, 3])
        assert not is_dataframe(None)
        assert not is_dataframe({"a": 1})

    def test_concat_batches_lists(self):
        """Test concat_batches concatenates lists correctly."""
        result = concat_batches([[1, 2], [3, 4], [5]])
        assert result == [1, 2, 3, 4, 5]

    def test_concat_batches_with_none_values(self):
        """Test concat_batches skips None values."""
        result = concat_batches([[1, 2], None, [3, 4], None])
        assert result == [1, 2, 3, 4]

    def test_concat_batches_empty_list(self):
        """Test concat_batches returns None for empty/all-None input."""
        assert concat_batches([]) is None
        assert concat_batches([None, None]) is None

    def test_concat_batches_dataframes(self):
        """Test concat_batches concatenates DataFrames correctly."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = concat_batches([df1, df2])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 3, 4]

    def test_concat_batches_mixed_none_dataframes(self):
        """Test concat_batches with mixed DataFrames and None."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = concat_batches([df1, None, df2])
        assert len(result) == 4

    def test_slice_batch_lists(self):
        """Test slice_batch slices lists correctly."""
        batch = [1, 2, 3, 4, 5]
        assert slice_batch(batch, 1, 3) == [2, 3]
        assert slice_batch(batch, 0, 2) == [1, 2]
        assert slice_batch(batch, 0, 0) == []

    def test_slice_batch_none(self):
        """Test slice_batch returns None for None input."""
        assert slice_batch(None, 0, 1) == []

    def test_slice_batch_dataframe(self):
        """Test slice_batch slices DataFrames correctly."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = slice_batch(df, 1, 3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["a"]) == [2, 3]

# ========== SequenceMixer Tests ==========
class TestSequenceMixer:
    """Test SequenceMixer - drains datasets sequentially."""

    def test_initialization_requires_datasets(self, rng):
        """Test SequenceMixer requires 'datasets' in spec."""
        with pytest.raises(ValueError):
            SequenceMixer({}, {}, rng)

        with pytest.raises(ValueError):
            SequenceMixer({"datasets": []}, {}, rng)

    def test_sequence_drain_single_dataset_lists(self, sample_batches_list, rng):
        """Test draining a single list-based dataset sequentially."""
        loader = MockLoader(sample_batches_list, "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)

        batches = []
        while True:
            train, val = mixer.next_batch()
            if train is None:
                break
            batches.append(train)
        total_rows = sum(len(batch) for batch in sample_batches_list)
        assert len(batches) == 1
        assert batch_len(batches[0]) == total_rows
        assert mixer.next_batch() == (None, None)

    def test_sequence_drain_single_dataset_dataframes(self, sample_batches_dataframe, rng):
        """Test draining a single DataFrame-based dataset."""
        loader = MockLoader(sample_batches_dataframe, "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)

        batches = []
        while True:
            train, val = mixer.next_batch()
            if train is None:
                break
            batches.append(train)

        total_rows = sum(len(df) for df in sample_batches_dataframe)
        assert len(batches) == 1
        assert batch_len(batches[0]) == total_rows

    def test_sequence_multiple_datasets(self, rng):
        """Test draining multiple datasets in sequence."""
        loader1 = MockLoader([[1, 2], [3]], "ds1")
        loader2 = MockLoader([[4, 5]], "ds2")

        spec = {"datasets": ["ds1", "ds2"]}
        mixer = SequenceMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)

        batches = []
        while True:
            train, val = mixer.next_batch()
            if train is None:
                break
            batches.append(batch_len(train))

        assert batches == [3, 2]

    def test_sequence_with_validation_split(self, sample_batches_list, rng):
        """Test SequenceMixer with validation split."""
        loader = MockLoader(sample_batches_list, "ds1")
        spec = {"datasets": ["ds1"], "validation_split": 0.3}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None
        assert val is not None
        total = batch_len(train) + batch_len(val)
        expected_total = sum(len(batch) for batch in sample_batches_list)
        assert total == expected_total
        expected_val = int(round(0.3 * expected_total))
        assert batch_len(val) == expected_val


    def test_sequence_no_validation_split(self, sample_batches_list, rng):
        """Test SequenceMixer without validation split."""
        loader = MockLoader(sample_batches_list, "ds1")
        spec = {"datasets": ["ds1"], "validation_split": 0.0}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None
        assert val is None

    def test_sequence_per_dataset_batch_size(self, rng):
        """Test per_dataset_batch_size parameter."""
        loader = MockLoader([[1, 2, 3, 4, 5]], "ds1")
        spec = {"datasets": ["ds1"], "per_dataset_batch_size": 3}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)  # batch_size ignored, per_dataset_batch_size used
        train, val = mixer.next_batch()
        assert train is not None
        assert batch_len(train) == 3

        train2, _ = mixer.next_batch()
        assert train2 is not None
        assert batch_len(train2) == 2

        train3, _ = mixer.next_batch()
        assert train3 is None

    def test_sequence_mix_plan_recording(self, sample_batches_list, rng):
        """Test that mix_plan records operations."""
        loader = MockLoader(sample_batches_list, "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        mixer.next_batch()

        plan = mixer.get_mix_plan()
        assert len(plan) == 1
        assert plan[0]["type"] == "sequence"
        assert plan[0]["dataset"] == "ds1"
        assert "train_count" in plan[0]
        assert "val_count" in plan[0]
        expected_total = sum(len(batch) for batch in sample_batches_list)
        assert plan[0]["dataset_batch_size"] == expected_total


# ========== RandomBatchesMixer Tests ==========
class TestRandomBatchesMixer:
    """Test RandomBatchesMixer - random mixed batches."""

    def test_initialization_requires_datasets(self, rng):
        """Test RandomBatchesMixer requires 'datasets'."""
        with pytest.raises(ValueError):
            RandomBatchesMixer({}, {}, rng)

    def test_weights_resized_and_normalized(self, rng):
        """Test custom weights are resized and normalized to dataset count."""
        loader1 = MockLoader([[1, 2]], "ds1")
        loader2 = MockLoader([[3, 4]], "ds2")
        mixer = RandomBatchesMixer(
            {"datasets": ["ds1", "ds2"], "weights": [1.0]},
            {"ds1": loader1, "ds2": loader2},
            rng
        )
        assert len(mixer.weights) == 2
        assert mixer.weights[0] == pytest.approx(0.5)
        assert mixer.weights[1] == pytest.approx(0.5)

    def test_random_unbalanced_with_weights(self, rng):
        """Test random mixing with custom weights."""
        loader1 = MockLoader([[1, 2, 3, 4, 5]], "ds1")
        loader2 = MockLoader([[6, 7, 8, 9]], "ds2")

        spec = {
            "datasets": ["ds1", "ds2"],
            "balanced": False,
            "weights": [0.7, 0.3]
        }
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None
        assert batch_len(train) <= 10

    def test_random_balanced_mixing(self, rng):
        """Test random mixing with balanced=True."""
        loader1 = MockLoader([[1, 2, 3, 4, 5, 6]], "ds1")
        loader2 = MockLoader([[7, 8, 9, 10, 11, 12]], "ds2")

        spec = {
            "datasets": ["ds1", "ds2"],
            "balanced": True
        }
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None
        total = batch_len(train) if train is not None else 0
        assert total <= 10

    def test_random_batches_default_weights(self, rng):
        """Test that default weights are uniform."""
        loader1 = MockLoader([[1, 2]], "ds1")
        loader2 = MockLoader([[3, 4]], "ds2")
        loader3 = MockLoader([[5, 6]], "ds3")

        spec = {"datasets": ["ds1", "ds2", "ds3"]}
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2, "ds3": loader3}, rng)

        assert mixer.weights[0] == pytest.approx(1/3)
        assert mixer.weights[1] == pytest.approx(1/3)
        assert mixer.weights[2] == pytest.approx(1/3)

    def test_random_batches_with_validation_split(self, rng):
        """Test RandomBatchesMixer with validation split."""
        loader1 = MockLoader([[1, 2, 3, 4, 5]], "ds1")
        loader2 = MockLoader([[6, 7, 8, 9]], "ds2")

        spec = {
            "datasets": ["ds1", "ds2"],
            "validation_split": 0.25
        }
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        if train is not None and val is not None:
            total = batch_len(train) + batch_len(val)
            val_ratio = batch_len(val) / total if total > 0 else 0
            assert 0.15 <= val_ratio <= 0.35

    def test_random_batches_mix_plan(self, rng):
        """Test that random mixer records in mix_plan."""
        loader1 = MockLoader([[1, 2]], "ds1")
        loader2 = MockLoader([[3, 4]], "ds2")

        spec = {"datasets": ["ds1", "ds2"]}
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        mixer.next_batch()

        plan = mixer.get_mix_plan()
        assert len(plan) == 1
        assert plan[0]["type"] == "random_batches"
        assert "counts_by_dataset" in plan[0]


# ========== BalancedByLabelMixer Tests ==========
class TestBalancedByLabelMixer:
        def test_balanced_by_label_strict_balancing_and_exhaustion(self, rng):
            """Balanced mixer should produce equal label counts until one label drains."""
            batches1 = [
                [{"label": "benign"}, {"label": "benign"}],
                [{"label": "benign"}, {"label": "benign"}]
            ]
            batches2 = [
                [{"label": "malicious"}, {"label": "malicious"}]
            ]

            loader1 = MockLoader(batches1, "ds1")
            loader2 = MockLoader(batches2, "ds2")

            spec = {
                "datasets": ["ds1", "ds2"],
                "labels": ["benign", "malicious"]
            }
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=4)

            train, _ = mixer.next_batch()
            assert train is not None
            label_counts = train["label"].value_counts().to_dict()
            assert label_counts["benign"] == label_counts["malicious"] == 2

            train2, _ = mixer.next_batch()
            assert train2 is None


        def test_initialization_requires_datasets(self, rng):
            """Test BalancedByLabelMixer requires 'datasets'."""
            with pytest.raises(ValueError):
                BalancedByLabelMixer({}, {}, rng)

        def test_label_discovery_from_list_data(self, rng):
            """Test automatic label discovery from list data."""
            batches1 = [
                [{"label": "benign"}, {"label": "benign"}],
                [{"label": "malicious"}]
            ]
            batches2 = [
                [{"label": "malicious"}, {"label": "benign"}]
            ]

            loader1 = MockLoader(batches1, "ds1")
            loader2 = MockLoader(batches2, "ds2")

            spec = {"datasets": ["ds1", "ds2"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=10)

            assert set(mixer.labels) == {"benign", "malicious"}

        def test_label_discovery_from_dataframe_data(self, rng):
            """Test label discovery from DataFrame data."""
            df1 = pd.DataFrame({"label": ["benign", "benign"]})
            df2 = pd.DataFrame({"label": ["malicious", "benign"]})

            loader1 = MockLoader([df1], "ds1")
            loader2 = MockLoader([df2], "ds2")

            spec = {"datasets": ["ds1", "ds2"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=10)

            assert set(mixer.labels) == {"benign", "malicious"}

        def test_provided_labels_override_discovery(self, rng):
            """Test provided labels override automatic discovery."""
            batches = [[{"label": "benign"}], [{"label": "malicious"}]]
            loader = MockLoader(batches, "ds1")

            spec = {
                "datasets": ["ds1"],
                "labels": ["type_a", "type_b", "type_c"]
            }
            mixer = BalancedByLabelMixer(spec, {"ds1": loader}, rng)

            mixer.reset_epoch(batch_size=10)

            assert mixer.labels == ["type_a", "type_b", "type_c"]

        def test_balanced_by_label_no_labels_defaults(self, rng):
            """Test that default labels are used when none discovered."""
            loader = MockLoader([[{"id": 1}]], "ds1")  # No 'label' field

            spec = {"datasets": ["ds1"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader}, rng)

            mixer.reset_epoch(batch_size=10)

            # Without label information the mixer cannot build label tables
            assert mixer.labels == []
            train, _ = mixer.next_batch()
            assert train is None

        def test_balanced_by_label_produces_mixed_batch(self, rng):
            """Test that balanced mixer tries to balance labels."""
            batches1 = [
                [{"label": "benign"}, {"label": "benign"}],
                [{"label": "benign"}, {"label": "benign"}]
            ]
            batches2 = [
                [{"label": "malicious"}, {"label": "malicious"}],
                [{"label": "malicious"}, {"label": "malicious"}]
            ]

            loader1 = MockLoader(batches1, "ds1")
            loader2 = MockLoader(batches2, "ds2")

            spec = {
                "datasets": ["ds1", "ds2"],
                "labels": ["benign", "malicious"]
            }
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=8)
            train, _ = mixer.next_batch()

            assert train is not None
            labels_in_batch = train["label"].value_counts().to_dict()
            assert set(labels_in_batch.keys()) == {"benign", "malicious"}
            assert labels_in_batch["benign"] == labels_in_batch["malicious"] == 4

        def test_balanced_by_label_mix_plan(self, rng):
            """Test that balanced mixer records in mix_plan."""
            batches = [[{"label": "benign"}], [{"label": "malicious"}]]
            loader = MockLoader(batches, "ds1")

            spec = {"datasets": ["ds1"], "labels": ["benign", "malicious"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader}, rng)

            mixer.reset_epoch(batch_size=2)
            mixer.next_batch()

            plan = mixer.get_mix_plan()
            assert len(plan) == 1
            assert plan[0]["type"] == "balanced_by_label"
            assert "counts_by_label" in plan[0]


class TestBalancedByLabelWithOversampling:
    def test_oversampling_balances_when_label_missing(self, rng):
        """Oversampling mixer should duplicate minority labels to fill batch."""
        majority = [[{"label": "benign"}, {"label": "benign"}]]
        minority = [[{"label": "malicious"}]]

        loader1 = MockLoader(majority, "ds1")
        loader2 = MockLoader(minority, "ds2")

        spec = {
            "datasets": ["ds1", "ds2"],
            "labels": ["benign", "malicious"]
        }
        mixer = BalancedByLabelWithOversamplingMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=4)
        train, _ = mixer.next_batch()

        assert train is not None
        counts = train["label"].value_counts().to_dict()
        assert counts["benign"] == counts["malicious"] == 2

        plan = mixer.get_mix_plan()
        assert plan[0]["type"] == "balanced_oversampling"
        assert plan[0]["resampled_by_label"]["malicious"] >= 1


# ========== Integration Tests ==========
class TestMixerIntegration:
    """Integration tests for mixers."""

    def test_reset_epoch_clears_plan(self, rng):
        """Test that reset_epoch clears mix_plan."""
        loader = MockLoader([[1, 2], [3, 4]], "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        mixer.next_batch()
        assert len(mixer.get_mix_plan()) > 0

        mixer.reset_epoch(batch_size=10)
        assert len(mixer.get_mix_plan()) == 0

    def test_multiple_epochs_with_reset(self, rng):
        """Test that resetting epochs resets loaders."""
        loader = MockLoader([[1, 2], [3, 4]], "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train1, _ = mixer.next_batch()

        mixer.reset_epoch(batch_size=10)
        train2, _ = mixer.next_batch()

        # Should get same batch after reset
        assert batch_len(train1) == batch_len(train2)

    def test_dataset_resolution_with_partial_match(self, rng):
        """Test that dataset keys can be resolved with partial matches."""
        loader = MockLoader([[1, 2]], "dataset_001")
        spec = {"datasets": ["001"]}  # Partial match
        mixer = SequenceMixer(spec, {"dataset_001": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None

    def test_no_replacement_exhaustion(self, rng):
        """Test that mixers stop when data is exhausted (no replacement)."""
        loader = MockLoader([[1], [2], [3]], "ds1")
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)

        batches = []
        while True:
            train, val = mixer.next_batch()
            if train is None:
                break
            batches.append(train)

        # Default behavior consumes full dataset in one batch
        assert len(batches) == 1
        assert batch_len(batches[0]) == 3
