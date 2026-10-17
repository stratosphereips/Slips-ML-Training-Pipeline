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
    _allocate_from_fractions,
)


def _zeek_like_frame(labels, start_id=0):
    """Frame shaped like ZeekDataset.as_dataframe(): label in ground_truth_label, empty 'label'."""
    return pd.DataFrame({
        "id": range(start_id, start_id + len(labels)),
        "ground_truth_label": labels,
        "label": [""] * len(labels),
    })


class MockLoader:
    """Mock loader exposing as_dataframe(), like ZeekDataset (mixers only call that)."""

    def __init__(self, batches):
        self.batches = batches

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

    def test_allocate_from_fractions_basic(self):
        """Fractions are turned into integer counts that sum to total."""
        assert _allocate_from_fractions([0.7, 0.3], 10) == [7, 3]
        assert _allocate_from_fractions([1, 1, 1], 10) == [4, 3, 3]
        assert _allocate_from_fractions([0.5, 0.5], 0) == [0, 0]
        # negative weights are clipped, all-zero weights fall back to uniform
        assert _allocate_from_fractions([-1.0, 1.0], 4) == [0, 4]
        assert _allocate_from_fractions([0.0, 0.0], 4) == [2, 2]

    def test_allocate_from_fractions_without_rng_is_stable(self):
        """Without an rng, leftover units go to tied labels in a stable (first-label) order."""
        for _ in range(20):
            assert _allocate_from_fractions([0.5, 0.5], 1) == [1, 0]

    @pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
    def test_allocate_from_fractions_with_rng_breaks_ties_randomly(self, make_rng):
        """With an rng, a leftover unit goes to either tied label, not always the first one."""
        rng = make_rng(0)
        draws = [_allocate_from_fractions([0.5, 0.5], 1, rng=rng) for _ in range(200)]
        assert all(sum(d) == 1 for d in draws)
        first = sum(d[0] for d in draws)
        assert 50 < first < 150
        # a non-tied remainder still goes to the label with the largest fractional part
        assert _allocate_from_fractions([0.6, 0.4], 1, rng=rng) == [1, 0]

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
        loader = MockLoader(sample_batches_list)
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
        loader = MockLoader(sample_batches_dataframe)
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
        loader1 = MockLoader([[1, 2], [3]])
        loader2 = MockLoader([[4, 5]])

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
        loader = MockLoader(sample_batches_list)
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
        assert sorted(train["id"].tolist() + val["id"].tolist()) == [1, 2, 3, 4, 5]

    def test_sequence_validation_split_is_cumulative_over_batches(self, rng):
        """Over several small batches the validation share is rounded cumulatively, not per batch."""
        frame = pd.DataFrame({"id": range(8)})
        spec = {"datasets": ["ds1"], "validation_split": 0.25, "per_dataset_batch_size": 2}
        mixer = SequenceMixer(spec, {"ds1": frame}, rng)
        mixer.reset_epoch(batch_size=2)

        val_counts, train_ids, val_ids = [], [], []
        for _ in range(100):
            train, val = mixer.next_batch()
            if train is None:
                break
            val_counts.append(batch_len(val))
            train_ids.extend(train["id"].tolist())
            if val is not None:
                val_ids.extend(val["id"].tolist())
        else:
            pytest.fail("epoch never ended")

        # per-batch rounding would give round(0.25 * 2) == 0 validation rows in every batch;
        # cumulative rounding gives round(0.25 * 8) == 2 over the epoch
        assert val_counts == [0, 1, 1, 0]
        assert [entry["val_count"] for entry in mixer.get_mix_plan()] == val_counts
        assert len(val_ids) == 2
        assert set(train_ids).isdisjoint(val_ids)
        assert sorted(train_ids + val_ids) == list(range(8))


    def test_sequence_no_validation_split(self, sample_batches_list, rng):
        """Test SequenceMixer without validation split."""
        loader = MockLoader(sample_batches_list)
        spec = {"datasets": ["ds1"], "validation_split": 0.0}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None
        assert val is None

    def test_sequence_per_dataset_batch_size(self, rng):
        """Test per_dataset_batch_size parameter."""
        loader = MockLoader([[1, 2, 3, 4, 5]])
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
        loader = MockLoader(sample_batches_list)
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
        loader1 = MockLoader([[1, 2]])
        loader2 = MockLoader([[3, 4]])
        mixer = RandomBatchesMixer(
            {"datasets": ["ds1", "ds2"], "weights": [1.0]},
            {"ds1": loader1, "ds2": loader2},
            rng
        )
        assert len(mixer.weights) == 2
        assert mixer.weights[0] == pytest.approx(0.5)
        assert mixer.weights[1] == pytest.approx(0.5)

    def test_random_unbalanced_with_weights(self, rng):
        """Per-batch dataset counts follow a multinomial draw with the given weights."""
        ds1 = pd.DataFrame({"value": range(0, 100)})
        ds2 = pd.DataFrame({"value": range(100, 200)})

        spec = {"datasets": ["ds1", "ds2"], "weights": [0.7, 0.3]}
        mixer = RandomBatchesMixer(spec, {"ds1": ds1, "ds2": ds2}, rng)
        # the first rng call of an epoch is the multinomial draw of the first batch
        expected = np.random.RandomState(42).multinomial(10, mixer.weights).tolist()

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert val is None
        assert batch_len(train) == 10
        counts = mixer.get_mix_plan()[0]["counts_by_dataset"]
        assert [counts["ds1"], counts["ds2"]] == expected
        # rows are consumed in order from each dataset
        assert sorted(train["value"]) == list(range(expected[0])) + list(range(100, 100 + expected[1]))

        # over many batches the share of ds1 approaches its weight
        for _ in range(9):
            mixer.next_batch()
        from_ds1 = sum(entry["counts_by_dataset"]["ds1"] for entry in mixer.get_mix_plan())
        assert 0.6 <= from_ds1 / 100 <= 0.8

    def test_random_batches_fill_from_other_datasets(self, rng):
        """A batch is filled up to batch_size from other datasets; the epoch ends when all are drained."""
        loader1 = MockLoader([[1, 2, 3, 4, 5, 6]])
        loader2 = MockLoader([[7, 8, 9, 10, 11, 12]])

        spec = {"datasets": ["ds1", "ds2"]}
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        train1, val1 = mixer.next_batch()
        train2, val2 = mixer.next_batch()

        assert val1 is None and val2 is None
        assert batch_len(train1) == 10
        assert batch_len(train2) == 2
        assert sorted(train1["value"].tolist() + train2["value"].tolist()) == list(range(1, 13))
        for entry in mixer.get_mix_plan():
            assert sum(entry["counts_by_dataset"].values()) == entry["train_count"]
        assert mixer.next_batch() == (None, None)

    def test_random_batches_default_weights(self, rng):
        """Test that default weights are uniform."""
        loader1 = MockLoader([[1, 2]])
        loader2 = MockLoader([[3, 4]])
        loader3 = MockLoader([[5, 6]])

        spec = {"datasets": ["ds1", "ds2", "ds3"]}
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2, "ds3": loader3}, rng)

        assert mixer.weights[0] == pytest.approx(1/3)
        assert mixer.weights[1] == pytest.approx(1/3)
        assert mixer.weights[2] == pytest.approx(1/3)

    def test_random_batches_with_validation_split(self, rng):
        """Test RandomBatchesMixer with validation split."""
        loader1 = MockLoader([[1, 2, 3, 4, 5]])
        loader2 = MockLoader([[6, 7, 8, 9]])

        spec = {
            "datasets": ["ds1", "ds2"],
            "validation_split": 0.25
        }
        mixer = RandomBatchesMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        # all 9 rows land in one batch; round(0.25 * 9) == 2 go to validation
        assert batch_len(train) == 7
        assert batch_len(val) == 2
        assert set(train["value"]).isdisjoint(val["value"])
        assert sorted(train["value"].tolist() + val["value"].tolist()) == list(range(1, 10))
        assert mixer.get_mix_plan()[0]["val_count"] == 2

    def test_random_batches_mix_plan(self, rng):
        """Test that random mixer records in mix_plan."""
        loader1 = MockLoader([[1, 2]])
        loader2 = MockLoader([[3, 4]])

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

            loader1 = MockLoader(batches1)
            loader2 = MockLoader(batches2)

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

            loader1 = MockLoader(batches1)
            loader2 = MockLoader(batches2)

            spec = {"datasets": ["ds1", "ds2"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=10)

            assert set(mixer.labels) == {"benign", "malicious"}

        def test_label_discovery_from_dataframe_data(self, rng):
            """Test label discovery from DataFrame data."""
            df1 = pd.DataFrame({"label": ["benign", "benign"]})
            df2 = pd.DataFrame({"label": ["malicious", "benign"]})

            loader1 = MockLoader([df1])
            loader2 = MockLoader([df2])

            spec = {"datasets": ["ds1", "ds2"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader1, "ds2": loader2}, rng)

            mixer.reset_epoch(batch_size=10)

            assert set(mixer.labels) == {"benign", "malicious"}

        def test_provided_labels_override_discovery(self, rng):
            """Test provided labels override automatic discovery."""
            batches = [[{"label": "benign"}], [{"label": "malicious"}]]
            loader = MockLoader(batches)

            spec = {
                "datasets": ["ds1"],
                "labels": ["type_a", "type_b", "type_c"]
            }
            mixer = BalancedByLabelMixer(spec, {"ds1": loader}, rng)

            mixer.reset_epoch(batch_size=10)

            assert mixer.labels == ["type_a", "type_b", "type_c"]

        def test_balanced_by_label_no_labels_defaults(self, rng):
            """Test that default labels are used when none discovered."""
            loader = MockLoader([[{"id": 1}]])  # No 'label' field

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

            loader1 = MockLoader(batches1)
            loader2 = MockLoader(batches2)

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
            loader = MockLoader(batches)

            spec = {"datasets": ["ds1"], "labels": ["benign", "malicious"]}
            mixer = BalancedByLabelMixer(spec, {"ds1": loader}, rng)

            mixer.reset_epoch(batch_size=2)
            mixer.next_batch()

            plan = mixer.get_mix_plan()
            assert len(plan) == 1
            assert plan[0]["type"] == "balanced_by_label"
            assert plan[0]["counts_by_label"] == {"benign": 1, "malicious": 1}

        def test_balanced_groups_by_ground_truth_label(self, rng):
            """ZeekDataset-shaped frames (label in ground_truth_label, empty 'label') are balanced."""
            ds1 = _zeek_like_frame(["Benign"] * 6 + ["Malicious"] * 2, start_id=0)
            ds2 = _zeek_like_frame(["Malicious"] * 4, start_id=100)

            mixer = BalancedByLabelMixer({"datasets": ["ds1", "ds2"]}, {"ds1": ds1, "ds2": ds2}, rng)
            mixer.reset_epoch(batch_size=4)
            assert mixer.labels == ["Benign", "Malicious"]

            seen = []
            for _ in range(3):
                train, val = mixer.next_batch()
                assert val is None
                assert train["ground_truth_label"].value_counts().to_dict() == {"Benign": 2, "Malicious": 2}
                seen.extend(train["id"].tolist())
            assert mixer.next_batch() == (None, None)
            # every row is used exactly once
            assert sorted(seen) == sorted(ds1["id"].tolist() + ds2["id"].tolist())

        def test_ground_truth_label_wins_over_label_column(self, rng):
            """When both columns exist, grouping uses ground_truth_label, not label."""
            frame = _zeek_like_frame(["Benign"] * 3 + ["Malicious"] * 3)
            frame["label"] = "something-else"

            mixer = BalancedByLabelMixer({"datasets": ["ds"]}, {"ds": frame}, rng)
            mixer.reset_epoch(batch_size=2)

            assert mixer.labels == ["Benign", "Malicious"]
            train, _ = mixer.next_batch()
            assert sorted(train["ground_truth_label"]) == ["Benign", "Malicious"]

        def test_take_for_label_returns_frame_or_none(self, rng):
            """_take_for_label returns a DataFrame of `need` fresh rows, or None if too few are left."""
            frame = _zeek_like_frame(["Benign"] * 3 + ["Malicious"] * 1)
            mixer = BalancedByLabelMixer({"datasets": ["ds"]}, {"ds": frame}, rng)
            mixer.reset_epoch(batch_size=2)

            chunk = mixer._take_for_label("Benign", 2)
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) == 2
            assert set(chunk["ground_truth_label"]) == {"Benign"}
            # only one fresh Benign row left -> asking for 2 gives None
            assert mixer._take_for_label("Benign", 2) is None
            assert mixer._take_for_label("unknown-label", 1) is None


class TestBalancedByLabelWithOversampling:
    def test_oversampling_balances_when_label_missing(self, rng):
        """Oversampling mixer should duplicate minority labels to fill batch."""
        majority = [[{"label": "benign"}, {"label": "benign"}]]
        minority = [[{"label": "malicious"}]]

        loader1 = MockLoader(majority)
        loader2 = MockLoader(minority)

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
        assert plan[0]["resampled_by_label"] == {"benign": 0, "malicious": 1}
        # no fresh rows left for any label -> epoch over
        assert mixer.next_batch() == (None, None)

    def test_epoch_ends_when_every_label_is_exhausted(self, rng):
        """The epoch runs until the largest label has no fresh rows; smaller labels are oversampled."""
        frame = _zeek_like_frame(["Benign"] * 3 + ["Malicious"] * 9)
        mixer = BalancedByLabelWithOversamplingMixer({"datasets": ["ds"]}, {"ds": frame}, rng)
        mixer.reset_epoch(batch_size=4)

        batches = []
        for _ in range(100):
            train, val = mixer.next_batch()
            if train is None and val is None:
                break
            assert val is None
            batches.append(train)
        else:
            pytest.fail("epoch never ended")

        # 2 Malicious per batch -> ceil(9 / 2) == 5 batches
        assert len(batches) == 5
        for train in batches[:-1]:
            assert train["ground_truth_label"].value_counts().to_dict() == {"Benign": 2, "Malicious": 2}
        # last batch: 1 fresh Malicious row, 1 Malicious and 2 Benign rows oversampled
        assert mixer.get_mix_plan()[-1]["resampled_by_label"] == {"Benign": 2, "Malicious": 1}
        # every fresh row was used
        assert set(pd.concat(batches)["id"]) == set(frame["id"])

    def test_validation_rows_are_never_trained_on(self, rng):
        """Only fresh rows are split; oversampled copies come from training rows only."""
        frame = _zeek_like_frame(["Benign"] * 8 + ["Malicious"] * 40)
        spec = {"datasets": ["ds"], "validation_split": 0.3}
        mixer = BalancedByLabelWithOversamplingMixer(spec, {"ds": frame}, rng)
        mixer.reset_epoch(batch_size=10)

        train_ids, val_ids = [], []
        for _ in range(100):
            train, val = mixer.next_batch()
            if train is None and val is None:
                break
            train_ids.extend(train["id"].tolist())
            if val is not None:
                val_ids.extend(val["id"].tolist())
        else:
            pytest.fail("epoch never ended")

        assert set(train_ids).isdisjoint(val_ids)
        assert len(val_ids) == len(set(val_ids))  # validation rows are never duplicated
        assert set(train_ids) | set(val_ids) == set(frame["id"])
        # the split is cumulative over the 48 fresh rows only: round(0.3 * 48) == 14
        assert len(val_ids) == 14
        # Benign (the minority) was oversampled from its training rows
        resampled = sum(entry["resampled_by_label"]["Benign"] for entry in mixer.get_mix_plan())
        assert resampled > 0


# ========== Integration Tests ==========
class TestMixerIntegration:
    """Integration tests for mixers."""

    def test_reset_epoch_clears_plan(self, rng):
        """Test that reset_epoch clears mix_plan."""
        loader = MockLoader([[1, 2], [3, 4]])
        spec = {"datasets": ["ds1"]}
        mixer = SequenceMixer(spec, {"ds1": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        mixer.next_batch()
        assert len(mixer.get_mix_plan()) > 0

        mixer.reset_epoch(batch_size=10)
        assert len(mixer.get_mix_plan()) == 0

    def test_multiple_epochs_with_reset(self, rng):
        """Test that resetting epochs resets loaders."""
        loader = MockLoader([[1, 2], [3, 4]])
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
        loader = MockLoader([[1, 2]])
        spec = {"datasets": ["001"]}  # Partial match
        mixer = SequenceMixer(spec, {"dataset_001": loader}, rng)

        mixer.reset_epoch(batch_size=10)
        train, val = mixer.next_batch()

        assert train is not None

    def test_no_replacement_exhaustion(self, rng):
        """Test that mixers stop when data is exhausted (no replacement)."""
        loader = MockLoader([[1], [2], [3]])
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
