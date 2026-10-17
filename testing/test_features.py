import pytest
import pandas as pd
import numpy as np
from src.features import FeatureExtraction
from src.commons import BENIGN, MALICIOUS


class TestFeatureExtraction:
    """Comprehensive test suite for FeatureExtraction class."""

    @pytest.fixture
    def extractor(self):
        return FeatureExtraction()

    @pytest.fixture
    def basic_flow(self):
        return {
            "starttime": "1609459200.001",
            "uid": "uid-123",
            "saddr": "192.168.1.1",
            "daddr": "10.0.0.1",
            "sport": 12345,
            "dport": 443,
            "proto": "tcp",
            "appproto": "https",
            "dur": 10.5,
            "sbytes": 1024,
            "dbytes": 2048,
            "spkts": 15,
            "dpkts": 14,
            "state": 1.0,
            "history": "ShADaFf",
            "ground_truth_label": "Benign",
            "type_": "flow",
            "dir_": "outgoing",
            "smac": "aa:bb:cc:dd:ee:ff",
            "dmac": "11:22:33:44:55:66",
        }

    # ========== Initialization Tests ==========
    def test_initialization_defaults_and_custom(self):
        """Test FeatureExtraction initialization with default and custom values."""
        # Test defaults
        fe_default = FeatureExtraction()
        assert fe_default.converter is not None
        assert "arp" in fe_default.protocols_to_discard
        assert "icmp" in fe_default.protocols_to_discard
        assert "" in fe_default.protocols_to_discard
        assert fe_default.columns_to_discard == []
        assert fe_default.column_types == {}
        assert fe_default.converter.default_label == "Benign"

        # Test custom values
        custom_protocols = ["tcp", "udp"]
        custom_cols = ["col1", "col2"]
        custom_types = {"sport": "int32"}
        fe_custom = FeatureExtraction(
            default_label="Custom",
            protocols_to_discard=custom_protocols,
            columns_to_discard=custom_cols,
            column_types=custom_types,
        )
        assert fe_custom.protocols_to_discard == custom_protocols
        assert fe_custom.columns_to_discard == custom_cols
        assert fe_custom.column_types == custom_types
        assert fe_custom.converter.default_label == "Custom"

    # ========== Label Extraction Tests ==========
    def test_extract_labels_exact_match_and_defaults(self, extractor):
        """Test label extraction with exact matches and defaults."""
        # Test exact matches for BENIGN and MALICIOUS
        df_benign = pd.DataFrame([{"ground_truth_label": "Benign"}])
        assert extractor._extract_labels(df_benign).iloc[0] == BENIGN

        df_malicious = pd.DataFrame([{"ground_truth_label": "Malicious"}])
        assert extractor._extract_labels(df_malicious).iloc[0] == MALICIOUS

        # Test case insensitivity (uppercased)
        df_benign_upper = pd.DataFrame([{"ground_truth_label": "BENIGN"}])
        assert extractor._extract_labels(df_benign_upper).iloc[0] == BENIGN

        df_malicious_upper = pd.DataFrame(
            [{"ground_truth_label": "MALICIOUS"}]
        )
        assert (
            extractor._extract_labels(df_malicious_upper).iloc[0] == MALICIOUS
        )

        # Test empty/missing defaults to BENIGN
        df_empty = pd.DataFrame([{"ground_truth_label": ""}])
        assert extractor._extract_labels(df_empty).iloc[0] == BENIGN

        df_no_label = pd.DataFrame([{"sport": 80}])
        assert extractor._extract_labels(df_no_label).iloc[0] == "Benign"

    def test_extract_labels_word_start_mal_matching(self, extractor):
        """'MAL' at the start of a word, or "1", marks a flow malicious."""
        test_cases_mal = [
            "Malicious",
            "Malware",
            "MALWARE",
            "MALWARE_SMB",
            "ATTACK_MALWARE",
            "ATTACK_MALICIOUS",
            "MAL",
            "1",
            " malicious ",
        ]
        for label in test_cases_mal:
            df = pd.DataFrame([{"ground_truth_label": label}])
            result = extractor._extract_labels(df).iloc[0]
            assert (
                result == MALICIOUS
            ), f"Failed for label '{label}', got {result}"

    @pytest.mark.parametrize("label_col", ["ground_truth_label", "detailed_ground_truth_label", "label", "detailed_label"])
    @pytest.mark.parametrize("label", ["Normal", "NORMAL", "From-Normal-V42", "Anomaly", "ANOMALY", "Benign", "", "0"])
    def test_extract_labels_mal_inside_word_is_benign(self, extractor, label_col, label):
        """'MAL' inside a word (Normal, Anomaly) does not make a flow malicious."""
        df = pd.DataFrame([{label_col: label}])
        assert extractor._extract_labels(df).iloc[0] == BENIGN

    def test_extract_labels_any_label_column_marks_malicious(self, extractor):
        """All label columns are ORed: the flow is malicious if any of them is malicious."""
        # a single label column of each kind
        df = pd.DataFrame([{"detailed_ground_truth_label": "Malicious"}])
        assert extractor._extract_labels(df).iloc[0] == MALICIOUS

        df = pd.DataFrame([{"label": "Benign"}])
        assert extractor._extract_labels(df).iloc[0] == BENIGN

        df = pd.DataFrame([{"module_labels": "Malicious"}])
        assert extractor._extract_labels(df).iloc[0] == MALICIOUS

        df = pd.DataFrame([{"detailed_label": "Malware"}])
        assert extractor._extract_labels(df).iloc[0] == MALICIOUS

        # no column has priority: a later malicious column wins over a benign ground truth
        df = pd.DataFrame(
            [
                {"ground_truth_label": "Benign", "label": "", "detailed_label": "Malware"},
                {"ground_truth_label": "Malicious", "label": "Benign", "detailed_label": ""},
                {"ground_truth_label": "Benign", "label": "Normal", "detailed_label": "From-Normal-V42"},
            ]
        )
        assert extractor._extract_labels(df).tolist() == [MALICIOUS, MALICIOUS, BENIGN]

    def test_extract_labels_nan_and_none_handling(self, extractor):
        """Test label extraction handles NaN/None gracefully."""
        df = pd.DataFrame(
            [
                {"ground_truth_label": None},
                {"ground_truth_label": np.nan},
                {"ground_truth_label": ""},
            ]
        )
        labels = extractor._extract_labels(df)
        assert all(labels == "Benign")

    # ========== Drop Labels Tests ==========
    def test_drop_labels_removes_and_preserves_columns(
        self, extractor, basic_flow
    ):
        """Test drop_labels removes label columns and preserves others."""
        df = pd.DataFrame([basic_flow])
        result = extractor.drop_labels(df)

        # Check removed columns
        removed_cols = [
            "ground_truth_label",
            "detailed_ground_truth_label",
            "label",
            "module_labels",
            "detailed_label",
        ]
        for col in removed_cols:
            assert col not in result.columns

        # Check preserved columns
        preserved_cols = [
            "sport",
            "dport",
            "proto",
            "sbytes",
            "saddr",
            "daddr",
        ]
        for col in preserved_cols:
            assert col in result.columns

        # Test with minimal DataFrame
        df_minimal = pd.DataFrame([{"sport": 80, "dport": 443}])
        result_minimal = extractor.drop_labels(df_minimal)
        assert result_minimal.shape == (1, 2)

    # ========== Feature Processing Tests ==========
    def test_process_features_proto_mapping_valid_protocols(self, extractor):
        """Test protocol mapping to numeric codes for protocols not in discard list."""
        # Only test TCP and UDP (not in default discard list)
        proto_mappings = [
            ("tcp", 0.0),
            ("TCP", 0.0),
            ("udp", 1.0),
            ("UDP", 1.0),
        ]
        for proto, expected in proto_mappings:
            df = pd.DataFrame(
                [
                    {
                        "proto": proto,
                        "sbytes": 100,
                        "dbytes": 50,
                        "spkts": 5,
                        "dpkts": 4,
                    }
                ]
            )
            result = extractor._process_features(df)
            assert not result.empty, f"Flow was filtered for {proto}"
            assert (
                result["proto"].iloc[0] == expected
            ), f"Proto {proto} should map to {expected}"

    def test_process_features_protocol_filtering_default(self, extractor):
        """Test filtering of unwanted protocols with default discard list."""
        df = pd.DataFrame(
            [
                {
                    "proto": "tcp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "arp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "icmp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "udp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
            ]
        )
        result = extractor._process_features(df)
        # Only tcp and udp should remain (arp, icmp filtered by default)
        assert len(result) == 2
        assert all(p in [0.0, 1.0] for p in result["proto"])

    def test_process_features_protocol_filtering_custom(self):
        """Test filtering with custom protocol discard list."""
        # Custom filter that replaces defaults
        fe_custom = FeatureExtraction(protocols_to_discard=["tcp", "udp"])
        df = pd.DataFrame(
            [
                {
                    "proto": "tcp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "udp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "arp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
                {
                    "proto": "icmp",
                    "sbytes": 100,
                    "dbytes": 50,
                    "spkts": 5,
                    "dpkts": 4,
                },
            ]
        )
        result = fe_custom._process_features(df)
        # tcp and udp filtered, arp and icmp remain (no longer in discard list)
        assert len(result) == 2
        assert all(p in [2.0, 4.0] for p in result["proto"])

    def test_process_features_byte_and_packet_computation(self, extractor):
        """Test bytes and packets computation."""
        df = pd.DataFrame(
            [
                {
                    "sbytes": 1000,
                    "dbytes": 2000,
                    "spkts": 10,
                    "dpkts": 15,
                    "proto": "tcp",
                },
                {
                    "sbytes": 500,
                    "dbytes": 500,
                    "spkts": 5,
                    "dpkts": 5,
                    "proto": "tcp",
                },
                {
                    "sbytes": np.nan,
                    "dbytes": 200,
                    "spkts": 0,
                    "dpkts": 5,
                    "proto": "tcp",
                },
            ]
        )
        result = extractor._process_features(df)

        # Check byte/packet computation
        assert result["bytes"].iloc[0] == 3000
        assert result["pkts"].iloc[0] == 25
        assert result["bytes"].iloc[1] == 1000
        assert result["pkts"].iloc[1] == 10
        assert result["bytes"].iloc[2] == 200  # 0 (nan) + 200
        assert result["pkts"].iloc[2] == 5

    def test_process_features_column_dropping(self, extractor):
        """Test dropping of built-in and custom columns."""
        df = pd.DataFrame(
            [
                {
                    "starttime": "2021-01-01",
                    "saddr": "192.168.1.1",
                    "daddr": "10.0.0.1",
                    "appproto": "https",
                    "history": "ShADaFf",
                    "uid": "uid-123",
                    "smac": "aa:bb",
                    "dmac": "cc:dd",
                    "type_": "flow",
                    "dir_": "out",
                    "endtime": "2021-01-01",
                    "flow_source": "zeek",
                    "proto": "tcp",
                    "sport": 80,
                    "dport": 443,
                    "sbytes": 100,
                    "dbytes": 200,
                    "spkts": 5,
                    "dpkts": 5,
                }
            ]
        )
        result = extractor._process_features(df)

        # Check built-in columns are dropped
        builtin_drop = [
            "starttime",
            "saddr",
            "daddr",
            "appproto",
            "history",
            "uid",
            "smac",
            "dmac",
            "type_",
            "dir_",
            "endtime",
            "flow_source",
        ]
        for col in builtin_drop:
            assert (
                col not in result.columns
            ), f"Column {col} should have been dropped"

        # Check features are preserved
        assert "sport" in result.columns
        assert "dport" in result.columns

    def test_process_features_custom_column_dropping(self):
        """Test dropping of custom columns."""
        fe_custom = FeatureExtraction(
            columns_to_discard=["custom1", "custom2"]
        )
        df = pd.DataFrame(
            [
                {
                    "custom1": "val1",
                    "custom2": "val2",
                    "proto": "tcp",
                    "sport": 80,
                    "dport": 443,
                    "sbytes": 100,
                    "dbytes": 200,
                    "spkts": 5,
                    "dpkts": 5,
                }
            ]
        )
        result = fe_custom._process_features(df)
        assert "custom1" not in result.columns
        assert "custom2" not in result.columns
        assert "sport" in result.columns

    def test_process_features_numeric_coercion_and_types(self, extractor):
        """Test numeric coercion and type checking."""
        df = pd.DataFrame(
            [
                {
                    "proto": "tcp",
                    "sport": "8080",
                    "dport": "443",
                    "dur": "10.5",
                    "sbytes": "1000",
                    "dbytes": "2000",
                    "spkts": "10",
                    "dpkts": "15",
                }
            ]
        )
        result = extractor._process_features(df)

        # Check numeric fields are numeric type
        numeric_cols = [
            "sport",
            "dport",
            "dur",
            "sbytes",
            "dbytes",
            "spkts",
            "dpkts",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(
                result[col]
            ), f"{col} is not numeric"

        # Check values
        assert result["sport"].iloc[0] == 8080
        assert result["dport"].iloc[0] == 443

    @pytest.mark.parametrize(
        "proto, expected",
        [
            ("tcp", 0.0),
            ("TCP", 0.0),
            ("udp", 1.0),
            (" UDP ", 1.0),
            ("icmp", 2.0),
            ("ICMP", 2.0),
            ("icmp-ipv6", 3.0),
            ("ICMP-IPv6", 3.0),
            ("arp", 4.0),
            ("ARP", 4.0),
            # anything else -> 0, exact match only (no substring matching)
            ("ipv6-icmp", 0.0),
            ("igmp", 0.0),
            ("udplite", 0.0),
            ("icmpv6", 0.0),
            ("sctp", 0.0),
            ("unknown", 0.0),
        ],
    )
    def test_proto_encoding_matches_slips(self, proto, expected):
        """Proto encoding is the exact-match SLIPS table; unknown protocols map to 0."""
        fe = FeatureExtraction(protocols_to_discard=[])
        df = pd.DataFrame([{"proto": proto, "sbytes": 1, "dbytes": 1, "spkts": 1, "dpkts": 1}])
        result = fe._process_features(df)
        assert result["proto"].tolist() == [expected]
        assert result["proto"].dtype == np.float64

    def test_proto_encoding_through_process_batch(self):
        """The public process_batch path uses the same proto codes."""
        fe = FeatureExtraction(protocols_to_discard=[])
        protos = ["tcp", "udp", "icmp", "icmp-ipv6", "arp", "igmp"]
        X, _ = fe.process_batch([{"proto": p, "ground_truth_label": "Benign"} for p in protos])
        assert X["proto"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]

    def test_missing_proto_is_discarded_as_empty(self, extractor):
        """A missing proto (None/NaN) becomes "" and is dropped when "" is in protocols_to_discard."""
        flows = [
            {"proto": None, "sbytes": 1, "ground_truth_label": "Malicious"},
            {"proto": "tcp", "sbytes": 2, "ground_truth_label": "Benign"},
            {"sbytes": 3, "ground_truth_label": "Malicious"},  # no proto key at all
        ]
        X, y = extractor.process_batch(flows)
        assert X["sbytes"].tolist() == [2]
        assert y.tolist() == [BENIGN]

        df = pd.DataFrame({"proto": [np.nan, "udp"], "sbytes": [1, 2]})
        assert extractor._process_features(df)["sbytes"].tolist() == [2]

        # without "" in the discard list the flow is kept (and encoded as 0)
        fe_keep = FeatureExtraction(protocols_to_discard=["arp"])
        X_keep, _ = fe_keep.process_batch(flows)
        assert X_keep["sbytes"].tolist() == [1, 2, 3]
        assert X_keep["proto"].tolist() == [0.0, 0.0, 0.0]

    def test_discard_ignores_case_and_whitespace(self):
        fe = FeatureExtraction(protocols_to_discard=[" ARP", "icmp "])
        df = pd.DataFrame({"proto": [" arp ", "ICMP", "tcp "], "sbytes": [1, 2, 3]})
        result = fe._process_features(df)
        assert result["sbytes"].tolist() == [3]
        assert result["proto"].tolist() == [0.0]

    def test_process_features_missing_byte_and_packet_columns(self, extractor):
        """Missing sbytes/dbytes/spkts/dpkts columns default to 0 instead of crashing."""
        df = pd.DataFrame([{"proto": "tcp", "dur": 1.0}, {"proto": "udp", "dur": 2.0}])
        result = extractor._process_features(df)
        for col in ["sbytes", "dbytes", "spkts", "dpkts", "bytes", "pkts"]:
            assert result[col].tolist() == [0, 0], col

        X, y = extractor.process_batch(df)
        assert list(X.columns) == ["dur", "proto", "spkts", "dpkts", "sbytes", "dbytes", "bytes", "pkts"]
        assert X["bytes"].tolist() == [0, 0]
        assert y.tolist() == [BENIGN, BENIGN]

    # ========== Process Batch Tests ==========
    def test_process_batch_various_input_formats(self, extractor, basic_flow):
        """DataFrame, list-of-dicts and single-dict inputs give the same features."""
        # DataFrame input
        df = pd.DataFrame([basic_flow, basic_flow])
        X_df, y_df = extractor.process_batch(df)
        assert len(X_df) == 2 and len(y_df) == 2

        # List of dicts
        X_list, y_list = extractor.process_batch([basic_flow, basic_flow])
        assert len(X_list) == 2 and len(y_list) == 2
        assert X_df["state"].tolist() == X_list["state"].tolist() == [1.0, 1.0]
        pd.testing.assert_frame_equal(X_df, X_list)
        pd.testing.assert_series_equal(y_df, y_list)

        # Single dict
        X_single, y_single = extractor.process_batch(basic_flow)
        assert len(X_single) == 1 and len(y_single) == 1
        pd.testing.assert_frame_equal(X_single, X_list.iloc[:1])

        # Empty list
        X_empty, y_empty = extractor.process_batch([])
        assert X_empty.empty and y_empty.empty

    def test_process_batch_raw_zeek_dataframe(self, extractor):
        """A DataFrame with raw Zeek columns is normalized like a list of raw Zeek dicts."""
        zeek = {
            "ts": "1609459200.001",
            "uid": "zeek-uid",
            "id.orig_h": "192.168.1.10",
            "id.orig_p": "54321",
            "id.resp_h": "8.8.8.8",
            "id.resp_p": "53",
            "proto": "udp",
            "service": "dns",
            "duration": "0.1",
            "orig_bytes": "60",
            "resp_bytes": "140",
            "conn_state": "SF",
            "orig_pkts": "1",
            "resp_pkts": "1",
            "label": "Benign",
        }
        flows = [zeek, dict(zeek, conn_state="S0", label="Malicious")]

        X_df, y_df = extractor.process_batch(pd.DataFrame(flows))
        X_list, y_list = extractor.process_batch(flows)

        pd.testing.assert_frame_equal(X_df, X_list)
        pd.testing.assert_series_equal(y_df, y_list)
        assert X_df["state"].tolist() == [1.0, 0.0]
        assert X_df["sport"].tolist() == [54321, 54321]
        assert X_df["bytes"].tolist() == [200, 200]
        assert y_df.tolist() == [BENIGN, MALICIOUS]

    def test_process_batch_slips_frame_with_string_state(self, extractor):
        """A SLIPS-style DataFrame whose state is still a string (e.g. "SF") gets normalized."""
        df = pd.DataFrame(
            [
                {"proto": "tcp", "sport": 1, "dport": 80, "dur": 1.0, "sbytes": 10, "dbytes": 20,
                 "spkts": 2, "dpkts": 2, "state": "SF", "ground_truth_label": "Malicious"},
                {"proto": "tcp", "sport": 2, "dport": 80, "dur": 1.0, "sbytes": 10, "dbytes": 20,
                 "spkts": 1, "dpkts": 0, "state": "S0", "ground_truth_label": "Benign"},
            ]
        )
        X, y = extractor.process_batch(df)
        assert X["state"].tolist() == [1.0, 0.0]
        assert X["state"].dtype == np.float64
        assert y.tolist() == [MALICIOUS, BENIGN]

    def test_process_batch_reraises_errors(self, extractor, basic_flow, monkeypatch, capsys):
        """Errors are re-raised (with a one-line context print), not turned into empty batches."""
        def boom(df):
            raise ValueError("feature processing failed")

        monkeypatch.setattr(extractor, "_process_features", boom)
        with pytest.raises(ValueError, match="feature processing failed"):
            extractor.process_batch([basic_flow])
        assert "Error in FeatureExtraction.process_batch()" in capsys.readouterr().out

    def test_process_batch_invalid_input_raises(self, extractor):
        """Input that is not a DataFrame, dict or iterable of dicts raises."""
        with pytest.raises(TypeError):
            extractor.process_batch(12345)

    def test_process_batch_zeek_auto_detection(self, extractor):
        """Test process_batch auto-detects and converts Zeek format."""
        zeek_flow = {
            "ts": "1609459200.001",
            "uid": "zeek-uid",
            "id.orig_h": "192.168.1.10",
            "id.orig_p": "54321",
            "id.resp_h": "8.8.8.8",
            "id.resp_p": "53",
            "proto": "udp",
            "service": "dns",
            "duration": "0.1",
            "orig_bytes": "60",
            "resp_bytes": "140",
            "conn_state": "SF",
            "orig_pkts": "1",
            "resp_pkts": "1",
            "label": "Benign",
        }

        X, y = extractor.process_batch([zeek_flow])
        assert not X.empty and not y.empty
        assert len(X) == len(y)
        assert y.iloc[0] == BENIGN

    def test_process_batch_output_structure_and_index(
        self, extractor, basic_flow
    ):
        """Test process_batch output structure, columns, and index."""
        flows = [basic_flow] * 3
        X, y = extractor.process_batch(flows)

        # Check alignment
        assert len(X) == len(y) == 3

        # Check columns are in expected order
        expected_cols = [
            "dur",
            "proto",
            "sport",
            "dport",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "state",
            "bytes",
            "pkts",
        ]
        assert list(X.columns) == expected_cols == extractor.slips_column_order

        # Check index reset
        assert X.index.equals(pd.RangeIndex(3))
        assert y.index.equals(pd.RangeIndex(3))

    def test_process_batch_filtering_and_labels(self, extractor):
        """Test process_batch filters flows and extracts labels correctly."""
        flows = [
            {
                "proto": "tcp",
                "sbytes": 100,
                "dbytes": 50,
                "spkts": 5,
                "dpkts": 4,
                "ground_truth_label": "Benign",
            },
            {
                "proto": "arp",
                "sbytes": 100,
                "dbytes": 50,
                "spkts": 5,
                "dpkts": 4,
                "ground_truth_label": "Benign",
            },
            {
                "proto": "udp",
                "sbytes": 100,
                "dbytes": 50,
                "spkts": 5,
                "dpkts": 4,
                "ground_truth_label": "Malicious",
            },
        ]
        X, y = extractor.process_batch(flows)

        # Should filter arp, keep tcp and udp
        assert len(X) == 2
        assert len(y) == 2
        assert y.iloc[0] == BENIGN
        assert y.iloc[1] == MALICIOUS

    def test_process_batch_custom_column_types_enforcement(self):
        """Test process_batch with custom column type specifications."""
        fe = FeatureExtraction(
            column_types={"sport": "int32", "dur": "float32"}
        )
        flows = [
            {
                "proto": "tcp",
                "sport": "8080",
                "dur": "10.5",
                "sbytes": 100,
                "dbytes": 50,
                "spkts": 5,
                "dpkts": 4,
                "ground_truth_label": "Benign",
            }
        ]
        X, y = fe.process_batch(flows)

        assert not X.empty
        assert X["sport"].dtype == np.int32
        assert X["dur"].dtype == np.float32
        assert X["sport"].tolist() == [8080]
        assert X["dur"].iloc[0] == pytest.approx(10.5)
        # columns without a requested type keep their default dtype
        assert X["dport"].dtype == np.int64
        assert X["proto"].dtype == np.float64

    # ========== Process Item Tests ==========
    def test_process_item_single_flow_processing(self, extractor, basic_flow):
        """Test process_item handles single flows correctly."""
        # Valid flow
        result = extractor.process_item(basic_flow)
        assert result is not None
        flow_dict, label = result
        assert isinstance(flow_dict, dict)
        assert label in [BENIGN, MALICIOUS]
        assert "ground_truth_label" not in flow_dict

    def test_process_item_filtered_flow_returns_none(self, extractor):
        """Test process_item returns None if flow is filtered out."""
        # Create a flow with only arp protocol (filtered by default)
        arp_flow = {
            "proto": "arp",
            "sbytes": 100,
            "dbytes": 50,
            "spkts": 1,
            "dpkts": 1,
        }
        result = extractor.process_item(arp_flow)
        assert result is None

    # ========== Unit Test with normalized data ==========
    def test_process_batch_with_normalized_slips_flows(self):
        """Unit test with already-normalized SLIPS format data."""
        fe = FeatureExtraction()

        # Use normalized SLIPS format (what comes out of conn_normalizer)
        normalized_flows = [
            {
                "starttime": "1609459200.001",
                "uid": "flow-1",
                "saddr": "10.0.0.10",
                "sport": 54321,
                "daddr": "8.8.8.8",
                "dport": 53,
                "proto": "udp",
                "appproto": "dns",
                "dur": 0.1,
                "sbytes": 60,
                "dbytes": 140,
                "spkts": 1,
                "dpkts": 1,
                "state": 1.0,
                "history": "Dd",
                "ground_truth_label": "Benign",
            },
            {
                "starttime": "1609459201.001",
                "uid": "flow-2",
                "saddr": "10.0.0.20",
                "sport": 48373,
                "daddr": "192.168.1.100",
                "dport": 445,
                "proto": "tcp",
                "appproto": "smb",
                "dur": 300.5,
                "sbytes": 5000,
                "dbytes": 10000,
                "spkts": 10,
                "dpkts": 8,
                "state": 0.0,
                "history": "S",
                "detailed_ground_truth_label": "MALWARE_SMB",
            },
        ]

        X, y = fe.process_batch(normalized_flows)

        # Should have 2 flows (both tcp and udp not filtered)
        assert len(X) == 2
        assert len(y) == 2

        # Check features exist and are numeric
        assert all(
            col in X.columns
            for col in ["sport", "dport", "proto", "bytes", "pkts"]
        )
        assert pd.api.types.is_numeric_dtype(X["sport"])
        assert pd.api.types.is_numeric_dtype(X["proto"])

        # Check labels (MALWARE_SMB starts with "MAL")
        assert y.iloc[0] == BENIGN
        assert y.iloc[1] == MALICIOUS
