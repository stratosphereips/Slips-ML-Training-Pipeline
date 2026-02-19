import pytest
import re
import tempfile
from numbers import Integral
from pathlib import Path
import pandas as pd
from src.dataset_wrapper import (
    ZeekDataset,
    find_and_load_datasets,
)
from src.commons import BENIGN, MALICIOUS, BACKGROUND
from src.conn_normalizer import CANONICAL_FIELDS


class TestZeekDataset:
    """Comprehensive test suite for ZeekDataset class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_conn_log(self, temp_dir):
        """Create a sample conn.log with headers and data."""
        conn_file = temp_dir / "conn.log"
        content = """#separator \t
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn
#open	2021-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	history	orig_pkts	resp_pkts	label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	string	count	count	string
1609459200.001	uid-1	192.168.1.1	12345	10.0.0.1	443	tcp	https	10.5	1024	2048	SF	ShADaFf	15	14	Benign
1609459201.001	uid-2	192.168.1.2	54321	8.8.8.8	53	udp	dns	0.1	60	140	SF	Dd	1	1	-
1609459202.001	uid-3	192.168.1.3	48373	192.168.1.100	445	tcp	smb	300.5	5000	10000	S0	S	10	8	Malicious
1609459203.001	uid-4	192.168.1.4	12345	10.0.0.2	80	tcp	http	5.0	512	1024	SF	ShADaFf	5	5	Background
1609459204.001	uid-5	192.168.1.5	50000	192.168.1.50	22	tcp	ssh	2.0	256	512	SF	ShADaFf	2	2	MALWARE
"""
        conn_file.write_text(content)
        return conn_file

    @pytest.fixture
    def sample_labeled_log(self, temp_dir):
        """Create a sample conn.log.labeled file."""
        labeled_file = temp_dir / "conn.log.labeled"
        content = """#separator \t
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn
#open	2021-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	history	orig_pkts	resp_pkts	label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	string	count	count	string
1609459200.001	uid-1	192.168.1.1	12345	10.0.0.1	443	tcp	https	10.5	1024	2048	SF	ShADaFf	15	14	Benign
1609459201.001	uid-2	192.168.1.2	54321	8.8.8.8	53	udp	dns	0.1	60	140	SF	Dd	1	1	Benign
1609459202.001	uid-3	192.168.1.3	48373	192.168.1.100	445	tcp	smb	300.5	5000	10000	S0	S	10	8	Malicious
1609459203.001	uid-4	192.168.1.4	12345	10.0.0.2	80	tcp	http	5.0	512	1024	SF	ShADaFf	5	5	Background
1609459204.001	uid-5	192.168.1.5	50000	192.168.1.50	22	tcp	ssh	2.0	256	512	SF	ShADaFf	2	2	MALWARE
"""
        labeled_file.write_text(content)
        return labeled_file

    # ========== Initialization and File Detection Tests ==========
    def test_initialization_and_file_detection(
        self, sample_conn_log, temp_dir
    ):
        """Test ZeekDataset initialization and file priority detection."""
        # Test with conn.log
        ds_plain = ZeekDataset(temp_dir, batch_size=2)
        assert ds_plain.current_file == sample_conn_log
        assert ds_plain.batch_size == 2
        assert ds_plain.seed is None

        # Test file priority: labeled > alt_labeled > plain
        labeled = temp_dir / "conn.log.labeled"
        labeled.write_text(sample_conn_log.read_text())
        ds_labeled = ZeekDataset(temp_dir, batch_size=3)
        assert ds_labeled.current_file == labeled

        # Test alt_labeled priority
        alt_labeled = temp_dir / "labeled-conn.log"
        labeled.unlink()
        alt_labeled.write_text(sample_conn_log.read_text())
        ds_alt = ZeekDataset(temp_dir, batch_size=2)
        assert ds_alt.current_file == alt_labeled

    def test_initialization_missing_file_raises_error(self, temp_dir):
        """Test that missing conn.log raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ZeekDataset(temp_dir, batch_size=2)

    def test_initialization_missing_root_raises_error(self):
        """Test that missing root directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ZeekDataset("/nonexistent/path", batch_size=2)

    # ========== Header and Type Parsing Tests ==========
    def test_headers_and_types_parsing(self, sample_conn_log, temp_dir):
        """Test correct parsing of #fields and #types from log file."""
        ds = ZeekDataset(temp_dir, batch_size=2)

        expected_headers = [
            "ts",
            "uid",
            "id.orig_h",
            "id.orig_p",
            "id.resp_h",
            "id.resp_p",
            "proto",
            "service",
            "duration",
            "orig_bytes",
            "resp_bytes",
            "conn_state",
            "history",
            "orig_pkts",
            "resp_pkts",
            "label",
        ]
        expected_types = [
            "time",
            "string",
            "addr",
            "port",
            "addr",
            "port",
            "enum",
            "string",
            "interval",
            "count",
            "count",
            "string",
            "string",
            "count",
            "count",
            "string",
        ]

        assert ds.headers == expected_headers
        assert ds.types == expected_types

    # ========== Label Mapping and Filtering Tests ==========
    def test_label_mapping_and_background_filtering(
        self, sample_conn_log, temp_dir
    ):
        """Test label mapping to BENIGN/MALICIOUS and BACKGROUND filtering."""
        ds = ZeekDataset(temp_dir, batch_size=10)

        # Should have 4 valid flows (Background filtered out)
        # Flow 1: "Benign" -> BENIGN
        # Flow 2: "-" (empty) -> BENIGN (default)
        # Flow 3: "Malicious" -> MALICIOUS
        # Flow 4: "Background" -> filtered out
        # Flow 5: "MALWARE" -> MALICIOUS (contains "MAL")
        assert ds.total_lines == 4
        assert len(ds.valid_indices) == 4
        assert len(ds.labels) == 4

        # Check label values
        labels_set = set(ds.labels)
        assert BENIGN in labels_set
        assert MALICIOUS in labels_set
        assert BACKGROUND not in labels_set

    def test_label_mapping_variations(self, temp_dir):
        """Test label mapping with various case and format variations."""
        conn_file = temp_dir / "conn.log"
        content = """#separator \t
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn
#open	2021-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	history	orig_pkts	resp_pkts	label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	string	count	count	string
1609459200.001	uid-1	192.168.1.1	12345	10.0.0.1	443	tcp	https	10.5	1024	2048	SF	ShADaFf	15	14	benign
1609459201.001	uid-2	192.168.1.2	54321	8.8.8.8	53	udp	dns	0.1	60	140	SF	Dd	1	1	BENIGN
1609459202.001	uid-3	192.168.1.3	48373	192.168.1.100	445	tcp	smb	300.5	5000	10000	S0	S	10	8	malicious
1609459203.001	uid-4	192.168.1.4	12345	10.0.0.2	80	tcp	http	5.0	512	1024	SF	ShADaFf	5	5	ATTACK_MALWARE
1609459204.001	uid-5	192.168.1.5	50000	192.168.1.50	22	tcp	ssh	2.0	256	512	SF	ShADaFf	2	2	1
1609459205.001	uid-6	192.168.1.6	40000	192.168.1.60	25	tcp	smtp	3.0	128	256	SF	ShADaFf	3	3	background
"""
        conn_file.write_text(content)
        ds = ZeekDataset(temp_dir, batch_size=10)

        # 5 valid flows (background filtered), 2 benign, 3 malicious
        assert ds.total_lines == 5
        assert ds.labels.count(str(BENIGN)) == 2
        assert ds.labels.count(str(MALICIOUS)) == 3

    def test_missing_label_defaults_to_benign(self, temp_dir):
        """Test that missing or empty label defaults to BENIGN."""
        conn_file = temp_dir / "conn.log"
        content = """#separator \t
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn
#open	2021-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	history	orig_pkts	resp_pkts	label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	string	count	count	string
1609459200.001	uid-1	192.168.1.1	12345	10.0.0.1	443	tcp	https	10.5	1024	2048	SF	ShADaFf	15	14	-
1609459201.001	uid-2	192.168.1.2	54321	8.8.8.8	53	udp	dns	0.1	60	140	SF	Dd	1	1
"""
        conn_file.write_text(content)
        ds = ZeekDataset(temp_dir, batch_size=10)

        assert ds.total_lines == 2
        assert all(label == str(BENIGN) for label in ds.labels)

    # ========== Type Casting Tests ==========
    def test_cast_conversions(self, sample_conn_log, temp_dir):
        """Test type casting for int, float, bool, and string types."""
        ds = ZeekDataset(temp_dir, batch_size=10)

        # Test _cast function
        assert ds._cast("1234", "count") == 1234
        assert ds._cast("12.34", "double") == 12.34
        assert ds._cast("t", "bool") is True
        assert ds._cast("f", "bool") is False
        assert ds._cast("hello", "string") == "hello"
        assert ds._cast("-", "count") is None
        assert ds._cast("", "string") is None

    def test_line_data_with_casting(self, sample_conn_log, temp_dir):
        """Test that retrieved lines have correct types after casting."""
        ds = ZeekDataset(temp_dir, batch_size=10)
        df = ds.as_dataframe()
        line = df.iloc[0]
        # Check types are correct
        assert isinstance(line["starttime"], float)
        assert isinstance(line["uid"], str)
        assert isinstance(line["sport"], Integral)
        assert isinstance(line["dur"], float)
        assert isinstance(line["sbytes"], Integral)
        assert line["ground_truth_label"] in [str(BENIGN), str(MALICIOUS)]

    # ========== DataFrame Access Tests ==========
    def test_as_dataframe_returns_all_valid_rows(self, sample_conn_log, temp_dir):
        """as_dataframe should return every valid (non-background) record."""
        ds = ZeekDataset(temp_dir, batch_size=10)
        df = ds.as_dataframe()

        assert len(df) == ds.total_lines == 4
        assert set(df.columns) == set(CANONICAL_FIELDS)
        assert "ground_truth_label" in df.columns

    def test_as_dataframe_cached_instance(self, sample_conn_log, temp_dir):
        """Repeated calls to as_dataframe should reuse cached DataFrame."""
        ds = ZeekDataset(temp_dir, batch_size=10)
        df1 = ds.as_dataframe()
        df2 = ds.as_dataframe()

        assert df1 is df2

    def test_len_matches_dataframe_length(self, sample_conn_log, temp_dir):
        """__len__ should match DataFrame length."""
        ds = ZeekDataset(temp_dir, batch_size=10)
        df = ds.as_dataframe()

        assert len(ds) == len(df) == ds.total_lines

    # ========== Seeding and Shuffling Tests ==========
    def test_seed_shuffles_consistently(self, sample_conn_log, temp_dir):
        """Same seed should shuffle valid indices deterministically."""
        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=42)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=42)

        assert ds1.valid_indices == ds2.valid_indices
        assert ds1.labels == ds2.labels

    def test_different_seeds_produce_different_order(self, temp_dir):
        """Different seeds should lead to different valid_index permutations."""
        conn_file = temp_dir / "conn.log"
        lines = [
            "#separator \t",
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\thistory\torig_pkts\tresp_pkts\tlabel",
            "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tstring\tcount\tcount\tstring",
        ]
        for i in range(20):
            lines.append(
                f"1609459200.{i:06d}\tuid-{i}\t192.168.1.{i}\t{10000 + i}\t10.0.0.{i}\t443\ttcp\thttps\t10.0\t100\t200\tSF\tShADaFf\t1\t1\tBenign"
            )
        conn_file.write_text("\n".join(lines))

        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=1)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=2)

        assert ds1.valid_indices != ds2.valid_indices
        assert sorted(ds1.valid_indices) == sorted(ds2.valid_indices)

    def test_no_seed_is_deterministic_on_same_file(
        self, sample_conn_log, temp_dir
    ):
        """Without seed, the loader should preserve file order deterministically."""
        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=None)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=None)

        assert ds1.valid_indices == ds2.valid_indices

    # ========== New Parameter Tests ==========
    def test_custom_labeled_filenames(self, temp_dir):
        """Test initialization with custom labeled_filenames parameter."""
        conn_file = temp_dir / "custom_name.log"
        content = """#separator \t
#fields\tts\tuid\tproto\tlabel
#types\ttime\tstring\tenum\tstring
1609459200.001\tuid-1\ttcp\tBenign
"""
        conn_file.write_text(content)

        ds = ZeekDataset(
            temp_dir, batch_size=10, labeled_filenames=["custom_name.log"]
        )
        assert ds.current_file == conn_file

    def test_custom_cache_dir(self, temp_dir, sample_conn_log):
        """Test initialization with custom cache_dir parameter."""
        custom_cache = temp_dir / "my_cache"
        custom_cache.mkdir()

        # Create large dataset to trigger caching
        lines = [
            "#separator \t",
            "#fields\tts\tuid\tproto\tlabel",
            "#types\ttime\tstring\tenum\tstring",
        ]
        for i in range(35000):
            lines.append(f"1609459200.{i:06d}\tuid-{i}\t\ttcp\tBenign")

        conn_file = temp_dir / "conn.log"
        conn_file.write_text("\n".join(lines))

        ds = ZeekDataset(temp_dir, batch_size=100, cache_dir=custom_cache)
        cache_file = ds._cache_path()

        # Cache should be in custom directory
        assert cache_file.parent == custom_cache

    def test_file_encoding_parameter(self, temp_dir):
        """Test file_encoding parameter is respected."""
        conn_file = temp_dir / "conn.log"
        content = """#separator \t
#fields\tts\tuid\tproto\tlabel
#types\ttime\tstring\tenum\tstring
1609459200.001\tuid-1\ttcp\tBenign
"""
        conn_file.write_text(content, encoding="utf-8")

        ds = ZeekDataset(
            temp_dir,
            batch_size=10,
            file_encoding="utf-8",
            file_errors="ignore",
        )
        assert ds.file_encoding == "utf-8"
        assert ds.file_errors == "ignore"
        assert ds.total_lines == 1

    def test_shuffle_per_epoch_false(self, sample_conn_log, temp_dir):
        """shuffle_per_epoch flag should be stored when disabled."""
        ds = ZeekDataset(temp_dir, batch_size=2, shuffle_per_epoch=False)

        assert ds.shuffle_per_epoch is False
        df1 = ds.as_dataframe().copy()
        df2 = ds.as_dataframe().copy()
        pd.testing.assert_frame_equal(df1, df2)

    def test_shuffle_per_epoch_true(self, sample_conn_log, temp_dir):
        """shuffle_per_epoch flag should be stored when enabled."""
        ds = ZeekDataset(
            temp_dir, batch_size=2, seed=42, shuffle_per_epoch=True
        )

        assert ds.shuffle_per_epoch is True
        df1 = ds.as_dataframe().copy()
        df2 = ds.as_dataframe().copy()
        pd.testing.assert_frame_equal(df1, df2)

    def test_cache_creation_for_large_dataset(self, temp_dir):
        """Test that cache is created for datasets above threshold."""
        # Create large dataset
        conn_file = temp_dir / "conn.log"
        lines = [
            "#separator \t",
            "#set_separator\t,",
            "#empty_field\t(empty)",
            "#unset_field\t-",
            "#path\tconn",
            "#open\t2021-01-01-00-00-00",
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\thistory\torig_pkts\tresp_pkts\tlabel",
            "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tstring\tcount\tcount\tstring",
        ]

        # Add 35000 lines (above 30000 threshold)
        for i in range(35000):
            lines.append(
                f"1609459200.{i:06d}\tuid-{i}\t192.168.1.{i%255}\t{12345+i%10000}\t10.0.0.{i%255}\t443\ttcp\thttps\t10.5\t1024\t2048\tSF\tShADaFf\t15\t14\tBenign"
            )

        conn_file.write_text("\n".join(lines))

        # Create dataset - should trigger cache creation
        ds = ZeekDataset(temp_dir, batch_size=100)
        cache_file = ds._cache_path()

        assert cache_file.exists()
        assert ds.total_lines == 35000

    def test_cache_reloaded_on_same_file(self, temp_dir):
        """Test that cache is reused for same file."""
        conn_file = temp_dir / "conn.log"
        lines = [
            "#separator \t",
            "#set_separator\t,",
            "#empty_field\t(empty)",
            "#unset_field\t-",
            "#path\tconn",
            "#open\t2021-01-01-00-00-00",
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\thistory\torig_pkts\tresp_pkts\tlabel",
            "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tstring\tcount\tcount\tstring",
        ]

        for i in range(35000):
            lines.append(
                f"1609459200.{i:06d}\tuid-{i}\t192.168.1.{i%255}\t{12345+i%10000}\t10.0.0.{i%255}\t443\ttcp\thttps\t10.5\t1024\t2048\tSF\tShADaFf\t15\t14\tBenign"
            )

        conn_file.write_text("\n".join(lines))

        # First load
        ds1 = ZeekDataset(temp_dir, batch_size=100)
        cache_file = ds1._cache_path()
        cache_mtime = cache_file.stat().st_mtime

        # Second load should use cache
        ds2 = ZeekDataset(temp_dir, batch_size=100)
        assert cache_file.stat().st_mtime == cache_mtime
        assert ds2.total_lines == ds1.total_lines

    def test_clear_cache(self, temp_dir):
        """Test clear_cache removes cache file."""
        conn_file = temp_dir / "conn.log"
        lines = [
            "#separator \t",
            "#set_separator\t,",
            "#empty_field\t(empty)",
            "#unset_field\t-",
            "#path\tconn",
            "#open\t2021-01-01-00-00-00",
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\thistory\torig_pkts\tresp_pkts\tlabel",
            "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tstring\tcount\tcount\tstring",
        ]

        for i in range(35000):
            lines.append(
                f"1609459200.{i:06d}\tuid-{i}\t192.168.1.{i%255}\t{12345+i%10000}\t10.0.0.{i%255}\t443\ttcp\thttps\t10.5\t1024\t2048\tSF\tShADaFf\t15\t14\tBenign"
            )

        conn_file.write_text("\n".join(lines))

        ds = ZeekDataset(temp_dir, batch_size=100)
        cache_file = ds._cache_path()
        assert cache_file.exists()

        ds.clear_cache()
        assert not cache_file.exists()

    # ========== Helper Functions Tests ==========
    @pytest.mark.parametrize(
        "params",
        [
            # Default parameters
            {},
            # Custom batch_size
            {"batch_size": 2},
            # Custom prefix_regex (match only 'abc')
            {"prefix_regex": r"^abc"},
            # Custom data_subdir
            {"data_subdir": "custom_data"},
            # Custom seed
            {"seed": 123},
            # Custom persist_cache_threshold
            {"persist_cache_threshold": 1},
            # Custom cache_dir
            {"cache_dir": None},
            # Custom labeled_filenames
            {"labeled_filenames": ["special.log"]},
            # Custom file_encoding and file_errors
            {"file_encoding": "utf-8", "file_errors": "replace"},
            # Custom shuffle_per_epoch
            {"shuffle_per_epoch": True},
        ],
    )
    def test_find_and_load_datasets_all_options(self, temp_dir, params):
        """Test find_and_load_datasets with all optional parameters."""
        # Setup directories and files
        ds1_dir = temp_dir / "001" / params.get("data_subdir", "data")
        ds2_dir = temp_dir / "abc" / params.get("data_subdir", "data")
        ds1_dir.mkdir(parents=True)
        ds2_dir.mkdir(parents=True)

        # Use custom labeled filename if specified
        filenames = params.get("labeled_filenames", None)
        file_name = filenames[0] if filenames else "conn.log"
        for ds_dir in [ds1_dir, ds2_dir]:
            conn_file = ds_dir / file_name
            content = """#separator \t
#fields\tts\tuid\tproto\tlabel
#types\ttime\tstring\tenum\tstring
1609459200.001\tuid-1\ttcp\tBenign
"""
            conn_file.write_text(
                content, encoding=params.get("file_encoding", "utf-8")
            )

        # Custom cache_dir if specified
        cache_dir = params.get("cache_dir", None)
        if cache_dir is None:
            cache_dir = temp_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            params["cache_dir"] = cache_dir

        # Run loader
        loaders = find_and_load_datasets(
            temp_dir,
            batch_size=params.get("batch_size", 1000),
            prefix_regex=params.get("prefix_regex", r"^\d{3}"),
            data_subdir=params.get("data_subdir", "data"),
            seed=params.get("seed", None),
            persist_cache_threshold=params.get(
                "persist_cache_threshold", 30000
            ),
            cache_dir=params.get("cache_dir", cache_dir),
            labeled_filenames=params.get("labeled_filenames", None),
            file_encoding=params.get("file_encoding", "utf-8"),
            file_errors=params.get("file_errors", "ignore"),
            shuffle_per_epoch=params.get("shuffle_per_epoch", False),
        )

        # Check correct datasets loaded according to prefix_regex
        expected_keys = []
        regex = params.get("prefix_regex", r"^\d{3}")
        if re.compile(regex).match("001"):
            expected_keys.append("001")
        if re.compile(regex).match("abc"):
            expected_keys.append("abc")
        for key in expected_keys:
            assert key in loaders
            assert isinstance(loaders[key], ZeekDataset)
            assert loaders[key].batch_size == params.get("batch_size", 1000)
            assert loaders[key].file_encoding == params.get(
                "file_encoding", "utf-8"
            )
            assert loaders[key].file_errors == params.get(
                "file_errors", "ignore"
            )
            assert loaders[key].shuffle_per_epoch == params.get(
                "shuffle_per_epoch", False
            )
            if params.get("seed", None) is not None:
                assert loaders[key].seed == params["seed"]
            if params.get("labeled_filenames", None):
                assert (
                    loaders[key].current_file.name
                    == params["labeled_filenames"][0]
                )
