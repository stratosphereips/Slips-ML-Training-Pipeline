import json
import pytest
import re
from numbers import Integral
from pathlib import Path
import src.dataset_wrapper as dataset_wrapper
from src.dataset_wrapper import (
    INDEX_CACHE_VERSION,
    ZeekDataset,
    find_and_load_datasets,
)
from src.commons import BENIGN, MALICIOUS, BACKGROUND, is_malicious_label
from src.conn_normalizer import CANONICAL_FIELDS, ConnToSlipsConverter


ZEEK_FIELDS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
    "duration", "orig_bytes", "resp_bytes", "conn_state", "history", "orig_pkts", "resp_pkts", "label",
]
ZEEK_TYPES = [
    "time", "string", "addr", "port", "addr", "port", "enum", "string",
    "interval", "count", "count", "string", "string", "count", "count", "string",
]


def zeek_row(i, label="Benign"):
    """One tab-separated conn.log row matching ZEEK_FIELDS."""
    return "\t".join([
        f"1609459200.{i:06d}", f"uid-{i}", "192.168.1.1", str(10000 + i), "10.0.0.1", "443",
        "tcp", "https", "1.5", "100", "200", "SF", "ShADaFf", "2", "2", label,
    ])


def write_zeek_log(directory, rows, filename="conn.log", with_header=True):
    """Write a small synthetic Zeek TSV log (header lines + tab-separated rows)."""
    lines = ["#separator \\x09", "#path\tconn"]
    if with_header:
        lines.append("#fields\t" + "\t".join(ZEEK_FIELDS))
        lines.append("#types\t" + "\t".join(ZEEK_TYPES))
    lines.extend(rows)
    path = Path(directory) / filename
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Malicious", True),
        ("Malware", True),
        ("MALWARE_SMB", True),
        ("ATTACK_MALWARE", True),
        ("Botnet-Malicious", True),
        ("1", True),
        (1, True),
        ("Normal", False),
        ("From-Normal-V42", False),
        ("Anomaly", False),
        ("Benign", False),
        ("", False),
        ("0", False),
        (None, False),
    ],
)
def test_is_malicious_label(value, expected):
    """'MAL' only counts at the start of a word; '1' is malicious."""
    assert is_malicious_label(value) is expected


class TestZeekDataset:
    """Comprehensive test suite for ZeekDataset class."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Temporary directory for test files."""
        return tmp_path

    @pytest.fixture
    def cache_dir(self, temp_dir):
        """Index cache directory inside the test's tmp dir (never src/cache)."""
        return temp_dir / "cache"

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

    # ========== Initialization and File Detection Tests ==========
    def test_initialization_and_file_detection(
        self, sample_conn_log, temp_dir, cache_dir
    ):
        """Test ZeekDataset initialization and file priority detection."""
        # Test with conn.log
        ds_plain = ZeekDataset(temp_dir, batch_size=2, cache_dir=cache_dir)
        assert ds_plain.current_file == sample_conn_log
        assert ds_plain.batch_size == 2
        assert ds_plain.seed is None

        # Test file priority: labeled > alt_labeled > plain
        labeled = temp_dir / "conn.log.labeled"
        labeled.write_text(sample_conn_log.read_text())
        ds_labeled = ZeekDataset(temp_dir, batch_size=3, cache_dir=cache_dir)
        assert ds_labeled.current_file == labeled

        # Test alt_labeled priority
        alt_labeled = temp_dir / "labeled-conn.log"
        labeled.unlink()
        alt_labeled.write_text(sample_conn_log.read_text())
        ds_alt = ZeekDataset(temp_dir, batch_size=2, cache_dir=cache_dir)
        assert ds_alt.current_file == alt_labeled

    def test_initialization_missing_file_raises_error(self, temp_dir, cache_dir):
        """Test that missing conn.log raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ZeekDataset(temp_dir, batch_size=2, cache_dir=cache_dir)

    def test_initialization_missing_root_raises_error(self, cache_dir):
        """Test that missing root directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ZeekDataset("/nonexistent/path", batch_size=2, cache_dir=cache_dir)

    # ========== Header and Type Parsing Tests ==========
    def test_headers_and_types_parsing(self, sample_conn_log, temp_dir, cache_dir, capsys):
        """Test correct parsing of #fields and #types from log file."""
        ds = ZeekDataset(temp_dir, batch_size=2, cache_dir=cache_dir)

        assert ds.headers == ZEEK_FIELDS
        assert ds.types == ZEEK_TYPES
        assert "[WARN]" not in capsys.readouterr().out

    def test_missing_fields_header_warns(self, temp_dir, cache_dir, capsys):
        """A log without a '#fields' header still loads, but prints a [WARN]."""
        write_zeek_log(temp_dir, [zeek_row(1), zeek_row(2)], with_header=False)

        ds = ZeekDataset(temp_dir, batch_size=2, cache_dir=cache_dir)

        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "#fields" in out
        assert ds.headers == []
        assert len(ds) == 2
        assert len(ds.as_dataframe()) == 2

    # ========== Label Mapping and Filtering Tests ==========
    def test_label_mapping_and_background_filtering(
        self, sample_conn_log, temp_dir, cache_dir
    ):
        """Test label mapping to BENIGN/MALICIOUS and BACKGROUND filtering."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

        # Should have 4 valid flows (Background filtered out)
        # Flow 1: "Benign" -> BENIGN
        # Flow 2: "-" (empty) -> BENIGN (default)
        # Flow 3: "Malicious" -> MALICIOUS
        # Flow 4: "Background" -> filtered out
        # Flow 5: "MALWARE" -> MALICIOUS ("MAL" starts the word)
        assert ds.total_lines == 4
        assert ds.valid_indices == [0, 1, 2, 4]
        assert ds.labels == [BENIGN, BENIGN, MALICIOUS, MALICIOUS]
        assert BACKGROUND not in set(ds.labels)

    def test_label_mapping_variations(self, temp_dir, cache_dir):
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
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

        # 5 valid flows (background filtered), 2 benign, 3 malicious
        assert ds.total_lines == 5
        assert ds.labels == [BENIGN, BENIGN, MALICIOUS, MALICIOUS, MALICIOUS]

    def test_mal_must_start_a_word(self, temp_dir, cache_dir):
        """Normal / From-Normal / Anomaly contain 'MAL' mid-word and stay benign."""
        raw_labels = [
            "Malicious", "Malware", "MALWARE_SMB", "ATTACK_MALWARE", "1",
            "Normal", "From-Normal-V42", "Anomaly", "Benign", "-",
        ]
        write_zeek_log(temp_dir, [zeek_row(i, lab) for i, lab in enumerate(raw_labels)])

        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

        expected = [MALICIOUS] * 5 + [BENIGN] * 5
        assert ds.labels == expected
        assert ds.as_dataframe()["ground_truth_label"].tolist() == expected

    def test_missing_label_defaults_to_benign(self, temp_dir, cache_dir):
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
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

        assert ds.total_lines == 2
        assert all(label == str(BENIGN) for label in ds.labels)

    # ========== Blank lines / empty datasets ==========
    @pytest.mark.parametrize("seed", [None, 3])
    def test_blank_lines_are_skipped_consistently(self, temp_dir, cache_dir, seed):
        """Blank lines are not flows: index and row iteration stay aligned (also via the cache)."""
        rows = [
            zeek_row(1, "Benign"),
            "",
            zeek_row(2, "Malicious"),
            "   ",
            "",
            zeek_row(3, "Background"),
            "\t",
            zeek_row(4, "Malicious"),
            "",
        ]
        write_zeek_log(temp_dir, rows)

        for _ in range(2):  # first build (cache written), then reload from the cache
            ds = ZeekDataset(temp_dir, batch_size=10, seed=seed, cache_dir=cache_dir, persist_cache_threshold=1)
            df = ds.as_dataframe()

            assert len(ds) == 3
            assert sorted(ds.valid_indices) == [0, 1, 3]
            assert df["uid"].tolist() == ["uid-1", "uid-2", "uid-4"]
            assert df["ground_truth_label"].tolist() == [BENIGN, MALICIOUS, MALICIOUS]
        assert ds._cache_path().exists()

    @pytest.mark.parametrize("rows", [[], [zeek_row(1, "Background"), zeek_row(2, "background")]])
    def test_zero_valid_rows_with_seed(self, temp_dir, cache_dir, rows):
        """A dataset without any valid row and a seed loads as empty instead of crashing."""
        write_zeek_log(temp_dir, rows)

        ds = ZeekDataset(temp_dir, batch_size=10, seed=42, cache_dir=cache_dir)

        assert len(ds) == 0
        assert ds.valid_indices == []
        assert ds.labels == []
        assert ds.as_dataframe().empty

    def test_normalize_errors_propagate(self, temp_dir, cache_dir, monkeypatch):
        """Errors while normalizing records are raised, not swallowed into an empty frame."""
        write_zeek_log(temp_dir, [zeek_row(1)])

        def boom(self, flows):
            raise RuntimeError("normalization failed")

        monkeypatch.setattr(ConnToSlipsConverter, "normalize_batch", boom)
        with pytest.raises(RuntimeError, match="normalization failed"):
            ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

    # ========== Type Casting Tests ==========
    def test_cast_conversions(self, sample_conn_log, temp_dir, cache_dir):
        """Test type casting for int, float, bool, and string types."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)

        # Test _cast function
        assert ds._cast("1234", "count") == 1234
        assert ds._cast("12.34", "double") == 12.34
        assert ds._cast("t", "bool") is True
        assert ds._cast("f", "bool") is False
        assert ds._cast("hello", "string") == "hello"
        assert ds._cast("-", "count") is None
        assert ds._cast("", "string") is None

    def test_line_data_with_casting(self, sample_conn_log, temp_dir, cache_dir):
        """Test that retrieved lines have correct types after casting."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)
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
    def test_as_dataframe_returns_all_valid_rows(self, sample_conn_log, temp_dir, cache_dir):
        """as_dataframe should return every valid (non-background) record."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)
        df = ds.as_dataframe()

        assert len(df) == ds.total_lines == 4
        assert set(df.columns) == set(CANONICAL_FIELDS)
        assert df["ground_truth_label"].tolist() == [BENIGN, BENIGN, MALICIOUS, MALICIOUS]
        # the SLIPS 'label' column stays empty; the ground truth lives in ground_truth_label
        assert set(df["label"]) == {""}

    def test_as_dataframe_cached_instance(self, sample_conn_log, temp_dir, cache_dir):
        """Repeated calls to as_dataframe should reuse cached DataFrame."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)
        df1 = ds.as_dataframe()
        df2 = ds.as_dataframe()

        assert df1 is df2

    def test_len_matches_dataframe_length(self, sample_conn_log, temp_dir, cache_dir):
        """__len__ should match DataFrame length."""
        ds = ZeekDataset(temp_dir, batch_size=10, cache_dir=cache_dir)
        df = ds.as_dataframe()

        assert len(ds) == len(df) == ds.total_lines

    # ========== Seeding and Shuffling Tests ==========
    def test_seed_shuffles_consistently(self, sample_conn_log, temp_dir, cache_dir):
        """Same seed should shuffle valid indices deterministically."""
        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=42, cache_dir=cache_dir)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=42, cache_dir=cache_dir)

        assert ds1.valid_indices == ds2.valid_indices
        assert ds1.labels == ds2.labels

    def test_different_seeds_produce_different_order(self, temp_dir, cache_dir):
        """Different seeds should lead to different valid_index permutations."""
        write_zeek_log(temp_dir, [zeek_row(i) for i in range(20)])

        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=1, cache_dir=cache_dir)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=2, cache_dir=cache_dir)

        assert ds1.valid_indices != ds2.valid_indices
        assert sorted(ds1.valid_indices) == sorted(ds2.valid_indices) == list(range(20))

    def test_no_seed_is_deterministic_on_same_file(
        self, sample_conn_log, temp_dir, cache_dir
    ):
        """Without seed, the loader should preserve file order deterministically."""
        ds1 = ZeekDataset(temp_dir, batch_size=2, seed=None, cache_dir=cache_dir)
        ds2 = ZeekDataset(temp_dir, batch_size=2, seed=None, cache_dir=cache_dir)

        assert ds1.valid_indices == ds2.valid_indices == [0, 1, 2, 4]

    @pytest.mark.parametrize("shuffle_per_epoch", [False, True])
    @pytest.mark.parametrize("seed", [None, 42])
    def test_shuffle_per_epoch_is_stored_and_rows_stay_in_file_order(
        self, sample_conn_log, temp_dir, cache_dir, shuffle_per_epoch, seed
    ):
        """shuffle_per_epoch is only stored on the dataset (mixers shuffle); as_dataframe keeps file order."""
        ds = ZeekDataset(
            temp_dir, batch_size=2, seed=seed, shuffle_per_epoch=shuffle_per_epoch, cache_dir=cache_dir
        )

        assert ds.shuffle_per_epoch is shuffle_per_epoch
        df = ds.as_dataframe()
        assert df["uid"].tolist() == ["uid-1", "uid-2", "uid-3", "uid-5"]
        assert df["ground_truth_label"].tolist() == [BENIGN, BENIGN, MALICIOUS, MALICIOUS]

    # ========== New Parameter Tests ==========
    def test_custom_labeled_filenames(self, temp_dir, cache_dir):
        """Test initialization with custom labeled_filenames parameter."""
        conn_file = temp_dir / "custom_name.log"
        content = """#separator \t
#fields\tts\tuid\tproto\tlabel
#types\ttime\tstring\tenum\tstring
1609459200.001\tuid-1\ttcp\tBenign
"""
        conn_file.write_text(content)

        ds = ZeekDataset(
            temp_dir, batch_size=10, labeled_filenames=["custom_name.log"], cache_dir=cache_dir
        )
        assert ds.current_file == conn_file

    def test_custom_cache_dir(self, temp_dir):
        """The index cache is written into the custom cache_dir."""
        custom_cache = temp_dir / "my_cache"
        custom_cache.mkdir()

        lines = [
            "#separator \t",
            "#fields\tts\tuid\tproto\tlabel",
            "#types\ttime\tstring\tenum\tstring",
        ]
        for i in range(20):
            lines.append(f"1609459200.{i:06d}\tuid-{i}\ttcp\tBenign")

        conn_file = temp_dir / "conn.log"
        conn_file.write_text("\n".join(lines))

        ds = ZeekDataset(temp_dir, batch_size=100, cache_dir=custom_cache, persist_cache_threshold=5)
        cache_file = ds._cache_path()

        assert cache_file.parent == custom_cache
        assert cache_file.exists()
        assert ds.total_lines == 20
        assert set(ds.as_dataframe()["proto"]) == {"tcp"}

    def test_file_encoding_parameter(self, temp_dir, cache_dir):
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
            cache_dir=cache_dir,
        )
        assert ds.file_encoding == "utf-8"
        assert ds.file_errors == "ignore"
        assert ds.total_lines == 1

    # ========== Index cache ==========
    def test_cache_creation_for_large_dataset(self, temp_dir, cache_dir):
        """A cache (tagged with INDEX_CACHE_VERSION) is written only above persist_cache_threshold."""
        write_zeek_log(temp_dir, [zeek_row(i) for i in range(10)])

        small = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=10)
        assert not small._cache_path().exists()

        ds = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=9)
        cache_file = ds._cache_path()
        assert cache_file.exists()
        assert ds.total_lines == 10

        data = json.loads(cache_file.read_text())
        assert data["__version"] == INDEX_CACHE_VERSION == 2
        assert data["valid_indices"] == list(range(10))
        assert data["labels"] == [BENIGN] * 10
        assert data["headers"] == ZEEK_FIELDS

    def test_cache_reloaded_on_same_file(self, temp_dir, cache_dir):
        """A valid cache is reused for the same file instead of re-indexing it."""
        write_zeek_log(temp_dir, [zeek_row(i) for i in range(10)])

        ds1 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=5)
        cache_file = ds1._cache_path()
        assert ds1.labels == [BENIGN] * 10

        # mark the cached labels so we can tell whether the second load used the cache
        data = json.loads(cache_file.read_text())
        data["labels"] = [MALICIOUS] * 10
        cache_file.write_text(json.dumps(data))
        cache_mtime = cache_file.stat().st_mtime

        ds2 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=5)
        assert ds2.labels == [MALICIOUS] * 10
        assert ds2.as_dataframe()["ground_truth_label"].tolist() == [MALICIOUS] * 10
        assert ds2.total_lines == ds1.total_lines
        assert cache_file.stat().st_mtime == cache_mtime

    @pytest.mark.parametrize("stale_version", [None, 1, INDEX_CACHE_VERSION + 1])
    def test_cache_with_other_version_is_rebuilt(self, temp_dir, cache_dir, stale_version):
        """A cache without __version or with another version is ignored and rewritten."""
        write_zeek_log(temp_dir, [zeek_row(0, "Normal"), zeek_row(1, "Anomaly"), zeek_row(2, "Malicious")])

        ds1 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=1)
        cache_file = ds1._cache_path()

        # simulate a cache from an older version (which mapped Normal/Anomaly to malicious)
        data = json.loads(cache_file.read_text())
        if stale_version is None:
            del data["__version"]
        else:
            data["__version"] = stale_version
        data["labels"] = [MALICIOUS] * 3
        cache_file.write_text(json.dumps(data))

        ds2 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=1)
        assert ds2.labels == [BENIGN, BENIGN, MALICIOUS]

        rebuilt = json.loads(cache_file.read_text())
        assert rebuilt["__version"] == INDEX_CACHE_VERSION
        assert rebuilt["labels"] == [BENIGN, BENIGN, MALICIOUS]

    def test_corrupt_cache_is_rebuilt(self, temp_dir, cache_dir):
        """An unreadable cache file is dropped and rebuilt."""
        write_zeek_log(temp_dir, [zeek_row(i) for i in range(3)])
        ds1 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=1)
        cache_file = ds1._cache_path()
        cache_file.write_text("{not json")

        ds2 = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=1)

        assert ds2.total_lines == 3
        assert json.loads(cache_file.read_text())["__version"] == INDEX_CACHE_VERSION

    def test_clear_cache(self, temp_dir, cache_dir):
        """Test clear_cache removes cache file."""
        write_zeek_log(temp_dir, [zeek_row(i) for i in range(10)])

        ds = ZeekDataset(temp_dir, batch_size=100, cache_dir=cache_dir, persist_cache_threshold=5)
        cache_file = ds._cache_path()
        assert cache_file.exists()

        ds.clear_cache()
        assert not cache_file.exists()
        ds.clear_cache()  # clearing a missing cache is a no-op

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
            # Custom persist_cache_threshold (0 -> every non-empty dataset is cached)
            {"persist_cache_threshold": 0},
            # Default cache_dir (None -> <module dir>/cache)
            {"cache_dir": None},
            # Custom labeled_filenames
            {"labeled_filenames": ["special.log"]},
            # Custom file_encoding and file_errors
            {"file_encoding": "utf-8", "file_errors": "replace"},
            # Custom shuffle_per_epoch
            {"shuffle_per_epoch": True},
        ],
    )
    def test_find_and_load_datasets_all_options(self, temp_dir, params, monkeypatch):
        """Test find_and_load_datasets with all optional parameters."""
        data_subdir = params.get("data_subdir", "data")
        filenames = params.get("labeled_filenames", None)
        file_name = filenames[0] if filenames else "conn.log"
        content = """#separator \t
#fields\tts\tuid\tproto\tlabel
#types\ttime\tstring\tenum\tstring
1609459200.001\tuid-1\ttcp\tBenign
"""
        for name in ("001", "abc"):
            ds_dir = temp_dir / name / data_subdir
            ds_dir.mkdir(parents=True)
            (ds_dir / file_name).write_text(content, encoding=params.get("file_encoding", "utf-8"))
        # decoys: a matching dir without the data subdir, and one whose data subdir has no log
        (temp_dir / "002").mkdir()
        (temp_dir / "003" / data_subdir).mkdir(parents=True)

        if "cache_dir" in params and params["cache_dir"] is None:
            # cache_dir=None defaults to <dir of dataset_wrapper.py>/cache; point that at tmp
            fake_module_dir = temp_dir / "fake_src"
            fake_module_dir.mkdir()
            monkeypatch.setattr(dataset_wrapper, "__file__", str(fake_module_dir / "dataset_wrapper.py"))
            cache_dir = None
            expected_cache_dir = fake_module_dir / "cache"
        else:
            cache_dir = temp_dir / "cache"
            expected_cache_dir = cache_dir
        threshold = params.get("persist_cache_threshold", 30000)

        loaders = find_and_load_datasets(
            temp_dir,
            batch_size=params.get("batch_size", 1000),
            prefix_regex=params.get("prefix_regex", r"^\d{3}"),
            data_subdir=data_subdir,
            seed=params.get("seed", None),
            persist_cache_threshold=threshold,
            cache_dir=cache_dir,
            labeled_filenames=filenames,
            file_encoding=params.get("file_encoding", "utf-8"),
            file_errors=params.get("file_errors", "ignore"),
            shuffle_per_epoch=params.get("shuffle_per_epoch", False),
        )

        # Exactly the datasets matching prefix_regex with a log file are loaded
        regex = re.compile(params.get("prefix_regex", r"^\d{3}"))
        expected_keys = {name for name in ("001", "abc") if regex.match(name)}
        assert set(loaders) == expected_keys
        for key in expected_keys:
            ds = loaders[key]
            assert isinstance(ds, ZeekDataset)
            assert ds.root == (temp_dir / key / data_subdir).resolve()
            assert ds.current_file.name == file_name
            assert ds.total_lines == 1
            assert ds.batch_size == params.get("batch_size", 1000)
            assert ds.seed == params.get("seed", None)
            assert ds.persist_cache_threshold == threshold
            assert ds.cache_dir.resolve() == expected_cache_dir.resolve()
            assert ds._cache_path().exists() == (ds.total_lines > threshold)
            assert ds.file_encoding == params.get("file_encoding", "utf-8")
            assert ds.file_errors == params.get("file_errors", "ignore")
            assert ds.shuffle_per_epoch == params.get("shuffle_per_epoch", False)
