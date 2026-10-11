"""Unit tests for exonit/reserved_holdout.py -- the guard that keeps the
reserved TOI hold-out out of every model's train/val split (spec Fase 1
annex flagged this as a gap: the guard was wired into both splitters but
never had a dedicated test). Every test uses a FICTITIOUS reserved-TIC file
via ``tmp_path`` (the ``path=`` kwarg every public function accepts) --
never the real 300-TIC ``data/eval/toi_holdout_reserved_v1.txt`` as the only
exercised path, per the Fase 3 briefing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exonit.reserved_holdout import (  # noqa: E402
    exclude_reserved,
    normalize_tic,
    reassign_reserved_split,
    reserved_holdout_tics,
)


@pytest.fixture
def reserved_file(tmp_path):
    """A fictitious reserved-TIC file -- 3 TICs, one with whitespace/blank
    lines to exercise the strip/skip-empty-lines logic."""
    p = tmp_path / "fake_reserved.txt"
    p.write_text("111111\n222222\n\n  333333  \n")
    return p


# --------------------------------------------------------------------------- #
# normalize_tic
# --------------------------------------------------------------------------- #

class TestNormalizeTic:
    def test_int_like_values_normalise_to_int_string(self):
        assert normalize_tic(111111) == "111111"
        assert normalize_tic("111111") == "111111"
        assert normalize_tic(111111.0) == "111111"

    def test_non_numeric_falls_back_to_stripped_string(self):
        assert normalize_tic(" TIC-abc ") == "TIC-abc"


# --------------------------------------------------------------------------- #
# reserved_holdout_tics
# --------------------------------------------------------------------------- #

class TestReservedHoldoutTics:
    def test_reads_and_normalises_fictitious_file(self, reserved_file):
        tics = reserved_holdout_tics(path=reserved_file)
        assert tics == {"111111", "222222", "333333"}

    def test_missing_file_returns_empty_set_with_warning(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.txt"
        tics = reserved_holdout_tics(path=missing)
        assert tics == set()
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert str(missing) in err

    def test_real_default_path_is_not_the_only_thing_exercised(self, reserved_file):
        # Sanity: the fixture path is genuinely different from the real
        # production reserved-hold-out file (this suite never depends on
        # data/eval/toi_holdout_reserved_v1.txt existing/being any size).
        from exonit.reserved_holdout import RESERVED_PATH
        assert Path(reserved_file) != RESERVED_PATH


# --------------------------------------------------------------------------- #
# exclude_reserved
# --------------------------------------------------------------------------- #

class TestExcludeReserved:
    def test_reserved_tic_is_dropped_non_reserved_kept(self, reserved_file):
        df = pd.DataFrame({"tic_id": ["111111", "999999", "222222"],
                           "value": [1, 2, 3]})
        out = exclude_reserved(df, "tic_id", path=reserved_file, label="test")
        assert set(out["tic_id"]) == {"999999"}
        assert list(out["value"]) == [2]

    def test_no_reserved_tics_present_returns_all_rows_untouched(self, reserved_file):
        df = pd.DataFrame({"tic_id": ["999999", "888888"]})
        out = exclude_reserved(df, "tic_id", path=reserved_file, label="test")
        assert set(out["tic_id"]) == {"999999", "888888"}

    def test_missing_file_is_a_noop_not_an_exception(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        df = pd.DataFrame({"tic_id": ["111111", "999999"]})
        out = exclude_reserved(df, "tic_id", path=missing, label="test")
        # No exclusion happened -- every row (including the TIC that WOULD
        # have been reserved had the file existed) survives. This is the
        # documented "fail loud (WARNING), not silent poisoning" contract:
        # the caller sees the warning on stderr but the guard itself does
        # not raise, so a mis-wired path is visible rather than crashing a
        # long-running build.
        assert len(out) == 2

    def test_missing_tic_column_is_a_noop_with_warning(self, reserved_file, capsys):
        df = pd.DataFrame({"not_a_tic_col": ["111111"]})
        out = exclude_reserved(df, "tic_id", path=reserved_file, label="test")
        assert len(out) == 1  # untouched
        assert "WARNING" in capsys.readouterr().err

    def test_empty_dataframe_is_returned_unchanged(self, reserved_file):
        df = pd.DataFrame({"tic_id": []})
        out = exclude_reserved(df, "tic_id", path=reserved_file, label="test")
        assert len(out) == 0

    def test_int_dtype_tic_column_still_matches_string_reserved_list(self, reserved_file):
        # The reserved file stores TICs as text; production tic_id columns
        # are sometimes int64 -- normalize_tic must bridge the two.
        df = pd.DataFrame({"tic_id": [111111, 999999]})
        out = exclude_reserved(df, "tic_id", path=reserved_file, label="test")
        assert set(out["tic_id"]) == {999999}


# --------------------------------------------------------------------------- #
# reassign_reserved_split -- the guard both splitters actually call
# --------------------------------------------------------------------------- #

class TestReassignReservedSplit:
    def test_reserved_tic_never_ends_up_in_train_or_val(self, reserved_file):
        df = pd.DataFrame({
            "tic_id": ["111111", "222222", "999999", "888888"],
            "split": ["train", "val", "train", "test"],
        })
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        by_tic = out.set_index("tic_id")["split"].to_dict()
        assert by_tic["111111"] == "reserved"
        assert by_tic["222222"] == "reserved"
        # Non-reserved TICs are untouched, whatever split they were in.
        assert by_tic["999999"] == "train"
        assert by_tic["888888"] == "test"

    def test_no_rows_dropped_only_relabelled(self, reserved_file):
        df = pd.DataFrame({
            "tic_id": ["111111", "222222", "333333", "999999"],
            "split": ["train", "train", "val", "test"],
        })
        n_before = len(df)
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        assert len(out) == n_before  # row-aligned, never drops (unlike exclude_reserved)
        assert set(out["split"]) == {"reserved", "test"}

    def test_custom_reserved_label(self, reserved_file):
        df = pd.DataFrame({"tic_id": ["111111"], "split": ["train"]})
        out = reassign_reserved_split(df, "tic_id", "split",
                                      reserved_label="holdout_v9",
                                      path=reserved_file, label="test")
        assert out.iloc[0]["split"] == "holdout_v9"

    def test_non_reserved_tic_is_not_touched(self, reserved_file):
        df = pd.DataFrame({"tic_id": ["555555"], "split": ["val"]})
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        assert out.iloc[0]["split"] == "val"

    def test_missing_file_is_a_noop_not_an_exception(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.txt"
        df = pd.DataFrame({"tic_id": ["111111"], "split": ["train"]})
        out = reassign_reserved_split(df, "tic_id", "split", path=missing,
                                      label="test")
        assert out.iloc[0]["split"] == "train"  # NOT reassigned -- no file, no guard
        assert "WARNING" in capsys.readouterr().err

    def test_missing_tic_column_is_a_noop_with_warning(self, reserved_file, capsys):
        df = pd.DataFrame({"split": ["train"]})
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        assert out.iloc[0]["split"] == "train"
        assert "WARNING" in capsys.readouterr().err

    def test_empty_dataframe_is_a_noop(self, reserved_file):
        df = pd.DataFrame({"tic_id": [], "split": []})
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        assert len(out) == 0

    def test_int_dtype_tic_column_still_matches(self, reserved_file):
        df = pd.DataFrame({"tic_id": [111111, 999999], "split": ["train", "val"]})
        out = reassign_reserved_split(df, "tic_id", "split", path=reserved_file,
                                      label="test")
        by_tic = out.set_index("tic_id")["split"].to_dict()
        assert by_tic[111111] == "reserved"
        assert by_tic[999999] == "val"


# --------------------------------------------------------------------------- #
# Integration-flavoured: the exact call pattern the two real splitters use
# --------------------------------------------------------------------------- #

class TestSplitterCallPattern:
    def test_matches_build_exominer_ffi_splits_v2_call_shape(self, reserved_file):
        """Mirrors scripts/build_exominer_ffi_splits_v2.py's exact call:
        keep = reassign_reserved_split(keep, "tic_id", "split",
                                       label="exominer_ffi.splits_v2")
        -- but against a fictitious reserved file, not the real 300-TIC one."""
        keep = pd.DataFrame({
            "h5_idx": [0, 1, 2], "tic_id": ["111111", "999999", "222222"],
            "label": ["FP", "PC", "KP"], "label_binary": [0, 1, 1],
            "split": ["train", "val", "test"],
        })
        out = reassign_reserved_split(keep, "tic_id", "split", path=reserved_file,
                                      label="exominer_ffi.splits_v2")
        by_tic = out.set_index("tic_id")["split"].to_dict()
        assert by_tic["111111"] == "reserved"
        assert by_tic["222222"] == "reserved"
        assert by_tic["999999"] == "val"
