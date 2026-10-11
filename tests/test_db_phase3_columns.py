import sqlite3
from pathlib import Path
from exonit.db import ExoNitDB


def test_candidates_table_has_exominer_ffi_columns(tmp_path):
    db = ExoNitDB(str(tmp_path / "test.db"))
    cols = db._query("PRAGMA table_info(candidates)")
    names = {row["name"] for row in cols}
    assert "exominer_ffi_score" in names, \
        f"missing exominer_ffi_score; have {sorted(names)}"
    assert "exominer_ffi_model_version" in names
