"""ExoNit scorecard: measure data quality vs TOI ground truth."""


HARMONIC_FACTORS_SUB = (0.5, 1/3, 0.25, 0.2)
HARMONIC_FACTORS_SUPER = (2, 3, 4, 5)

DEPTH_BINS = ["<500", "500-1000", "1000-5000", ">5000"]
PERIOD_BINS = ["<3d", "3-10d", "10-30d", ">30d"]
SECTOR_BINS = ["1-2", "3-6", "7-15", ">15"]


def bin_depth(depth_ppm):
    if depth_ppm < 500:
        return "<500"
    if depth_ppm < 1000:
        return "500-1000"
    if depth_ppm < 5000:
        return "1000-5000"
    return ">5000"


def bin_period(period_days):
    if period_days < 3:
        return "<3d"
    if period_days < 10:
        return "3-10d"
    if period_days < 30:
        return "10-30d"
    return ">30d"


def bin_sectors(n_sectors):
    if n_sectors <= 2:
        return "1-2"
    if n_sectors <= 6:
        return "3-6"
    if n_sectors <= 15:
        return "7-15"
    return ">15"


import pandas as pd


def load_toi_catalog(csv_path):
    """Read TOI catalog CSV and return normalized DataFrame.

    Normalized columns:
        tic_id (str), disposition (str), period_days (float),
        depth_ppm (float), duration_hours (float), n_observations (int),
        epoch_bjd (float)
    """
    df = pd.read_csv(csv_path, low_memory=False)
    out = pd.DataFrame({
        "tic_id": df["TIC ID"].astype(str),
        "disposition": df["TFOPWG Disposition"].astype(str).replace("nan", None),
        "period_days": pd.to_numeric(df["Period (days)"], errors="coerce"),
        "depth_ppm": pd.to_numeric(df["Depth (ppm)"], errors="coerce"),
        "duration_hours": pd.to_numeric(df["Duration (hours)"], errors="coerce"),
        "n_observations": pd.to_numeric(
            df["Time Series Observations"], errors="coerce"
        ).fillna(0).astype(int),
        # Transit epoch (full BJD, ~2457000+); needed for window-function
        # detectability. Absent in some derived catalogs -> NaN.
        "epoch_bjd": pd.to_numeric(
            df.get("Epoch (BJD)", pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        ),
    })
    return out


# --- Window-function-aware detectability ----------------------------------
#
# A "miss" in the plain recall is ambiguous: the pipeline may have failed, OR
# the transit was effectively unobservable given TESS's real cadence coverage
# (gaps between sectors, discontinuous baselines). For long periods especially,
# a planet whose transits all fall in observing gaps cannot be recovered by any
# BLS. These helpers compute the real window function from a TIC's observed
# cadence timestamps and the TOI ephemeris, so recall can be reported over the
# subset that was actually detectable.

# Minimum distinct transits a box-search needs to fold a signal (matches the
# worker's >=2-transit guard in worker.py).
_MIN_TRANSITS_DETECTABLE = 2
# Minimum in-transit cadences to have any shot at an SNR detection. Lenient so
# even 30-min FFI cadence (~a few points per transit) can qualify.
_MIN_IN_TRANSIT_CADENCES = 3


def _epoch_to_lc_frame(epoch_bjd, time_arr):
    """Convert a full-BJD epoch to the same frame as ``time_arr``.

    TESS lightkurve time is BTJD (= BJD - 2457000, values ~1300-3500). TOI
    epochs are full BJD (~2457000+). Auto-detect the LC frame from its median
    so the phase fold aligns to the real transit center.
    """
    import numpy as np
    med = float(np.nanmedian(time_arr))
    if med < 100000.0:  # time_arr is BTJD
        return epoch_bjd - 2457000.0
    return epoch_bjd  # time_arr already in full BJD


def compute_window_function(time_arr, period_days, epoch_bjd, duration_hours):
    """Window function of a periodic transit against observed cadences.

    Returns a dict:
        observed_transits      distinct transit epochs with >=1 in-transit point
        in_transit_cadences    total cadences falling inside a transit
        n_transits_ideal       transits expected under continuous coverage
        duty_cycle             observed_transits / n_transits_ideal
        phase_coverage         fraction of 20 phase bins with any data
        detectability_class    'detectable' | 'marginal' | 'impossible'
    """
    import numpy as np
    t = np.asarray(time_arr, dtype=np.float64)
    t = t[np.isfinite(t)]
    if (t.size == 0 or not np.isfinite(period_days) or period_days <= 0
            or not np.isfinite(epoch_bjd) or not np.isfinite(duration_hours)):
        return {
            "observed_transits": 0, "in_transit_cadences": 0,
            "n_transits_ideal": 0, "duty_cycle": 0.0,
            "phase_coverage": 0.0, "detectability_class": "unknown",
        }

    t0 = _epoch_to_lc_frame(epoch_bjd, t)
    half = (duration_hours / 24.0) / 2.0
    span = float(t.max() - t.min())

    cycle = np.round((t - t0) / period_days)
    dt = t - (t0 + cycle * period_days)          # signed offset from center
    in_transit = np.abs(dt) <= half

    observed_transits = int(np.unique(cycle[in_transit]).size)
    in_transit_cadences = int(in_transit.sum())
    n_transits_ideal = int(np.floor(span / period_days)) + 1

    phase = ((t - t0) / period_days) % 1.0
    n_bins = 20
    phase_coverage = float(np.unique((phase * n_bins).astype(int)).size) / n_bins

    cls = classify_detectability(observed_transits, in_transit_cadences)
    return {
        "observed_transits": observed_transits,
        "in_transit_cadences": in_transit_cadences,
        "n_transits_ideal": n_transits_ideal,
        "duty_cycle": (observed_transits / n_transits_ideal
                       if n_transits_ideal else 0.0),
        "phase_coverage": phase_coverage,
        "detectability_class": cls,
    }


def classify_detectability(observed_transits, in_transit_cadences):
    """Bucket a TOI by whether TESS coverage made it recoverable.

    - 'impossible': fewer than 2 observed transits -> no box search can fold it.
    - 'marginal':   exactly the minimum transits, or too few in-transit points
                    to build SNR.
    - 'detectable': enough transits and in-transit cadences for a fair shot.
    """
    if observed_transits < _MIN_TRANSITS_DETECTABLE:
        return "impossible"
    if (observed_transits == _MIN_TRANSITS_DETECTABLE
            or in_transit_cadences < _MIN_IN_TRANSIT_CADENCES):
        return "marginal"
    return "detectable"


from exonit.completeness_store import completeness_pg_enabled
from exonit.db import ExoNitDB
from exonit.db_backend import connect_operational, is_pg_url


def _row_values(row):
    """Return a positional list of a fetched row's values.

    SQLite rows are tuples; the PG backend uses ``dict_row`` so a raw fetch is
    a dict. Both are normalised to a positional list so callers can unpack by
    index regardless of backend.
    """
    if row is None:
        return []
    if isinstance(row, dict):
        return list(row.values())
    return list(row)


def _read_sql_df(con, sql, is_pg):
    """Read a SELECT into a DataFrame from a borrowed connection.

    SQLite: ``pd.read_sql_query`` directly (the historical path).
    PG: pandas does NOT correctly consume a psycopg3 connection (esp. with the
    pooled ``dict_row`` factory), so we run a plain-tuple cursor and build the
    DataFrame from the result + column names. Same data, same column order.
    """
    if not is_pg:
        return pd.read_sql_query(sql, con)
    from psycopg.rows import tuple_row
    with con.cursor(row_factory=tuple_row) as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_db_processed(db_path):
    """Read stars, candidates, completeness from an ExoNit operational DB.

    Returns (stars_df, candidates_df, completeness_df).

    Accepts a SQLite path or a ``postgresql://`` DSN — the backend layer routes
    the reads. Note ``completeness`` stays on SQLite even in PG mode (hybrid
    backend); against a pure PG DSN that table does not exist, so this full
    loader is the SQLite-only path. The memory-bounded ``_for_toi`` loader is
    what the scorecard/miss path actually uses and it is PG-aware.
    """
    is_pg = is_pg_url(str(db_path))
    backend = connect_operational(str(db_path))
    try:
        with backend.raw_connection() as con:
            stars = _read_sql_df(con, "SELECT * FROM stars", is_pg)
            candidates = _read_sql_df(con, "SELECT * FROM candidates", is_pg)
            completeness = _read_sql_df(
                con, "SELECT tic_id FROM completeness", is_pg)
    finally:
        backend.close()
    stars["tic_id"] = stars["tic_id"].astype(str)
    candidates["tic_id"] = candidates["tic_id"].astype(str)
    completeness["tic_id"] = completeness["tic_id"].astype(str)
    return stars, candidates, completeness


def load_db_processed_for_toi(db_path, toi_tics):
    """Memory-bounded loader for the scorecard / miss-diagnosis path.

    Loads only the stars and candidates rows whose tic_id is in the TOI
    catalogue (the per-TIC metrics never touch non-TOI rows), plus the global
    coverage aggregates computed in SQL. Peak RAM scales with the TOI catalogue
    (~7.6k TICs), not the accumulated DB.

    Works against a SQLite path or a ``postgresql://`` DSN. The whole sequence
    (CREATE TEMP TABLE → populate → filtered SELECTs → aggregates) runs on one
    borrowed connection so the session-scoped temp table is visible throughout.

    completeness lives on SQLite even in PG mode (hybrid backend, the
    ``EXONIT_COMPLETENESS_PG`` flag OFF/default), so the
    ``distinct_completeness_tics`` aggregate is only queried on SQLite in that
    case; against a pure PG DSN with the flag OFF it is reported as 0 (the
    scorecard reads completeness coverage from the dedicated SQLite
    completeness DB elsewhere).

    Phase 4a Task 4 (``EXONIT_COMPLETENESS_PG=1``): completeness has moved off
    that SQLite sidecar into PG staging + bucketed Parquet aggregates.
    ``count_completeness_tics()`` (T1) is the exact count in that mode, but it
    lives on ``ExoNitDB`` (needs ``self._cmpl_pg``/``self.backend``) rather
    than the raw :class:`Backend` this loader otherwise borrows via
    ``connect_operational`` — so this branch opens a short-lived ``ExoNitDB``
    just for that one aggregate and closes it immediately after.

    Returns (stars_df, candidates_df, coverage_aggregates) where
    coverage_aggregates has keys: total_stars, completed_stars,
    distinct_candidate_tics, distinct_completeness_tics.
    """
    is_pg = is_pg_url(str(db_path))
    backend = connect_operational(str(db_path))
    dialect = backend.dialect
    ins_sql = dialect.translate(
        dialect.insert_ignore("_toi_tics", ["tic_id"]))
    try:
        with backend.raw_connection() as con:
            con.execute("CREATE TEMP TABLE _toi_tics (tic_id TEXT PRIMARY KEY)")
            con.cursor().executemany(
                ins_sql, [(str(t),) for t in toi_tics])
            stars = _read_sql_df(
                con,
                "SELECT * FROM stars WHERE tic_id IN "
                "(SELECT tic_id FROM _toi_tics)", is_pg)
            candidates = _read_sql_df(
                con,
                "SELECT * FROM candidates WHERE tic_id IN "
                "(SELECT tic_id FROM _toi_tics)", is_pg)
            # Alias every column: the PG backend uses dict_row, so unnamed
            # COUNT() columns would collide on the same key and only one would
            # survive. Aliases keep all three distinct on both backends.
            agg = con.execute(
                "SELECT (SELECT COUNT(*) FROM stars) AS total_stars, "
                "(SELECT COUNT(*) FROM stars WHERE status='completed') "
                "AS completed_stars, "
                "(SELECT COUNT(DISTINCT tic_id) FROM candidates) "
                "AS distinct_candidate_tics"
            ).fetchone()
            total_stars, completed_stars, distinct_candidate_tics = (
                _row_values(agg))
            if is_pg:
                if completeness_pg_enabled():
                    # Phase 4a Task 4: exact PG-backed count via T1's
                    # count_completeness_tics(), on its own short-lived
                    # ExoNitDB (see the docstring above for why it can't
                    # reuse the raw `backend`/`con` this loader already
                    # holds). Opening it here — rather than up front for
                    # every call — keeps the flag-OFF/SQLite paths exactly
                    # as cheap as before.
                    cmpl_db = ExoNitDB(str(db_path))
                    try:
                        distinct_completeness_tics = (
                            cmpl_db.count_completeness_tics())
                    finally:
                        cmpl_db.close()
                else:
                    # completeness is SQLite-only (hybrid backend, flag OFF);
                    # not in PG.
                    distinct_completeness_tics = 0
            else:
                row = con.execute(
                    "SELECT COUNT(DISTINCT tic_id) FROM completeness"
                ).fetchone()
                distinct_completeness_tics = _row_values(row)[0]
    finally:
        backend.close()
    stars["tic_id"] = stars["tic_id"].astype(str)
    candidates["tic_id"] = candidates["tic_id"].astype(str)
    coverage_aggregates = {
        "total_stars": int(total_stars),
        "completed_stars": int(completed_stars),
        "distinct_candidate_tics": int(distinct_candidate_tics),
        "distinct_completeness_tics": int(distinct_completeness_tics),
    }
    return stars, candidates, coverage_aggregates


def _best_match_for_known(candidates_for_tic, known_period, tolerance=0.01):
    """Return (candidate_row, kind) for the best match (or (None, None))."""
    best = None
    best_kind = None
    for _, cand in candidates_for_tic.iterrows():
        matched, kind = match_period(cand["period_days"], known_period,
                                     tolerance=tolerance)
        if matched:
            if kind == "direct":
                return cand, "direct"
            if best is None:
                best, best_kind = cand, kind
    return best, best_kind


def _recall_over(toi, candidates, tic_scope, tolerance):
    """Recall of the KP/CP planets whose TIC is in *tic_scope*.

    ``recall`` is **None**, never 0.0, when the scope contains no KP/CP at all
    (2026-07-14). A run over 50k random discovery targets typically contains
    ZERO confirmed planets, and reporting 0.0 there would read as "we detected
    none of them" — a catastrophic-looking score for a question that was never
    asked. None says the only true thing: this scope cannot measure recall.
    """
    kp_cp = toi[toi["disposition"].isin(["KP", "CP"])]
    in_scope = kp_cp[kp_cp["tic_id"].isin(tic_scope)]

    detected = 0
    for _, row in in_scope.iterrows():
        tic_candidates = candidates[candidates["tic_id"] == row["tic_id"]]
        match, _ = _best_match_for_known(
            tic_candidates, row["period_days"], tolerance)
        if match is not None:
            detected += 1

    n = len(in_scope)
    return {
        "recall": (detected / n) if n > 0 else None,
        "measurable": n > 0,
        "n_kp_cp": int(n),
        "n_detected": int(detected),
    }


def compute_recall_and_coverage(toi, stars, candidates, completeness,
                                tolerance=0.01, *, coverage_aggregates=None,
                                lot_tics=None):
    """Compute recall (per-lot AND cumulative) plus coverage metrics.

    TWO recalls, deliberately, and never conflated (the 2026-07-14 audit):

    * ``lot``        — the KP/CP planets in the TIC set of THIS RUN's lot. This
                       is the only number that can validate a change to the
                       pipeline, because it is the only one that reacts to the
                       run. Requires *lot_tics*.
    * ``cumulative`` — every KP/CP ever processed into this DB. Useful as a
                       health signal for the sector; useless as an A/B metric.

    Before this split there was one number, ``recall_global``, which WAS the
    cumulative one but was read (in STATUS.md, in dossiers, by me) as the run's
    result. It doesn't move: three consecutive runs over 100k / 300k / 400k
    completely different stars all reported recall=0.7231270358306189 with
    n=307, to the last decimal, because the sector's KP/CP were already all
    processed and 400k fresh stars add virtually no new confirmed planets. A
    metric that cannot be wrong cannot be right either — every A/B comparison
    made against it has to be re-examined.

    When coverage_aggregates is provided (a dict with total_stars,
    completed_stars, distinct_candidate_tics, distinct_completeness_tics),
    the coverage block uses those pre-computed SQL aggregates instead of
    scanning the full DataFrames. completeness may be None in that case.
    """
    processed_tics = set(stars["tic_id"].astype(str))
    cumulative = _recall_over(toi, candidates, processed_tics, tolerance)

    if lot_tics is None:
        lot = {"recall": None, "measurable": False, "n_kp_cp": 0,
               "n_detected": 0,
               "reason": "no lot csv provided (--sample)"}
    else:
        # Intersect with processed_tics: a TIC in the lot that the run never
        # actually completed must not count as a miss against it.
        lot = _recall_over(toi, candidates,
                           processed_tics & {str(t) for t in lot_tics},
                           tolerance)
        lot["lot_size"] = len(set(lot_tics))
        if not lot["measurable"]:
            lot["reason"] = "the lot contains no KP/CP — recall is not measurable"

    if coverage_aggregates is not None:
        total = coverage_aggregates["total_stars"]
        completed = coverage_aggregates["completed_stars"]
        with_candidate = coverage_aggregates["distinct_candidate_tics"]
        with_completeness = coverage_aggregates["distinct_completeness_tics"]
    else:
        completed = (stars["status"] == "completed").sum()
        with_candidate = candidates["tic_id"].nunique()
        with_completeness = completeness["tic_id"].nunique()
        total = len(stars)
    coverage = {
        "completed": completed / total if total else 0.0,
        "with_candidate": with_candidate / total if total else 0.0,
        "with_completeness": with_completeness / total if total else 0.0,
    }

    return {
        "lot": lot,
        "cumulative": cumulative,
        "coverage": coverage,
    }


import numpy as np


def compute_recall_by_bin(toi, stars, candidates, tolerance=0.01):
    """Return nested dict of recall ratios by depth/period/sectors bin."""
    stars_idx = stars.set_index("tic_id")
    processed_tics = set(stars_idx.index.astype(str))
    kp_cp = toi[toi["disposition"].isin(["KP", "CP"])].copy()
    kp_cp = kp_cp[kp_cp["tic_id"].isin(processed_tics)]

    depth_totals = {b: [0, 0] for b in DEPTH_BINS}
    period_totals = {b: [0, 0] for b in PERIOD_BINS}
    sector_totals = {b: [0, 0] for b in SECTOR_BINS}

    for _, row in kp_cp.iterrows():
        tic_candidates = candidates[candidates["tic_id"] == row["tic_id"]]
        match, _ = _best_match_for_known(
            tic_candidates, row["period_days"], tolerance
        )
        detected = 1 if match is not None else 0

        if pd.notna(row["depth_ppm"]):
            db = bin_depth(row["depth_ppm"])
            depth_totals[db][0] += detected
            depth_totals[db][1] += 1

        if pd.notna(row["period_days"]):
            pb = bin_period(row["period_days"])
            period_totals[pb][0] += detected
            period_totals[pb][1] += 1

        n_sec = int(stars_idx.loc[row["tic_id"]]["n_sectors"] or 0)
        sb = bin_sectors(n_sec)
        sector_totals[sb][0] += detected
        sector_totals[sb][1] += 1

    def ratios(totals):
        return {b: (d / t if t > 0 else 0.0) for b, (d, t) in totals.items()}

    return {
        "recall_by_depth": ratios(depth_totals),
        "recall_by_period": ratios(period_totals),
        "recall_by_sectors": ratios(sector_totals),
    }


def _rel_err(det, known):
    if known == 0 or pd.isna(det) or pd.isna(known):
        return None
    return abs(det - known) / known


def compute_parameter_precision(toi, stars, candidates, tolerance=0.01):
    """Median and sigma of relative error for period/depth/duration."""
    processed_tics = set(stars["tic_id"].astype(str))
    kp_cp = toi[toi["disposition"].isin(["KP", "CP"])].copy()
    kp_cp = kp_cp[kp_cp["tic_id"].isin(processed_tics)]

    period_errs, depth_errs, duration_errs = [], [], []
    for _, row in kp_cp.iterrows():
        tic_candidates = candidates[candidates["tic_id"] == row["tic_id"]]
        match, kind = _best_match_for_known(
            tic_candidates, row["period_days"], tolerance
        )
        if match is None:
            continue
        multiplier = 1.0
        if kind and kind.startswith("harmonic_"):
            multiplier = float(kind.split("_", 1)[1])
        elif kind and kind.startswith("sub_harmonic_"):
            multiplier = float(kind.split("_", 2)[2])
        target_period = row["period_days"] * multiplier

        err_p = _rel_err(match["period_days"], target_period)
        if err_p is not None:
            period_errs.append(err_p)

        err_d = _rel_err(match["depth_ppm"], row["depth_ppm"])
        if err_d is not None:
            depth_errs.append(err_d)

        cand_duration_hours = match["duration_days"] * 24
        err_dur = _rel_err(cand_duration_hours, row["duration_hours"])
        if err_dur is not None:
            duration_errs.append(err_dur)

    def summarize(errs):
        if not errs:
            return {"median_rel_err": None, "sigma": None, "n": 0}
        return {
            "median_rel_err": float(np.median(errs)),
            "sigma": float(np.std(errs)),
            "n": len(errs),
        }

    return {
        "period_precision": summarize(period_errs),
        "depth_precision": summarize(depth_errs),
        "duration_precision": summarize(duration_errs),
    }


def compute_exominer_metrics(toi, stars, candidates, threshold=0.5):
    """ExoMiner recall + specificity over the processed sample.

    Reports three flavours of recall on KP+CP and the matching specificity
    on FP+FA, plus null counts so a glance tells you how much of the
    catalogue is genuinely scored vs missing data:

    * **docker** — the ExoMiner++ Docker score from the server-side path
      (column ``exominer_score``). Available on TICs with SPOC DV products.
    * **ffi** — the Phase 3 worker-side score (``exominer_ffi_score``).
      Available on every TIC whose light curve loaded successfully.
    * **combined** — ``max(docker, ffi)`` ≥ threshold. Mirrors the verdict
      v2.1 fallback rule when the Docker score is NULL.

    Aggregated per-TIC (multi-candidate TICs collapse to the max), which
    is the right granularity for catalogue-level recall vs the published
    TOI list.
    """
    cands = candidates.copy()
    cands["tic_id"] = cands["tic_id"].astype(str)
    if "exominer_score" not in cands.columns:
        cands["exominer_score"] = float("nan")
    if "exominer_ffi_score" not in cands.columns:
        cands["exominer_ffi_score"] = float("nan")
    per_tic = cands.groupby("tic_id").agg(
        em=("exominer_score", "max"),
        em_ffi=("exominer_ffi_score", "max"),
    )

    processed_tics = set(stars["tic_id"].astype(str))

    def _tally(disp_set):
        sel = toi[toi["disposition"].isin(disp_set)
                  & toi["tic_id"].isin(processed_tics)]
        n = len(sel)
        if n == 0:
            return {"n": 0, "em_pos": 0, "em_null": 0,
                    "ffi_pos": 0, "ffi_null": 0, "combined_pos": 0}
        em_pos = em_null = ffi_pos = ffi_null = combined_pos = 0
        for tic in sel["tic_id"]:
            row = per_tic.loc[tic] if tic in per_tic.index else None
            em = row["em"] if row is not None else None
            em_ffi = row["em_ffi"] if row is not None else None
            # em==-1 is the no-DV cache-negative sentinel (see
            # progress/investigation-exominer-minus1-2026-07-02.md), not a
            # real Docker score — bucket it with "null" (not attempted) so
            # recall_docker / specificity_docker report honest coverage
            # instead of silently dropping these rows from BOTH buckets.
            # exominer_ffi_score never carries -1 (worker path only), so
            # em_ffi's bucketing is unchanged.
            em_missing = em is None or pd.isna(em) or em == -1
            em_ok = (not em_missing) and em >= threshold
            ffi_ok = em_ffi is not None and not pd.isna(em_ffi) and em_ffi >= threshold
            if em_missing: em_null += 1
            elif em_ok:    em_pos += 1
            if em_ffi is None or pd.isna(em_ffi): ffi_null += 1
            elif ffi_ok:                          ffi_pos += 1
            if em_ok or ffi_ok:                   combined_pos += 1
        return {"n": n, "em_pos": em_pos, "em_null": em_null,
                "ffi_pos": ffi_pos, "ffi_null": ffi_null,
                "combined_pos": combined_pos}

    pos = _tally({"KP", "CP"})
    neg = _tally({"FP", "FA"})

    def _ratio(num, den):
        return num / den if den else 0.0

    return {
        "exominer": {
            "threshold": threshold,
            "n_kp_cp_processed": pos["n"],
            "n_fp_fa_processed": neg["n"],
            "recall_docker":    _ratio(pos["em_pos"],       pos["n"]),
            "recall_ffi":       _ratio(pos["ffi_pos"],      pos["n"]),
            "recall_combined":  _ratio(pos["combined_pos"], pos["n"]),
            "specificity_docker":   _ratio(neg["n"] - neg["em_pos"] - neg["em_null"],   neg["n"]),
            "specificity_ffi":      _ratio(neg["n"] - neg["ffi_pos"] - neg["ffi_null"], neg["n"]),
            "kp_cp_em_null":  pos["em_null"],
            "kp_cp_ffi_null": pos["ffi_null"],
            "fp_fa_em_null":  neg["em_null"],
            "fp_fa_ffi_null": neg["ffi_null"],
        }
    }


def compute_fp_rate(toi, stars, candidates, cnn_threshold=0.5):
    """Fraction of known FPs that ExoNit labels as planet."""
    processed_tics = set(stars["tic_id"].astype(str))
    fps = toi[toi["disposition"] == "FP"]
    fps_processed = fps[fps["tic_id"].isin(processed_tics)]

    cnn_flagged = 0
    claude_flagged = 0
    n = len(fps_processed)
    for _, row in fps_processed.iterrows():
        tic_candidates = candidates[candidates["tic_id"] == row["tic_id"]]
        if tic_candidates.empty:
            continue
        if (tic_candidates["cnn_score"].fillna(0) > cnn_threshold).any():
            cnn_flagged += 1
        if (tic_candidates["claude_verdict"] == "planet_candidate").any():
            claude_flagged += 1

    return {
        "fp_rate_cnn": cnn_flagged / n if n else 0.0,
        "fp_rate_claude": claude_flagged / n if n else 0.0,
        "n_fp_processed": int(n),
    }


from datetime import datetime


#: Bumped to 2 on 2026-07-14, when metrics.recall_global (a CUMULATIVE number
#: that every reader took for the run's own result) was replaced by the
#: explicit metrics.lot / metrics.cumulative split. Tools that compare
#: scorecards across time MUST branch on this: a v1 file's recall_global is
#: today's metrics.cumulative.recall, and it says nothing about that run's lot.
SCHEMA_VERSION = 2


def build_scorecard(toi, stars, candidates, completeness,
                    *, run_id, sample_size, stratification,
                    pipeline_version, seed, tolerance=0.01,
                    coverage_aggregates=None, lot_tics=None, lot_csv=None):
    """Assemble full scorecard dict from raw inputs.

    ``lot_tics`` (+ ``lot_csv``, for provenance) scope the run's own recall.
    Omitted => metrics.lot is emitted as not-measurable with a reason, never as
    a number: a scorecard that silently reports the cumulative figure as if it
    were the run's is exactly the failure this schema exists to prevent.
    """
    rc = compute_recall_and_coverage(
        toi, stars, candidates, completeness, tolerance,
        coverage_aggregates=coverage_aggregates, lot_tics=lot_tics,
    )
    bins = compute_recall_by_bin(toi, stars, candidates, tolerance)
    prec = compute_parameter_precision(toi, stars, candidates, tolerance)
    fp = compute_fp_rate(toi, stars, candidates)
    em = compute_exominer_metrics(toi, stars, candidates)

    lot = dict(rc["lot"])
    if lot_csv:
        lot["lot_csv"] = lot_csv

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sample_size": sample_size,
        "stratification": stratification,
        "pipeline_version": pipeline_version,
        "seed": seed,
        "tolerance": tolerance,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            # The run's own result. The ONLY number an A/B may be read from.
            "lot": lot,
            # Whole-DB health. Moves by ~1 detection per run at 400k stars;
            # never use it to judge a pipeline change.
            "cumulative": rc["cumulative"],
            # Everything below is cumulative too (bins over 2 KP/CP would be
            # noise, not statistics) — see scorecard_to_markdown's header.
            **bins,
            **prec,
            "fp_rate_cnn": fp["fp_rate_cnn"],
            "fp_rate_claude": fp["fp_rate_claude"],
            "n_fp_processed": fp["n_fp_processed"],
            **em,
        },
        "coverage": rc["coverage"],
    }


def scorecard_to_markdown(card):
    """Human-readable markdown report for terminal or docs.

    Leads with the LOT recall and states plainly when it is not measurable.
    The cumulative figure is reported second and explicitly labelled as unable
    to validate the run — a reader who skims must not be able to walk away with
    the cumulative number as if it were this run's score. That is precisely
    what happened for months.
    """
    m = card["metrics"]
    lot, cum = m["lot"], m["cumulative"]

    if lot["measurable"]:
        lot_lines = [
            f"- **recall (lot) = {lot['recall']:.3f}** "
            f"({lot['n_detected']}/{lot['n_kp_cp']} KP/CP in the lot)",
            "",
            "This is the only figure that measures THIS run and the only one on "
            "which a pipeline A/B can be read.",
        ]
    else:
        lot_lines = [
            "- **recall (lot) = n/a** — "
            f"{lot.get('reason', 'not measurable')}.",
            "",
            "⚠️ This run **cannot be validated with recall**: its lot "
            "contains no confirmed planet (KP/CP). The cumulative figure below "
            "does **NOT** validate this run — it does not react to what you changed.",
        ]

    cum_recall = ("n/a" if cum["recall"] is None
                  else f"{cum['recall']:.3f}")
    lines = [
        f"# Scorecard: {card['run_id']}",
        "",
        f"- Sample size: **{card['sample_size']}**",
        f"- Pipeline version: `{card['pipeline_version']}`",
        f"- Stratification: {card.get('stratification', 'n/a')}",
        f"- Seed: {card.get('seed')}",
        f"- Generated: {card.get('generated_at')}",
        "",
        "## Lot recall (this run)",
        "",
        *lot_lines,
        "",
        "## Cumulative recall (whole DB — NOT the result of this run)",
        "",
        f"- recall (cumulative) = {cum_recall} "
        f"({cum['n_detected']}/{cum['n_kp_cp']} KP/CP ever processed)",
        "",
        "Global catalogue health. It moves ~1 detection per 400k-star run: "
        "it is not useful for judging any pipeline change.",
        "",
        "## Recall by Depth (ppm) — cumulative",
        "",
        "| Bin | Recall |",
        "|-----|--------|",
    ]
    for b, v in m["recall_by_depth"].items():
        lines.append(f"| {b} | {v:.3f} |")
    lines += ["", "## Recall by Period", "", "| Bin | Recall |", "|-----|--------|"]
    for b, v in m["recall_by_period"].items():
        lines.append(f"| {b} | {v:.3f} |")
    lines += ["", "## Recall by Sectors", "", "| Bin | Recall |", "|-----|--------|"]
    for b, v in m["recall_by_sectors"].items():
        lines.append(f"| {b} | {v:.3f} |")

    def fmt_prec(label, p):
        if p["median_rel_err"] is None:
            return f"- {label}: n/a (n=0)"
        return (f"- {label}: median_rel_err={p['median_rel_err']:.4f}, "
                f"sigma={p['sigma']:.4f}, n={p['n']}")

    lines += ["", "## Parameter Precision", "",
              fmt_prec("period", m["period_precision"]),
              fmt_prec("depth", m["depth_precision"]),
              fmt_prec("duration", m["duration_precision"]),
              "",
              "## FP Rate",
              "",
              f"- fp_rate_cnn    = {m['fp_rate_cnn']:.3f} "
              f"(n_fp={m['n_fp_processed']})",
              f"- fp_rate_claude = {m['fp_rate_claude']:.3f}",
              ""]
    if "exominer" in m:
        e = m["exominer"]
        lines += ["## ExoMiner",
                  "",
                  f"Threshold: {e['threshold']:.2f}. "
                  f"KP+CP processed: {e['n_kp_cp_processed']}, "
                  f"FP+FA processed: {e['n_fp_fa_processed']}.",
                  "",
                  "| Stream | Recall (KP+CP) | Specificity (FP+FA) | KP+CP null | FP+FA null |",
                  "|--------|----------------|---------------------|------------|------------|",
                  f"| Docker  | {e['recall_docker']:.3f}   | {e['specificity_docker']:.3f}   | {e['kp_cp_em_null']}  | {e['fp_fa_em_null']}  |",
                  f"| FFI     | {e['recall_ffi']:.3f}      | {e['specificity_ffi']:.3f}      | {e['kp_cp_ffi_null']} | {e['fp_fa_ffi_null']} |",
                  f"| Combined| {e['recall_combined']:.3f} | n/a                                 | —          | —          |",
                  ""]
    lines += ["## Coverage", ""]
    for k, v in card["coverage"].items():
        lines.append(f"- {k}: {v:.3f}")
    return "\n".join(lines)


def match_period(det_period, known_period, tolerance=0.01, factors=None,
                  inclusive=False):
    """Return (matched, kind) for detected vs known period.

    kind is one of: "direct", "harmonic_{k}", "sub_harmonic_{k}", or None
    (or "alias_{k}" when a custom ``factors`` override is given — see below).
    A direct match takes precedence over harmonic matches.

    factors: optional iterable of alias multipliers to test INSTEAD of the
        default HARMONIC_FACTORS_SUPER + HARMONIC_FACTORS_SUB combination
        (the direct 1:1 check always runs first, regardless). Existing
        callers (factors=None, the default) get the original harmonic /
        sub-harmonic behavior, unchanged.
    inclusive: if True, use a "<=" tolerance comparison instead of the
        default strict "<". Existing callers (inclusive=False, the default)
        get the original strict comparison, unchanged. Added (2026-07-02,
        D10 dedup) so callers with a legacy inclusive-tolerance predicate can
        delegate here without any change in behavior — see
        docs/reviews/2026-07-02-opus48-cleanup-review.md item D10 and
        progress/impl-training-consolidation-2026-07-02.md.
    """
    if known_period is None or known_period <= 0:
        return False, None
    if det_period is None or det_period <= 0:
        return False, None

    def _within(target):
        rel = abs(det_period - target) / target
        return rel <= tolerance if inclusive else rel < tolerance

    if _within(known_period):
        return True, "direct"

    if factors is not None:
        for k in factors:
            target = known_period * k
            if _within(target):
                return True, f"alias_{k}"
        return False, None

    for k in HARMONIC_FACTORS_SUPER:
        target = known_period * k
        if _within(target):
            return True, f"harmonic_{k}"

    for k in HARMONIC_FACTORS_SUB:
        target = known_period * k
        if _within(target):
            return True, f"sub_harmonic_{k}"

    return False, None
