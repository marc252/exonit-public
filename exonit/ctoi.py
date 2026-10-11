"""cTOI submission tools — phase-folded plots, TOI cross-check, ExoFOP export.

Generates publication-ready phase-folded transit images from FITS data,
cross-checks candidates against the TOI catalog to avoid duplicates,
and exports candidates in ExoFOP-compatible CSV format.
"""

import csv
import io
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_phase_fold_plot(tic_id, candidate, time, flux, output_path=None):
    """Generate a publication-ready phase-folded transit plot.

    Args:
        tic_id: TIC identifier.
        candidate: Dict with period_days, t0, duration_days, depth_ppm, snr.
        time: Light curve time array.
        flux: Normalized flux array.
        output_path: If set, save PNG here. Otherwise return bytes.

    Returns:
        PNG bytes if output_path is None, else the output_path.
    """
    period = candidate['period_days']
    t0 = candidate['t0']
    duration = candidate.get('duration_days', 0.05)
    depth = candidate.get('depth_ppm', 0)
    snr = candidate.get('snr', 0)
    cnn = candidate.get('cnn_score', None)
    fpp = candidate.get('triceratops_fpp', None)

    # Phase-fold
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    phase_hours = phase * period * 24

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                     gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor('#1a1a2e')

    # Top: full light curve with transit markers
    ax1.set_facecolor('#16213e')
    ax1.scatter(time, flux, s=0.2, c='#a8dadc', alpha=0.4, rasterized=True)
    for t in np.arange(t0, time[-1], period):
        if t >= time[0]:
            ax1.axvline(t, color='#e63946', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Time (BTJD)', color='white')
    ax1.set_ylabel('Normalized Flux', color='white')
    ax1.set_title(f'TIC {tic_id} — Full Light Curve', color='white',
                  fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')

    # Bottom: phase-folded with binned data
    ax2.set_facecolor('#16213e')
    win = duration * 24 * 4  # 4x transit duration window
    mask = np.abs(phase_hours) < win
    ax2.scatter(phase_hours[mask], flux[mask], s=0.3, c='#a8dadc',
                alpha=0.2, rasterized=True)

    # Bin the data
    n_bins = 200
    bins = np.linspace(-win, win, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned = []
    for j in range(n_bins):
        m = (phase_hours >= bins[j]) & (phase_hours < bins[j + 1])
        binned.append(np.nanmedian(flux[m]) if np.sum(m) > 0 else np.nan)
    binned = np.array(binned)
    ax2.plot(bin_centers, binned, 'o-', color='#e63946', markersize=3,
             linewidth=1.5, zorder=10)

    # Transit depth indicator
    ax2.axhline(1.0, color='#444', linestyle='--', linewidth=0.5)
    ax2.axhline(1.0 - depth / 1e6, color='#f4a261', linestyle='--',
                linewidth=0.5, alpha=0.7)

    ax2.set_xlabel('Hours from Transit Center', color='white')
    ax2.set_ylabel('Normalized Flux', color='white')
    ax2.set_xlim(-win, win)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')

    # Info box
    info_lines = [
        f'P = {period:.6f} d',
        f'Depth = {depth:.1f} ppm',
        f'Duration = {duration * 24:.2f} h',
        f'SNR = {snr:.1f}',
    ]
    if cnn is not None:
        info_lines.append(f'CNN = {cnn:.3f}')
    if fpp is not None and fpp >= 0:
        info_lines.append(f'FPP = {fpp:.4f}')
    info_text = '\n'.join(info_lines)
    ax2.text(0.02, 0.02, info_text, transform=ax2.transAxes,
             fontsize=9, color='white', fontfamily='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))

    ax2.set_title(f'Phase-Folded Transit — Candidate {candidate.get("candidate_num", 1)}',
                  color='white', fontsize=12, fontweight='bold')

    # Footer
    fig.text(0.99, 0.01, f'ExoNit — {datetime.now().strftime("%Y-%m-%d")}',
             ha='right', fontsize=8, color='#666')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return output_path
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        buf.seek(0)
        return buf.read()


def load_light_curve(tic_id, metadata_cache_path):
    """Load light curve from FITS files via metadata cache.

    Returns (time, flux) arrays or (None, None) on failure.
    """
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import lightkurve as lk
        from exonit.metadata_cache import MetadataCache

        cache = MetadataCache(metadata_cache_path)
        entry = cache.get_entry(str(tic_id))
        if not entry:
            cache.close()
            return None, None

        paths = []
        for f in entry['fits_files'][:6]:
            p = cache.get_fits_path(f)
            if p and os.path.exists(p):
                paths.append(p)
        cache.close()

        if not paths:
            return None, None

        lcs = [lk.read(p) for p in paths]
        lc = lcs[0] if len(lcs) == 1 else lk.LightCurveCollection(lcs).stitch()
        lc = lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=float('inf'))
        lc_flat = lc.flatten(window_length=1001)

        time_arr = np.array(lc_flat.time.value, dtype=np.float64)
        flux_arr = np.array(lc_flat.flux.value, dtype=np.float64)
        good = np.isfinite(time_arr) & np.isfinite(flux_arr)
        return time_arr[good], flux_arr[good]

    except Exception as e:
        print(f"  ctoi: failed to load LC for TIC {tic_id}: {e}")
        return None, None


_toi_cache = {}
_toi_cache_mtime = 0


def _load_toi_catalog(catalog_path):
    """Load TOI catalog into memory, cached by file modification time."""
    global _toi_cache, _toi_cache_mtime
    if not os.path.exists(catalog_path):
        return {}
    mtime = os.path.getmtime(catalog_path)
    if mtime == _toi_cache_mtime and _toi_cache:
        return _toi_cache

    catalog = {}
    try:
        with open(catalog_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tic = row.get('TIC ID', '').strip()
                if not tic:
                    continue
                if tic not in catalog:
                    catalog[tic] = []
                toi = row.get('TOI', '').strip()
                disposition = row.get('TFOPWG Disposition', '').strip()
                known_period = row.get('Period (days)', '').strip()
                kp = None
                if known_period:
                    try:
                        kp = float(known_period)
                    except ValueError:
                        pass
                catalog[tic].append({
                    'toi': toi,
                    'disposition': disposition,
                    'known_period': kp,
                })
        _toi_cache = catalog
        _toi_cache_mtime = mtime
    except Exception as e:
        print(f"  ctoi: failed to load TOI catalog: {e}")
    return catalog


def crosscheck_toi(tic_id, period, catalog_path):
    """Check if a candidate matches a known TOI.

    Returns list of match dicts or None if no match.
    Matches by TIC ID, then checks if periods are compatible
    (same period or integer multiple within 1%).
    """
    catalog = _load_toi_catalog(catalog_path)
    entries = catalog.get(str(tic_id))
    if not entries:
        return None

    matches = []
    for entry in entries:
        match = {
            'toi': entry['toi'],
            'disposition': entry['disposition'],
            'known_period': entry['known_period'],
            'period_match': False,
            'is_known_planet': entry['disposition'] in ('KP', 'CP'),
        }
        kp = entry['known_period']
        if kp and kp > 0 and period > 0:
            ratio = period / kp
            for mult in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
                if abs(ratio - mult) < 0.01:
                    match['period_match'] = True
                    match['period_ratio'] = round(ratio, 3)
                    break
        matches.append(match)

    return matches if matches else None


def export_exofop_csv(backend, catalog_path, min_snr=10.0, min_cnn=0.7,
                      max_fpp=0.5, exclude_known=True):
    """Export candidates as ExoFOP-compatible CSV.

    Filters candidates by quality thresholds and excludes known TOIs.

    ``backend`` is a db_backend.Backend (SQLite or PG); the query runs through
    it so this works on both engines (the old direct ``sqlite3.connect`` broke
    under PostgreSQL, where the configured db path is a DSN, not a file).

    Returns CSV string ready for download.
    """
    candidates = backend.query("""
        SELECT c.tic_id, c.candidate_num, c.period_days, c.t0,
               c.duration_days, c.depth_ppm, c.snr, c.cnn_score,
               c.triceratops_fpp, c.claude_verdict,
               s.ra, s.dec, s.tmag, s.n_sectors
        FROM candidates c
        JOIN stars s ON c.tic_id = s.tic_id
        WHERE c.snr >= ? AND c.cnn_score >= ?
        AND s.status = 'completed'
        ORDER BY c.snr DESC
    """, (min_snr, min_cnn))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'TIC ID', 'Candidate', 'Period (days)', 'Epoch (BJD)',
        'Duration (hours)', 'Depth (ppm)', 'SNR', 'CNN Score',
        'FPP', 'Claude Verdict', 'RA', 'Dec', 'TESS Mag',
        'N Sectors', 'TOI Match', 'Disposition', 'New cTOI'
    ])

    n_new = 0
    for c in candidates:
        # FPP filter
        fpp = c['triceratops_fpp']
        if fpp is not None and fpp >= 0 and fpp > max_fpp:
            continue
        if fpp is None:
            fpp = -1.0

        # Cross-check TOI
        toi_matches = crosscheck_toi(c['tic_id'], c['period_days'], catalog_path)
        toi_str = ''
        disp_str = ''
        is_new = True

        if toi_matches:
            for m in toi_matches:
                if m['period_match']:
                    toi_str = m['toi']
                    disp_str = m['disposition']
                    if m['is_known_planet']:
                        is_new = False
                    break
            if not toi_str:
                toi_str = toi_matches[0]['toi']
                disp_str = toi_matches[0]['disposition']

        if exclude_known and not is_new:
            continue

        n_new += 1
        writer.writerow([
            c['tic_id'], c['candidate_num'],
            f"{c['period_days']:.6f}",
            f"{c['t0']:.6f}",
            f"{c['duration_days'] * 24:.4f}" if c['duration_days'] else '',
            f"{c['depth_ppm']:.1f}",
            f"{c['snr']:.1f}",
            f"{c['cnn_score']:.4f}" if c['cnn_score'] else '',
            f"{fpp:.4f}" if fpp and fpp >= 0 else '',
            c['claude_verdict'] or '',
            f"{c['ra']:.6f}" if c['ra'] else '',
            f"{c['dec']:.6f}" if c['dec'] else '',
            f"{c['tmag']:.3f}" if c['tmag'] else '',
            c['n_sectors'] or '',
            toi_str, disp_str,
            'YES' if is_new else 'NO'
        ])

    return output.getvalue(), n_new
