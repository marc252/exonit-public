"""Native port of ``src_preprocessing/lc_preprocessing/preprocess.py``.

Scope: pure-compute orchestration for a single TCE — phase-folding the
detrended timeseries, binning into global/local views, odd/even splits,
weak-secondary, centroid offset, momentum-dump, plus the derived stat
scalars.  In the container the pipeline's final step serializes all
these arrays into a ``tf.train.Example`` and writes a TFRecord shard.
The native worker path skips TFRecords entirely:

  * :func:`preprocess_tce` replaces the container's ``process_single_tce``
    + ``generate_example_for_tce`` chain.  It takes the already phase-
    folded dict (as produced by the detrend + phase-fold sibling
    modules) along with the per-TCE config and returns a flat ``dict``
    mapping ExoMiner feature names to ``np.ndarray`` (or scalar floats
    for the ``*_stat_*`` fields).  No ``tf.train.Example``, no file I/O,
    no plotting.
  * The intermediate helpers (``generate_flux_binned_views`` etc.) are
    kept verbatim except for:
      - ``plot_preprocessing_tce`` branches stripped (worker never plots).
      - ``utils_visualization`` import removed.
      - ``report_exclusion`` from ``utils_preprocessing_io`` replaced by a
        local :func:`_report_exclusion` log-only stub (same convention as
        :mod:`.diff_img` / :mod:`.phase_fold`).
      - Foreign imports rewritten to the sibling modules under
        :mod:`worker.exominer_native.preprocess`.

Trimmed from the container source (not ported):
  - ``sample_ephemerides`` — training-time augmentation only.
  - ``process_target_tces`` — reads FITS and iterates multiple TCEs;
    orchestration is done by the worker's own TIC pipeline.
  - ``process_light_curve`` / ``flux_preprocessing`` /
    ``centroid_preprocessing`` — rely on FITS I/O and on lightkurve
    (for the SG detrend branch), not on the native critical path.
  - ``process_single_tce`` — calls the periodogram pipeline and the TF
    example writer; replaced by :func:`preprocess_tce`.
  - ``generate_example_for_tce`` — TFRecord writer.
  - All ``utils_visualization`` plot branches.
  - ``__main__`` / multiprocessing wrappers.

Patches vs oracle imports:
  - ``src_preprocessing.lc_preprocessing.utils_ephemeris`` → ``.ephemeris``
  - ``src_preprocessing.lc_preprocessing.utils_imputing`` → ``.imputing``
  - ``src_preprocessing.lc_preprocessing.utils_odd_even`` → ``.odd_even``
  - ``src_preprocessing.lc_preprocessing.utils_preprocessing`` → ``.utils_common``
  - ``src_preprocessing.lc_preprocessing.utils_preprocessing_io.report_exclusion``
    → local :func:`_report_exclusion` stub.
  - ``src_preprocessing.lc_preprocessing.phase_fold_and_binning`` → ``.phase_fold``
  - ``src_preprocessing.lc_preprocessing.utils_centroid_preprocessing``
    → ``.centroid`` (only used by the dropped ``centroid_preprocessing``).
  - ``src_preprocessing.tf_util.example_util`` — removed (no TFRecords).
  - ``src_preprocessing.lc_preprocessing.utils_visualization`` — removed.
  - ``src_preprocessing.lc_preprocessing.lc_periodogram`` — removed
    (periodogram features are added by the caller if needed).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from astropy.stats import mad_std

from . import ephemeris as _ephem
from . import imputing as _imputing
from . import odd_even as _odd_even
from . import phase_fold as _phase_fold
from . import utils_common as _utils


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (copied verbatim from the oracle module)
# ---------------------------------------------------------------------------

NORM_SIGMA_EPS = 1e-12

# Mapping from internal view names to the ExoMiner feature names that the
# downstream normalizer (C.14) / ONNX model consumes.  Copied verbatim from
# the oracle so any external caller that relies on these keys stays in sync.
MAP_VIEWS_TO_OLD_NAMES = {

    'flux_global': 'global_flux_view',
    'flux_local': 'local_flux_view',
    'flux_global_norm': 'global_flux_view_fluxnorm',
    'flux_local_norm': 'local_flux_view_fluxnorm',

    'flux_global_unfolded': 'unfolded_global_flux_view',
    'flux_global_unfolded_norm': 'unfolded_global_flux_view_fluxnorm',
    'flux_local_unfolded': 'unfolded_local_flux_view',
    'flux_local_unfolded_norm': 'unfolded_local_flux_view_fluxnorm',

    'flux_odd_local': 'local_flux_odd_view',
    'flux_even_local': 'local_flux_even_view',
    'flux_odd_local_norm': 'local_flux_odd_view_fluxnorm',
    'flux_even_local_norm': 'local_flux_even_view_fluxnorm',

    'flux_weak_secondary_local': 'local_weak_secondary_view',
    'flux_weak_secondary_local_norm': 'local_weak_secondary_view_selfnorm',

    'centroid_offset_distance_to_target_global': 'global_centr_view',
    'centroid_offset_distance_to_target_local': 'local_centr_view',

    'flux_trend_global': 'flux_trend_global',
    'flux_trend_local': 'flux_trend_local',
    'flux_trend_global_norm': 'flux_trend_global_norm',
    'flux_trend_local_norm': 'flux_trend_local_norm',

    'flux_trend_global_unfolded': 'flux_trend_global_unfolded',
    'flux_trend_global_unfolded_norm': 'flux_trend_global_unfolded_norm',
    'flux_trend_local_unfolded': 'flux_trend_local_unfolded',
    'flux_trend_local_unfolded_norm': 'flux_trend_local_unfolded_norm',

    'momentum_dump_local': 'local_momentum_dump_view',
}


# ---------------------------------------------------------------------------
# Helpers that the port-side entry points need
# ---------------------------------------------------------------------------


def _report_exclusion(msg: str, *_args, **_kwargs) -> None:
    """Worker-local stand-in for the oracle's ``report_exclusion``.

    The oracle version appends to a per-target exclusion file; the worker
    just logs a warning.  ``*args, **kwargs`` accept (and discard) the
    extra positional ``output_file_path`` argument for signature parity.
    """
    _log.warning(str(msg))


# Sibling-imported names used by the kept helpers.
create_binary_time_series = _ephem.create_binary_time_series
find_first_epoch_after_this_time = _ephem.find_first_epoch_after_this_time
imputing_gaps = _imputing.imputing_gaps
create_odd_even_views = _odd_even.create_odd_even_views
phase_fold_and_sort_light_curve_odd_even = _odd_even.phase_fold_and_sort_light_curve_odd_even
phase_fold_and_sort_light_curve = _phase_fold.phase_fold_and_sort_light_curve
phase_split_light_curve = _phase_fold.phase_split_light_curve
global_view = _phase_fold.global_view
local_view = _phase_fold.local_view
centering_and_normalization = _phase_fold.centering_and_normalization
generate_view_momentum_dump = _phase_fold.generate_view_momentum_dump
check_inputs_generate_example = _utils.check_inputs_generate_example
remove_outliers = _utils.remove_outliers


# ---------------------------------------------------------------------------
# Pure-compute helpers ported verbatim from the oracle (plot/visualization
# branches removed, ``report_exclusion`` → local stub).
# ---------------------------------------------------------------------------


def find_intransit_cadences(tces_in_target, time_arrs, transit_duration_factor):
    """Find in-transit cadences (primary + secondary) for all TCEs in a target.

    Ported verbatim from
    ``src_preprocessing/lc_preprocessing/preprocess.py::find_intransit_cadences``.
    """
    n_tces_in_target = len(tces_in_target)

    target_intransit_cadences_arr = [np.zeros((n_tces_in_target, len(time_arr)), dtype='bool')
                                     for time_arr in time_arrs]
    for tce_in_target_i, tce_in_target in tces_in_target.iterrows():

        duration_primary, duration_secondary = set_duration_gap_for_tce(
            tce_in_target['tce_duration'],
            tce_in_target['tce_period'],
            transit_duration_factor,
            tce_in_target['tce_maxmesd'],
        )

        first_transit_time = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'],
                                                               tce_in_target['tce_period'],
                                                               time[0])
                              for time in time_arrs]

        tce_intransit_cadences_arr = [create_binary_time_series(time,
                                                                first_transit_time,
                                                                duration_primary,
                                                                tce_in_target['tce_period'])
                                      for first_transit_time, time in zip(first_transit_time, time_arrs)]

        if 'tce_maxmesd' in tce_in_target and duration_secondary is not None:
            first_transit_time_secondary = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'] +
                                                                             tce_in_target['tce_maxmesd'],
                                                                             tce_in_target['tce_period'],
                                                                             time[0])
                                            for time in time_arrs]

            tce_intransit_cadences_arr_secondary = [create_binary_time_series(time,
                                                                              first_transit_time,
                                                                              duration_secondary,
                                                                              tce_in_target['tce_period'])
                                                    for first_transit_time, time in
                                                    zip(first_transit_time_secondary, time_arrs)]

            tce_intransit_cadences_arr = [np.logical_or(primary_arr, secondary_arr)
                                          for primary_arr, secondary_arr
                                          in zip(tce_intransit_cadences_arr, tce_intransit_cadences_arr_secondary)]

        for arr_i, intransit_cadences_arr in enumerate(tce_intransit_cadences_arr):
            target_intransit_cadences_arr[arr_i][tce_in_target_i] = intransit_cadences_arr

    return target_intransit_cadences_arr


def gap_intransit_cadences_other_tces(target_intransit_cadences_arr, idx_tce, data,
                                      gap_keep_overlap=True, impute_gaps=False,
                                      seed=None, tce=None, config=None):
    """Gap (or impute) in-transit cadences belonging to other TCEs.

    Ported verbatim from
    ``src_preprocessing/lc_preprocessing/preprocess.py::gap_intransit_cadences_other_tces``.
    """
    other_tces_idxs = np.ones(len(target_intransit_cadences_arr[0]), dtype='bool')
    other_tces_idxs[idx_tce] = False

    it_cadences_other_tces = [np.sum(target_intransit_cadences[other_tces_idxs, :], axis=0) != 0
                              for target_intransit_cadences in target_intransit_cadences_arr]
    if gap_keep_overlap:
        for arr_i, it_cadences_other_tces_arr in enumerate(it_cadences_other_tces):
            it_cadences_other_tces_arr[target_intransit_cadences_arr[arr_i][idx_tce]] = False

    for arr_i, it_cadences_other_tces_arr in enumerate(it_cadences_other_tces):
        for timeseries in data:
            if impute_gaps:
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i] = (
                        imputing_gaps(data[timeseries]['x'][arr_i], it_cadences_other_tces_arr,
                                      seed=seed, tce=tce, config=config))
                    data[timeseries]['y'][arr_i] = imputing_gaps(data[timeseries]['y'][arr_i],
                                                                 it_cadences_other_tces_arr,
                                                                 seed=seed, tce=tce, config=config)
                else:
                    data[timeseries][arr_i] = (
                        imputing_gaps(data[timeseries][arr_i], it_cadences_other_tces_arr,
                                      seed=seed, tce=tce, config=config))
            else:
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i][it_cadences_other_tces_arr] = np.nan
                    data[timeseries]['y'][arr_i][it_cadences_other_tces_arr] = np.nan
                else:
                    data[timeseries][arr_i][it_cadences_other_tces_arr] = np.nan

    return data


def set_duration_gap_for_tce(tce_duration, tce_period, transit_duration_factor, tce_secondary_offset=None):
    """Set primary (and secondary, if applicable) transit gap durations.

    Ported verbatim.
    """
    if tce_secondary_offset is not None:
        duration_gapped_primary = max(
            min(transit_duration_factor * tce_duration,
                2 * np.abs(tce_secondary_offset) - tce_duration,
                tce_period),
            tce_duration,
        )
        duration_gapped_secondary = duration_gapped_primary
    else:
        duration_gapped_primary = min(transit_duration_factor * tce_duration, tce_period)
        duration_gapped_secondary = None

    return duration_gapped_primary, duration_gapped_secondary


def phase_fold_timeseries(data, config, tce, plot_preprocessing_tce=False):
    """Phase-fold the detrended timeseries for a TCE.

    Ported from the oracle; ``plot_preprocessing_tce`` is accepted for
    signature parity but ignored (worker never plots).
    """
    del plot_preprocessing_tce  # unused — worker never plots

    data = check_inputs_generate_example(data, tce, config)

    for key, val in data.items():
        if val is None:
            raise KeyError(f'No detrended data for {key} time series before creating the phase folded time series.')

    phasefolded_timeseries = {field: 3 * (None,)
                              for field in ['flux', 'flux_unfolded', 'flux_trend', 'flux_trend_unfolded', 'flux_odd',
                                            'flux_even', 'flux_weak_secondary', 'centroid_offset_distance_to_target',
                                            'momentum_dump']}

    time, flux, n_transits_flux = phase_fold_and_sort_light_curve(data['flux_time'],
                                                                  data['flux'],
                                                                  tce['tce_period'],
                                                                  tce['tce_time0bk'],
                                                                  augmentation=False)
    phasefolded_timeseries['flux'] = (time, flux, n_transits_flux)

    time_trend, flux_trend, n_transits_flux_trend = phase_fold_and_sort_light_curve(data['flux_time'],
                                                                                    data['flux_trend'],
                                                                                    tce['tce_period'],
                                                                                    tce['tce_time0bk'],
                                                                                    augmentation=False)
    phasefolded_timeseries['flux_trend'] = (time_trend, flux_trend, n_transits_flux_trend)

    time_wksecondary, flux_wksecondary, n_transits_wksecondary = (
        phase_fold_and_sort_light_curve(data['flux_time'],
                                        data['flux'],
                                        tce['tce_period'],
                                        tce['tce_time0bk'] + tce['tce_maxmesd'],
                                        augmentation=False))
    phasefolded_timeseries['flux_weak_secondary'] = (time_wksecondary, flux_wksecondary, n_transits_wksecondary)

    (odd_time, odd_flux, n_transits_odd, odd_time_nophased), \
        (even_time, even_flux, n_transits_even, even_time_nophased) = (
        phase_fold_and_sort_light_curve_odd_even(data['flux_time'],
                                                 data['flux'],
                                                 tce['tce_period'],
                                                 tce['tce_time0bk'],
                                                 augmentation=False))
    phasefolded_timeseries['flux_odd'] = (odd_time, odd_flux, n_transits_odd)
    phasefolded_timeseries['flux_even'] = (even_time, even_flux, n_transits_even)

    time_centroid_dist, centroid_dist, n_transits_centroid = (
        phase_fold_and_sort_light_curve(data['centroid_dist_time'],
                                        data['centroid_dist'],
                                        tce['tce_period'],
                                        tce['tce_time0bk'],
                                        augmentation=False))
    phasefolded_timeseries['centroid_offset_distance_to_target'] = \
        (time_centroid_dist, centroid_dist, n_transits_centroid)

    if config['get_momentum_dump']:
        time_momentum_dump, momentum_dump, n_transits_momentum_dump = (
            phase_fold_and_sort_light_curve(data['momentum_dump_time'],
                                            data['momentum_dump'],
                                            tce['tce_period'],
                                            tce['tce_time0bk'],
                                            augmentation=False))
        phasefolded_timeseries['momentum_dump'] = (time_momentum_dump, momentum_dump, n_transits_momentum_dump)

    time_split, flux_split, n_phases_split, _ = phase_split_light_curve(
        data['flux_time'],
        data['flux'],
        tce['tce_period'],
        tce['tce_time0bk'],
        tce['tce_duration'],
        config['n_max_phases'],
        config['keep_odd_even_order'],
        config['frac_it_cadences_thr'],
        config['sampling_rate_h'],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config['quarter_sampling']
        else None,
        remove_outliers_params={'sigma': config['outlier_removal_sigma'],
                                'fill': config['outlier_removal_fill'],
                                'outlier_type': 'upper',
                                } if config['outlier_removal'] else None,
    )
    phasefolded_timeseries['flux_unfolded'] = (time_split, flux_split, n_phases_split)

    if n_phases_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_split} phase(s) for flux, need at least {config["min_n_phases"]} to '
                         f'create example.')

    if n_phases_split == 0:
        phasefolded_timeseries['flux_unfolded'] = (
            np.tile(phasefolded_timeseries['flux'][0], (config['n_max_phases'], 1)),
            np.tile(phasefolded_timeseries['flux'][1], (config['n_max_phases'], 1)),
            config['n_max_phases'],
        )

    time_trend_split, flux_trend_split, n_phases_trend_split, _ = phase_split_light_curve(
        data['flux_time'],
        data['flux_trend'],
        tce['tce_period'],
        tce['tce_time0bk'],
        tce['tce_duration'],
        config['n_max_phases'],
        config['keep_odd_even_order'],
        0,
        config['sampling_rate_h'],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config['quarter_sampling']
        else None,
        remove_outliers_params={'sigma': config['outlier_removal_sigma'],
                                'fill': config['outlier_removal_fill'],
                                'outlier_type': 'upper',
                                } if config['outlier_removal'] else None,
    )
    phasefolded_timeseries['flux_trend_unfolded'] = (time_trend_split, flux_trend_split, n_phases_trend_split)

    if n_phases_trend_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_trend_split} phase(s) for flux trend, need at least '
                         f'{config["min_n_phases"]} to create example.')

    if config['outlier_removal']:
        for ts_name, (ts_time, ts_values, n_transits_ts) in phasefolded_timeseries.items():
            if ts_name in ['momentum_dump', 'centroid_offset_distance_to_target']:
                continue
            if 'unfolded' in ts_name:
                continue
            if 'trend' in ts_name:
                ts_values, _ = remove_outliers(
                    ts_values,
                    sigma=config['outlier_removal_sigma'],
                    fill=config['outlier_removal_fill'],
                    outlier_type='both',
                )
            else:
                ts_values, _ = remove_outliers(
                    ts_values,
                    sigma=config['outlier_removal_sigma'],
                    fill=config['outlier_removal_fill'],
                    outlier_type='upper',
                )

    return phasefolded_timeseries


def generate_flux_binned_views(data, tce, config, plot_preprocessing_tce=False):
    """Bin the phase-folded flux timeseries into global + local views.

    Ported verbatim; plot branch stripped.
    """
    del plot_preprocessing_tce

    glob_flux_view, glob_binned_time, glob_flux_view_var, _, bin_counts = \
        global_view(data['flux'][0],
                    data['flux'][1],
                    tce['tce_period'],
                    tce=tce,
                    normalize=False,
                    centering=False,
                    num_bins=config['num_bins_glob'],
                    bin_width_factor=config['bin_width_factor_glob'],
                    report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                    tce_duration=tce['tce_duration'])
    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    glob_flux_view_var /= np.sqrt(bin_counts)

    unfolded_glob_flux_view = []
    unfolded_glob_flux_view_var = []
    for phase_i in range(data['flux_unfolded'][2]):
        unfolded_glob_flux_view_phase, binned_time, unfolded_glob_flux_view_var_phase, _, bin_counts = \
            global_view(data['flux_unfolded'][0][phase_i],
                        data['flux_unfolded'][1][phase_i],
                        tce['tce_period'],
                        tce=tce,
                        normalize=False,
                        centering=False,
                        num_bins=config['num_bins_glob'],
                        bin_width_factor=config['bin_width_factor_glob'],
                        report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                        tce_duration=tce['tce_duration'])
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_glob_flux_view_var_phase /= np.sqrt(bin_counts)
        unfolded_glob_flux_view.append(unfolded_glob_flux_view_phase)
        unfolded_glob_flux_view_var.append(unfolded_glob_flux_view_var_phase)
    unfolded_glob_flux_view = np.array(unfolded_glob_flux_view)
    unfolded_glob_flux_view_var = np.array(unfolded_glob_flux_view_var)

    loc_flux_view, loc_binned_time, loc_flux_view_var, _, bin_counts = \
        local_view(data['flux'][0],
                   data['flux'][1],
                   tce['tce_period'],
                   tce['tce_duration'],
                   tce=tce,
                   normalize=False,
                   centering=False,
                   num_durations=config['num_durations'],
                   num_bins=config['num_bins_loc'],
                   bin_width_factor=config['bin_width_factor_loc'],
                   report={'config': config, 'tce': tce, 'view': 'local_flux_view'})

    if np.all(np.isnan(loc_flux_view)):
        _report_exclusion(
            f'No data available for local flux view. Setting it to median and var view to mad std using '
            f'statistics from the flux phase folded time series.')
        mu, sigma = np.nanmedian(data['flux'][1]), mad_std(data['flux'][1], ignore_nan=True)
        if np.isfinite(mu) and np.isfinite(sigma):
            loc_flux_view = mu * np.ones(config['num_bins_loc'])
            loc_flux_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_flux_view = np.zeros(config['num_bins_loc'])
            loc_flux_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    unfolded_loc_flux_view = []
    unfolded_loc_flux_view_var = []
    for phase_i in range(data['flux_unfolded'][2]):
        unfolded_loc_flux_view_phase, binned_time, unfolded_loc_flux_view_var_phase, _, bin_counts = \
            local_view(data['flux_unfolded'][0][phase_i],
                       data['flux_unfolded'][1][phase_i],
                       tce['tce_period'],
                       tce['tce_duration'],
                       tce=tce,
                       normalize=False,
                       centering=False,
                       num_durations=config['num_durations'],
                       num_bins=config['num_bins_loc'],
                       bin_width_factor=config['bin_width_factor_loc'],
                       report={'config': config, 'tce': tce, 'view': 'local_flux_view'})
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_loc_flux_view_var_phase /= np.sqrt(bin_counts)
        unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
        unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)
    unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
    unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

    binned_timeseries = {
        'flux_global': (glob_binned_time, glob_flux_view, glob_flux_view_var, data['flux'][2]),
        'flux_local': (loc_binned_time, loc_flux_view, loc_flux_view_var, data['flux'][2]),
        'flux_global_unfolded': (None, unfolded_glob_flux_view, unfolded_glob_flux_view_var,
                                 data['flux_unfolded'][2]),
        'flux_local_unfolded': (None, unfolded_loc_flux_view, unfolded_loc_flux_view_var,
                                data['flux_unfolded'][2]),
    }

    norm_flux_global_stats = {'mu': np.median(glob_flux_view)}
    norm_flux_global_stats['sigma'] = np.abs(np.min(glob_flux_view - norm_flux_global_stats['mu'])) + NORM_SIGMA_EPS

    norm_flux_local_stats = {'mu': np.median(loc_flux_view)}
    norm_flux_local_stats['sigma'] = np.abs(np.min(loc_flux_view - norm_flux_local_stats['mu'])) + NORM_SIGMA_EPS

    binned_timeseries_norm = {f'{ts_name}_norm': (binned_time,
                                                  (binned_values - norm_flux_global_stats['mu']) /
                                                  norm_flux_global_stats['sigma'],
                                                  binned_values_var / norm_flux_global_stats['sigma'],
                                                  n_transits)
                              for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
                              binned_timeseries.items() if 'global' in ts_name}
    binned_timeseries_norm.update(
        {f'{ts_name}_norm': (binned_time,
                             (binned_values - norm_flux_local_stats['mu']) /
                             norm_flux_local_stats['sigma'],
                             binned_values_var / norm_flux_local_stats['sigma'],
                             n_transits)
         for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
         binned_timeseries.items() if 'local' in ts_name}
    )

    binned_timeseries.update(binned_timeseries_norm)

    binned_timeseries.update(
        {
            'flux_global_stat_abs_min': norm_flux_global_stats['sigma'],
            'flux_local_stat_abs_min': norm_flux_local_stats['sigma'],
        }
    )

    return binned_timeseries, norm_flux_global_stats, norm_flux_local_stats


def generate_flux_trend_binned_views(data, tce, config, plot_preprocessing_tce=False):
    """Bin the phase-folded flux trend into global + local views.

    Ported verbatim; plot branch stripped.
    """
    del plot_preprocessing_tce

    glob_flux_view, glob_binned_time, glob_flux_view_var, _, bin_counts = \
        global_view(data['flux_trend'][0],
                    data['flux_trend'][1],
                    tce['tce_period'],
                    tce=tce,
                    normalize=False,
                    centering=False,
                    num_bins=config['num_bins_glob'],
                    bin_width_factor=config['bin_width_factor_glob'],
                    report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                    tce_duration=tce['tce_duration'])
    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    glob_flux_view_var /= np.sqrt(bin_counts)

    unfolded_glob_flux_view = []
    unfolded_glob_flux_view_var = []
    for phase_i in range(data['flux_trend_unfolded'][2]):
        unfolded_glob_flux_view_phase, binned_time, unfolded_glob_flux_view_var_phase, _, bin_counts = \
            global_view(data['flux_trend_unfolded'][0][phase_i],
                        data['flux_trend_unfolded'][1][phase_i],
                        tce['tce_period'],
                        tce=tce,
                        normalize=False,
                        centering=False,
                        num_bins=config['num_bins_glob'],
                        bin_width_factor=config['bin_width_factor_glob'],
                        report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                        tce_duration=tce['tce_duration'])
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_glob_flux_view_var_phase /= np.sqrt(bin_counts)
        unfolded_glob_flux_view.append(unfolded_glob_flux_view_phase)
        unfolded_glob_flux_view_var.append(unfolded_glob_flux_view_var_phase)
    unfolded_glob_flux_view = np.array(unfolded_glob_flux_view)
    unfolded_glob_flux_view_var = np.array(unfolded_glob_flux_view_var)

    loc_flux_view, loc_binned_time, loc_flux_view_var, _, bin_counts = \
        local_view(data['flux_trend'][0],
                   data['flux_trend'][1],
                   tce['tce_period'],
                   tce['tce_duration'],
                   tce=tce,
                   normalize=False,
                   centering=False,
                   num_durations=config['num_durations'],
                   num_bins=config['num_bins_loc'],
                   bin_width_factor=config['bin_width_factor_loc'],
                   report={'config': config, 'tce': tce, 'view': 'local_flux_view'})

    if np.all(np.isnan(loc_flux_view)):
        _report_exclusion(
            f'No data available for local flux trend view. Setting it to median and var view to mad std using '
            f'statistics from the flux trend phase folded time series.')
        mu, sigma = np.nanmedian(data['flux_trend'][1]), mad_std(data['flux_trend'][1], ignore_nan=True)
        if np.isfinite(mu) and np.isfinite(sigma):
            loc_flux_view = mu * np.ones(config['num_bins_loc'])
            loc_flux_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_flux_view = np.zeros(config['num_bins_loc'])
            loc_flux_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    unfolded_loc_flux_view = []
    unfolded_loc_flux_view_var = []
    for phase_i in range(data['flux_trend_unfolded'][2]):
        unfolded_loc_flux_view_phase, binned_time, unfolded_loc_flux_view_var_phase, _, bin_counts = \
            local_view(data['flux_trend_unfolded'][0][phase_i],
                       data['flux_trend_unfolded'][1][phase_i],
                       tce['tce_period'],
                       tce['tce_duration'],
                       tce=tce,
                       normalize=False,
                       centering=False,
                       num_durations=config['num_durations'],
                       num_bins=config['num_bins_loc'],
                       bin_width_factor=config['bin_width_factor_loc'],
                       report={'config': config, 'tce': tce, 'view': 'local_flux_view'})

        if np.all(~np.isfinite(unfolded_loc_flux_view_phase)):
            unfolded_loc_flux_view_phase = np.median(glob_flux_view) * np.ones(config['num_bins_loc'])
            unfolded_loc_flux_view_var_phase = (
                mad_std(glob_flux_view, ignore_nan=False) * np.ones(config['num_bins_loc']))

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_loc_flux_view_var_phase /= np.sqrt(bin_counts)
        unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
        unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)
    unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
    unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

    norm_flux_max = np.max(glob_flux_view)
    norm_flux_min = np.min(glob_flux_view)
    delta_flux_max_min = norm_flux_max - norm_flux_min + NORM_SIGMA_EPS

    binned_timeseries = {
        'flux_trend_global': (glob_binned_time, glob_flux_view, glob_flux_view_var, data['flux'][2]),
        'flux_trend_local': (loc_binned_time, loc_flux_view, loc_flux_view_var, data['flux'][2]),
        'flux_trend_global_unfolded': (None, unfolded_glob_flux_view, unfolded_glob_flux_view_var,
                                       data['flux_unfolded'][2]),
        'flux_trend_local_unfolded': (None, unfolded_loc_flux_view, unfolded_loc_flux_view_var,
                                      data['flux_unfolded'][2]),
    }

    binned_timeseries_norm = {f'{ts_name}_norm': (binned_time,
                                                  (binned_values - norm_flux_min) / delta_flux_max_min,
                                                  binned_values_var / delta_flux_max_min,
                                                  n_transits)
                              for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
                              binned_timeseries.items()}

    binned_timeseries.update(binned_timeseries_norm)

    binned_timeseries.update(
        {
            'flux_trend_global_stat_max': norm_flux_max,
            'flux_trend_global_stat_min': norm_flux_min,
        }
    )

    return binned_timeseries


def generate_odd_even_binned_views(data, tce, config, norm_stats, plot_preprocessing_tce=False):
    """Generate odd-even binned views from phase-folded flux.

    Ported verbatim; plot branch stripped.
    """
    del plot_preprocessing_tce

    odd_data, even_data, odd_even_flag = create_odd_even_views(data['flux_odd'][0],
                                                               data['flux_odd'][1],
                                                               data['flux_even'][0],
                                                               data['flux_even'][1],
                                                               data['flux_odd'][2],
                                                               data['flux_even'][2],
                                                               tce,
                                                               config)

    odd_data['se_oot'] = odd_data['se_oot'] / norm_stats['sigma']
    odd_data['std_oot_bin'] = odd_data['std_oot_bin'] / norm_stats['sigma']
    even_data['se_oot'] = even_data['se_oot'] / norm_stats['sigma']
    even_data['std_oot_bin'] = even_data['std_oot_bin'] / norm_stats['sigma']

    odd_local_abs_min = np.abs(np.min(odd_data['local_flux_view'] - np.nanmedian(odd_data['local_flux_view'])))
    flux_odd_local_norm = (odd_data['local_flux_view'] - norm_stats['mu']) / norm_stats['sigma']
    flux_odd_local_var_norm = odd_data['local_flux_view_se'] / norm_stats['sigma']

    even_local_abs_min = np.abs(np.min(even_data['local_flux_view'] - np.nanmedian(even_data['local_flux_view'])))
    flux_even_local_norm = (even_data['local_flux_view'] - norm_stats['mu']) / norm_stats['sigma']
    flux_even_local_var_norm = even_data['local_flux_view_se'] / norm_stats['sigma']

    binned_timeseries = {
        'flux_odd_local': (odd_data['binned_time'], odd_data['local_flux_view'], odd_data['local_flux_view_se'],
                           data['flux_odd'][2]),
        'flux_odd_local_norm': (odd_data['binned_time'], flux_odd_local_norm, flux_odd_local_var_norm,
                                data['flux_odd'][2]),
        'flux_even_local': (even_data['binned_time'], even_data['local_flux_view'], even_data['local_flux_view_se'],
                            data['flux_even'][2]),
        'flux_even_local_norm': (even_data['binned_time'], flux_even_local_norm, flux_even_local_var_norm,
                                 data['flux_even'][2]),
    }

    binned_timeseries.update(
        {
            'flux_odd_local_stat_se_oot': odd_data['se_oot'],
            'flux_odd_local_stat_std_oot_bin': odd_data['std_oot_bin'],
            'flux_even_local_stat_se_oot': even_data['se_oot'],
            'flux_even_local_stat_std_oot_bin': even_data['std_oot_bin'],
            'flux_odd_local_stat_abs_min': odd_local_abs_min,
            'flux_even_local_stat_abs_min': even_local_abs_min,
        }
    )

    return binned_timeseries, odd_even_flag


def generate_weak_secondary_binned_views(data, tce, config, norm_stats=None, plot_preprocessing_tce=False):
    """Generate weak-secondary local binned view.

    Ported verbatim; plot branch stripped.
    """
    del plot_preprocessing_tce

    midpoint_primary_transit_time = -tce['tce_maxmesd']
    tr_dur_f_primary_gap = 1

    primary_gap_indices = np.logical_and(data['flux_weak_secondary'][0] > midpoint_primary_transit_time -
                                         tr_dur_f_primary_gap * tce['tce_duration'],
                                         data['flux_weak_secondary'][0] < midpoint_primary_transit_time +
                                         tr_dur_f_primary_gap * tce['tce_duration'])
    mu = np.nanmedian(data['flux_weak_secondary'][1])
    sigma = mad_std(data['flux_weak_secondary'][1], ignore_nan=True)
    rng = np.random.default_rng(seed=config['random_seed'])
    data['flux_weak_secondary'][1][primary_gap_indices] = rng.normal(mu, sigma, primary_gap_indices.sum())

    loc_weak_secondary_view, binned_time, loc_weak_secondary_view_var, _, bin_counts = \
        local_view(data['flux_weak_secondary'][0],
                   data['flux_weak_secondary'][1],
                   tce['tce_period'],
                   tce['tce_duration'],
                   tce=tce,
                   normalize=False,
                   centering=False,
                   num_durations=config['num_durations'],
                   num_bins=config['num_bins_loc'],
                   bin_width_factor=config['bin_width_factor_loc'],
                   report={'config': config, 'tce': tce, 'view': 'local_wks_view'})

    if np.all(np.isnan(loc_weak_secondary_view)):
        _report_exclusion(
            f'No data available for local weak secondary view. Setting it to median and var view to mad std using '
            f'statistics from the weak secondary phase folded time series.')
        mu, sigma = np.nanmedian(data['flux_weak_secondary'][1]), mad_std(data['flux_weak_secondary'][1],
                                                                          ignore_nan=True)
        if np.isfinite(mu) and np.isfinite(sigma):
            loc_weak_secondary_view = mu * np.ones(config['num_bins_loc'])
            loc_weak_secondary_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_weak_secondary_view = np.zeros(config['num_bins_loc'])
            loc_weak_secondary_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_weak_secondary_view_var /= np.sqrt(bin_counts)

    if norm_stats is None:
        norm_stats = {'mu': np.median(loc_weak_secondary_view)}
        norm_stats['sigma'] = np.abs(np.min(loc_weak_secondary_view - norm_stats['mu'])) + NORM_SIGMA_EPS

    loc_weak_secondary_view_norm = \
        centering_and_normalization(loc_weak_secondary_view,
                                    norm_stats['mu'],
                                    norm_stats['sigma'],
                                    report={'config': config, 'tce': tce, 'view': 'weak secondary flux'})
    loc_weak_secondary_view_var_norm = loc_weak_secondary_view_var / norm_stats['sigma']

    binned_timeseries = {
        'flux_weak_secondary_local': (binned_time,
                                      loc_weak_secondary_view,
                                      loc_weak_secondary_view_var,
                                      data['flux_weak_secondary'][2]),
        'flux_weak_secondary_local_norm': (binned_time,
                                           loc_weak_secondary_view_norm,
                                           loc_weak_secondary_view_var_norm,
                                           data['flux_weak_secondary'][2]),
        'flux_weak_secondary_local_stat_abs_min': norm_stats['sigma'],
    }

    return binned_timeseries


def generate_centroid_binned_views(data, tce, config):
    """Generate centroid-offset binned views.

    Ported verbatim.
    """
    glob_centr_view, glob_binned_time, glob_centr_view_var, _, bin_counts = \
        global_view(data['centroid_offset_distance_to_target'][0],
                    data['centroid_offset_distance_to_target'][1],
                    tce['tce_period'],
                    tce=tce,
                    centroid=True,
                    normalize=False,
                    centering=False,
                    num_bins=config['num_bins_glob'],
                    bin_width_factor=config['bin_width_factor_glob'],
                    report={'config': config, 'tce': tce,
                            'view': 'global_centr_view'},
                    tce_duration=tce['tce_duration'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    glob_centr_view_var /= np.sqrt(bin_counts)

    loc_centr_view, loc_binned_time, loc_centr_view_var, _, bin_counts = \
        local_view(data['centroid_offset_distance_to_target'][0],
                   data['centroid_offset_distance_to_target'][1],
                   tce['tce_period'],
                   tce['tce_duration'],
                   tce=tce,
                   centroid=True,
                   normalize=False,
                   centering=False,
                   num_durations=config['num_durations'],
                   num_bins=config['num_bins_loc'],
                   bin_width_factor=config['bin_width_factor_loc'],
                   report={'config': config, 'tce': tce,
                           'view': 'local_centr_view'})

    if np.isnan(loc_centr_view).all():
        _report_exclusion(
            f'No data available for local centroid view. Setting it to median and var view to mad std using '
            f'statistics from the phase folded time series.')
        t_min = max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations'])
        t_max = min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations'])
        loc_binned_time = np.linspace(t_min, t_max, config['num_bins_loc'], endpoint=True)
        mu = np.nanmedian(data['centroid_offset_distance_to_target'][1])
        sigma = mad_std(data['centroid_offset_distance_to_target'][1], ignore_nan=True)
        if np.isfinite(mu) and np.isfinite(sigma):
            loc_centr_view = mu * np.ones(config['num_bins_loc'], dtype='float')
            loc_centr_view_var = sigma * np.ones(config['num_bins_loc'], dtype='float')
        else:
            loc_centr_view = np.zeros(config['num_bins_loc'])
            loc_centr_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_centr_view_var /= np.sqrt(bin_counts)

    binned_timeseries = {
        'centroid_offset_distance_to_target_global': (glob_binned_time, glob_centr_view, glob_centr_view_var,
                                                      data['centroid_offset_distance_to_target'][2]),
        'centroid_offset_distance_to_target_local': (loc_binned_time, loc_centr_view, loc_centr_view_var,
                                                     data['centroid_offset_distance_to_target'][2]),
        'centroid_offset_distance_to_target_global_stat_abs_max': np.nanmax(glob_centr_view),
        'centroid_offset_distance_to_target_local_stat_abs_max': np.nanmax(loc_centr_view),
    }

    return binned_timeseries


def generate_momentum_dump_views(data, tce, config, plot_preprocessing_tce=False):
    """Generate momentum-dump binned local view.

    Ported verbatim; plot branch stripped.
    """
    del plot_preprocessing_tce

    loc_mom_dump_view, loc_mom_dump_view_var, binned_time = generate_view_momentum_dump(
        data['momentum_dump'][0],
        data['momentum_dump'][1],
        config['num_bins_loc'],
        config['bin_width_factor_loc'] * tce['tce_duration'],
        max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations']),
        min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations']),
    )

    binned_timeseries = {
        'momentum_dump_local': (binned_time, loc_mom_dump_view, loc_mom_dump_view_var, -1),
    }

    return binned_timeseries


def create_example_stats(binned_timeseries, odd_even_flag, t_min_transit, t_max_transit, config):
    """Compute summary scalars for a preprocessed TCE.

    Ported verbatim from the oracle.
    """
    example_stats = {}

    example_stats.update({f'num_transits_{ts_name}': ts_data[3] for ts_name, ts_data in binned_timeseries.items()
                          if 'stat' not in ts_name})
    example_stats.update({stat_name: stat_val for stat_name, stat_val in binned_timeseries.items()
                          if 'stat' in stat_name})

    glob_centr_oot_bins = (binned_timeseries['centroid_offset_distance_to_target_global'][0] < t_min_transit) | \
                          (binned_timeseries['centroid_offset_distance_to_target_global'][0] > t_max_transit)
    loc_centr_it_bins = (binned_timeseries['centroid_offset_distance_to_target_local'][0] >= t_min_transit) & \
                        (binned_timeseries['centroid_offset_distance_to_target_local'][0] <= t_max_transit)
    example_stats['avg_oot_centroid_offset'] = np.median(
        binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['std_oot_centroid_offset'] = mad_std(
        binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['peak_centroid_offset'] = np.max(
        binned_timeseries['centroid_offset_distance_to_target_local'][1][loc_centr_it_bins]) - \
        example_stats['avg_oot_centroid_offset']

    example_stats['mid_local_flux_shift'] = np.argmin(
        binned_timeseries['centroid_offset_distance_to_target_local'][1]) - int(config['num_bins_loc'] / 2)
    example_stats['mid_global_flux_shift'] = np.argmin(
        binned_timeseries['centroid_offset_distance_to_target_global'][1]) - int(config['num_bins_glob'] / 2)

    loc_flux_it_bins = (binned_timeseries['flux_local'][0] >= t_min_transit) & \
                       (binned_timeseries['flux_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_local'][1][loc_flux_it_bins])
    example_stats['transit_depth_hat'] = (1 - binned_timeseries['flux_local'][1][loc_flux_it_bins][ind]) * 1e6
    example_stats['transit_depth_hat_se'] = binned_timeseries['flux_local'][1][loc_flux_it_bins][ind] * 1e6

    example_stats['odd_even_flag'] = odd_even_flag

    loc_flux_odd_it_bins = (binned_timeseries['flux_odd_local'][0] >= t_min_transit) & \
                           (binned_timeseries['flux_odd_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_odd_local'][1][loc_flux_odd_it_bins])
    example_stats['transit_depth_odd_hat'] = (1 - binned_timeseries['flux_odd_local'][1][loc_flux_odd_it_bins][
        ind]) * 1e6
    example_stats['transit_depth_odd_hat_se'] = binned_timeseries['flux_odd_local'][2][loc_flux_odd_it_bins][ind] * 1e6

    loc_flux_even_it_bins = (binned_timeseries['flux_even_local'][0] >= t_min_transit) & \
                            (binned_timeseries['flux_even_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins])
    example_stats['transit_depth_even_hat'] = (1 - binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins][
        ind]) * 1e6
    example_stats['transit_depth_even_hat_se'] = binned_timeseries['flux_even_local'][2][loc_flux_even_it_bins][
        ind] * 1e6

    loc_flux_wks_it_bins = (binned_timeseries['flux_weak_secondary_local'][0] >= t_min_transit) & \
                           (binned_timeseries['flux_weak_secondary_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins])
    example_stats['wks_transit_depth_hat'] = (1 -
                                              binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins][
                                                  ind]) * 1e6
    example_stats['wks_transit_depth_hat_se'] = binned_timeseries['flux_weak_secondary_local'][2][loc_flux_wks_it_bins][
        ind] * 1e6

    return example_stats


# ---------------------------------------------------------------------------
# Top-level refactored entry point — dict return instead of tf.train.Example.
# ---------------------------------------------------------------------------


def preprocess_tce(phase_folded_data: dict, tce, config: dict) -> dict[str, Any]:
    """Preprocess the phase-folded timeseries for a single TCE.

    Worker-native replacement of the container's
    ``process_single_tce`` + ``generate_example_for_tce`` chain.

    Parameters
    ----------
    phase_folded_data:
        Output of :func:`phase_fold_timeseries` (or the equivalent
        produced by ``worker.exominer_native.preprocess.phase_fold``).
        Must contain per-timeseries ``(time, values, n_transits)`` tuples
        for ``flux``, ``flux_unfolded``, ``flux_trend``,
        ``flux_trend_unfolded``, ``flux_odd``, ``flux_even``,
        ``flux_weak_secondary``, ``centroid_offset_distance_to_target``,
        and optionally ``momentum_dump``.
    tce:
        Pandas Series (or mapping) holding the TCE ephemeris and scalar
        diagnostics.  Must provide at least ``tce_period``,
        ``tce_duration``, ``tce_maxmesd``, plus anything else the caller
        wants carried through to the returned ``tce_table`` block.
    config:
        Preprocessing parameters — ``num_bins_glob``, ``num_bins_loc``,
        ``num_durations``, ``bin_width_factor_glob``,
        ``bin_width_factor_loc``, ``get_momentum_dump``, ``random_seed``.

    Returns
    -------
    dict
        ``{
            "views":        {feature_name → np.ndarray},
            "views_var":    {feature_name → np.ndarray},
            "num_transits": {feature_name → int},
            "stats":        {stat_name → float},
            "example_stats": {stat_name → float},
            "odd_even_flag": str,
            "tce_table":    {name → scalar},   # raw TCE fields (str/num)
        }``

    The ``views`` / ``views_var`` / ``num_transits`` keys use the same
    ExoMiner feature names the ONNX model consumes (see
    :data:`MAP_VIEWS_TO_OLD_NAMES`), so the output can be fed directly
    to the downstream normalizer (C.14).  ``views_var[name]`` is paired
    with ``views[name]`` exactly like the oracle's ``{name}_var`` suffix
    convention.
    """
    t_min_transit, t_max_transit = (max(-tce['tce_period'] / 2, -tce['tce_duration'] / 2),
                                    min(tce['tce_period'] / 2, tce['tce_duration'] / 2))

    binned_timeseries: dict[str, Any] = {}

    binned_timeseries_flux, _, binned_flux_local_norm_stats = (
        generate_flux_binned_views(phase_folded_data, tce, config))
    binned_timeseries.update(binned_timeseries_flux)

    binned_timeseries_flux_trend = generate_flux_trend_binned_views(phase_folded_data, tce, config)
    binned_timeseries.update(binned_timeseries_flux_trend)

    binned_timeseries_odd_even_flux, odd_even_flag = generate_odd_even_binned_views(
        phase_folded_data, tce, config, binned_flux_local_norm_stats)
    binned_timeseries.update(binned_timeseries_odd_even_flux)

    binned_timeseries_wksecondary = generate_weak_secondary_binned_views(
        phase_folded_data, tce, config, norm_stats=None)
    binned_timeseries.update(binned_timeseries_wksecondary)

    binned_timeseries_centroid = generate_centroid_binned_views(phase_folded_data, tce, config)
    binned_timeseries.update(binned_timeseries_centroid)

    if config.get('get_momentum_dump'):
        binned_timeseries_momentum_dump = generate_momentum_dump_views(phase_folded_data, tce, config)
        binned_timeseries.update(binned_timeseries_momentum_dump)

    # ------------------------------------------------------------------
    # Validate + pack into the flat feature dict the model expects.
    # Mirrors the validation the oracle does while serializing to
    # ``tf.train.Example``.
    # ------------------------------------------------------------------
    views: dict[str, np.ndarray] = {}
    views_var: dict[str, np.ndarray] = {}
    num_transits: dict[str, int] = {}
    stats: dict[str, float] = {}

    for binned_ts_name, binned_ts_data in binned_timeseries.items():
        if 'stat' in binned_ts_name:
            if not np.isfinite(binned_ts_data):
                _log.warning(
                    "Statistic %s is non-finite (%s); replacing with 0 "
                    "(native-pipeline graceful-degradation path).",
                    binned_ts_name, binned_ts_data,
                )
                binned_ts_data = 0.0
            stats[binned_ts_name] = float(binned_ts_data)
            continue

        # binned_ts_data: (time, values, values_var, n_transits)
        _, values, values_var, n_transits_ts = binned_ts_data
        # Graceful degradation: the oracle ``preprocess_tce`` raises on any
        # non-finite bin — that kills the whole task in the worker. For the
        # native pipeline we prefer to emit a zero-filled view (which ONNX's
        # nan_to_num path would reach anyway) + log a warning, so the TIC
        # still gets a (low, consistent-with-"no planet") score instead of
        # a hard failure that drops it from the scorecard entirely.
        values = np.asarray(values)
        values_var = np.asarray(values_var)
        if np.any(~np.isfinite(values)):
            n_bad = int(np.sum(~np.isfinite(values)))
            _log.warning(
                "Binned timeseries %s has %d non-finite bin(s); filling with 0 "
                "(native-pipeline graceful-degradation path).",
                binned_ts_name, n_bad,
            )
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(values_var)):
            values_var = np.nan_to_num(values_var, nan=0.0, posinf=0.0, neginf=0.0)

        feature_name = MAP_VIEWS_TO_OLD_NAMES.get(binned_ts_name, binned_ts_name)
        views[feature_name] = np.asarray(values)
        views_var[f'{feature_name}_var'] = np.asarray(values_var)
        num_transits[f'{feature_name}_num_transits'] = int(n_transits_ts) if n_transits_ts is not None else -1

    example_stats = create_example_stats(binned_timeseries, odd_even_flag, t_min_transit, t_max_transit, config)

    tce_table: dict[str, Any] = {}
    try:
        iterator = tce.items()
    except AttributeError:
        iterator = dict(tce).items()
    for name, value in iterator:
        if name == 'Public Comment':
            continue
        tce_table[name] = value

    return {
        'views': views,
        'views_var': views_var,
        'num_transits': num_transits,
        'stats': stats,
        'example_stats': example_stats,
        'odd_even_flag': odd_even_flag,
        'tce_table': tce_table,
    }
