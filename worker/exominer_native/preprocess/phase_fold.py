"""Native port of ``src_preprocessing/lc_preprocessing/phase_fold_and_binning.py``.

**Critical module** — generates the global and local views that feed the
ExoMiner ONNX model. The output dicts/arrays from this module map directly
onto the model's input feature names, so preserving numerical parity with
the oracle is load-bearing.

Patches vs the oracle:
  - ``from src_preprocessing.light_curve import median_filter, util`` →
    ``from . import lc_median_filter as median_filter, lc_util as util``
  - ``from ..utils_imputing import impute_binned_ts`` →
    ``from .imputing import impute_binned_ts``
  - ``from ..utils_preprocessing_io import report_exclusion`` → local
    ``_report_exclusion`` stub (log-only)
  - ``from ..utils_preprocessing import remove_outliers`` →
    ``from .utils_common import remove_outliers``
"""
from __future__ import annotations

import logging

import numpy as np
from astropy.stats import mad_std

from . import lc_median_filter as median_filter
from . import lc_util as util
from .imputing import impute_binned_ts
from .utils_common import remove_outliers


NORM_SIGMA_EPS = 1e-12


_log = logging.getLogger(__name__)


def _report_exclusion(msg, output_file_path=None):
    """Worker-local stand-in for the oracle's ``report_exclusion``."""
    _log.warning(str(msg))


def phase_fold_and_sort_light_curve(time, timeseries, period, t0, augmentation=False):
    """Phase-fold a light curve and sort by ascending time.

    Returns ``(time, timeseries, num_transits)``.
    """
    if not augmentation:
        time = util.phase_fold_time(time, period, t0)
    else:
        time, sampled_idxs, num_transits = util.phase_fold_time_aug(time, period, t0)
        timeseries = timeseries[sampled_idxs]

    num_transits = np.sum(np.diff(time) < 0) + 1

    sorted_i = np.argsort(time)
    time = time[sorted_i]
    timeseries = timeseries[sorted_i]

    return time, timeseries, num_transits


def phase_split_light_curve(
    time,
    timeseries,
    period,
    t0,
    duration,
    n_max_phases,
    keep_odd_even_order,
    it_cadences_per_thr,
    num_cadences_per_h,
    extend_method,
    quarter_timestamps=None,
    seed=None,
    remove_outliers_params=None,
):
    """Split a 1D time series into phases; extract at most ``n_max_phases``."""
    duration_f = 1.5
    tmin_it, tmax_it = max(-period / 2, -duration_f * duration), min(period / 2, duration_f * duration)

    expected_num_it_cadences = int(2 * duration_f * duration * 24 * num_cadences_per_h)

    k_min, k_max = int(np.ceil((t0 - time[0]) / period)), int(np.ceil((time[-1] - t0) / period))
    epochs = [t0 - k * period for k in range(k_min)]
    epochs += [t0 + k * period for k in range(1, k_max)]

    n_phases = len(epochs)
    odd_even_id_arr = -1 * np.ones(n_phases, dtype="int")
    odd_even_id_arr[::2] = 0
    odd_even_id_arr[1::2] = 1

    phase = util.phase_fold_time(time, period, t0)

    diff_phase = np.diff(phase, prepend=np.nan)
    idx_splits = np.where(diff_phase < 0)[0]
    diff_time = np.diff(time, prepend=np.nan)
    idx_splits_per = np.where(diff_time > period)[0]
    idx_splits_same_phase_sign = np.where(np.logical_and(diff_time > period / 2, diff_time < period))[0]
    idx_splits_same_phase_sign_valid = [
        idx for idx in idx_splits_same_phase_sign if phase[idx - 1] * phase[idx] == 1
    ]
    idx_splits = np.concatenate([idx_splits, idx_splits_per, idx_splits_same_phase_sign_valid]).astype("uint")
    idx_splits.sort()
    time_split = np.array_split(time, idx_splits)
    phase_split = np.array_split(phase, idx_splits)
    timeseries_split = np.array_split(timeseries, idx_splits)
    n_obs_phases = len(phase_split)

    odd_even_obs = -1 * np.ones(n_obs_phases, dtype="int")

    epoch_idx = 0
    for phase_i in range(n_obs_phases):
        for epoch_i in range(epoch_idx, n_phases):
            if np.logical_and(
                np.all(epochs[epoch_i] - period / 2 < time_split[phase_i]),
                np.all(epochs[epoch_i] + period / 2 > time_split[phase_i]),
            ):
                odd_even_obs[phase_i] = odd_even_id_arr[epoch_i]
                epoch_idx = epoch_i + 1
                break

    idx_valid_phases = odd_even_obs != -1
    time_split = [time_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    timeseries_split = [timeseries_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    phase_split = [phase_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    odd_even_obs = [odd_even_obs[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    n_obs_phases = len(time_split)
    if n_obs_phases == 0:
        return None, None, 0, None

    if remove_outliers_params:
        timeseries_split, idxs_out_lst = remove_outliers(
            timeseries_split,
            seed=seed,
            **remove_outliers_params,
        )
        idxs_valid_phases = [phase_idx for phase_idx in range(len(timeseries_split)) if len(timeseries_split) > 0]
        timeseries_split = [timeseries_split[phase_i] for phase_i in range(len(timeseries_split))
                            if phase_i in idxs_valid_phases]
        time_split = [time_split[phase_i] for phase_i in idxs_valid_phases]
        phase_split = [phase_split[phase_i] for phase_i in idxs_valid_phases]
        odd_even_obs = [odd_even_obs[phase_i] for phase_i in idxs_valid_phases]
        n_obs_phases = len(time_split)
        if n_obs_phases == 0:
            return None, None, 0, None

    num_it_cadences_phases = np.array([
        np.isfinite(timeseries_split[phase_i][
            np.logical_and(phase_split[phase_i] > tmin_it, phase_split[phase_i] < tmax_it)
        ]).sum()
        for phase_i in range(n_obs_phases)
    ])
    idx_valid_phases = (num_it_cadences_phases / expected_num_it_cadences) > it_cadences_per_thr
    time_split = [time_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    timeseries_split = [timeseries_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    phase_split = [phase_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    odd_even_obs = [odd_even_obs[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    n_obs_phases = len(time_split)
    if n_obs_phases == 0:
        return None, None, 0, None

    if keep_odd_even_order:
        keep_idxs = [0]
        curr_phase_type = odd_even_obs[0]
        for phase_i in range(1, n_obs_phases):
            if odd_even_obs[phase_i] != curr_phase_type:
                keep_idxs.append(phase_i)
                curr_phase_type = odd_even_obs[phase_i]
        time_split = [time_split[keep_idx] for keep_idx in keep_idxs]
        timeseries_split = [timeseries_split[keep_idx] for keep_idx in keep_idxs]
        phase_split = [phase_split[keep_idx] for keep_idx in keep_idxs]
        odd_even_obs = [odd_even_obs[keep_idx] for keep_idx in keep_idxs]
        n_obs_phases = len(time_split)
        if n_obs_phases == 0:
            return None, None, 0, None

    if quarter_timestamps is None:
        if n_obs_phases > n_max_phases:

            rng = np.random.default_rng(seed=seed)
            chosen_phases_st = rng.integers(0, n_obs_phases - n_max_phases, size=None)
            chosen_phases = np.arange(chosen_phases_st, chosen_phases_st + n_max_phases)

            time_split = [time_split[chosen_phase] for chosen_phase in chosen_phases]
            timeseries_split = [timeseries_split[chosen_phase] for chosen_phase in chosen_phases]
            phase_split = [phase_split[chosen_phase] for chosen_phase in chosen_phases]
            odd_even_obs = [odd_even_obs[chosen_phase] for chosen_phase in chosen_phases]

        elif n_obs_phases < n_max_phases:

            if extend_method == "zero_padding":
                n_miss_phases = n_max_phases - n_obs_phases

                med_n_cadences = np.median([time_split[phase_i] for phase_i in range(n_obs_phases)])
                med_timeseries_val = np.median([timeseries_split[phase_i] for phase_i in range(n_obs_phases)])
                phase_split_miss = np.linspace(-period / 2, period / 2, med_n_cadences, endpoint=True)

                phase_split += [phase_split_miss] * n_miss_phases
                timeseries_split += [med_timeseries_val * np.ones(med_n_cadences, dtype="float")] * n_miss_phases
                time_split += [np.nan * np.ones(med_n_cadences, dtype="float")] * n_miss_phases
                odd_even_obs += [np.nan] * n_miss_phases

            elif extend_method == "copy_phases":

                n_full_group_phases = n_max_phases // n_obs_phases
                n_partial_group_phases = n_max_phases % n_obs_phases
                phase_split = phase_split * n_full_group_phases + phase_split[:n_partial_group_phases]

                time_split = list(time_split) * n_full_group_phases + list(time_split[:n_partial_group_phases])
                timeseries_split = (
                    list(timeseries_split) * n_full_group_phases
                    + list(timeseries_split[:n_partial_group_phases])
                )
                odd_even_obs = odd_even_obs * n_full_group_phases + odd_even_obs[:n_partial_group_phases]

            else:
                raise NotImplementedError(f"Extend method for phases `{extend_method}` not implemented.")

    else:
        n_seasons = 4
        n_phases_per_season = n_max_phases // n_seasons
        t0_time_split = [time_phase[0] for time_phase in time_split]
        season_phases = np.nan * np.ones(len(time_split))
        for t_i, t in enumerate(t0_time_split):
            for q, q_t in quarter_timestamps.items():
                if t >= q_t[0] and t <= q_t[1]:
                    season_phases[t_i] = q % n_seasons
                    break

        chosen_idx_phases_all_seasons = []
        not_chosen_idxs_phases_all_seasons = []
        for season_i in range(n_seasons):
            idxs_phases_in_season = np.where(season_phases == season_i)[0]
            n_phases_in_season = len(idxs_phases_in_season)
            if n_phases_in_season >= n_phases_per_season:
                rng = np.random.default_rng(seed=seed)
                chosen_phases_st = rng.integers(0, n_phases_in_season - n_phases_per_season + 1, size=None)
                chosen_idx_phases_season = idxs_phases_in_season[chosen_phases_st: chosen_phases_st + n_phases_per_season]
                not_chosen_idxs_phases_all_seasons.append(np.setdiff1d(idxs_phases_in_season, chosen_idx_phases_season))
            else:
                chosen_idx_phases_season = idxs_phases_in_season

            chosen_idx_phases_all_seasons.append(chosen_idx_phases_season)

        chosen_idx_phases_all_seasons = np.concatenate(chosen_idx_phases_all_seasons)
        if len(not_chosen_idxs_phases_all_seasons) != 0:
            not_chosen_idxs_phases_all_seasons = np.concatenate(not_chosen_idxs_phases_all_seasons)

        n_chosen_phases_all_seasons = len(chosen_idx_phases_all_seasons)

        n_phases_left = n_max_phases - n_chosen_phases_all_seasons

        if len(not_chosen_idxs_phases_all_seasons) != 0:
            chosen_idx_phases_all_seasons = np.concatenate(
                [chosen_idx_phases_all_seasons, not_chosen_idxs_phases_all_seasons[:n_phases_left]],
                dtype="int",
            )

        n_chosen_phases_all_seasons = len(chosen_idx_phases_all_seasons)
        n_phases_left = n_max_phases - n_chosen_phases_all_seasons
        if n_phases_left > 0:
            if extend_method == "copy_phases":
                n_full_group_phases = n_max_phases // n_chosen_phases_all_seasons
                n_partial_group_phases = n_max_phases % n_chosen_phases_all_seasons
                chosen_idx_phases_all_seasons = np.concatenate(
                    [np.tile(chosen_idx_phases_all_seasons, n_full_group_phases),
                     chosen_idx_phases_all_seasons[:n_partial_group_phases]],
                    axis=0,
                )
            elif extend_method == "zero_padding":
                chosen_idx_phases_all_seasons = np.concatenate(
                    (chosen_idx_phases_all_seasons, -1 * np.ones(n_phases_left, dtype="int")),
                    axis=1,
                )
                med_timeseries_val = np.median([time_split[phase_i] for phase_i in chosen_idx_phases_all_seasons])
                med_n_cadences = np.median([time_split[phase_i] for phase_i in chosen_idx_phases_all_seasons])

                phase_split_zeropadding = np.linspace(-period / 2, period / 2, med_n_cadences, endpoint=True)
                time_split_zeroppading = np.nan * np.ones(med_n_cadences, dtype="float")
                timeseries_split_zeropadding = med_timeseries_val * np.ones(med_n_cadences, dtype="float")
            else:
                raise ValueError(f"Extend method for phases `{extend_method}` not implemented.")

        time_split_n, timeseries_split_n, phase_split_n, odd_even_obs_n = [], [], [], []
        for chosen_phase in chosen_idx_phases_all_seasons:
            if chosen_phase == -1:
                phase_split_n.append(phase_split_zeropadding)
                time_split_n.append(time_split_zeroppading)
                timeseries_split_n.append(timeseries_split_zeropadding)
                odd_even_obs_n.append(np.nan)
            else:
                time_split_n.append(time_split[chosen_phase])
                timeseries_split_n.append(timeseries_split[chosen_phase])
                phase_split_n.append(phase_split[chosen_phase])
                odd_even_obs_n.append(odd_even_obs[chosen_phase])

        time_split, timeseries_split, phase_split, odd_even_obs = (
            time_split_n, timeseries_split_n, phase_split_n, odd_even_obs_n
        )

    n_obs_phases = len(time_split)

    if n_obs_phases == 0:
        return None, None, 0, None

    return phase_split, timeseries_split, n_obs_phases, odd_even_obs


def generate_view(
    time, flux, num_bins, bin_width, t_min, t_max, tce,
    centering=True, normalize=True, centroid=False,
    **kwargs,
):
    """Generate a view of a phase-folded light curve using a median filter."""
    view, time_bins, view_var, bin_counts = median_filter.median_filter(
        time, flux, num_bins, bin_width, t_min, t_max
    )

    view, view_var, inds_nan = impute_binned_ts(
        time_bins, view, tce["tce_period"], tce["tce_duration"], view_var
    )

    if centering:
        view -= np.median(view)

    if normalize:
        view = normalize_view(view, val=None, centroid=centroid, **kwargs)

    return view, time_bins, view_var, inds_nan, bin_counts


def generate_view_momentum_dump(time, momentum_dumps, num_bins, bin_width, t_min, t_max):
    """Generate a view of a phase-folded momentum dump array using a mean filter."""
    view, time_bins, view_var, _ = median_filter.median_filter(
        time, momentum_dumps, num_bins, bin_width, t_min, t_max,
        bin_fn=np.mean, bin_var_fn=np.std,
    )

    inds_nan = np.isnan(view)
    view[inds_nan] = 0
    view_var[inds_nan] = 0

    return view, view_var, time_bins


def normalize_view(view, val=None, centroid=False, **kwargs):
    """Normalize the phase-folded time series by ``|max|`` (centroid) or ``|min|`` (flux)."""
    if val is None:
        val = np.abs(np.max(view)) if centroid else np.abs(np.min(view))

    if val == 0:
        if "report" in kwargs:
            out_path = None
            config = kwargs["report"].get("config")
            tce = kwargs["report"].get("tce")
            if config and "exclusion_logs_dir" in config and tce and "uid" in tce:
                out_path = config["exclusion_logs_dir"] / f'exclusions-{tce["uid"]}.txt'
            _report_exclusion(
                f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.',
                out_path,
            )

    return view / (val + NORM_SIGMA_EPS)


def centering_and_normalization(view, val_centr, val_norm, **kwargs):
    """Center and normalize a 1D time series: ``(view - val_centr) / (val_norm + eps)``."""
    if val_norm == 0:
        if "report" in kwargs:
            out_path = None
            config = kwargs["report"].get("config")
            tce = kwargs["report"].get("tce")
            if config and "exclusion_logs_dir" in config and tce and "uid" in tce:
                out_path = config["exclusion_logs_dir"] / f'exclusions-{tce["uid"]}.txt'
            _report_exclusion(
                f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.',
                out_path,
            )

    return (view - val_centr) / (val_norm + NORM_SIGMA_EPS)


def global_view(
    time, flux, period, tce,
    num_bins=2001, bin_width_factor=1 / 2001,
    centroid=False, normalize=True, centering=True,
    **kwargs,
):
    """Generate the 'global view' of a phase-folded light curve."""
    view, time_bins, view_var, inds_nan, bin_counts = generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=max(period * bin_width_factor, kwargs["tce_duration"] * 0.16),
        t_min=-period / 2,
        t_max=period / 2,
        tce=tce,
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        **kwargs,
    )

    return view, time_bins, view_var, inds_nan, bin_counts


def local_view(
    time, flux, period, duration, tce,
    num_bins=201, bin_width_factor=0.16, num_durations=4,
    centroid=False, normalize=True, centering=True,
    **kwargs,
):
    """Generate the 'local view' of a phase-folded light curve."""
    t_min = max(-period / 2, -duration * num_durations)
    t_max = min(period / 2, duration * num_durations)

    if t_min > time[-1] or t_max < time[0]:
        out_path = None
        if "report" in kwargs:
            config = kwargs["report"].get("config")
            if config and "exclusion_logs_dir" in config:
                out_path = config["exclusion_logs_dir"] / f'exclusions-{tce["uid"]}.txt'
        _report_exclusion(
            f'No in-transit cadences in view {kwargs.get("report", {}).get("view", "?")}. Setting local view to '
            "median and var view to mad std using med/mad_std statistics from the phase folded time series.",
            out_path,
        )
        time_bins = np.linspace(t_min, t_max, num_bins, endpoint=True)
        med = np.median(flux)
        std_rob_estm = mad_std(flux)
        view_var = std_rob_estm * np.ones(num_bins, dtype="float")
        view = med + np.ones(num_bins, dtype="float")
        inds_nan = {
            "oot": False * np.ones(num_bins, dtype="bool"),
            "it": False * np.ones(num_bins, dtype="bool"),
        }
        bin_counts = np.ones(num_bins, dtype="float")
    else:
        view, time_bins, view_var, inds_nan, bin_counts = generate_view(
            time,
            flux,
            tce=tce,
            num_bins=num_bins,
            bin_width=duration * bin_width_factor,
            t_min=t_min,
            t_max=t_max,
            centroid=centroid,
            normalize=normalize,
            centering=centering,
            **kwargs,
        )

    return view, time_bins, view_var, inds_nan, bin_counts
