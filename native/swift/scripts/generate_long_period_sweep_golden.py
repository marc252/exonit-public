#!/usr/bin/env python3
"""Generate the deterministic CPU reference for the long-period sweep."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["EXONIT_WOTAN_BACKEND"] = "cpu"


def biweight_location(values, cval=5.0):
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad == 0:
        return center
    inverse_cmad = 1.0 / (cval * mad)
    old_center = center
    delta = np.inf
    while abs(delta) > 1e-6:
        numerator = 0.0
        denominator = 0.0
        for value in values:
            scaled = (value - center) * inverse_cmad
            if abs(scaled) < 1:
                base = 1 - scaled * scaled
                weight = base * base
                numerator += (value - center) * weight
                denominator += weight
        center += numerator / denominator
        delta = old_center - center
        old_center = center
    return center


def detrend_with_wotan(time, flux, window_length, transit_mask):
    """Offline CPU reference of Wotan's biweight running-location path."""
    valid = np.isfinite(time) & np.isfinite(flux)
    compressed_time = time[valid]
    compressed_flux = flux[valid]
    compressed_mask = transit_mask[valid]
    if compressed_time.size == 0:
        return np.full(time.size, np.nan)
    boundaries = [0]
    boundaries.extend(np.flatnonzero(np.diff(compressed_time) > 0.5) + 1)
    boundaries.append(compressed_time.size + 1)
    trend_compressed = []
    half_window = window_length / 2.0
    edge_cutoff = min(0.5, half_window)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        upper = min(upper, compressed_time.size)
        segment_time = compressed_time[lower:upper]
        segment_flux = compressed_flux[lower:upper]
        segment_mask = compressed_mask[lower:upper]
        segment_trend = np.full(segment_time.size, np.nan)
        low = segment_time.min() + edge_cutoff
        high = segment_time.max() - edge_cutoff
        for index, center_time in enumerate(segment_time):
            if center_time < low or center_time > high:
                continue
            start = int(np.searchsorted(segment_time, center_time - half_window, side="left"))
            end = int(np.searchsorted(segment_time, center_time + half_window, side="left"))
            if start >= end:
                continue
            values = [
                value for value, masked in zip(segment_flux[start:end], segment_mask[start:end])
                if not masked and value > -1e-12
            ]
            if values:
                segment_trend[index] = biweight_location(values)
        trend_compressed.extend(segment_trend)
    trend = np.full(time.size, np.nan)
    trend[valid] = np.asarray(trend_compressed)
    trend[trend == 0] = np.nan
    return flux / trend


def period_grid(period_min, period_max, count):
    return 1.0 / np.linspace(1.0 / period_min, 1.0 / period_max, count)


def duration_bins(bin_count, minimum, maximum, steps):
    start = max(1, int(minimum * bin_count))
    end = min(bin_count // 3, int(maximum * bin_count))
    return np.unique(np.linspace(start, end, steps).astype(np.int64))


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def bls(time, flux, period_min, period_max, period_count, bin_count,
        duration_steps, duration_min_frac, duration_max_frac):
    relative = time - time[0]
    periods = period_grid(period_min, period_max, period_count)
    widths = duration_bins(bin_count, duration_min_frac, duration_max_frac, duration_steps)
    total = float(np.sum(flux))
    powers = []
    records = []
    for period in periods:
        phase = (relative / period) % 1.0
        bins = np.minimum((phase * bin_count).astype(np.int64), bin_count - 1)
        sums = np.bincount(bins, weights=flux, minlength=bin_count)
        counts = np.bincount(bins, minlength=bin_count)
        ext_sums = np.concatenate((sums, sums))
        ext_counts = np.concatenate((counts, counts))
        cumulative_sums = np.concatenate(([0.0], np.cumsum(ext_sums)))
        cumulative_counts = np.concatenate(([0], np.cumsum(ext_counts)))
        best = (-1e30, 0, 0, 0.0)
        for width in widths:
            for phase_bin in range(bin_count):
                inside_count = int(cumulative_counts[phase_bin + width] - cumulative_counts[phase_bin])
                outside_count = len(time) - inside_count
                if inside_count < 3 or outside_count < 3:
                    continue
                inside_sum = cumulative_sums[phase_bin + width] - cumulative_sums[phase_bin]
                depth = (total - inside_sum) / outside_count - inside_sum / inside_count
                if depth <= 0:
                    continue
                power = depth * depth * inside_count * outside_count / (inside_count + outside_count)
                if power > best[0]:
                    best = (power, int(width), phase_bin, depth)
        powers.append(best[0])
        records.append(best)
    powers = np.asarray(powers)
    index = int(np.argmax(powers))
    med = median(powers)
    mad = median(np.abs(powers - med))
    snr = 0.0 if mad == 0 else float((powers[index] - med) / (1.4826 * mad))
    power, width, phase_bin, depth = records[index]
    return {
        "period": float(periods[index]),
        "power": float(power),
        "duration": float(width / bin_count * periods[index]),
        "t0": float(phase_bin / bin_count * periods[index] + time[0]),
        "depth": float(depth),
        "snr": snr,
    }


def clean(values):
    return [None if not np.isfinite(value) else float(value) for value in values]


rng = np.random.RandomState(90210)
time = np.arange(641, dtype=np.float64) * 0.25
# A >100-day baseline with a gap; the signal is intentionally long-period.
time[420:] += 1.25
period_signal = 61.5
t0_signal = 5.25
duration_signal = 1.5
phase = np.mod(time - t0_signal, period_signal)
in_transit = (phase < duration_signal) & np.isfinite(time)
flux = 1.0 + 0.0003 * np.sin(time * 0.17) + 0.0001 * np.cos(time * 0.043)
flux -= np.where(in_transit, 0.020, 0.0)
flux += rng.normal(0.0, 0.00005, size=time.size)
flux[37] = np.nan
bad_mask = np.zeros(time.size, dtype=bool)
bad_mask[500] = True

period_max = (time[-1] - time[0]) / 2.5
options = {
    "baselineThresholdDays": 100.0,
    "windowsDays": [2.0, 4.0],
    "periodMinimumDays": 30.0,
    "legacyPeriodMaximumDays": 50.0,
    "periodMaximumCapDays": 200.0,
    "minimumTransits": 2.5,
    "basePeriodCount": 241,
    "periodCountScaleReferenceMaximumDays": None,
    "binCount": 64,
    "durationStepCount": 7,
    "durationMinimumFraction": 0.02,
    "durationMaximumFraction": 0.16,
    "minimumSNR": 3.0,
}
results = []
for window in options["windowsDays"]:
    flattened = detrend_with_wotan(time, flux, window_length=window, transit_mask=bad_mask)
    good = flattened[~bad_mask & np.isfinite(flattened)]
    fill = median(good) if good.size else 1.0
    flattened = flattened.copy()
    flattened[bad_mask | ~np.isfinite(flattened)] = fill
    result = bls(
        time, flattened, options["periodMinimumDays"],
        min(period_max, options["periodMaximumCapDays"]),
        options["basePeriodCount"], options["binCount"],
        options["durationStepCount"], options["durationMinimumFraction"],
        options["durationMaximumFraction"],
    )
    result["windowDays"] = window
    results.append(result)
best = max(results, key=lambda item: item["snr"])
payload = {
    "time": clean(time),
    "flux": clean(flux),
    "accumulatedMask": bad_mask.tolist(),
    "options": options,
    "expected": best,
    "perWindow": results,
}
out = ROOT / "Fixtures" / "long_period_sweep_golden.json"
out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
print(out)
print(json.dumps({key: best[key] for key in ("windowDays", "period", "t0", "duration", "depth", "snr", "power")}, indent=2))
