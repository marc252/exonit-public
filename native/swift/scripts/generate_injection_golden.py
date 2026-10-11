#!/usr/bin/env python3
"""Generate the deterministic CPU injection-recovery contract fixture."""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-python"))

# Keep fixture generation independent of the host's accelerator availability.
os.environ["EXONIT_INJECTION_METAL"] = "0"
os.environ["EXONIT_INJECTION_OPENCL"] = "0"

from injection import disable_gpu, run_injection_recovery  # noqa: E402


def main() -> None:
    disable_gpu()
    time = np.arange(0.0, 12.000001, 0.05, dtype=np.float64)
    # A deliberately flattened synthetic light curve: all signal is produced
    # by injection.py, making the fixture stable and easy to inspect.
    flux = np.ones_like(time)
    radius_grid = np.array([1.0, 2.0], dtype=np.float64)
    period_grid = np.array([1.0, 2.0], dtype=np.float64)
    result = run_injection_recovery(
        time,
        flux,
        stellar_radius=1.0,
        radius_grid=radius_grid,
        period_grid=period_grid,
        n_injections_per_cell=2,
        snr_threshold=7.0,
        bls_n_periods=101,
        bls_n_bins=30,
        bls_n_dur_steps=5,
    )
    result = {
        "time": time.tolist(),
        "flux": flux.tolist(),
        "stellar_radius": 1.0,
        "radius_grid": radius_grid.tolist(),
        "period_grid": period_grid.tolist(),
        "injections_per_cell": 2,
        "snr_threshold": 7.0,
        "bls_n_periods": 101,
        "bls_n_bins": 30,
        "bls_n_dur_steps": 5,
        "expected": {
            "radius_grid": result["radius_grid"],
            "period_grid": result["period_grid"],
            "recovered_counts": [
                [cell["n_recovered"] for cell in result["details"][row * 2:(row + 1) * 2]]
                for row in range(2)
            ],
            "completeness_map": result["completeness_map"],
            "truncated": result["truncated"],
            "periods_evaluated": result["periods_evaluated"],
        },
    }
    output = ROOT / "Fixtures/injection_recovery_golden.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
