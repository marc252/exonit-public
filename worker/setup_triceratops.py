#!/usr/bin/env python3
"""Install and patch triceratops for ExoNit workers.

Usage:
    pip install triceratops pytransit
    python setup_triceratops.py

Patches:
  1. funcs.py: delim_whitespace → sep=r'\s+' (pandas 3.0 compat)
  2. marginal_likelihoods.py: LDC lookup duplicate fix (.flat[0])
"""

import os
import sys
import re


def find_triceratops_dir():
    try:
        import triceratops
        return os.path.dirname(triceratops.__file__)
    except ImportError:
        print("ERROR: triceratops not installed. Run:")
        print("  pip install triceratops pytransit")
        sys.exit(1)


def patch_file(path, patches):
    """Apply text replacements to a file. Returns number of patches applied."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    applied = 0
    for old, new in patches:
        if old in content:
            content = content.replace(old, new)
            applied += 1
    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return applied


def main():
    tri_dir = find_triceratops_dir()
    print(f"triceratops found at: {tri_dir}")

    # Patch 1: funcs.py — pandas 3.0 compat
    funcs_path = os.path.join(tri_dir, "funcs.py")
    n = patch_file(funcs_path, [
        ("read_csv(output_url, delim_whitespace=True)",
         "read_csv(output_url, sep=r'\\s+')"),
    ])
    print(f"  funcs.py: {n} patch(es) applied" if n else "  funcs.py: already patched")

    # Patch 2: marginal_likelihoods.py — LDC duplicate fix
    ml_path = os.path.join(tri_dir, "marginal_likelihoods.py")
    patches = [
        # u1, u2 = ldc_u1s[mask], ldc_u2s[mask]  →  .flat[0]
        ("    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]",
         "    u1, u2 = ldc_u1s[mask].flat[0], ldc_u2s[mask].flat[0]"),
        # u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask], u2s_at_Z[mask]
        ("        u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask], u2s_at_Z[mask]",
         "        u1_vals, u2_vals = u1s_at_Z[mask], u2s_at_Z[mask]\n"
         "        u1s_comp[i] = u1_vals.flat[0] if len(u1_vals) > 0 else 0.0\n"
         "        u2s_comp[i] = u2_vals.flat[0] if len(u2_vals) > 0 else 0.0"),
        # u1s_comp[i], u2s_comp[i] = ldc_u1s[mask], ldc_u2s[mask]
        ("        u1s_comp[i], u2s_comp[i] = ldc_u1s[mask], ldc_u2s[mask]",
         "        u1_vals, u2_vals = ldc_u1s[mask], ldc_u2s[mask]\n"
         "        u1s_comp[i] = u1_vals.flat[0] if len(u1_vals) > 0 else 0.0\n"
         "        u2s_comp[i] = u2_vals.flat[0] if len(u2_vals) > 0 else 0.0"),
        # u1s_possible[i], u2s_possible[i] = ldc_u1s[mask], ldc_u2s[mask]
        ("        u1s_possible[i], u2s_possible[i] = ldc_u1s[mask], ldc_u2s[mask]",
         "        u1_vals, u2_vals = ldc_u1s[mask], ldc_u2s[mask]\n"
         "        u1s_possible[i] = u1_vals.flat[0] if len(u1_vals) > 0 else 0.0\n"
         "        u2s_possible[i] = u2_vals.flat[0] if len(u2_vals) > 0 else 0.0"),
    ]
    n = patch_file(ml_path, patches)
    print(f"  marginal_likelihoods.py: {n} patch(es) applied" if n else
          "  marginal_likelihoods.py: already patched")

    # Verify
    print("\nVerificació:")
    try:
        from triceratops import triceratops as tri
        print("  import triceratops: OK")
    except Exception as e:
        print(f"  import triceratops: FAIL ({e})")
    try:
        from triceratops_fpp import calc_fpp
        print("  import triceratops_fpp: OK")
    except Exception as e:
        print(f"  import triceratops_fpp: FAIL ({e})")

    print("\nDone! Restart the worker to activate TRICERATOPS FPP.")


if __name__ == "__main__":
    main()
