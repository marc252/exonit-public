#!/bin/bash
set -euo pipefail

usage() {
  echo "usage: $0 --bundle DIR --root DIR --allowed-signers FILE --signer IDENTITY | --rollback --root DIR" >&2
  exit 2
}

bundle= root= allowed_signers= signer= rollback=false
while (($#)); do
  case "$1" in
    --bundle) bundle=${2-}; shift 2 ;;
    --root) root=${2-}; shift 2 ;;
    --allowed-signers) allowed_signers=${2-}; shift 2 ;;
    --signer) signer=${2-}; shift 2 ;;
    --rollback) rollback=true; shift ;;
    *) usage ;;
  esac
done

[[ -n $root ]] || usage
mkdir -p "$root/releases"

valid_target() {
  [[ $1 =~ ^releases/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] && [[ -d "$root/$1" ]]
}
replace_link() {
  local name=$1 target=$2
  # -f overwrites the existing link; -h (macOS/BSD ln) makes it operate on
  # the link itself rather than following it into a directory, which is
  # what a stale `current -> releases/<dir>` symlink would otherwise do
  # to a plain `mv -f tmp link` (the tmp link lands *inside* that dir
  # instead of replacing `link`, and the install silently exits 0).
  ln -sfh "$target" "$root/$name"
}

if [[ $rollback == true ]]; then
  [[ -L $root/current && -L $root/previous ]] || { echo "no release to roll back" >&2; exit 1; }
  current_target=$(readlink "$root/current")
  previous_target=$(readlink "$root/previous")
  valid_target "$current_target" && valid_target "$previous_target" || { echo "invalid release links" >&2; exit 1; }
  replace_link previous "$current_target"
  replace_link current "$previous_target"
  [[ $(readlink "$root/current") == "$previous_target" ]] || {
    echo "rollback verification failed: current was not flipped to $previous_target" >&2
    exit 1
  }
  exit 0
fi

for required in "$bundle" "$allowed_signers" "$signer"; do
  [[ -n $required ]] || usage
done
[[ -d $bundle && -f $bundle/manifest.json && -f $bundle/manifest.json.sig && -f $allowed_signers ]] || {
  echo "invalid release bundle or allowed signers file" >&2
  exit 1
}

ssh-keygen -Y verify -q -f "$allowed_signers" -I "$signer" -n "$signer" -s "$bundle/manifest.json.sig" < "$bundle/manifest.json"

version=$(python3 - "$bundle" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve(strict=True)
try:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"files", "version"} or not isinstance(manifest["files"], dict):
        raise ValueError("invalid manifest schema")
    version = manifest["version"]
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("invalid manifest version")
    expected = manifest["files"]
    actual = {}
    for directory, dirs, names in os.walk(bundle, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in dirs + names):
            raise ValueError("release bundle contains a symlink")
        for name in names:
            path = base / name
            relative = path.relative_to(bundle).as_posix()
            if relative in {"manifest.json", "manifest.json.sig"}:
                continue
            if not path.is_file():
                raise ValueError("release bundle contains a non-file")
            actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("manifest hashes do not match bundle")
    required = {
        "exonit-swift-worker",
        "Sidecars/exonit_science_sidecar.py",
    }
    if (not required.issubset(actual)
            or not any(key.startswith("Models/cnn_ffi_v1.mlpackage/") for key in actual)
            or not any(key.startswith("Models/exominer_ffi_v6_ensemble.mlpackage/") for key in actual)):
        raise ValueError("release bundle is missing required payload")
except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"invalid release bundle: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(version)
PY
)

final="$root/releases/$version"
[[ ! -e $final && ! -L $final ]] || { echo "release already installed: $version" >&2; exit 1; }
stage=$(mktemp -d "$root/releases/.${version}.tmp.XXXXXX")
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT
cp -R -p "$bundle/." "$stage"
mv "$stage" "$final"
trap - EXIT

if ! "$final/exonit-swift-worker" --help >/dev/null 2>&1 || \
   ! /usr/bin/python3 "$final/Sidecars/exonit_science_sidecar.py" --self-test >/dev/null 2>&1; then
  echo "release smoke test failed: $version" >&2
  rm -rf "$final"
  exit 1
fi

new_target="releases/$version"
if [[ -L $root/current ]]; then
  old_target=$(readlink "$root/current")
  valid_target "$old_target" || { echo "invalid current release link" >&2; exit 1; }
  replace_link previous "$old_target"
fi
replace_link current "$new_target"
[[ $(readlink "$root/current") == "$new_target" ]] || {
  echo "install verification failed: current was not flipped to $new_target" >&2
  exit 1
}
