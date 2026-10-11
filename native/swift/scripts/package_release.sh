#!/bin/bash
set -euo pipefail

usage() {
  echo "usage: $0 --version VERSION --binary PATH --cnn PATH --exominer PATH --sidecar PATH --signing-key PATH --signer IDENTITY --output DIR" >&2
  exit 2
}

version= binary= cnn= exominer= sidecar= signing_key= signer= output=
while (($#)); do
  case "$1" in
    --version) version=${2-}; shift 2 ;;
    --binary) binary=${2-}; shift 2 ;;
    --cnn) cnn=${2-}; shift 2 ;;
    --exominer) exominer=${2-}; shift 2 ;;
    --sidecar) sidecar=${2-}; shift 2 ;;
    --signing-key) signing_key=${2-}; shift 2 ;;
    --signer) signer=${2-}; shift 2 ;;
    --output) output=${2-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid version" >&2; exit 2; }
for required in "$binary" "$cnn" "$exominer" "$sidecar" "$signing_key" "$signer" "$output"; do
  [[ -n $required ]] || usage
done
[[ -f $binary && -x $binary ]] || { echo "binary must be an executable file" >&2; exit 2; }
[[ -d $cnn && -d $exominer ]] || { echo "model packages must be directories" >&2; exit 2; }
[[ -f $sidecar && -f $signing_key ]] || { echo "sidecar and signing key must be files" >&2; exit 2; }
[[ ! -e $output && ! -L $output ]] || { echo "output already exists: $output" >&2; exit 1; }

for input in "$binary" "$cnn" "$exominer" "$sidecar"; do
  [[ -z $(find "$input" -type l -print -quit) ]] || { echo "release inputs may not contain symlinks: $input" >&2; exit 2; }
done

output_parent=$(dirname "$output")
output_name=$(basename "$output")
mkdir -p "$output_parent"
stage=$(mktemp -d "$output_parent/.${output_name}.tmp.XXXXXX")
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT

mkdir -p "$stage/Models" "$stage/Sidecars"
cp -p "$binary" "$stage/exonit-swift-worker"
cp -R -p "$cnn" "$stage/Models/cnn_ffi_v1.mlpackage"
cp -R -p "$exominer" "$stage/Models/exominer_ffi_v6_ensemble.mlpackage"
cp -p "$sidecar" "$stage/Sidecars/exonit_science_sidecar.py"

python3 - "$stage" "$version" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = sys.argv[2]
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink() or not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {"files": files, "version": version}
(root / "manifest.json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

ssh-keygen -Y sign -q -f "$signing_key" -n "$signer" "$stage/manifest.json"
mv "$stage" "$output"
trap - EXIT
