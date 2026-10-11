"""worker.exominer_native.preprocess — native port of ExoMiner preprocessing.

Each module mirrors a chunk of the NASA container's ``src_preprocessing/``
tree, adapted to run in the worker's modern venv (numpy 2.x, no TF). Every
ported function is validated against the Docker oracle via a parity gate in
``tests/worker/exominer_native/preprocess/test_parity_<module>.py``.

See ``docs/superpowers/specs/2026-04-18-exominer-native-design.md`` §3.2 for
the full module-by-module mapping.
"""
