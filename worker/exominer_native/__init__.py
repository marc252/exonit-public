"""worker.exominer_native — native (Docker-free, TF-free) ExoMiner pipeline.

See docs/superpowers/specs/2026-04-18-exominer-native-design.md for the full
design. This package replaces the old venv_tf + TFRecord flow: preprocessing
stays in pure Python + numpy, inference uses ONNX Runtime.
"""
