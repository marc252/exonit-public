"""worker — ExoNit worker-side code (run as scripts or imported as a package).

The `worker/` directory historically hosted flat top-level modules that are
executed directly (e.g. `python worker.py`). New worker-side Python is being
reorganised into proper subpackages (e.g. `worker.exominer_native`). This
`__init__.py` marks `worker/` as a regular package so submodules under it can
be imported as `worker.<subpackage>`.
"""
