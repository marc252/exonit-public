"""exonit.exominer_native — server-side proxies for external data.

Workers fetch MAST DV XML, TICv8 stellar params, and Gaia DR2 + RUWE
through these endpoints rather than calling astroquery themselves (design
decision D4). Caching is SQLite + filesystem, per-resource.
"""
