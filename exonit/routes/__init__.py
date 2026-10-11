"""Domain-split route registration modules for exonit.server.create_app().

Each sibling module exposes ``register(app, deps) -> None``, which attaches a
group of related endpoints (by domain) to *app* using the shared objects in
*deps* (see :mod:`exonit.routes.deps`). This package intentionally has no
import-time side effects: importing it (or any of its submodules) must never
create a DB connection, open a socket, or otherwise touch the outside world —
all such construction happens inside ``create_app()`` and is threaded through
via ``deps``.

Background: docs/reviews/2026-07-02-opus48-cleanup-review.md flagged
``create_app()`` (~4,550 lines, 84 endpoints as closures) as the root cause of
server-side logic duplication. This package is the "Fase D" split: mechanical
extraction of route registration by domain, behavior-neutral (same paths,
methods, status codes, response shapes, background-thread/executor
semantics). See progress/impl-createapp-split-2026-07-02.md for the module
map and design rationale.
"""
