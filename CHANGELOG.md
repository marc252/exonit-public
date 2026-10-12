# Changelog

## sync 2026-10-12 (internal fbdd171)
- feat(release): publish_results.sh (single orphan commit to exonit-results) + README; fix gh visibility command
- fix(release): replace internal [[memory]] links and list skipped legacy scorecards in results INDEX
- feat(release): curated public results tree for exonit-results (scorecards, dossiers, strong candidates, TOI benchmark, ledger)
- fix(discovery): P1 review — task-row sector, scoped wipe, require-fpp, tests
- fix(test): P4 review — deterministic prep, CSV-driven ephemeris/tolerance
- fix(worker): P2 review — opt-in defaults, cap, ensemble tests
- fix(db): add_candidate lookup must include sector (prerequisite for cross-sector recurrence)
- feat(discovery): cross-sector recurrence review queue + re-observation lot generator (P1)
- feat(worker): P2 refine — wider t0 window + second pass
- feat(worker): refine BLS duration/t0 and merge same-period candidates (P2, Python)
- feat(verdict): persist verdict_proba (P3)

## sync 2026-10-11 (internal 6779974)
- (no user-facing changes)