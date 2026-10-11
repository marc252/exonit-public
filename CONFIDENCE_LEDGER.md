# ExoNit — Confidence Ledger

**Current Level: L0 — `vibe-coded`**

This ledger tracks our confidence in ExoNit's *scientific* validity, not its software quality. The pipeline can be technically excellent while still producing astrophysically misleading results. Higher levels = more independent validation. **Going public requires L3 as a minimum**; L4 enables peer-reviewed publication.

> Why this exists: ExoNit was built largely with AI assistance and without an astrophysicist on the team. The Confidence Ledger is the discipline that prevents that fact from leaking out as misleading public claims. Each level has explicit gating criteria — we move up only when every box for the next level is checked.

---

## Levels

| Level | Tag | Meaning | What we can do at this level |
|---|---|---|---|
| **L0** | `vibe-coded` | Pipeline runs end-to-end. No formal validation against external truth, known scientific bugs may exist undetected. | Use it personally. Show the code privately. |
| **L1** | `documented-experimental` | All known bugs catalogued; honest scorecards against TOI catalog committed; regression tests cover bugs found so far; README contains a clear experimental disclaimer. | Show to technical friends with the disclaimer. |
| **L2** | `internally-validated` | Sustained TOI parity over multiple months; CNN-FFI / ExoMiner-FFI deployed with per-source metrics; no critical KP/CP recall regressions for 6 months rolling; formal internal code audit done. | Approach an external reviewer with confidence. |
| **L3** | `externally-reviewed` | At least one astrophysicist (PhD/postdoc in TESS exoplanets or TOI Working Group member) has read the code and validated the scientific decisions (vetting thresholds, BLS settings, TRICERATOPS calibration, CNN training set). Their findings are applied. | Publish on GitHub with their name in acknowledgements. Submit no candidates to ExoFOP without independent re-validation. |
| **L4** | `co-authored` | Active collaboration with an astrophysicist who signs future decisions. Submitted poster/paper to a TESS conference or journal. Repository is shared with co-authors. | Treat ExoNit as a real scientific instrument. ExoFOP submissions OK with co-author sign-off. |

---

## Where we are

**L0 — `vibe-coded`** (declared 2026-05-06).

Most of the L1 boxes are partially complete from prior work. The next push (L0 → L1) is concrete and achievable.

### Gating criteria — L0 → L1 (`documented-experimental`)

- [ ] **All known bugs tracked in one checkable place** with severity, status, and reproducer, so open bugs are not scattered across commit messages and notes.
- [ ] **Honest scorecard against the TOI catalog** committed to the repository (for example `data/scorecards/toi-parity-YYYY-MM-DD.{json,md}`). Must include: precision, recall, F1 split by KP / CP / FP / APC, plus per-stellar-type breakdown. Numbers as they are, not cherry-picked.
- [ ] **Regression tests** for at least the 6 major bugs fixed during 2026-04 / 2026-05 sessions. New code must not silently re-break them.
- [ ] **README** with first-paragraph experimental disclaimer ("not validated by an astrophysicist; do NOT submit candidates to ExoFOP based on this output").
- [ ] **Reproducible install** documented end-to-end for at least one OS, including server + at least one worker. Someone with no context should be able to run a 10-TIC scorecard from a clean clone.
- [ ] **License** chosen (e.g. MIT or Apache-2.0) with the disclaimer baked into the LICENSE notice.

### Gating criteria — L1 → L2 (`internally-validated`)

- [ ] **Monthly TOI parity check** automated and committed. Trend line, not a single snapshot. No quarterly regression in recall on KP/CP.
- [ ] **CNN-FFI deployment validated**: per-source AUC / AP for QLP, TGLC, SPOC matching the per-source training metrics within tolerance, on a held-out catalog slice.
- [ ] **ExoMiner-FFI integration into verdict** (Phase 3 deploy) with scorecard demonstrating no recall loss vs the SPOC-only verdict on overlapping TICs.
- [ ] **6 months rolling** with no critical bug introducing systematic FP or KP/CP loss. Tracked in this ledger.
- [ ] **Internal code audit**: TODOs, dead code, OWASP-style security issues, secrets leak check. One write-up committed.
- [ ] **Astropy-style contribution guide** ([CONTRIBUTING.md](CONTRIBUTING.md)) so an external reviewer knows how to engage.

### Gating criteria — L2 → L3 (`externally-reviewed`)

- [ ] **Outreach attempted** to at least 2 astrophysicists in the TESS / TOI WG / Astropy ecosystem. Documented (who, when, response).
- [ ] **Formal code review** by a PhD or postdoc with astrophysical training. Output: written review with concrete findings.
- [ ] **Reviewer's findings applied** to the code, with commits referencing each finding.
- [ ] **List of FP-causing bugs still present** (after fixes) published in the README. We don't hide what we couldn't fix.

### Gating criteria — L3 → L4 (`co-authored`)

- [ ] **Active co-author** (astrophysicist) named in the repo, signing future significant scientific decisions.
- [ ] **Public submission** (poster, conference talk, journal paper, ApJ Letters note, or arXiv preprint) accepted or submitted.
- [ ] **Shared repository** with co-author write access; release tagging requires co-author sign-off.
- [ ] **ExoFOP submission protocol** agreed with co-author; submissions go out under joint names.

---

## Log

Format: each entry is a dated row. New entries go *below*, oldest stays at the top so the trajectory reads top-to-bottom.

| Date | Event | Category | Level impact |
|---|---|---|---|
| 2026-05-06 | Confidence Ledger system established. Project rescoped from "future public" to "personal until L3". | system | L0 declared |

---

## How to use this ledger (for maintainers and contributors)

- **Don't claim a level you haven't earned.** All criteria for level N must be checked before you write "we are now Ln".
- **Each gating box that gets ticked is a Log entry.** Date, what evidence, where the artefact is committed.
- **A regression downgrades the level.** If a critical KP/CP recall bug is found at L2, drop back to L1 until fixed *and* there's a regression test.
- **Before publishing anything (blog post, tweet, GitHub release), check the current level and adjust the framing.** L1 = "experimental, look but don't trust". L3 = "reviewed, use with caveats". L4 = "scientific tool".
- **A level is not a value judgement on the maintainers' work.** It reflects only how independently the science has been validated.
