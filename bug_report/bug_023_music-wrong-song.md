# Bug #023 — Music played the wrong video (tutorials, covers, unrelated clips)

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/media/music.py`
- **Commit(s):** e2cc602

## Symptom
"Play Hotel California" produced a guitar tutorial, a cover, or something totally unrelated ("A shocking incident for a nomadic family...").

## Root cause
`ytsearch1` took the literal top YouTube hit for the query, with no filtering.

## Fix
`music.py` `_search_candidates` / `_pick_best` / `_pick_target`: fetch 5 candidates, filter junk (tutorial/lesson/cover/tab/reaction/karaoke/backing), prefer official / "- Topic" channels and query-word matches. Verified with Eagles / Queen / Pink Floyd.

## How to verify
Ask for three well-known songs; the journal logs the chosen title and it is the original recording each time.

## Will it come back?
Occasionally for obscure queries — the heuristics are keyword-based. Extend the junk-word list if a new pattern shows up.
