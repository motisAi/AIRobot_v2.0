# tests — smoke and invariant tests

Guardian runs `tests/` on every deploy (`deploy/deploy.sh` → `evolution/guardian.py`); a failing test blocks the
deploy and triggers the rollback. The `tests` step (`pytest tests/ -q`) is added to Guardian in the 2026-09-27 deep
clean — until it lands, Guardian's `imports` check mirrors `test_imports.py` and the other tests must be run by hand
before `deploy.sh`.

| Test | Checks | Status |
|---|---|---|
| `test_imports.py` | Every project module that `main.py` imports (directly or lazily) loads under the venv. Self‑updating: it parses `main.py`. | live |
| `test_no_orphans.py` | Every `.py` under `modules/` and `parts_used/` (except `__init__` / README) is imported by some other repo file. Catches dead modules that Guardian and `test_imports` never see (both derive their list from `main.py`). | planned |
| `test_config_keys.py` | Every key in `config/config.yaml` maps to a dataclass field in `config/settings.py` **and** its name appears in code outside `settings.py`. Blocks decoy keys (14 existed before the deep clean). | planned |
| `test_audio_threading.py` | With a fake PyAudio: a `MicStream` opened in thread A raises `RuntimeError` on `close()` from thread B; close from A succeeds; `open_input(None)` raises; `pause_listening()` returns False and opens no command mic when the wake thread never sets `_stream_closed_event`. | planned |
| `test_mic_resolution.py` | Mocked `/dev/snd/by-id` + PortAudio table: normal, renumbered card, device missing, table without `hw:N`. Expects the `hw:N`‑matching index; `None` (never a default) when missing; the re‑init path only when zero streams are open. | planned |
| `test_paths.py` | Every model path in config resolves to a file (bug_041 class). | planned |

```bash
venv/bin/python tests/test_imports.py          # plain
venv/bin/python -m pytest tests -q             # all tests (pytest in the venv)
```
Add a test when a bug had no test that would have caught it; name it after the bug
(`test_bug_026_hallucination_filter.py`). The invariants these tests protect are listed in
[docs/architecture/stella-architecture.md §4.4 and §12](../docs/architecture/stella-architecture.md); the live
checklist (journal greps, bench reproductions) is §13.
