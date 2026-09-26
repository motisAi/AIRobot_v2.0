# tests — smoke tests

| Test | Checks |
|---|---|
| `test_imports.py` | Every project module that `main.py` imports (directly or lazily) loads under the venv. Self‑updating: it parses `main.py`. |

```bash
venv/bin/python tests/test_imports.py          # plain
venv/bin/python -m pytest tests -q             # if pytest is installed
```
Guardian (`evolution/guardian.py`) runs the same check as part of every deploy.
Add a test when a bug had no test that would have caught it; name it after the bug
(`test_bug_026_hallucination_filter.py`).
