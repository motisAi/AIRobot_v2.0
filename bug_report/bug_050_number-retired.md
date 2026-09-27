# Bug #050 — Number retired — never assigned (numbering gap closed 2026-09-27)

- **Date found:** n/a
- **Status:** retired
- **Area:** n/a
- **Files touched:** none
- **Commit(s):** n/a
- **Superseded by:** —
- **Reverted:** —

## Symptom
No bug. The 2026-09-19 audit created bug_047, bug_048, bug_049 and bug_051; number 050 was skipped and never assigned.

## Root cause
Numbering gap in the ledger. The rule is that numbers are sequential with no gaps so that a bug number cited in a commit
message can always be checked against `bug_report/`.

## Fix
This placeholder closes the gap. Do not reuse 050.

## How to verify
`ls bug_report/bug_0*.md | wc -l` counts every number from 001 to the latest with none missing.

## Will it come back?
No, if every fix commit carries its bug file with the next free number (architecture doc §12, rule 7).
