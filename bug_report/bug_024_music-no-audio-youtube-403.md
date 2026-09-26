# Bug #024 — Music: no sound at all — YouTube 403s the raw stream URL

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/media/music.py`; `~/.deno` (deno 2.9.6 installed as yt-dlp JS runtime)
- **Commit(s):** e2cc602

## Symptom
Stella announced the song but nothing played ("it's not playing"); every play silently failed.

## Root cause
YouTube now requires a PO token for the default web client, so the raw stream URL that yt-dlp handed to ffmpeg returned HTTP 403.

## Fix
yt-dlp fetches with the **ANDROID** player client (no PO token needed) and **pipes** audio into ffmpeg, so yt-dlp performs the HTTP with the right headers. deno 2.9.6 was installed to `~/.deno` via a Python unzip (apt/unzip unavailable) for future extractor needs; the android client works without it.

## How to verify
Ask for a song: continuous playback, correct title in the journal, no `HTTP Error 403` lines.

## Will it come back?
Yes, likely — YouTube changes its client requirements regularly. First step: `pip install -U yt-dlp` in the venv; then re-check which player client still works.
