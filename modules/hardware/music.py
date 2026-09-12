"""Music playback for Stella — dependency-free (no mpv/PipeWire needed).

Pipeline: yt-dlp resolves a YouTube audio URL -> ffmpeg decodes it to raw PCM ->
a Python pump thread scales the volume live and writes to `aplay` on Stella's
speaker. Because we own the pump loop, we get live volume / pause / resume /
stop / duck with only tools already on the Pi (ffmpeg, aplay, numpy, yt-dlp).
Personal/home use.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time

import numpy as np

logger = logging.getLogger("Music")

RATE = 44100
CHANNELS = 2
CHUNK = 4096  # bytes per pump iteration (~23 ms)


class MusicPlayer:
    def __init__(self, alsa_device=None, ytdlp_path="yt-dlp", default_volume=70,
                 duck_volume=25, volume_step=20):
        self.alsa_device = alsa_device
        self.ytdlp = ytdlp_path
        self.default_volume = int(default_volume)
        self.duck_volume = int(duck_volume)
        self.volume_step = int(volume_step)

        self._ff = None          # ffmpeg (decoder) process
        self._yt = None          # yt-dlp (fetch) process -> pipes into ffmpeg
        self._ap = None          # aplay (output) process
        self._ap_lock = threading.Lock()
        self._out_paused = False  # output device released (e.g. Stella is speaking)
        self._thread = None
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()   # set == user-paused
        self._volume = int(default_volume)     # 0..150 (user-facing)
        self._factor = default_volume / 100.0  # applied gain
        self._ducked = False
        self._title = None

    # -- availability ------------------------------------------------------
    @property
    def available(self) -> bool:
        has_ytdlp = os.path.exists(self.ytdlp) or shutil.which(self.ytdlp) is not None
        return (shutil.which("ffmpeg") is not None
                and shutil.which("aplay") is not None and has_ytdlp)

    @property
    def title(self):
        return self._title

    # -- discovery / smart pick -------------------------------------------
    # Titles that are almost never the song the user asked for.
    _JUNK = ("tutorial", "lesson", "how to play", "cover", "guitar tab",
             "bass tab", " tab ", "tabs", "reaction", "karaoke", "backing track",
             "instrumental", "sped up", "slowed", "8 bit", "8-bit", "nightcore",
             "chords", "drum cover", "piano cover", "loop", "1 hour", "1hour")

    def _search_candidates(self, query: str, n: int = 5):
        """Return [(id, title, duration, channel)] for the top n search hits."""
        try:
            r = subprocess.run(
                [self.ytdlp, "--no-playlist", "--skip-download", "--no-warnings",
                 *self._YT_CLIENT, "--flat-playlist", "--print",
                 "%(id)s\t%(title)s\t%(duration)s\t%(channel)s",
                 f"ytsearch{n}:{query}"],
                capture_output=True, text=True, timeout=35)
            out = []
            for line in (r.stdout or "").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip():
                    vid = parts[0].strip()
                    title = parts[1].strip()
                    dur = parts[2].strip() if len(parts) > 2 else ""
                    chan = parts[3].strip() if len(parts) > 3 else ""
                    out.append((vid, title, dur, chan))
            return out
        except Exception as exc:
            logger.warning("search candidates failed: %s", exc)
            return []

    def _pick_best(self, query: str, cands):
        """Choose the best real-song candidate; skip tutorials/covers/etc."""
        if not cands:
            return None
        qwords = [w for w in query.lower().split() if len(w) > 2]
        want_live = "live" in query.lower()
        best, best_score = None, -1e9
        for i, (vid, title, dur, chan) in enumerate(cands):
            tl = title.lower(); cl = chan.lower()
            score = 100 - i * 5           # slight preference for higher rank
            if any(j in tl for j in self._JUNK):
                score -= 300
            if ("live" in tl) and not want_live:
                score -= 120
            if cl.endswith("- topic") or "official" in tl or "vevo" in cl:
                score += 60
            score += 8 * sum(1 for w in qwords if w in tl)  # matches request
            if score > best_score:
                best, best_score = (vid, title), score
        return best

    def search_title(self, query: str):
        if query.startswith("http"):
            return None
        picked = self._pick_best(query, self._search_candidates(query))
        return picked[1] if picked else None

    def _resolve_url(self, query: str):
        # Direct URL -> resolve straight through.
        if query.startswith("http"):
            target = query
        else:
            picked = self._pick_best(query, self._search_candidates(query))
            if picked:
                self._picked_title = picked[1]
                target = "https://www.youtube.com/watch?v=" + picked[0]
            else:
                target = f"ytsearch1:{query}"   # last-resort fallback
        try:
            r = subprocess.run(
                [self.ytdlp, "-f", "bestaudio/best", "--no-playlist",
                 "--no-warnings", "-g", target],
                capture_output=True, text=True, timeout=40)
            url = (r.stdout or "").strip().split("\n")[-1].strip()
            return url or None
        except Exception as exc:
            logger.warning("resolve url failed: %s", exc)
            return None

    # YouTube client that needs no PO token (default web client 403s the stream).
    _YT_CLIENT = ["--extractor-args", "youtube:player_client=android"]

    def _pick_target(self, query: str):
        """Return a concrete YouTube watch URL (or ytsearch fallback) for the
        best real-song match, and remember its title."""
        if query.startswith("http"):
            return query
        picked = self._pick_best(query, self._search_candidates(query))
        if picked:
            self._picked_title = picked[1]
            return "https://www.youtube.com/watch?v=" + picked[0]
        return f"ytsearch1:{query}"

    # -- playback ----------------------------------------------------------
    def play(self, query: str, title=None, volume=None) -> bool:
        self.stop()
        target = self._pick_target(query)
        if not target:
            return False
        vol = self.default_volume if volume is None else int(volume)
        self._volume = vol
        self._factor = vol / 100.0
        self._ducked = False
        self._title = title or getattr(self, "_picked_title", None) or query
        try:
            # yt-dlp fetches (android client, no 403) and streams to stdout;
            # ffmpeg decodes that pipe to raw PCM for our volume pump.
            self._yt = subprocess.Popen(
                [self.ytdlp, "-f", "bestaudio/best", "--no-playlist",
                 "--no-warnings", *self._YT_CLIENT, "-o", "-", target],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._ff = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-i", "pipe:0",
                 "-f", "s16le", "-ar", str(RATE), "-ac", str(CHANNELS), "pipe:1"],
                stdin=self._yt.stdout, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL)
            self._yt.stdout.close()  # let ffmpeg own the read end (SIGPIPE on stop)
            self._ap = self._start_aplay()
        except Exception as exc:
            logger.error("failed to start yt-dlp/ffmpeg/aplay: %s", exc)
            self._terminate_procs()
            return False
        self._stop_evt.clear()
        self._pause_evt.clear()
        self._out_paused = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        logger.info("playing: %s (vol %d)", self._title, vol)
        return True

    def _start_aplay(self):
        ap = ["aplay", "-q", "-t", "raw", "-f", "S16_LE",
              "-r", str(RATE), "-c", str(CHANNELS)]
        if self.alsa_device:
            ap += ["-D", self.alsa_device]
        return subprocess.Popen(ap, stdin=subprocess.PIPE)

    def _pump(self):
        try:
            while not self._stop_evt.is_set():
                data = self._ff.stdout.read(CHUNK)
                if not data:
                    break  # stream ended
                # Hold the chunk while user-paused OR while the output device is
                # released for Stella to speak (no audio lost — ffmpeg blocks).
                while ((self._pause_evt.is_set() or self._out_paused)
                       and not self._stop_evt.is_set()):
                    time.sleep(0.03)
                if self._stop_evt.is_set():
                    break
                f = self._factor
                if abs(f - 1.0) > 0.001:
                    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) * f
                    np.clip(a, -32768, 32767, out=a)
                    data = a.astype(np.int16).tobytes()
                with self._ap_lock:
                    ap = self._ap
                if ap is None or ap.stdin is None:
                    continue
                try:
                    ap.stdin.write(data)
                except (BrokenPipeError, ValueError, OSError):
                    # Intentional device release (speaking) or stop -> keep going.
                    if self._out_paused or self._stop_evt.is_set():
                        continue
                    break
        finally:
            self._terminate_procs()
            self._title = None

    # -- release/reacquire the speaker so Stella can talk over the music ----
    def pause_output(self):
        """Free the ALSA device (e.g. while Stella speaks) — music stays queued."""
        if not self.is_playing():
            return
        with self._ap_lock:
            self._out_paused = True
            ap = self._ap
            self._ap = None
        if ap:
            try:
                if ap.stdin:
                    ap.stdin.close()
            except Exception:
                pass
            try:
                if ap.poll() is None:
                    ap.terminate()
            except Exception:
                pass

    def resume_output(self):
        """Re-open the speaker after Stella finished talking."""
        if not self.is_playing():
            return
        with self._ap_lock:
            if self._out_paused:
                try:
                    self._ap = self._start_aplay()
                except Exception as exc:
                    logger.error("could not reopen aplay: %s", exc)
                    self._ap = None
                self._out_paused = False

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- live control ------------------------------------------------------
    def pause(self):
        self._pause_evt.set()

    def resume(self):
        self._pause_evt.clear()

    def set_volume(self, vol: int) -> int:
        self._volume = max(0, min(150, int(vol)))
        if not self._ducked:
            self._factor = self._volume / 100.0
        return self._volume

    def change_volume(self, delta: int) -> int:
        return self.set_volume(self._volume + delta)

    def louder(self):
        return self.change_volume(self.volume_step)

    def quieter(self):
        return self.change_volume(-self.volume_step)

    def duck(self):
        if self.is_playing() and not self._ducked:
            self._ducked = True
            self._factor = self.duck_volume / 100.0

    def unduck(self):
        if self.is_playing() and self._ducked:
            self._ducked = False
            self._factor = self._volume / 100.0

    def stop(self):
        self._stop_evt.set()
        self._pause_evt.clear()
        # Terminate procs first so a blocked stdin.write unblocks, then join.
        self._terminate_procs()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def _terminate_procs(self):
        for p in (self._ap, self._ff, self._yt):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        try:
            if self._ap and self._ap.stdin:
                self._ap.stdin.close()
        except Exception:
            pass
