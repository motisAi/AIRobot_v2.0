"""Liveness (anti-photo) monitor for the primary face.

A printed photo, or a phone held up to the camera, is *recognised* by the face
pipeline exactly like the real person — dlib only compares appearance. This
module watches the largest face at ~15 fps with MediaPipe FaceMesh (478
landmarks, eyelids included) and counts BLINKS through the Eye Aspect Ratio.
Real people blink every few seconds; a photo never does.

Design notes (measured on Stella's Pi 5, 2026-10-31)
- FaceMesh runs on a padded CROP of the face, never the full 1280x720 frame:
  on the wide lens it finds nothing in the full frame (face ~250 px of 1280)
  but returns 478 landmarks on the crop in ~20 ms.
- The crop is re-seeded from the face module's YuNet box every time that
  module processes a frame (every Nth frame) and self-tracked from the
  previous landmarks in between, so blinks are not missed between detections.
- Blink = a "V-dip" in ratio = EAR / local baseline (median of the EAR
  0.35-1.5 s earlier): a local minimum below `close_ratio`, with at least
  `min_low_frames` frames below 0.90 within +-0.2 s, and the ratio back above
  `open_ratio` both 0.15-0.5 s BEFORE and AFTER it. A blink is a 100-300 ms
  symmetric dip; single-frame landmark jitter has no width, and head pitch /
  glasses / looking down move the EAR slowly and one-sidedly (so the sliding
  baseline follows them). Decisions lag 0.5 s (the "after" context).
  Designed on recordings from Stella's own camera (2026-10-31, master at his
  usual 0.4 m, camera looking up): cued blinks dip to 0.40-0.60 (one shallow
  0.84), noise stays >= 0.85; the detector found 10 blinks / 40 s incl. 4-5
  of 5 cued ones. A fixed-threshold + hysteresis detector missed most of them.
- Verdict per face track: None (undecided: tracked < decide_after_s, or no
  face), True (a blink within blink_window_s), False (tracked >= decide_after_s
  with no blink => photo).
- Known limitation: a VIDEO playing on a phone blinks. Catching that needs a
  texture/moire model (MiniFASNet) or real depth (LiDAR) — planned.

Consumers: FaceRecognitionModule puts `live` on every face event / log line
and refuses `learn_face()` without a blink; RobotBrain / main.py do not grant
master privileges, disarm the guard or greet on a face that is not live.
"""
import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the venv
    mp = None
    MP_AVAILABLE = False

# FaceMesh landmark ids (p1..p6 per eye, EAR = (|p2p6| + |p3p5|) / (2 |p1p4|))
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP = 1

Box = Tuple[int, int, int, int]


def eye_aspect_ratio(pts: np.ndarray) -> float:
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0


class LivenessMonitor:
    """Blink-based liveness for the largest face in view (see module doc)."""

    def __init__(self, camera_manager, fps: int = 15, blink_window_s: float = 30.0,
                 decide_after_s: float = 15.0, close_ratio: float = 0.82,
                 open_ratio: float = 0.92, min_low_frames: int = 2,
                 logger: Optional[logging.Logger] = None):
        self.camera_manager = camera_manager
        self.fps = max(5, int(fps))
        self.blink_window_s = float(blink_window_s)
        self.decide_after_s = float(decide_after_s)
        self.close_ratio = float(close_ratio)
        self.open_ratio = float(open_ratio)
        self.min_low_frames = int(min_low_frames)
        self.logger = logger or logging.getLogger("Liveness")
        self.available = MP_AVAILABLE
        self.running = False

        self._lock = threading.Lock()
        self._mesh = None
        self._thread: Optional[threading.Thread] = None
        self._new_frame = threading.Event()
        self._frame = None
        self._frame_time = 0.0
        self._seed_box: Optional[Box] = None      # from the face module's detector
        self._seed_time = 0.0
        self._track_box: Optional[Box] = None     # from our own last landmarks
        self._track_time = 0.0
        self._miss = 0

        # per-track state
        self.track_start: Optional[float] = None
        self.last_seen = 0.0
        self.blink_times: deque = deque(maxlen=64)
        self.ear = 0.0
        self.baseline = 0.0                            # local (sliding) baseline
        self._ear_hist: deque = deque(maxlen=90)       # (t, ear), ~3-6 s
        self._ratio_hist: deque = deque(maxlen=900)    # (t, ear/baseline), ~60 s
        self._last_eval = 0.0                          # newest frame already judged for a V-dip
        self._nose: deque = deque(maxlen=45)
        self.frames_processed = 0
        self.frames_offered = 0          # camera callbacks seen (before rate limiting)
        self.proc_ms = 0.0               # last FaceMesh crop+process time
        self.mesh_hits = 0
        self._verdict_logged: Optional[bool] = None
        self._last_summary = time.time() - 45.0   # first summary ~15 s after start

    # ------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        if not MP_AVAILABLE:
            self.logger.warning("mediapipe not available — liveness monitor disabled")
            return False
        if self.camera_manager is None:
            self.logger.warning("no camera manager — liveness monitor disabled")
            return False
        try:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as exc:
            self.logger.warning("FaceMesh init failed (%s) — liveness monitor disabled", exc)
            return False
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="liveness", daemon=True)
        self._thread.start()
        self.logger.info("Liveness monitor started (FaceMesh blink check, %d fps, "
                         "window %.0fs, decide after %.0fs)",
                         self.fps, self.blink_window_s, self.decide_after_s)
        return True

    def stop(self):
        self.running = False
        self._new_frame.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._mesh is not None:
                self._mesh.close()
        except Exception:
            pass
        self._mesh = None

    # ------------------------------------------------------------ inputs
    def _on_frame(self, frame):
        """Feed a frame from outside (tests / tools). The service loop polls the
        CameraManager's latest frame itself: subscribers only receive every
        frame_skip-th frame (~3.7 fps), far too slow to catch a 150 ms blink."""
        if not self.running:
            return
        self.frames_offered += 1
        now = time.time()
        if now - self._frame_time < 1.0 / self.fps:
            return
        self._frame = frame.image
        self._frame_time = now
        self._new_frame.set()

    def set_face_box(self, box: Box, when: Optional[float] = None):
        """Seed/re-seed the crop from the face module's detector (x, y, w, h)."""
        with self._lock:
            self._seed_box = tuple(int(v) for v in box)
            self._seed_time = when or time.time()

    # ------------------------------------------------------------ outputs
    def is_live(self, box: Optional[Box] = None) -> Optional[bool]:
        """True = blinked within the window; False = watched long enough, no
        blink (photo); None = undecided / not tracking / different face."""
        if not self.running:
            return None
        now = time.time()
        if self.track_start is None or now - self.last_seen > 2.0:
            return None
        if box is not None and not self._same_face(box):
            return None
        if any(now - t <= self.blink_window_s for t in self.blink_times):
            return True
        if now - self.track_start >= self.decide_after_s:
            return False
        return None

    def wait_for_blink(self, timeout: float) -> bool:
        """Block up to `timeout` s until the tracked face blinks (enrolment gate)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_live() is True:
                return True
            time.sleep(0.1)
        return self.is_live() is True

    def ratio_stats(self) -> dict:
        """Distribution of ear/baseline over the last ~60 s (blink tuning)."""
        if len(self._ratio_hist) < 10:
            return {}
        r = np.asarray([x for _, x in self._ratio_hist])
        return {"n": int(r.size), "min": round(float(r.min()), 2),
                "p1": round(float(np.percentile(r, 1)), 2), "p5": round(float(np.percentile(r, 5)), 2),
                "lt_075": int((r < 0.75).sum()), "lt_085": int((r < 0.85).sum())}

    def status(self) -> dict:
        now = time.time()
        last_blink = max(self.blink_times) if self.blink_times else None
        motion = 0.0
        if len(self._nose) >= 10:
            arr = np.asarray(self._nose)
            motion = float(arr.std(axis=0).mean())
        return {
            "ratio_60s": self.ratio_stats(),
            "active": self.running,
            "tracking": self.track_start is not None and now - self.last_seen <= 2.0,
            "tracked_s": round(now - self.track_start, 1) if self.track_start else 0.0,
            "blinks": len(self.blink_times),
            "last_blink_age_s": round(now - last_blink, 1) if last_blink else None,
            "ear": round(self.ear, 3),
            "ear_baseline": round(self.baseline, 3),
            "head_motion": round(motion, 4),
            "verdict": self.is_live(),
            "frames": self.frames_processed,
            "frames_offered": self.frames_offered,
            "proc_ms": round(self.proc_ms, 1),
            "mesh_hits": self.mesh_hits,
        }

    # ------------------------------------------------------------ internals
    def _same_face(self, box: Box) -> bool:
        with self._lock:
            tb = self._track_box
        if tb is None:
            return True  # nothing to compare against yet
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        tx, ty = tb[0] + tb[2] / 2.0, tb[1] + tb[3] / 2.0
        tol = 0.75 * max(box[2], tb[2], 1)
        return abs(cx - tx) <= tol and abs(cy - ty) <= tol

    def _pick_box(self) -> Optional[Box]:
        now = time.time()
        with self._lock:
            seed, seed_t = self._seed_box, self._seed_time
            track, track_t = self._track_box, self._track_time
        if track is not None and now - track_t < 1.0 and (seed is None or track_t >= seed_t):
            return track
        if seed is not None and now - seed_t < 2.5:
            return seed
        if track is not None and now - track_t < 1.0:
            return track
        return None

    def _reset_track(self):
        self.track_start = None
        self.blink_times.clear()
        self._ear_hist.clear()
        self._ratio_hist.clear()
        self._nose.clear()
        self.baseline = 0.0
        self._last_eval = 0.0
        self._verdict_logged = None
        with self._lock:
            self._track_box = None

    def _next_frame(self, period: float):
        """Newest camera frame at up to `fps`: poll CameraManager.get_latest_frame()
        (updated every captured frame) or, without a camera manager, whatever
        _on_frame() delivered. Returns None when nothing new arrived in time."""
        if self.camera_manager is not None:
            deadline = time.time() + max(period, 0.2)
            while self.running and time.time() < deadline:
                try:
                    f = self.camera_manager.get_latest_frame()
                except Exception:
                    f = None
                img = getattr(f, "image", None)
                if img is not None and img is not self._frame:
                    self._frame = img
                    self.frames_offered += 1
                    return img
                time.sleep(period / 4.0)
            return None
        if self._new_frame.wait(0.5):
            self._new_frame.clear()
            return self._frame
        return None

    def _loop(self):
        period = 1.0 / self.fps
        while self.running:
            t0 = time.time()
            frame = self._next_frame(period)
            if frame is None:
                if self.track_start is not None and time.time() - self.last_seen > 2.0:
                    self._reset_track()
                continue
            box = self._pick_box()
            if box is None:
                if self.track_start is not None and time.time() - self.last_seen > 2.0:
                    self._reset_track()
            else:
                try:
                    tp = time.time()
                    self._process(frame, box)
                    self.proc_ms = (time.time() - tp) * 1000.0
                except Exception as exc:
                    self.logger.debug("liveness frame error: %s", exc)
            # pace to ~fps so FaceMesh never eats a whole core
            rest = period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    def _process(self, frame: np.ndarray, box: Box):
        x, y, w, h = box
        H, W = frame.shape[:2]
        pad = int(0.5 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return
        scale = 1.0
        if w < 160:  # FaceMesh wants a reasonably large face
            scale = min(3.0, 160.0 / max(w, 1))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        self.frames_processed += 1
        now = time.time()
        if not res.multi_face_landmarks:
            self._miss += 1
            if self._miss > self.fps:  # ~1 s without a mesh: drop the self-track
                with self._lock:
                    self._track_box = None
            return
        self._miss = 0
        self.mesh_hits += 1
        lm = res.multi_face_landmarks[0].landmark
        ch, cw = rgb.shape[:2]
        pts = np.array([(p.x * cw, p.y * ch) for p in lm], dtype=np.float32)
        xs = pts[:, 0] / scale + x0
        ys = pts[:, 1] / scale + y0
        bx0, by0, bx1, by1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        with self._lock:
            self._track_box = (int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0))
            self._track_time = now
        if self.track_start is None:
            self.track_start = now
        self.last_seen = now
        ear = 0.5 * (eye_aspect_ratio(pts[list(LEFT_EYE)]) + eye_aspect_ratio(pts[list(RIGHT_EYE)]))
        self._update_blink(ear, now)
        fw = max(1.0, bx1 - bx0)
        self._nose.append(((xs[NOSE_TIP] - bx0) / fw, (ys[NOSE_TIP] - by0) / fw))
        self._log_verdict(now)
        self._maybe_summary(now)

    def _maybe_summary(self, now: float):
        """One INFO line per minute while tracking (first after ~15 s) — enough
        to judge thresholds from the journal without DEBUG logging."""
        if now - self._last_summary < 60.0:
            return
        self._last_summary = now
        self.logger.info("Liveness: tracked %.0fs blinks=%d ear=%.3f base=%.3f mesh=%d/%d "
                         "(offered %d, %.0f ms/frame) ratio60s=%s verdict=%s",
                         now - (self.track_start or now), len(self.blink_times), self.ear,
                         self.baseline, self.mesh_hits, self.frames_processed,
                         self.frames_offered, self.proc_ms, self.ratio_stats(), self.is_live())

    def _update_blink(self, ear: float, now: float):
        self.ear = ear
        self._ear_hist.append((now, ear))
        # Local baseline: what the eyes looked like 0.35-1.5 s ago (the most
        # recent frames are excluded so a blink's own dip can't lower it).
        ref = [e for t, e in self._ear_hist if 0.35 <= now - t <= 1.5]
        if len(ref) < 3:
            return
        base = float(np.median(ref))
        self.baseline = base
        if base <= 1e-6:
            return
        self._ratio_hist.append((now, ear / base))
        self._detect_vdip(now)

    def _detect_vdip(self, now: float):
        """Judge every frame that is now >= 0.5 s old (so its 'after' context
        exists) exactly once: is it the bottom of a short, symmetric dip?"""
        pts = [(t, r) for t, r in self._ratio_hist if now - t <= 1.2]
        cands = [(t, r) for t, r in pts if self._last_eval < t <= now - 0.5]
        if not cands:
            return
        self._last_eval = cands[-1][0]
        for tc, rc in cands:
            if rc >= self.close_ratio:
                continue
            around = [r for t, r in pts if abs(t - tc) <= 0.35]
            if rc > min(around):
                continue                                   # not the local minimum
            near = [r for t, r in pts if abs(t - tc) <= 0.2]
            need = self.min_low_frames if len(near) >= 4 else 1  # low fps: 1 frame
            if sum(1 for r in near if r < 0.90) < need:
                continue
            before = [r for t, r in pts if 0.15 <= tc - t <= 0.5]
            after = [r for t, r in pts if 0.15 <= t - tc <= 0.5]
            if len(before) < 2 or len(after) < 2:
                continue
            if float(np.median(before)) < self.open_ratio or float(np.median(after)) < self.open_ratio:
                continue                                   # one-sided: pose change, not a blink
            last = self.blink_times[-1] if self.blink_times else 0.0
            if tc - last < 0.3:
                continue                                   # refractory: no double count
            self.blink_times.append(tc)
            self.logger.debug("blink %.2fs ago (ratio %.2f, base %.3f)", now - tc, rc, self.baseline)

    def _log_verdict(self, now: float):
        v = self.is_live()
        if v is None or v == self._verdict_logged:
            return
        self._verdict_logged = v
        if v:
            self.logger.info("Liveness: blink confirmed after %.1fs (blinks=%d)",
                             now - self.track_start, len(self.blink_times))
        else:
            self.logger.info("Liveness: NO blink in %.0fs — treating this face as not live (photo?)",
                             now - self.track_start)
