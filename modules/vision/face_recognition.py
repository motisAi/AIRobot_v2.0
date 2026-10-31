"""
Face Recognition Module
=======================
Handles face detection, recognition, and learning using DeepFace.
Manages the face database and provides real-time face identification.

Now uses the shared CameraManager instead of opening its own camera
device, eliminating camera conflicts with other vision modules.
"""

import cv2
import numpy as np
import logging
import time
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import queue

# Deep learning libraries (optional – heavy deps)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, SystemExit):
    DeepFace = None
    DEEPFACE_AVAILABLE = False

try:
    import face_recognition as face_recognition_lib
    FACE_RECOGNITION_AVAILABLE = True
except (ImportError, SystemExit):
    face_recognition_lib = None
    FACE_RECOGNITION_AVAILABLE = False

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, model_config, hardware_config, security_config, system_config
from core.robot_brain import RobotEvent


@dataclass
class Face:
    """Face data structure"""
    
    id: str                          # Unique face ID
    name: str                        # Person's name
    embeddings: List[np.ndarray]    # Face embeddings (multiple for accuracy)
    images: List[np.ndarray]        # Face images
    first_seen: datetime
    last_seen: datetime
    interaction_count: int = 0
    is_master: bool = False
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    

def encode_face_from_frame(frame_bgr: np.ndarray, box, target_face_px: int = 200,
                           max_scale: float = 3.0):
    """ONE face-encoding pipeline for recognition AND enrolment (tools/reenroll.py
    uses it too), so stored embeddings match what runtime produces.

    box = (x, y, w, h) from the detector. Takes a generously padded crop and —
    because the 120° lens makes faces small in pixels — upscales it so the face
    is ~target_face_px wide before dlib detects/aligns/encodes (its landmarks
    degrade below ~140 px). Falls back to forcing the detector box if dlib's own
    detector finds nothing in the crop. Returns a 128-d embedding or None.
    """
    if not FACE_RECOGNITION_AVAILABLE or frame_bgr is None or box is None:
        return None
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        return None
    fh, fw = frame_bgr.shape[:2]
    pady, padx = int(h * 0.6), int(w * 0.6)
    ctop, cbot = max(0, y - pady), min(fh, y + h + pady)
    cleft, cright = max(0, x - padx), min(fw, x + w + padx)
    crop = frame_bgr[ctop:cbot, cleft:cright]
    if crop.size == 0:
        return None
    scale = 1.0
    if w < target_face_px:
        scale = min(max_scale, float(target_face_px) / float(w))
    if scale > 1.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        encs = face_recognition_lib.face_encodings(rgb)
    except Exception:
        encs = []
    if not encs:
        # Force the (expanded) detector box, mapped into the scaled crop.
        bx, by = (x - cleft) * scale, (y - ctop) * scale
        bw, bh = w * scale, h * scale
        ch, cw = rgb.shape[:2]
        top = int(max(0, by - 0.35 * bh)); bot = int(min(ch, by + bh + 0.15 * bh))
        left = int(max(0, bx - 0.20 * bw)); right = int(min(cw, bx + bw + 0.20 * bw))
        if bot - top > 10 and right - left > 10:
            try:
                encs = face_recognition_lib.face_encodings(rgb, [(top, right, bot, left)])
            except Exception:
                encs = []
    return encs[0] if encs else None


def estimate_distance_m(face_w_px: float, frame_w_px: int, hfov_deg: float = 102.0,
                        face_width_m: float = 0.16) -> float:
    """Rough camera-to-face distance from the face box width (pinhole model).
    Average adult face ~0.16 m wide; the Camera Module 3 Wide has ~102° HFOV.
    Good to roughly ±20 % — enough for near/far behaviour, head/eye convergence
    and a sanity check on real depth sensors later."""
    try:
        if face_w_px <= 0 or frame_w_px <= 0:
            return 0.0
        f_px = (frame_w_px / 2.0) / float(np.tan(np.radians(hfov_deg / 2.0)))
        return round(float(face_width_m * f_px / face_w_px), 2)
    except Exception:
        return 0.0


class FaceRecognitionModule:
    """
    Face recognition system using DeepFace and face_recognition.
    Provides real-time face detection, recognition, and learning capabilities.
    """
    
    def __init__(self, brain=None, camera_manager=None):
        """
        Initialize face recognition module
        
        Args:
            brain: Reference to robot brain for event emission
            camera_manager: Shared CameraManager instance (preferred)
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Face Recognition Module")
        
        # Brain reference
        self.brain = brain
        
        # Shared camera manager (replaces per-module camera ownership)
        self.camera_manager = camera_manager
        
        # Configuration
        self.model_name = model_config.face_model
        self.backend = model_config.face_backend
        self.distance_metric = model_config.face_distance_metric
        self.threshold = model_config.face_recognition_threshold
        # Accept a match outright when this confident; otherwise require the
        # winner to beat the runner-up by match_margin, else -> unknown
        # (prevents Moti/Orr flip-flop when both land near the threshold).
        self.confident_distance = 0.50
        self.match_margin = 0.08
        # Upscale small faces to ~this width before encoding (120° wide lens).
        self.target_face_px = 200
        # No identity below this face width (see _process_frame).
        self.min_recognize_px = int(getattr(model_config, 'face_min_recognize_px', 70))
        self.camera_hfov_deg = float(getattr(hardware_config, 'camera_hfov_deg', 102.0))
        
        # Face database
        self.known_faces: Dict[str, Face] = {}
        self.database_path = Path(config.PROJECT_ROOT) / "data" / "faces" / "face_db.pkl"
        self.images_path = Path(config.PROJECT_ROOT) / "data" / "faces" / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing faces
        self.load_face_database()
        
        self.camera_index = hardware_config.camera_index
        self.resolution = hardware_config.camera_resolution
        self.fps = hardware_config.camera_fps
        
        # Processing variables
        self.current_frame = None
        self.processed_frame = None
        self.face_locations = []
        self.face_names = []
        self.processing = False
        self.running = False
        
        # Threading
        self.recognition_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Frame pacing: the shared CameraManager ALREADY delivers only every
        # system_config.frame_skip-th frame to subscribers, so we process every
        # frame we are given. (Skipping again here meant 1 frame in 64 — one
        # look every ~3.5 s — which made her slow to notice anyone.)
        self.frame_skip_counter = 0
        self.frame_skip = 1
        
        # Face tracking
        self.tracked_faces = {}
        self.face_tracker_timeout = 2.0
        
        # Learning mode
        self.learning_mode = False
        self.learning_face_id = None
        self.learning_samples = []

        # Liveness (anti-photo) monitor — created in start(); None = unavailable.
        self.liveness = None
        self.last_learn_reason = ''   # '' | 'not_live' | 'no_samples' | 'no_embeddings'


    def start(self):
        """Start face recognition system"""
        self.logger.info("Starting face recognition")
        self.running = True
        
        if not self.camera_manager:
            # The shared CameraManager is the ONLY camera owner (no cv2.VideoCapture
            # here — the legacy own-camera path was removed in the 2026-09-27 clean-up).
            self.logger.error("Face recognition needs the shared CameraManager — not started")
            return False
        self.camera_manager.subscribe("face_recognition", self._on_shared_frame)
        self.logger.info("Face recognition subscribed to shared camera")

        # Liveness monitor: FaceMesh blink check on the biggest face, in its own
        # thread at ~15 fps (we only look every Nth frame — too slow for blinks).
        if getattr(model_config, 'liveness_enabled', True):
            try:
                from modules.vision.liveness import LivenessMonitor
                mon = LivenessMonitor(
                    self.camera_manager,
                    fps=getattr(model_config, 'liveness_fps', 15),
                    blink_window_s=getattr(security_config, 'liveness_blink_window_s', 20.0),
                    decide_after_s=getattr(security_config, 'liveness_decide_after_s', 12.0),
                    logger=logging.getLogger("Liveness"))
                self.liveness = mon if mon.start() else None
            except Exception as exc:
                self.logger.warning("Liveness monitor unavailable: %s", exc)
                self.liveness = None

        # Start recognition thread (processes frames from queue)
        self.recognition_thread = threading.Thread(target=self._recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        self.logger.info("Face recognition started")
        return True
    
    def _on_shared_frame(self, frame):
        """Callback from shared CameraManager — push frame into queue."""
        if not self.running:
            return
        if not self.frame_queue.full():
            self.frame_queue.put(frame.image)
    
    def stop(self):
        """Stop face recognition system"""
        self.logger.info("Stopping face recognition")
        self.running = False
        
        # Unsubscribe from shared camera
        if self.camera_manager:
            self.camera_manager.unsubscribe("face_recognition")
        if self.liveness is not None:
            try:
                self.liveness.stop()
            except Exception:
                pass

        # Wait for the recognition thread
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        
        # Save database
        self.save_face_database()
        self.logger.info("Face recognition stopped")
    

    def _recognition_loop(self):
        """Process frames for face recognition"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.5)
                
                # Skip frames for performance
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.frame_skip != 0:
                    continue
                
                # Process frame
                self._process_frame(frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Recognition error: {e}")
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Adaptively brighten dark frames (gamma) for low-light face recog."""
        if not getattr(hardware_config, 'camera_low_light_boost', True):
            return frame
        try:
            mean = float(np.mean(frame))
            if mean >= 110:            # already bright enough
                return frame
            gamma = 0.5 if mean < 60 else 0.7   # smaller gamma = brighter
            table = ((np.linspace(0, 1, 256) ** gamma) * 255).astype('uint8')
            return cv2.LUT(frame, table)
        except Exception:
            return frame

    def _process_frame(self, frame: np.ndarray):
        """
        Process a single frame for face recognition

        Args:
            frame: Video frame to process
        """
        self.processing = True

        try:
            # Brighten dark frames for DETECTION only (adaptive gamma; well-lit
            # frames pass through unchanged). Encoding uses the untouched frame
            # so runtime embeddings match enrolment: tools/reenroll.py encodes
            # raw frames, and a gamma-lifted face encodes measurably differently
            # (enrolment self-test 0.11 became 0.45-0.50 live, and the lift
            # depends on the *background* brightness, not the face).
            det_frame = self._enhance_low_light(frame)
            # Detect faces
            faces = self._detect_faces(det_frame)
            
            if not faces:
                self.face_locations = []
                self.face_names = []
                self.processing = False
                return

            # Seed the liveness monitor with the biggest face so its FaceMesh
            # crop follows the person between our (every Nth frame) detections.
            if self.liveness is not None:
                big = max(faces, key=lambda f: f['area'][2] * f['area'][3])
                self.liveness.set_face_box(big['area'])

            # Recognize faces
            recognized_faces = []

            for face_data in faces:
                face_region = face_data['face']
                face_area = face_data['area']
                confidence = face_data.get('confidence', 0)

                # Identity needs pixels: below min_recognize_px (70 px ≈ 1.2 m on
                # the 120° lens) dlib guesses — a 30 px face at 2.8 m was called
                # "Moti" and a 56 px one "Orr". Tiny faces are still tracked and
                # drawn (presence, gaze) but get no identity and no event.
                too_small = face_area[2] < self.min_recognize_px
                if too_small:
                    person_id, similarity = 'unknown', 0.0
                else:
                    person_id, similarity = self._recognize_face(face_region, frame, face_area)
                # True = blinked recently, False = watched 12 s without a blink
                # (photo), None = undecided / monitor off. Consumers gate on it.
                live = self.liveness.is_live(face_area) if self.liveness is not None else None

                # Create result
                result = {
                    'id': person_id,
                    'name': self._get_person_name(person_id),
                    'location': face_area,
                    'confidence': similarity,
                    # Rough range from face size (pinhole model) — feeds the
                    # eyes/head later and is a sanity check for real depth.
                    'distance_m': estimate_distance_m(
                        face_area[2], frame.shape[1], self.camera_hfov_deg),
                    'live': live,
                    'timestamp': time.time()
                }
                
                recognized_faces.append(result)
                
                # Track face
                self._update_face_tracking(person_id, face_area)
                
                # Emit event if brain is connected (throttle: 1 event per face per 10s)
                if self.brain and not too_small:
                    now = time.time()
                    last_emit = getattr(self, '_last_emit_times', {})
                    if not hasattr(self, '_last_emit_times'):
                        self._last_emit_times = {}
                        last_emit = self._last_emit_times
                    emit_key = person_id if person_id != 'unknown' else 'unknown'
                    if emit_key not in last_emit or (now - last_emit[emit_key]) > 10:
                        last_emit[emit_key] = now
                        self.logger.info(
                            f"Face detected: {result['name']} (id={person_id}, "
                            f"sim={similarity:.2f}, w={face_area[2]}px, ~{result['distance_m']:.1f} m, "
                            f"live={'yes' if live else ('NO' if live is False else '?')})")
                        self.brain.emit_event(RobotEvent(
                            type='face_detected',
                            source='face_recognition',
                            data={
                                'face_id': person_id,
                                'name': result['name'],
                                'confidence': similarity,
                                'is_master': self._is_master(person_id),
                                'location': face_area,
                                'frame_size': (frame.shape[1], frame.shape[0]),
                                'distance_m': result['distance_m'],
                                'live': live,
                                'liveness_active': bool(self.liveness is not None and self.liveness.running),
                            },
                            priority=4
                        ))
                
                # Handle learning mode — store (full frame, face box) so we can
                # encode exactly like recognition does (a bare crop makes dlib's
                # compute_face_descriptor raise 'incompatible function arguments').
                if self.learning_mode and person_id == 'unknown' and not too_small:
                    self.learning_samples.append((frame.copy(), face_area))
            
            # Update results
            self.face_locations = [f['location'] for f in recognized_faces]
            self.face_names = [f['name'] for f in recognized_faces]
            self.processed_frame = self._draw_faces(frame.copy(), recognized_faces)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
        
        finally:
            self.processing = False
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame using configured backend
        
        Args:
            frame: Video frame
            
        Returns:
            List of detected faces with locations
        """
        faces = []
        
        try:
            if self.backend == 'yunet':
                return self._detect_yunet(frame)

            if self.backend == 'opencv':
                # Use OpenCV Haar Cascades (faster but less accurate)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                face_rects = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=8,
                    minSize=(80, 80)
                )
                
                if getattr(self, '_haar_eye', None) is None:
                    self._haar_eye = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_eye.xml')
                for (x, y, w, h) in face_rects:
                    # Phantom filter: a real frontal face shows at least one eye.
                    # Wall texture / paintings that fool Haar almost never do, so an
                    # eyeless box is dropped (kills a "new person" on a blank wall).
                    roi = gray[y:y + h, x:x + w]
                    eyes = self._haar_eye.detectMultiScale(
                        roi, scaleFactor=1.1, minNeighbors=6,
                        minSize=(max(12, w // 8), max(12, h // 8)))
                    if len(eyes) < 1:
                        continue
                    face_img = frame[y:y+h, x:x+w]
                    faces.append({
                        'face': face_img,
                        'area': (x, y, w, h),
                        'confidence': 0.9  # OpenCV doesn't provide confidence
                    })
            
            else:
                # Use DeepFace with specified backend
                if not DEEPFACE_AVAILABLE:
                    self.logger.warning("DeepFace not available, skipping non-opencv detection")
                    return faces
                detections = DeepFace.extract_faces(
                    img_path=frame,
                    target_size=(224, 224),
                    detector_backend=self.backend,
                    enforce_detection=False
                )
                
                for detection in detections:
                    if detection['confidence'] > model_config.face_detection_confidence:
                        face_img = (detection['face'] * 255).astype(np.uint8)
                        area = detection['area']
                        
                        faces.append({
                            'face': face_img,
                            'area': (area['x'], area['y'], area['w'], area['h']),
                            'confidence': detection['confidence']
                        })
        
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
        
        return faces
    
    _YUNET_MODEL = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "face_detection_yunet_2023mar.onnx"

    def _detect_yunet(self, frame: np.ndarray) -> List[Dict]:
        """DNN face detector (opencv FaceDetectorYN). Far fewer wall/painting
        false-positives than Haar; returns a per-face confidence. Falls back to
        Haar if the model is absent or the API errors, so detection never dies."""
        faces = []
        h, w = frame.shape[:2]
        if getattr(self, '_yunet', None) is None:
            if not self._YUNET_MODEL.exists():
                self.logger.warning("YuNet model missing (%s) — falling back to Haar", self._YUNET_MODEL)
                self.backend = 'opencv'
                return self._detect_faces(frame)
            self._yunet = cv2.FaceDetectorYN.create(
                str(self._YUNET_MODEL), "", (w, h), 0.7, 0.3, 50)
            self._yunet_size = (w, h)
            self.logger.info("YuNet DNN face detector active (%s)", self._YUNET_MODEL.name)
        if self._yunet_size != (w, h):
            self._yunet.setInputSize((w, h))
            self._yunet_size = (w, h)
        try:
            _, dets = self._yunet.detect(frame)
        except cv2.error as e:
            self.logger.error("YuNet detect error (%s) — falling back to Haar", e)
            self.backend = 'opencv'
            return self._detect_faces(frame)
        if dets is not None:
            for d in dets:
                x, y, bw, bh = (int(v) for v in d[:4])
                x, y = max(0, x), max(0, y)
                bw, bh = min(bw, w - x), min(bh, h - y)
                if bw <= 0 or bh <= 0:
                    continue
                faces.append({
                    'face': frame[y:y + bh, x:x + bw],
                    'area': (x, y, bw, bh),
                    'confidence': round(float(d[-1]), 3),
                })
        return faces

    def _recognize_face(self, face_image: np.ndarray,
                        full_frame: np.ndarray = None,
                        face_area: tuple = None) -> Tuple[str, float]:
        """
        Recognize a face against known faces
        
        Args:
            face_image: Face region image
            full_frame: Original full camera frame
            face_area: (x, y, w, h) face location in full_frame
            
        Returns:
            Tuple of (person_id, similarity_score)
        """
        if not self.known_faces:
            return ('unknown', 0.0)
        
        try:
            # ONE encoding pipeline shared with enrolment and tools/reenroll.py
            # (encode_face_from_frame): padded crop, upscaled when the face is
            # small (120° lens), then dlib detect+align+encode. Matching the
            # enrolment path keeps runtime distances near the enrolled
            # self-distance (~0.2) instead of drifting toward the threshold.
            embedding = None
            if FACE_RECOGNITION_AVAILABLE:
                if full_frame is not None and face_area is not None:
                    embedding = encode_face_from_frame(
                        full_frame, face_area, self.target_face_px)
                else:
                    # Fallback: try with the bare cropped face image
                    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    encs = face_recognition_lib.face_encodings(rgb_face)
                    embedding = encs[0] if encs else None

            if embedding is None:
                if not DEEPFACE_AVAILABLE:
                    return ('unknown', 0.0)
                # Fallback to DeepFace
                embedding = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False
                )[0]['embedding']
            
            # Compare with known faces: best distance PER PERSON, then a margin
            # gate so two people who both land near the threshold don't flip-flop
            # (that made her call Moti "Orr"). If the top two are too close she
            # returns unknown instead of guessing.
            per_person = {}
            for person_id, face_data in self.known_faces.items():
                dmin = float('inf')
                for known_embedding in face_data.embeddings:
                    if self.distance_metric == 'cosine':
                        distance = 1 - np.dot(embedding, known_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                        )
                    else:  # euclidean / euclidean_l2
                        distance = np.linalg.norm(embedding - known_embedding)
                    if distance < dmin:
                        dmin = distance
                per_person[person_id] = dmin
            
            if not per_person:
                return ('unknown', 0.0)
            
            ranked = sorted(per_person.items(), key=lambda kv: kv[1])
            best_match, best_distance = ranked[0]
            second_id, second_distance = ranked[1] if len(ranked) > 1 else (None, float('inf'))
            
            if best_distance < self.threshold:
                margin = second_distance - best_distance
                if best_distance <= self.confident_distance or margin >= self.match_margin:
                    similarity = max(0.0, 1.0 - best_distance)
                    self.known_faces[best_match].last_seen = datetime.now()
                    self.known_faces[best_match].interaction_count += 1
                    self.logger.debug("match %s d=%.3f (runner-up %s d=%.3f)",
                                      best_match, best_distance, second_id, second_distance)
                    return (best_match, similarity)
                self.logger.info(
                    "Face ambiguous: %s d=%.2f vs %s d=%.2f (margin<%.2f) -> unknown",
                    best_match, best_distance, second_id, second_distance, self.match_margin)
                return ('unknown', 0.0)
            
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
        
        return ('unknown', 0.0)
    
    def learn_face(self, name: str, samples: int = 10) -> bool:
        """
        Learn a new face
        
        Args:
            name: Name of the person
            samples: Number of samples to collect
            
        Returns:
            bool: True if successful
        """
        self.logger.info(f"Learning face for {name}")
        self.last_learn_reason = ''

        # Anti-photo: never enrol a face that does not blink. Returns as soon as
        # the liveness monitor has seen a blink (usually within a few seconds).
        if self.liveness is not None and getattr(security_config, 'liveness_required', True):
            wait = float(getattr(security_config, 'liveness_enroll_wait_s', 8.0))
            if not self.liveness.wait_for_blink(wait):
                self.last_learn_reason = 'not_live'
                self.logger.warning("learn_face(%s): no blink seen in %.0fs — not enrolling "
                                    "(a photo, or nobody in front of the camera)", name, wait)
                return False

        # Enter learning mode
        self.learning_mode = True
        self.learning_face_id = f"person_{int(time.time())}"
        self.learning_samples = []
        
        # Collect samples
        self.logger.info(f"Please look at the camera. Collecting {samples} samples...")
        
        start_time = time.time()
        timeout = 15  # seconds
        min_needed = max(3, samples // 3)

        while len(self.learning_samples) < samples and time.time() - start_time < timeout:
            time.sleep(0.3)

        # Exit learning mode
        self.learning_mode = False
        self.logger.info("Collected %d face sample(s) for %s",
                         len(self.learning_samples), name)

        if len(self.learning_samples) < min_needed:
            self.logger.warning("Not enough face samples (%d, need %d) — likely "
                                "poor lighting or no face in view",
                                len(self.learning_samples), min_needed)
            self.last_learn_reason = 'no_samples'
            return False
        
        # Process samples — each is (full_frame, (x, y, w, h))
        embeddings = []

        for sample in self.learning_samples:
            try:
                if not FACE_RECOGNITION_AVAILABLE:
                    break
                frame_img, box = sample
                # Same pipeline as recognition (and tools/reenroll.py), so the
                # stored embeddings match what runtime produces.
                enc = encode_face_from_frame(frame_img, box, self.target_face_px)
                if enc is not None:
                    embeddings.append(enc)
            except Exception as e:
                self.logger.error(f"Failed to process sample: {e}")
        
        if not embeddings:
            self.logger.error("No valid embeddings extracted")
            self.last_learn_reason = 'no_embeddings'
            return False
        
        # Create face entry
        new_face = Face(
            id=self.learning_face_id,
            name=name,
            embeddings=embeddings,
            images=[],  # don't store raw frames in the DB (keeps it small)
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            is_master=(name == security_config.master_user_id),
            permissions=['basic'] if name != security_config.master_user_id else ['all'],
            metadata={'learned_samples': len(self.learning_samples)}
        )
        
        # Add to database
        self.known_faces[self.learning_face_id] = new_face
        
        # Save to disk
        self.save_face_database()
        
        # Save sample images
        for i, img in enumerate(new_face.images[:3]):
            img_path = self.images_path / f"{self.learning_face_id}_{i}.jpg"
            cv2.imwrite(str(img_path), img)
        
        self.logger.info(f"Successfully learned face for {name} with ID {self.learning_face_id}")
        return True
    
    def forget_face(self, person_id: str) -> bool:
        """
        Remove a face from the database
        
        Args:
            person_id: ID of person to forget
            
        Returns:
            bool: True if successful
        """
        if person_id in self.known_faces:
            # Don't allow removing master
            if self.known_faces[person_id].is_master:
                self.logger.warning("Cannot remove master user")
                return False
            
            del self.known_faces[person_id]
            
            # Remove images
            for img_file in self.images_path.glob(f"{person_id}_*.jpg"):
                img_file.unlink()
            
            self.save_face_database()
            self.logger.info(f"Removed face {person_id}")
            return True
        
        return False
    
    def update_face_name(self, person_id: str, new_name: str) -> bool:
        """
        Update the name of a known face
        
        Args:
            person_id: ID of person
            new_name: New name
            
        Returns:
            bool: True if successful
        """
        if person_id in self.known_faces:
            self.known_faces[person_id].name = new_name
            self.save_face_database()
            self.logger.info(f"Updated name for {person_id} to {new_name}")
            return True
        
        return False
    
    def _get_person_name(self, person_id: str) -> str:
        """
        Get person's name from ID
        
        Args:
            person_id: Person ID
            
        Returns:
            Person's name or 'Unknown'
        """
        if person_id == 'unknown':
            return 'Unknown'
        
        if person_id in self.known_faces:
            return self.known_faces[person_id].name
        
        return 'Unknown'
    
    def _is_master(self, person_id: str) -> bool:
        """
        Check if person is master user
        
        Args:
            person_id: Person ID
            
        Returns:
            bool: True if master
        """
        if person_id in self.known_faces:
            return self.known_faces[person_id].is_master
        
        return False
    
    def _update_face_tracking(self, person_id: str, location: Tuple[int, int, int, int]):
        """
        Update face tracking information
        
        Args:
            person_id: Person ID
            location: Face location (x, y, w, h)
        """
        self.tracked_faces[person_id] = {
            'location': location,
            'last_seen': time.time(),
            'tracking_id': f"{person_id}_{int(time.time())}"
        }
        
        # Clean old tracked faces
        current_time = time.time()
        self.tracked_faces = {
            pid: data for pid, data in self.tracked_faces.items()
            if current_time - data['last_seen'] < self.face_tracker_timeout
        }
    
    def _draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw face detection results on frame
        
        Args:
            frame: Video frame
            faces: List of detected faces
            
        Returns:
            Frame with drawings
        """
        for face in faces:
            x, y, w, h = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Choose color based on recognition
            if face['id'] == 'unknown':
                color = (0, 0, 255)  # Red for unknown
            elif self._is_master(face['id']):
                color = (255, 215, 0)  # Gold for master
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def save_face_database(self):
        """Save face database to disk"""
        try:
            # Convert Face objects to serializable format
            save_data = {}
            for person_id, face in self.known_faces.items():
                save_data[person_id] = {
                    'id': face.id,
                    'name': face.name,
                    'embeddings': [emb.tolist() for emb in face.embeddings],
                    'first_seen': face.first_seen.isoformat(),
                    'last_seen': face.last_seen.isoformat(),
                    'interaction_count': face.interaction_count,
                    'is_master': face.is_master,
                    'permissions': face.permissions,
                    'metadata': face.metadata
                }
            
            # Save to pickle file
            with open(self.database_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Saved {len(self.known_faces)} faces to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save face database: {e}")
    
    def load_face_database(self):
        """Load face database from disk.
        
        If any entry was enrolled without embeddings (e.g. on Windows where
        dlib is hard to build), this method computes them from the saved
        sample images automatically.
        """
        if not self.database_path.exists():
            self.logger.info("No existing face database found")
            return
        
        try:
            # Load from pickle file
            with open(self.database_path, 'rb') as f:
                save_data = pickle.load(f)
            
            needs_save = False
            
            # Convert to Face objects
            for person_id, data in save_data.items():
                embeddings = [np.array(emb) for emb in data['embeddings']]
                
                # Auto-compute missing embeddings from saved images
                metadata = data.get('metadata', {})
                if not embeddings and metadata.get('needs_encoding', False):
                    self.logger.info(f"Computing embeddings for {data['name']} from saved images...")
                    computed = self._compute_embeddings_from_images(person_id)
                    if computed:
                        embeddings = computed
                        data['embeddings'] = [e.tolist() for e in embeddings]
                        metadata['needs_encoding'] = False
                        data['metadata'] = metadata
                        needs_save = True
                        self.logger.info(f"  Computed {len(embeddings)} embeddings for {data['name']}")
                
                face = Face(
                    id=data['id'],
                    name=data['name'],
                    embeddings=embeddings,
                    images=[],
                    first_seen=datetime.fromisoformat(data['first_seen']),
                    last_seen=datetime.fromisoformat(data['last_seen']),
                    interaction_count=data['interaction_count'],
                    is_master=data['is_master'],
                    permissions=data.get('permissions', ['basic']),
                    metadata=metadata
                )
                self.known_faces[person_id] = face
            
            self.logger.info(f"Loaded {len(self.known_faces)} faces from database")
            
            # Persist newly computed embeddings
            if needs_save:
                with open(self.database_path, 'wb') as f:
                    pickle.dump(save_data, f)
                self.logger.info("Saved updated embeddings to database")
            
        except Exception as e:
            self.logger.error(f"Failed to load face database: {e}")
    
    def _compute_embeddings_from_images(self, person_id: str) -> List[np.ndarray]:
        """Compute face embeddings from saved sample images on disk."""
        embeddings = []
        for img_path in sorted(self.images_path.glob(f"{person_id}_*.jpg")):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition_lib.face_encodings(rgb) if FACE_RECOGNITION_AVAILABLE else []
                if encodings:
                    embeddings.append(encodings[0])
            except Exception as e:
                self.logger.debug(f"Could not encode {img_path.name}: {e}")
        return embeddings
    

    def get_processed_frame(self) -> Optional[np.ndarray]:
        """
        Get the processed frame with face annotations
        
        Returns:
            Processed frame or None
        """
        return self.processed_frame
    
    def get_detected_faces(self) -> List[Dict[str, Any]]:
        """
        Get currently detected faces
        
        Returns:
            List of detected faces
        """
        faces = []
        for i, location in enumerate(self.face_locations):
            name = self.face_names[i] if i < len(self.face_names) else 'Unknown'
            faces.append({
                'name': name,
                'location': location,
                'timestamp': time.time()
            })
        
        return faces
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get module statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'running': self.running,
            'processing': self.processing,
            'known_faces_count': len(self.known_faces),
            'currently_detected': len(self.face_locations),
            'tracked_faces': len(self.tracked_faces),
            'learning_mode': self.learning_mode,
            'frame_skip': self.frame_skip,
            'camera_fps': self.fps,
            'resolution': self.resolution
        }
    
    def calibrate_recognition_threshold(self, test_samples: int = 20):
        """
        Calibrate recognition threshold using test samples
        
        Args:
            test_samples: Number of test samples to use
        """
        self.logger.info("Starting threshold calibration")
        
        if not self.known_faces:
            self.logger.warning("No known faces for calibration")
            return
        
        # Collect test samples
        samples = []
        start_time = time.time()
        
        while len(samples) < test_samples and time.time() - start_time < 30:
            if self.current_frame is not None:
                faces = self._detect_faces(self.current_frame)
                if faces:
                    samples.append(faces[0]['face'])
            time.sleep(0.5)
        
        if not samples:
            self.logger.warning("No samples collected for calibration")
            return
        
        # Test different thresholds
        thresholds = np.arange(0.2, 0.8, 0.05)
        results = []
        
        for threshold in thresholds:
            self.threshold = threshold
            correct = 0
            
            for sample in samples:
                person_id, _ = self._recognize_face(sample)
                # Assume the most common result is correct
                # (In practice, this would need ground truth)
                if person_id != 'unknown':
                    correct += 1
            
            accuracy = correct / len(samples)
            results.append((threshold, accuracy))
            self.logger.debug(f"Threshold {threshold:.2f}: Accuracy {accuracy:.2%}")
        
        # Find optimal threshold
        optimal_threshold = max(results, key=lambda x: x[1])[0]
        self.threshold = optimal_threshold
        
        self.logger.info(f"Calibration complete. Optimal threshold: {optimal_threshold:.2f}")
    
    def enable_continuous_capture(self):
        """Enable continuous capture mode for observation"""
        self.frame_skip = 1  # Process every frame
        self.logger.info("Continuous capture enabled")
    
    def disable_continuous_capture(self):
        """Disable continuous capture mode (the CameraManager does the skipping)"""
        self.frame_skip = 1
        self.logger.info("Continuous capture disabled")
    
    def shutdown(self):
        """Clean shutdown of the module"""
        self.stop()


# Import for system configuration
from config.settings import system_config

if __name__ == "__main__":
    """Test the face recognition module"""
    
    import sys
    
    # Create module instance
    face_module = FaceRecognitionModule()
    
    # Start module
    face_module.start()
    
    print("Face Recognition Module Test")
    print("=" * 50)
    print("Commands:")
    print("  l <name> - Learn a new face")
    print("  f <id>   - Forget a face")
    print("  s        - Show statistics")
    print("  c        - Calibrate threshold")
    print("  q        - Quit")
    print()
    
    try:
        while True:
            cmd = input("Command: ").strip().lower()
            
            if cmd.startswith('l '):
                name = cmd[2:].strip()
                if face_module.learn_face(name):
                    print(f"Successfully learned face for {name}")
                else:
                    print("Failed to learn face")
            
            elif cmd.startswith('f '):
                face_id = cmd[2:].strip()
                if face_module.forget_face(face_id):
                    print(f"Forgot face {face_id}")
                else:
                    print("Failed to forget face")
            
            elif cmd == 's':
                stats = face_module.get_statistics()
                print("\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
            
            elif cmd == 'c':
                face_module.calibrate_recognition_threshold()
            
            elif cmd == 'q':
                break
            
            # Show current detections
            faces = face_module.get_detected_faces()
            if faces:
                print(f"Currently detecting: {[f['name'] for f in faces]}")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Stop module
        face_module.stop()
        print("\nModule stopped")