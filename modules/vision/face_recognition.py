"""
Face Recognition Module
=======================
Handles face detection, recognition, and learning using DeepFace.
Manages the face database and provides real-time face identification.


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

# Deep learning libraries
from deepface import DeepFace
from deepface.commons import functions
import face_recognition

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, model_config, hardware_config, security_config


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
    

class FaceRecognitionModule:
    """
    Face recognition system using DeepFace and face_recognition.
    Provides real-time face detection, recognition, and learning capabilities.
    """
    
    def __init__(self, brain=None):
        """
        Initialize face recognition module
        
        Args:
            brain: Reference to robot brain for event emission
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Face Recognition Module")
        
        # Brain reference
        self.brain = brain
        
        # Configuration
        self.model_name = model_config.face_model
        self.backend = model_config.face_backend
        self.distance_metric = model_config.face_distance_metric
        self.threshold = model_config.face_recognition_threshold
        
        # Face database
        self.known_faces: Dict[str, Face] = {}
        self.database_path = Path(config.PROJECT_ROOT) / "data" / "faces" / "face_db.pkl"
        self.images_path = Path(config.PROJECT_ROOT) / "data" / "faces" / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing faces
        self.load_face_database()
        
        # Camera setup
        self.camera = None
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
        self.capture_thread = None
        self.recognition_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.frame_skip = system_config.frame_skip
        
        # Face tracking
        self.tracked_faces = {}  # Track faces across frames
        self.face_tracker_timeout = 2.0  # Seconds before face is considered lost
        
        # Learning mode
        self.learning_mode = False
        self.learning_face_id = None
        self.learning_samples = []
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera for face detection
        
        Returns:
            bool: True if successful
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, hardware_config.camera_buffer_size)
            
            # Test camera
            ret, frame = self.camera.read()
            if ret:
                self.logger.info(f"Camera initialized successfully at index {self.camera_index}")
                return True
            else:
                self.logger.error("Failed to read from camera")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start(self):
        """Start face recognition system"""
        self.logger.info("Starting face recognition")
        
        # Initialize camera if not already done
        if not self.camera:
            if not self.initialize_camera():
                self.logger.error("Cannot start without camera")
                return False
        
        # Set running flag
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self._recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        self.logger.info("Face recognition started")
        return True
    
    def stop(self):
        """Stop face recognition system"""
        self.logger.info("Stopping face recognition")
        
        self.running = False
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Save database
        self.save_face_database()
        
        self.logger.info("Face recognition stopped")
    
    def _capture_loop(self):
        """Continuously capture frames from camera"""
        while self.running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    
                    if ret:
                        # Update current frame
                        self.current_frame = frame
                        
                        # Add to queue if not full
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    else:
                        self.logger.warning("Failed to capture frame")
                        time.sleep(0.1)
                else:
                    self.logger.warning("Camera not available")
                    time.sleep(1.0)
                    
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
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
    
    def _process_frame(self, frame: np.ndarray):
        """
        Process a single frame for face recognition
        
        Args:
            frame: Video frame to process
        """
        self.processing = True
        
        try:
            # Detect faces
            faces = self._detect_faces(frame)
            
            if not faces:
                self.face_locations = []
                self.face_names = []
                self.processing = False
                return
            
            # Recognize faces
            recognized_faces = []
            
            for face_data in faces:
                face_region = face_data['face']
                face_area = face_data['area']
                confidence = face_data.get('confidence', 0)
                
                # Recognize face
                person_id, similarity = self._recognize_face(face_region)
                
                # Create result
                result = {
                    'id': person_id,
                    'name': self._get_person_name(person_id),
                    'location': face_area,
                    'confidence': similarity,
                    'timestamp': time.time()
                }
                
                recognized_faces.append(result)
                
                # Track face
                self._update_face_tracking(person_id, face_area)
                
                # Emit event if brain is connected
                if self.brain:
                    self.brain.emit_event({
                        'type': 'face_detected',
                        'source': 'face_recognition',
                        'data': {
                            'face_id': person_id,
                            'name': result['name'],
                            'confidence': similarity,
                            'is_master': self._is_master(person_id)
                        }
                    })
                
                # Handle learning mode
                if self.learning_mode and person_id == 'unknown':
                    self.learning_samples.append(face_region)
            
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
            if self.backend == 'opencv':
                # Use OpenCV Haar Cascades (faster but less accurate)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                face_rects = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in face_rects:
                    face_img = frame[y:y+h, x:x+w]
                    faces.append({
                        'face': face_img,
                        'area': (x, y, w, h),
                        'confidence': 0.9  # OpenCV doesn't provide confidence
                    })
            
            else:
                # Use DeepFace with specified backend
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
    
    def _recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face against known faces
        
        Args:
            face_image: Face region image
            
        Returns:
            Tuple of (person_id, similarity_score)
        """
        if not self.known_faces:
            return ('unknown', 0.0)
        
        try:
            # Get face embedding using face_recognition for speed
            face_encoding = face_recognition.face_encodings(face_image)
            
            if not face_encoding:
                # Fallback to DeepFace
                embedding = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False
                )[0]['embedding']
            else:
                embedding = face_encoding[0]
            
            # Compare with known faces
            best_match = None
            best_distance = float('inf')
            
            for person_id, face_data in self.known_faces.items():
                for known_embedding in face_data.embeddings:
                    # Calculate distance
                    if self.distance_metric == 'cosine':
                        distance = 1 - np.dot(embedding, known_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                        )
                    elif self.distance_metric == 'euclidean':
                        distance = np.linalg.norm(embedding - known_embedding)
                    else:  # euclidean_l2
                        distance = np.linalg.norm(embedding - known_embedding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = person_id
            
            # Check threshold
            if best_distance < self.threshold:
                similarity = 1 - (best_distance / self.threshold)
                
                # Update last seen
                self.known_faces[best_match].last_seen = datetime.now()
                self.known_faces[best_match].interaction_count += 1
                
                return (best_match, similarity)
            
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
        
        # Enter learning mode
        self.learning_mode = True
        self.learning_face_id = f"person_{int(time.time())}"
        self.learning_samples = []
        
        # Collect samples
        self.logger.info(f"Please look at the camera. Collecting {samples} samples...")
        
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while len(self.learning_samples) < samples and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        # Exit learning mode
        self.learning_mode = False
        
        if len(self.learning_samples) < samples // 2:
            self.logger.error(f"Not enough samples collected ({len(self.learning_samples)})")
            return False
        
        # Process samples
        embeddings = []
        
        for sample in self.learning_samples:
            try:
                # Get embedding
                encoding = face_recognition.face_encodings(sample)
                if encoding:
                    embeddings.append(encoding[0])
                    
            except Exception as e:
                self.logger.error(f"Failed to process sample: {e}")
        
        if not embeddings:
            self.logger.error("No valid embeddings extracted")
            return False
        
        # Create face entry
        new_face = Face(
            id=self.learning_face_id,
            name=name,
            embeddings=embeddings,
            images=self.learning_samples[:5],  # Keep first 5 images
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
        """Load face database from disk"""
        if not self.database_path.exists():
            self.logger.info("No existing face database found")
            return
        
        try:
            # Load from pickle file
            with open(self.database_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Convert to Face objects
            for person_id, data in save_data.items():
                face = Face(
                    id=data['id'],
                    name=data['name'],
                    embeddings=[np.array(emb) for emb in data['embeddings']],
                    images=[],  # Images loaded separately if needed
                    first_seen=datetime.fromisoformat(data['first_seen']),
                    last_seen=datetime.fromisoformat(data['last_seen']),
                    interaction_count=data['interaction_count'],
                    is_master=data['is_master'],
                    permissions=data.get('permissions', ['basic']),
                    metadata=data.get('metadata', {})
                )
                self.known_faces[person_id] = face
            
            self.logger.info(f"Loaded {len(self.known_faces)} faces from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load face database: {e}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current camera frame
        
        Returns:
            Current frame or None
        """
        return self.current_frame
    
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
        """Disable continuous capture mode"""
        self.frame_skip = model_config.frame_skip
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