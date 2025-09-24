"""
Configuration Settings Module - No Hailo Version
================================================
Settings optimized to work without Hailo accelerator.
Uses CPU-based models for all AI tasks.

"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class ModelConfigNoHailo:
    """AI Model configurations for CPU-only operation"""
    
    # Face Recognition Settings (CPU optimized)
    face_model: str = "VGG-Face"  # Lighter than Facenet
    face_backend: str = "opencv"  # Fastest backend
    face_distance_metric: str = "cosine"
    face_recognition_threshold: float = 0.4
    face_detection_confidence: float = 0.7
    
    # Object Detection Settings (CPU version)
    # Instead of YOLO on Hailo, use lighter alternatives
    object_detection_method: str = "mobilenet"  # Options: mobilenet, cascade, none
    object_model: str = "ssd_mobilenet_v2"  # Lightweight model
    object_confidence_threshold: float = 0.5
    object_nms_threshold: float = 0.4
    object_max_detections: int = 50  # Reduced for CPU
    use_hailo: bool = False  # Explicitly disable Hailo
    
    # Speech Recognition (CPU optimized)
    whisper_model: str = "tiny"  # Use tiny model for CPU (base is too heavy)
    whisper_language: str = "en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # Fastest computation
    
    # Text-to-Speech (lightweight)
    tts_use_lite_model: bool = True  # Use lighter TTS models
    tts_model: str = "tts_models/en/ljspeech/glow-tts"  # Lighter than tacotron
    tts_vocoder: str = None  # Skip vocoder for speed
    tts_speed: float = 1.0
    
    # Wake Word Detection
    wake_word: str = "hey robot"
    wake_word_sensitivity: float = 0.5
    picovoice_access_key: str = os.getenv("PICOVOICE_ACCESS_KEY", "")


@dataclass
class SystemConfigOptimized:
    """System settings optimized for Raspberry Pi without Hailo"""
    
    # Performance Settings (CPU-optimized)
    max_threads: int = 4  # Raspberry Pi 5 has 4 cores
    vision_thread_count: int = 1  # Reduce threads
    audio_thread_count: int = 1
    enable_gpu: bool = False
    enable_hailo: bool = False  # Explicitly disabled
    
    # Processing Optimization (more aggressive)
    frame_skip: int = 5  # Process every 5th frame (was 3)
    batch_processing: bool = False  # Disable batch processing
    model_quantization: bool = True  # Always use quantized models
    cache_enabled: bool = True
    cache_size_mb: int = 256  # Reduced cache
    
    # Reduced processing intervals
    vision_process_interval: float = 0.2  # 200ms (was 100ms)
    audio_process_interval: float = 0.1  # 100ms (was 50ms)
    
    # Lower resolution for better performance
    camera_resolution: tuple = (320, 240)  # Reduced from 640x480
    camera_fps: int = 15  # Reduced from 30
    
    # Memory Management (more conservative)
    max_memory_percent: float = 70.0  # Lower threshold
    clear_cache_threshold: float = 60.0


class SimplifiedObjectDetection:
    """
    Simplified object detection without Hailo
    Uses OpenCV's built-in cascades and MobileNet
    """
    
    def __init__(self):
        """Initialize CPU-based object detection"""
        import cv2
        
        # Use Haar Cascades for specific objects
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # For general object detection, use MobileNet SSD
        self.use_mobilenet = False
        self.net = None
        self.classes = []
        
        # Try to load MobileNet if available
        try:
            prototxt = "models/MobileNetSSD_deploy.prototxt"
            model = "models/MobileNetSSD_deploy.caffemodel"
            
            if Path(prototxt).exists() and Path(model).exists():
                self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
                self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                              "sofa", "train", "tvmonitor"]
                self.use_mobilenet = True
                print("MobileNet loaded for object detection")
        except:
            print("MobileNet not available, using basic detection only")
    
    def detect_objects(self, frame):
        """
        Detect objects in frame using CPU
        
        Args:
            frame: Input image
            
        Returns:
            List of detected objects
        """
        import cv2
        import numpy as np
        
        detections = []
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            detections.append({
                'class': 'person',
                'confidence': 0.8,
                'bbox': (x, y, w, h)
            })
        
        # Use MobileNet if available
        if self.use_mobilenet and self.net:
            # Prepare the frame
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                0.007843, (300, 300), 127.5
            )
            
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            # Process detections
            for i in range(outputs.shape[2]):
                confidence = outputs[0, 0, i, 2]
                
                if confidence > 0.5:
                    idx = int(outputs[0, 0, i, 1])
                    if idx < len(self.classes):
                        box = outputs[0, 0, i, 3:7] * np.array(
                            [frame.shape[1], frame.shape[0], 
                             frame.shape[1], frame.shape[0]]
                        )
                        (x, y, x2, y2) = box.astype("int")
                        
                        detections.append({
                            'class': self.classes[idx],
                            'confidence': float(confidence),
                            'bbox': (x, y, x2-x, y2-y)
                        })
        
        return detections


# Performance tips for running without Hailo
OPTIMIZATION_TIPS = """
Performance Optimization Tips (No Hailo):
=========================================

1. **Use Smaller Models:**
   - Whisper: Use 'tiny' instead of 'base'
   - Face Recognition: Use 'opencv' backend
   - TTS: Use glow-tts instead of tacotron2

2. **Reduce Processing:**
   - Increase frame_skip to 5-10
   - Lower camera resolution to 320x240
   - Reduce FPS to 10-15

3. **Disable Features:**
   - Turn off continuous observation
   - Disable batch processing
   - Reduce number of tracked objects

4. **System Optimization:**
   - Enable GPU memory split (Raspberry Pi):
     sudo raspi-config > Advanced > Memory Split > 128
   - Increase swap file:
     sudo dphys-swapfile swapconf
   - Overclock CPU (if cooling available):
     sudo raspi-config > Overclock

5. **Alternative Approaches:**
   - Use cloud APIs for heavy processing
   - Implement edge-cloud hybrid approach
   - Use simpler algorithms (Haar Cascades vs Neural Networks)
"""


def check_system_compatibility():
    """
    Check if system can run without Hailo
    
    Returns:
        Dict with compatibility information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.machine(),
        'processor': platform.processor(),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'cpu_cores': psutil.cpu_count(),
        'python_version': platform.python_version(),
        'is_raspberry_pi': 'arm' in platform.machine().lower(),
        'can_run': True,
        'warnings': [],
        'suggestions': []
    }
    
    # Check minimum requirements
    if info['ram_gb'] < 2:
        info['warnings'].append("Low RAM: Less than 2GB available")
        info['suggestions'].append("Reduce model sizes and resolution")
    
    if info['cpu_cores'] < 4:
        info['warnings'].append(f"Only {info['cpu_cores']} CPU cores available")
        info['suggestions'].append("Increase frame_skip and reduce threads")
    
    # Check for Raspberry Pi specific
    if info['is_raspberry_pi']:
        info['suggestions'].append("Enable GPU memory split: sudo raspi-config")
        info['suggestions'].append("Consider overclocking if cooling available")
    
    # Check Python version
    if platform.python_version() < '3.8':
        info['warnings'].append("Python version < 3.8, some features may not work")
        info['can_run'] = False
    
    return info


# Create simplified config class
class RobotConfigNoHailo:
    """Configuration without Hailo support"""
    
    def __init__(self):
        """Initialize configuration for CPU-only operation"""
        self.model = ModelConfigNoHailo()
        self.system = SystemConfigOptimized()
        
        # Use the original configs for non-performance critical settings
        from config.settings import (
            HardwareConfig, 
            SecurityConfig, 
            BehaviorConfig
        )
        
        self.hardware = HardwareConfig()
        # Override camera settings for better CPU performance
        self.hardware.camera_resolution = (320, 240)
        self.hardware.camera_fps = 15
        
        self.security = SecurityConfig()
        self.behavior = BehaviorConfig()
        
        # Disable Hailo explicitly
        self.hardware.hailo_device_id = None
        self.hardware.enable_hailo = False
        
        # Create directories
        self._create_directories()
        
        # Check system
        self.compatibility = check_system_compatibility()
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            PROJECT_ROOT / "data" / "models",
            PROJECT_ROOT / "data" / "faces",
            PROJECT_ROOT / "data" / "voices",
            PROJECT_ROOT / "data" / "logs",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def print_optimization_status(self):
        """Print optimization status and recommendations"""
        print("\n" + "=" * 60)
        print("   RUNNING WITHOUT HAILO ACCELERATOR")
        print("=" * 60)
        
        print("\nSystem Compatibility Check:")
        for key, value in self.compatibility.items():
            if key not in ['warnings', 'suggestions']:
                print(f"  {key}: {value}")
        
        if self.compatibility['warnings']:
            print("\n⚠ Warnings:")
            for warning in self.compatibility['warnings']:
                print(f"  - {warning}")
        
        if self.compatibility['suggestions']:
            print("\n💡 Suggestions:")
            for suggestion in self.compatibility['suggestions']:
                print(f"  - {suggestion}")
        
        print("\nOptimizations Applied:")
        print(f"  ✓ Using lightweight models (Whisper: {self.model.whisper_model})")
        print(f"  ✓ Reduced resolution: {self.system.camera_resolution}")
        print(f"  ✓ Frame skip: {self.system.frame_skip}")
        print(f"  ✓ CPU-only processing")
        
        if not self.compatibility['can_run']:
            print("\n❌ System may not meet minimum requirements!")
        else:
            print("\n✅ System ready for CPU-only operation")
        
        print("\n" + "=" * 60)


# Example usage for downloading lightweight models
def download_lightweight_models():
    """
    Download lightweight models for CPU operation
    """
    import os
    import urllib.request
    
    models_dir = PROJECT_ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # MobileNet SSD for object detection (lightweight)
    mobilenet_files = {
        "MobileNetSSD_deploy.prototxt": 
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel": 
            "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    }
    
    print("Downloading lightweight models...")
    
    for filename, url in mobilenet_files.items():
        filepath = models_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ {filename} downloaded")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                print(f"    You can manually download from: {url}")


if __name__ == "__main__":
    """Test configuration without Hailo"""
    
    # Create config
    config = RobotConfigNoHailo()
    
    # Print status
    config.print_optimization_status()
    
    # Print optimization tips
    print(OPTIMIZATION_TIPS)
    
    # Test object detection
    print("\nTesting CPU-based object detection...")
    detector = SimplifiedObjectDetection()
    
    print("\nConfiguration complete for CPU-only operation!")