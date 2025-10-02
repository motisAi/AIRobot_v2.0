#!/usr/bin/env python3
"""
Quick hardware test to find the best camera and microphone settings
"""

import cv2
import pyaudio
import sys

def test_camera():
    """Test camera devices"""
    print("Testing camera devices...")
    
    for i in range(5):  # Test first 5 camera indices
        print(f"Testing camera index {i}...")
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Camera {i}: Working - Resolution: {frame.shape}")
                else:
                    print(f"✗ Camera {i}: No frame")
                cap.release()
            else:
                print(f"✗ Camera {i}: Cannot open")
        except Exception as e:
            print(f"✗ Camera {i}: Error - {e}")
    
def test_microphones():
    """Test microphone devices"""
    print("\nTesting microphone devices...")
    
    try:
        p = pyaudio.PyAudio()
        
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Input device
                print(f"  Device {i}: {info['name']} - {info['maxInputChannels']} channels")
        
        p.terminate()
    except Exception as e:
        print(f"Error testing microphones: {e}")

if __name__ == "__main__":
    test_camera()
    test_microphones()
    
    print("\nRecommendations:")
    print("- Use camera index that showed 'Working'")
    print("- Use microphone device index for your preferred mic")
    print("- Update your .env file with these values")