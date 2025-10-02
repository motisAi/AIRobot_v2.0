#!/usr/bin/env python3
"""
Simple test script for minimal AI Robot system
Tests only the core functionality with SIM7600X
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def test_basic_imports():
    """Test if all basic imports work"""
    print("Testing basic imports...")
    
    try:
        from config.settings import config, behavior_config
        print("✓ Configuration imports successful")
    except Exception as e:
        print(f"✗ Configuration import failed: {e}")
        return False
    
    try:
        from complete_integration import integrate_everything
        print("✓ Integration imports successful")
    except Exception as e:
        print(f"✗ Integration import failed: {e}")
        return False
    
    return True

def test_groq_connection():
    """Test Groq API connection"""
    print("\nTesting Groq API connection...")
    
    try:
        from complete_integration import GroqIntegration
        groq = GroqIntegration()
        
        if groq.client:
            # Test a simple chat
            response = groq.chat("Hello, are you working?")
            print(f"✓ Groq API working: {response[:50]}...")
            return True
        else:
            print("✗ Groq API key not configured")
            return False
    except Exception as e:
        print(f"✗ Groq API test failed: {e}")
        return False

def test_sim7600x():
    """Test SIM7600X connection"""
    print("\nTesting SIM7600X connection...")
    
    try:
        from modules.hardware.sim7600x_controller import SIM7600XController
        
        # Just try to create the controller (don't connect yet)
        controller = SIM7600XController()
        print("✓ SIM7600X controller created successfully")
        return True
    except Exception as e:
        print(f"✗ SIM7600X controller failed: {e}")
        return False

def test_minimal_robot():
    """Test creating robot with minimal configuration"""
    print("\nTesting minimal robot initialization...")
    
    try:
        from main import AIRobot
        
        # Create robot with minimal config
        robot = AIRobot(config_file=str(PROJECT_ROOT / "config" / "minimal_config.json"))
        print("✓ Robot created successfully")
        
        # Initialize modules (should skip missing hardware)
        robot.initialize_modules()
        print("✓ Module initialization completed")
        
        print(f"Active modules: {list(robot.modules.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Robot initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("AI Robot Minimal System Test")
    print("=" * 40)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        test_basic_imports,
        test_groq_connection,
        test_sim7600x,
        test_minimal_robot
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\nTest Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()