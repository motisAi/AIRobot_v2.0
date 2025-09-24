#!/usr/bin/env python3
"""
SIM7600X Test Script
===================
Test the SIM7600X controller functionality with the connected module.

Run this to verify SIM7600X is working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hardware.sim7600x_controller import SIM7600XController
import logging
import time

def test_sim7600x():
    """Test SIM7600X controller functionality"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 SIM7600X Controller Test")
    print("=" * 40)
    
    try:
        # Create controller
        print("📱 Creating SIM7600X controller...")
        controller = SIM7600XController()
        
        print(f"   Port: {controller.port}")
        print(f"   Baudrate: {controller.baudrate}")
        print(f"   APN: {controller.apn}")
        
        # Test basic AT communication first
        print("\n🔌 Testing basic AT communication...")
        import serial
        
        ser = serial.Serial(controller.port, controller.baudrate, timeout=3.0)
        ser.write(b'AT\r\n')
        time.sleep(1.0)
        response = ser.read_all().decode('utf-8', errors='ignore')
        ser.close()
        
        if 'OK' in response:
            print("✅ Basic AT communication working")
        else:
            print(f"❌ AT communication failed: {response}")
            return False
        
        # Test controller connection
        print("\n📡 Testing controller connection...")
        if controller.connect():
            print("✅ SIM7600X controller connected!")
            
            # Get network status
            print("\n📶 Getting network status...")
            network = controller.get_network_status()
            
            print(f"   Signal strength: {network.signal_strength}/31")
            print(f"   Network registered: {controller.network_registered}")
            print(f"   Operator: {network.operator}")
            print(f"   Network type: {network.network_type}")
            
            # Test GPS
            print("\n🛰️ Testing GPS...")
            if controller.enable_gps():
                print("   GPS enabled - waiting for fix...")
                time.sleep(5)  # Wait a bit for GPS
                
                gps = controller.get_gps_data()
                if gps.valid:
                    print(f"   GPS Location: {gps.latitude}, {gps.longitude}")
                    print(f"   Satellites: {gps.satellites}")
                else:
                    print("   GPS fix not available yet (normal for indoor testing)")
            else:
                print("   GPS not available")
            
            # Show statistics
            print("\n📊 Statistics:")
            stats = controller.get_statistics()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Disconnect
            controller.disconnect()
            print("\n✅ Test completed successfully!")
            return True
            
        else:
            print("❌ Failed to connect controller")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sim7600x()
    sys.exit(0 if success else 1)