#!/usr/bin/env python3
"""
SIM7600X Network Connectivity Test
==================================
Comprehensive test to verify SIM7600X 4G connectivity and compare with WiFi
"""

import sys
import time
import subprocess
import socket
import requests
import json
from pathlib import Path

def run_cmd(cmd, timeout=10):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_network_interfaces():
    """Check network interfaces and their IPs"""
    print("🌐 Network Interface Analysis")
    print("-" * 40)
    
    success, output, error = run_cmd("ip addr show")
    if not success:
        print(f"❌ Failed to get interfaces: {error}")
        return {}
    
    interfaces = {}
    current_iface = None
    
    for line in output.split('\n'):
        if ': ' in line and 'inet' not in line:
            parts = line.split(': ')
            if len(parts) >= 2:
                current_iface = parts[1].split('@')[0]
                state = "UP" if "state UP" in line else "DOWN"
                interfaces[current_iface] = {'state': state, 'ips': []}
                
                # Color code the status
                status_icon = "🟢" if state == "UP" else "🔴"
                print(f"{status_icon} {current_iface}: {state}")
                
        elif 'inet ' in line and current_iface:
            ip = line.strip().split()[1]
            interfaces[current_iface]['ips'].append(ip)
            print(f"  📍 {ip}")
    
    return interfaces

def test_routing_table():
    """Check routing table to see which interface is default"""
    print("\n🛣️ Routing Table Analysis")
    print("-" * 40)
    
    success, output, error = run_cmd("ip route show")
    if not success:
        print(f"❌ Failed to get routes: {error}")
        return
    
    default_routes = []
    for line in output.split('\n'):
        if 'default' in line:
            default_routes.append(line.strip())
            print(f"🛤️ {line.strip()}")
    
    if len(default_routes) > 1:
        print("ℹ️ Multiple default routes found - traffic may use both WiFi and SIM7600X")
    elif len(default_routes) == 1:
        print("ℹ️ Single default route")
    else:
        print("⚠️ No default route found")

def test_public_ip():
    """Test what public IP we're using"""
    print("\n📍 Public IP Detection")
    print("-" * 40)
    
    ip_services = [
        "http://httpbin.org/ip",
        "http://ipinfo.io/ip",
        "http://api.ipify.org",
        "http://icanhazip.com"
    ]
    
    results = {}
    
    for service in ip_services:
        try:
            print(f"🔄 Testing {service}...")
            response = requests.get(service, timeout=10)
            
            if response.status_code == 200:
                # Extract IP from different response formats
                content = response.text.strip()
                if service == "http://httpbin.org/ip":
                    try:
                        ip = json.loads(content)['origin']
                    except:
                        ip = content
                else:
                    ip = content
                
                results[service] = ip
                print(f"  ✅ {ip}")
            else:
                print(f"  ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Check if all services return the same IP
    unique_ips = set(results.values())
    if len(unique_ips) == 1:
        public_ip = list(unique_ips)[0]
        print(f"\n🎯 Your public IP: {public_ip}")
        return public_ip
    elif len(unique_ips) > 1:
        print(f"\n⚠️ Multiple IPs detected: {unique_ips}")
        print("This might indicate load balancing or multiple connections")
        return list(unique_ips)
    else:
        print("\n❌ Could not determine public IP")
        return None

def test_specific_connectivity():
    """Test connectivity to specific services"""
    print("\n🌍 Service Connectivity Test")
    print("-" * 40)
    
    services = [
        ("Google DNS", "8.8.8.8"),
        ("Cloudflare DNS", "1.1.1.1"),
        ("Google", "https://www.google.com"),
        ("Groq API", "https://api.groq.com"),
        ("GitHub", "https://github.com"),
        ("OpenAI", "https://api.openai.com")
    ]
    
    for name, target in services:
        print(f"🔄 Testing {name} ({target})...")
        
        if target.startswith('http'):
            # HTTP test
            try:
                response = requests.get(target, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ HTTP OK ({response.status_code})")
                else:
                    print(f"  ⚠️ HTTP {response.status_code}")
            except Exception as e:
                print(f"  ❌ HTTP Failed: {e}")
        else:
            # Ping test
            success, output, error = run_cmd(f"ping -c 3 -W 5 {target}")
            if success:
                # Extract packet loss
                if "0% packet loss" in output:
                    print(f"  ✅ Ping OK")
                else:
                    print(f"  ⚠️ Ping partial success")
            else:
                print(f"  ❌ Ping failed")

def test_sim7600x_specific():
    """Test SIM7600X specific functionality"""
    print("\n📱 SIM7600X Module Test")
    print("-" * 40)
    
    # Check for USB modem interface
    success, output, error = run_cmd("lsusb")
    if success:
        usb_devices = output.lower()
        if 'qualcomm' in usb_devices or 'modem' in usb_devices or 'sierra' in usb_devices:
            print("✅ USB modem device detected")
        else:
            print("ℹ️ No obvious USB modem in lsusb output")
    
    # Check for modem manager
    success, output, error = run_cmd("systemctl is-active ModemManager")
    if success and 'active' in output:
        print("✅ ModemManager is running")
        
        # Get modem info if available
        success, output, error = run_cmd("mmcli -L")
        if success and output.strip():
            print("📱 Modem Manager detected modems:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
    else:
        print("ℹ️ ModemManager not active")
    
    # Check for ppp interfaces (SIM7600X often uses PPP)
    success, output, error = run_cmd("ip link show | grep ppp")
    if success and output.strip():
        print("📡 PPP interfaces found:")
        for line in output.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("ℹ️ No PPP interfaces found")

def compare_with_without_wifi():
    """Instructions for testing with/without WiFi"""
    print("\n🔄 WiFi vs SIM7600X Comparison")
    print("-" * 40)
    print("To test if SIM7600X is working independently:")
    print("1. Note your current public IP above")
    print("2. Disconnect WiFi: sudo nmcli radio wifi off")
    print("3. Wait 30 seconds")
    print("4. Re-run this test")
    print("5. Compare public IPs")
    print("6. Reconnect WiFi: sudo nmcli radio wifi on")
    print("")
    print("If public IP changes, SIM7600X is working as backup!")

def main():
    print("SIM7600X Network Connectivity Test")
    print("=" * 50)
    print("Testing network connectivity and SIM7600X status...\n")
    
    # Test network interfaces
    interfaces = test_network_interfaces()
    
    # Test routing
    test_routing_table()
    
    # Test public IP
    public_ip = test_public_ip()
    
    # Test specific services
    test_specific_connectivity()
    
    # Test SIM7600X specific features
    test_sim7600x_specific()
    
    # Provide comparison instructions
    compare_with_without_wifi()
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
    
    # Summary
    up_interfaces = [name for name, info in interfaces.items() 
                    if info['state'] == 'UP' and info['ips']]
    
    print(f"📊 Active interfaces with IP: {len(up_interfaces)}")
    for iface in up_interfaces:
        print(f"  • {iface}")
    
    if public_ip:
        print(f"🌍 Internet access: ✅ Working")
        print(f"📍 Public IP: {public_ip}")
    else:
        print(f"🌍 Internet access: ❌ Issues detected")

if __name__ == "__main__":
    main()