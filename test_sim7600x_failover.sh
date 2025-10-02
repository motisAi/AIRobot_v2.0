#!/bin/bash
# SIM7600X Failover Test Script
# =============================
# This script tests SIM7600X by temporarily disabling WiFi

echo "🔄 SIM7600X Failover Test"
echo "========================="

# Get current public IP
echo "📍 Getting current public IP (via WiFi)..."
WIFI_IP=$(curl -s http://ipinfo.io/ip 2>/dev/null || echo "Failed")
echo "Current IP (WiFi): $WIFI_IP"

echo ""
echo "⚠️  DISABLING WiFi for 60 seconds..."
echo "    This will temporarily disconnect WiFi"
echo "    Press Ctrl+C to abort in next 5 seconds"

# 5 second countdown
for i in 5 4 3 2 1; do
    echo "    $i..."
    sleep 1
done

echo ""
echo "📵 Disabling WiFi..."
sudo nmcli radio wifi off

echo "⏱️  Waiting 30 seconds for SIM7600X to take over..."
sleep 30

echo "🔍 Checking network status without WiFi..."

# Check interfaces
ip addr show | grep -E "(wlan0|ppp0|eth0)" | grep "state UP"

echo ""
echo "📍 Testing internet via SIM7600X..."

# Try to get IP via SIM7600X
SIM_IP=$(curl -s --max-time 15 http://ipinfo.io/ip 2>/dev/null || echo "Failed")
echo "IP via SIM7600X: $SIM_IP"

if [ "$SIM_IP" != "Failed" ] && [ "$SIM_IP" != "$WIFI_IP" ]; then
    echo "✅ SUCCESS! SIM7600X is working!"
    echo "   WiFi IP: $WIFI_IP"
    echo "   SIM IP:  $SIM_IP"
else
    echo "❌ SIM7600X not working or no internet"
fi

echo ""
echo "🔄 Re-enabling WiFi..."
sudo nmcli radio wifi on

echo "⏱️  Waiting 15 seconds for WiFi to reconnect..."
sleep 15

echo "📍 Final IP check..."
FINAL_IP=$(curl -s http://ipinfo.io/ip 2>/dev/null || echo "Failed")
echo "Final IP (should be WiFi): $FINAL_IP"

echo ""
echo "📊 SUMMARY:"
echo "   WiFi IP:  $WIFI_IP"
echo "   SIM IP:   $SIM_IP"
echo "   Final IP: $FINAL_IP"

if [ "$SIM_IP" != "Failed" ] && [ "$SIM_IP" != "$WIFI_IP" ]; then
    echo "✅ SIM7600X WORKING - Can provide internet when WiFi is down!"
else
    echo "❌ SIM7600X needs configuration - Only WiFi working"
fi