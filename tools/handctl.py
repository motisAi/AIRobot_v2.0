#!/usr/bin/env python3
"""Send commands to the ESP32 hand-test firmware over serial and print replies.

Usage: handctl.py "sweep 0" "sleep 1" "s 0 1500"
Opening the port auto-resets the ESP32 (fine — it just re-runs setup()).
"""
import sys
import time

import serial

import os
PORT = os.getenv("HAND_PORT", "/dev/ttyACM0")   # S3 UART bridge (was ttyUSB0 on classic ESP32)
BAUD = 115200


def drain(ser, dur):
    end = time.time() + dur
    while time.time() < end:
        line = ser.readline()
        if line:
            print(line.decode(errors="replace").rstrip())


def main():
    cmds = sys.argv[1:]
    ser = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(2.5)          # wait for ESP32 boot after auto-reset
    drain(ser, 0.6)          # print boot banner
    for c in cmds:
        if c.startswith("sleep"):
            time.sleep(float(c.split()[1]))
            continue
        ser.write((c + "\n").encode())
        print(">>", c)
        drain(ser, 1.2)
    ser.close()


if __name__ == "__main__":
    main()
