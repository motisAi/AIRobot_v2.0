// servos.h - Stella's arm (5x MG996R) + 5-finger hand on one PCA9685 (I2C SDA GPIO12 / SCL GPIO9).
// Runs on core 1 (the Arduino loop). Never blocks: every move is a target that a 20 ms tick
// eases toward, and finger gestures are a short pose sequence played by the same tick.
#pragma once

void servosBegin();             // load calibration, find the PCA9685, cold/warm start
void servosLoop();              // call every loop(): PCA retry, motion tick, gesture player
void servoCommand(char* s);     // one serial line (lowercased in place); see HELP in servos.cpp
