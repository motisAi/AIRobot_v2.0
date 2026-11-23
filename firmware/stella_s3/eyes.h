// eyes.h - Stella's two GC9D01 160x160 eye LCDs (custom hardware-SPI driver, no TFT_eSPI).
// The eyes render on their OWN task on core 0, so drawing never delays the servos.
#pragma once

void eyesBegin();                 // start the render task (panel init happens inside it)
bool isEyeCommand(const char* s); // G/B/C/E/P/M/T: one uppercase letter, then a space or end of line
void eyeCommand(const char* s);   // apply it (called from the serial loop on core 1)
