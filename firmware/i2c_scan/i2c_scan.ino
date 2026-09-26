// Stella ESP32 — first-flash sanity check + I2C scanner.
// Proves the Pi<->ESP32 link works AND detects the PCA9685 (expected at 0x40).
// Wiring: PCA9685 SDA->GPIO21, SCL->GPIO22, VCC->3V3, GND->GND (common ground).

#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define LED_PIN 2   // onboard LED on most WROOM-32 devkits

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  delay(400);
  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.println();
  Serial.println("Stella ESP32 online - I2C scanner starting");
  Serial.println("(expecting PCA9685 at 0x40)");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  int found = 0;
  Serial.println("--- scanning I2C bus ---");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  found device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.print(addr, HEX);
      if (addr == 0x40) Serial.print("  <-- PCA9685 (servo driver) OK!");
      Serial.println();
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  no I2C devices found - check SDA/SCL/GND wiring & 3V3 to VCC");
  } else {
    Serial.print("  total devices: ");
    Serial.println(found);
  }
  digitalWrite(LED_PIN, LOW);
  delay(2500);
}
