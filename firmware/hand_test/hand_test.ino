// Stella robotic hand — safe interactive servo tester (ESP32 + PCA9685).
// Raw PCA9685 register access (no library). Does NOT move servos on boot.
//
// Channels (per wiring): 0=pinky, 1=ring, 2=middle, 3=index, 4=thumb.
//
// Serial commands @115200:
//   s <ch> <us>   set channel to a pulse in microseconds (clamped 700..2300)
//   c <ch>        center a channel (1500 us)
//   sweep <ch>    gently wiggle a channel +-150us so you can SEE which finger
//   off <ch>      release a channel (servo goes limp)
//   off all       release all channels
//   scan          list I2C devices (expect 0x40)

#include <Wire.h>

#define SDA_PIN 8    // ESP32-S3: I2C on GPIO8 (SDA) / GPIO9 (SCL) — S3 has no GPIO22
#define SCL_PIN 9
#define PCA 0x40
#define MODE1 0x00
#define PRESCALE 0xFE
#define LED0_ON_L 0x06
#define SERVO_MIN_US 400
#define SERVO_MAX_US 2600

void wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(PCA); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}
uint8_t rd(uint8_t reg) {
  Wire.beginTransmission(PCA); Wire.write(reg); Wire.endTransmission();
  Wire.requestFrom(PCA, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}
void setFreq(float f) {
  uint8_t prescale = (uint8_t)(25000000.0 / (4096.0 * f) - 1 + 0.5);
  wr(MODE1, 0x10);   // SLEEP=1 (required before changing PRE_SCALE)
  wr(PRESCALE, prescale);
  wr(MODE1, 0x20);   // WAKE: SLEEP=0, AI=1 (auto-increment) -> oscillator ON
  delay(5);
  wr(MODE1, 0xA0);   // RESTART + AI, SLEEP stays 0 (PWM now actually outputs)
}
void setPWM(uint8_t ch, uint16_t on, uint16_t off) {
  Wire.beginTransmission(PCA);
  Wire.write(LED0_ON_L + 4 * ch);
  Wire.write(on & 0xFF); Wire.write(on >> 8);
  Wire.write(off & 0xFF); Wire.write(off >> 8);
  Wire.endTransmission();
}
void setUS(uint8_t ch, uint16_t us) {
  us = constrain(us, SERVO_MIN_US, SERVO_MAX_US);
  uint16_t ticks = (uint32_t)us * 4096 / 20000;   // 50 Hz -> 20000 us period
  setPWM(ch, 0, ticks);
}
void releaseCh(uint8_t ch) { setPWM(ch, 0, 4096); }  // full-off = limp

void scan() {
  Serial.println("I2C scan:");
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print("  0x"); Serial.print(a, HEX);
      if (a == 0x40) Serial.print("  <- PCA9685");
      Serial.println();
    }
  }
}
void sweep(uint8_t ch) {
  for (int us = 1500; us <= 1650; us += 10) { setUS(ch, us); delay(40); }
  for (int us = 1650; us >= 1350; us -= 10) { setUS(ch, us); delay(40); }
  for (int us = 1350; us <= 1500; us += 10) { setUS(ch, us); delay(40); }
  Serial.println("swept ch" + String(ch));
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);
  Serial.println();
  Wire.beginTransmission(PCA);
  bool ok = (Wire.endTransmission() == 0);
  Serial.println(String("Stella hand test. PCA9685 @0x40: ") + (ok ? "FOUND" : "NOT FOUND"));
  setFreq(50);
  // Intentionally do NOT drive servos on boot -> they stay limp/safe.
  Serial.println("cmds: s <ch> <us> | c <ch> | sweep <ch> | off <ch> | off all | scan");
}

String buf;
void handle(String line) {
  line.trim();
  if (line.startsWith("s ")) {
    int sp = line.indexOf(' ', 2);
    int ch = line.substring(2, sp).toInt();
    int us = line.substring(sp + 1).toInt();
    setUS(ch, us);
    Serial.println("ch" + String(ch) + " -> " + String(constrain(us, SERVO_MIN_US, SERVO_MAX_US)) + "us");
  } else if (line.startsWith("c ")) {
    int ch = line.substring(2).toInt();
    setUS(ch, 1500); Serial.println("ch" + String(ch) + " center 1500us");
  } else if (line.startsWith("sweep ")) {
    sweep(line.substring(6).toInt());
  } else if (line.startsWith("off")) {
    if (line.indexOf("all") >= 0) { for (int i = 0; i < 16; i++) releaseCh(i); Serial.println("all off"); }
    else { int ch = line.substring(4).toInt(); releaseCh(ch); Serial.println("ch" + String(ch) + " off"); }
  } else if (line == "scan") {
    scan();
  } else if (line.length()) {
    Serial.println("? " + line);
  }
}
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(buf); buf = ""; }
    else if (c != '\r') buf += c;
  }
}
