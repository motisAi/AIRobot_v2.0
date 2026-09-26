// Stella robotic hand — motion engine + gesture library (ESP32 + PCA9685).
// Open-loop servos (no position feedback): we TRACK position in software and
// establish truth by homing to a closed fist on boot.
//
// Channels: 0=pinky 1=ring 2=middle 3=index 4=thumb
// Calibrated pulses (us):  open {1500,1500,1500,1500,2000}  closed {2600,2600,2600,2600,500}
//
// Serial @115200:
//   fist | open | home            whole-hand poses (home = fist)
//   hello | middle | point | peace | thumbs        gestures
//   count <0-5>                    hold up N fingers
//   f <ch> open|close              one finger
//   m <ch> <us>                    smooth-move one finger to a pulse
//   s <ch> <us>                    instant set (no smoothing)
//   speed slow|med|fast            motion speed for following moves
//   pos                            report tracked positions
//   off <ch> | off all             release servo(s)

#include <Wire.h>

#define SDA_PIN 8    // ESP32-S3: I2C on GPIO8 (SDA) / GPIO9 (SCL) — S3 has no GPIO22
#define SCL_PIN 9
#define PCA 0x40
#define MODE1 0x00
#define PRESCALE 0xFE
#define LED0_ON_L 0x06
#define US_MIN 400
#define US_MAX 2600

const int OPEN_US[5]  = {1500, 1500, 1500, 1500, 2000};
const int CLOSE_US[5] = {2600, 2600, 2600, 2600,  500};
int curUS[5];
int gStep = 18;   // us per 15ms tick (medium speed)

// ---- PCA9685 low level ----
void wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(PCA); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}
void setFreq(float f) {
  uint8_t prescale = (uint8_t)(25000000.0 / (4096.0 * f) - 1 + 0.5);
  wr(MODE1, 0x10); wr(PRESCALE, prescale); wr(MODE1, 0x20); delay(5); wr(MODE1, 0xA0);
}
void setPWM(uint8_t ch, uint16_t on, uint16_t off) {
  Wire.beginTransmission(PCA);
  Wire.write(LED0_ON_L + 4 * ch);
  Wire.write(on & 0xFF); Wire.write(on >> 8);
  Wire.write(off & 0xFF); Wire.write(off >> 8);
  Wire.endTransmission();
}
void applyUS(int ch, int us) {
  us = constrain(us, US_MIN, US_MAX);
  setPWM(ch, 0, (uint16_t)((uint32_t)us * 4096 / 20000));
}
void releaseCh(int ch) { setPWM(ch, 0, 4096); }   // limp

// ---- motion engine (smooth, all fingers together) ----
void moveTo(const int target[5], int stepUS) {
  bool moving = true;
  while (moving) {
    moving = false;
    for (int i = 0; i < 5; i++) {
      int tgt = constrain(target[i], US_MIN, US_MAX);
      if (curUS[i] != tgt) {
        int d = tgt - curUS[i];
        int s = (abs(d) < stepUS) ? d : (d > 0 ? stepUS : -stepUS);
        curUS[i] += s; applyUS(i, curUS[i]); moving = true;
      }
    }
    if (moving) delay(15);
  }
}
void setNow(int ch, int us) { us = constrain(us, US_MIN, US_MAX); applyUS(ch, us); curUS[ch] = us; }

// ---- poses & gestures ----
void poseMask(const bool openF[5], int step) {
  int t[5];
  for (int i = 0; i < 5; i++) t[i] = openF[i] ? OPEN_US[i] : CLOSE_US[i];
  moveTo(t, step);
}
void fist()  { moveTo(CLOSE_US, gStep); }
void openH() { moveTo(OPEN_US, gStep); }

void hello() {                       // friendly wave: thumb half, fingers curl 3x
  int thumbHalf = (OPEN_US[4] + CLOSE_US[4]) / 2;
  int base[5] = {OPEN_US[0], OPEN_US[1], OPEN_US[2], OPEN_US[3], thumbHalf};
  moveTo(base, 40);
  int p80 = OPEN_US[0] + (int)(0.8 * (CLOSE_US[0] - OPEN_US[0]));   // ~2380
  int curl[5] = {p80, p80, p80, p80, thumbHalf};
  int flat[5] = {OPEN_US[0], OPEN_US[1], OPEN_US[2], OPEN_US[3], thumbHalf};
  for (int k = 0; k < 3; k++) { moveTo(curl, 35); moveTo(flat, 35); }
  fist();   // momentary gesture -> return to the resting fist (home)
}
void middleFinger() {                // rest closed, middle up, hold 3s, back to fist
  bool m[5] = {false, false, true, false, false};
  poseMask(m, 45);
  delay(3000);
  fist();
}
void middleHold() {                  // middle up and STAY up until 'fist'/'lower'
  bool m[5] = {false, false, true, false, false};
  poseMask(m, 45);
}
void point()   { bool m[5] = {false, false, false, true, false};  poseMask(m, gStep); }
void peace()   { bool m[5] = {false, false, true,  true, false};  poseMask(m, gStep); }
void thumbsUp(){ bool m[5] = {false, false, false, false, true};  poseMask(m, gStep); }
void countN(int n) {                 // hold up N fingers: index,middle,ring,pinky,thumb
  int order[5] = {3, 2, 1, 0, 4};
  bool m[5] = {false, false, false, false, false};
  for (int i = 0; i < n && i < 5; i++) m[order[i]] = true;
  poseMask(m, gStep);
}

// ---- serial command parser ----
String buf;
int arg(String s, int idx) {         // idx-th whitespace-separated token as int
  int start = 0, count = 0;
  while (start < s.length()) {
    int sp = s.indexOf(' ', start);
    if (sp < 0) sp = s.length();
    if (count == idx) return s.substring(start, sp).toInt();
    start = sp + 1; count++;
  }
  return 0;
}
void handle(String line) {
  line.trim(); String l = line; l.toLowerCase();
  if (l == "fist" || l == "home") { fist(); Serial.println("fist"); }
  else if (l == "open") { openH(); Serial.println("open"); }
  else if (l == "hello") { hello(); Serial.println("hello"); }
  else if (l == "middle") { middleFinger(); Serial.println("middle"); }
  else if (l == "middlehold") { middleHold(); Serial.println("middlehold"); }
  else if (l == "lower" || l == "rest") { fist(); Serial.println("rest"); }
  else if (l == "point") { point(); Serial.println("point"); }
  else if (l == "peace") { peace(); Serial.println("peace"); }
  else if (l == "thumbs" || l == "thumbsup") { thumbsUp(); Serial.println("thumbs"); }
  else if (l.startsWith("count")) { countN(arg(l, 1)); Serial.println("count"); }
  else if (l.startsWith("speed")) {
    if (l.indexOf("slow") > 0) gStep = 8; else if (l.indexOf("fast") > 0) gStep = 45; else gStep = 18;
    Serial.println("speed step=" + String(gStep));
  }
  else if (l.startsWith("set ")) {
    // "set 11111" -> pinky,ring,middle,index,thumb  (1=open/extended, 0=closed)
    String m = l.substring(4); m.trim();
    if (m.length() >= 5) {
      int t[5];
      for (int i = 0; i < 5; i++) t[i] = (m[i] == '1') ? OPEN_US[i] : CLOSE_US[i];
      moveTo(t, 45);            // fast, for live mirroring
      Serial.println("set " + m.substring(0, 5));
    }
  }
  else if (l.startsWith("f ")) {
    int ch = arg(l, 1); bool op = l.indexOf("open") > 0;
    int t[5]; for (int i = 0; i < 5; i++) t[i] = curUS[i];
    t[ch] = op ? OPEN_US[ch] : CLOSE_US[ch]; moveTo(t, gStep);
    Serial.println("f" + String(ch) + (op ? " open" : " close"));
  }
  else if (l.startsWith("m ")) {
    int ch = arg(l, 1), us = arg(l, 2);
    int t[5]; for (int i = 0; i < 5; i++) t[i] = curUS[i]; t[ch] = us; moveTo(t, gStep);
    Serial.println("m" + String(ch) + " " + String(constrain(us, US_MIN, US_MAX)));
  }
  else if (l.startsWith("s ")) { int ch = arg(l, 1), us = arg(l, 2); setNow(ch, us); Serial.println("s" + String(ch) + " " + String(us)); }
  else if (l.startsWith("off")) {
    if (l.indexOf("all") > 0) { for (int i = 0; i < 16; i++) releaseCh(i); Serial.println("all off"); }
    else { int ch = arg(l, 1); releaseCh(ch); Serial.println("ch" + String(ch) + " off"); }
  }
  else if (l == "pos") {
    Serial.print("pos:");
    for (int i = 0; i < 5; i++) { Serial.print(" ch"); Serial.print(i); Serial.print("="); Serial.print(curUS[i]); }
    Serial.println();
  }
  else if (l.length()) Serial.println("? " + line);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);
  Wire.beginTransmission(PCA);
  bool ok = (Wire.endTransmission() == 0);
  setFreq(50);
  // Gentle home: flick to OPEN (away from stops), then smoothly close into fist.
  for (int i = 0; i < 5; i++) { curUS[i] = OPEN_US[i]; applyUS(i, OPEN_US[i]); }
  delay(400);
  fist();
  Serial.println();
  Serial.println(String("Stella hand ready. PCA9685: ") + (ok ? "OK" : "NOT FOUND") + ". Home = closed fist.");
  Serial.println("cmds: fist|open|hello|middle|point|peace|thumbs|count N|f <ch> open|close|m <ch> <us>|speed slow|med|fast|pos|off");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(buf); buf = ""; }
    else if (c != '\r') buf += c;
  }
}
