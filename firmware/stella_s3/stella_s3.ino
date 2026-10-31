/*
 * stella_s3.ino  —  Stella's single ESP32-S3: HAND (servos) + EYES (LCDs)
 * ===========================================================================
 * ONE ESP32-S3 drives BOTH:
 *   - the 5-finger hand, via a PCA9685 16-ch servo driver on I2C (GPIO8/9), and
 *   - two GC9D01 160x160 round eye LCDs on a shared SPI bus.
 * The Pi talks to this board over ONE USB cable (the S3's NATIVE USB port) at
 * 115200 and sends hand gestures AND eye commands on the same line stream.
 *
 * PINS (none overlap; GPIO8/9 are the PCA9685 I2C, so the eyes avoid them) -----
 *   Hand  : I2C  SDA=GPIO8  SCL=GPIO9   -> PCA9685 @0x40, servo ch0..4
 *   Eyes  : SPI  SCLK=GPIO7 MOSI/DIN=GPIO4  DC=GPIO10
 *           RST1(left)=GPIO0  RST2(right)=GPIO5   CS_L=GPIO2 CS_R=GPIO6
 *           BL_L=GPIO1  BL_R=GPIO3            (module VCC->3V3, GND->GND)
 *
 * SERIAL PROTOCOL (one stream; routed by the first character) -----------------
 *   EYES  (uppercase single letter):
 *     G <x> <y>   gaze -100..100 (x: + = her right, y: + = down)
 *     B           blink      C  center      E <n> emotion 0-4    P <n> pupil 0-100
 *   HAND  (lowercase words, unchanged from the standalone hand firmware):
 *     fist|open|home  hello|middle|middlehold|point|peace|thumbs
 *     count <0-5>   set <5 bits>   f <ch> open|close   m <ch> <us>   s <ch> <us>
 *     speed slow|med|fast   off <ch>|off all   pos
 *   Routing: a line whose first non-space char is G/B/C/E/P (followed by space
 *   or end) is an EYE command; everything else goes to the hand parser. Hand
 *   words are lowercase, so there is never a clash (eye 'C' vs hand "count").
 *
 * BUILD (PlatformIO, board esp32-s3-devkitc-1): -DARDUINO_USB_CDC_ON_BOOT=1
 *   -DARDUINO_USB_MODE=1 so Serial == the native USB port the Pi is wired to.
 *   Library: the display supplier's TFT_eSPI vendored at firmware/libraries
 *   (GC9D01 + dual reset; User_Setup.h has TFT_DC 10, TFT_RST 0, TFT2_RST 5,
 *   TFT_MOSI 4, TFT_SCLK 7, TFT_CS -1). Recipe: firmware/README.md.
 *
 * NOTE: a hand gesture (hello/middle) runs a short BLOCKING motion, during which
 * the eyes hold their last frame; they resume the moment the gesture finishes.
 * ===========================================================================
 */

#include <Wire.h>
#include <SPI.h>
#include <TFT_eSPI.h>

// ===========================================================================
//  HAND  (PCA9685 servos on I2C GPIO8/9)
// ===========================================================================
#define SDA_PIN 8
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
void releaseCh(int ch) { setPWM(ch, 0, 4096); }

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

void poseMask(const bool openF[5], int step) {
  int t[5];
  for (int i = 0; i < 5; i++) t[i] = openF[i] ? OPEN_US[i] : CLOSE_US[i];
  moveTo(t, step);
}
void fist()  { moveTo(CLOSE_US, gStep); }
void openH() { moveTo(OPEN_US, gStep); }
void hello() {
  int thumbHalf = (OPEN_US[4] + CLOSE_US[4]) / 2;
  int base[5] = {OPEN_US[0], OPEN_US[1], OPEN_US[2], OPEN_US[3], thumbHalf};
  moveTo(base, 40);
  int p80 = OPEN_US[0] + (int)(0.8 * (CLOSE_US[0] - OPEN_US[0]));
  int curl[5] = {p80, p80, p80, p80, thumbHalf};
  int flat[5] = {OPEN_US[0], OPEN_US[1], OPEN_US[2], OPEN_US[3], thumbHalf};
  for (int k = 0; k < 3; k++) { moveTo(curl, 35); moveTo(flat, 35); }
  fist();
}
void middleFinger() { bool m[5] = {false, false, true, false, false}; poseMask(m, 45); delay(3000); fist(); }
void middleHold()   { bool m[5] = {false, false, true, false, false}; poseMask(m, 45); }
void point()        { bool m[5] = {false, false, false, true, false};  poseMask(m, gStep); }
void peace()        { bool m[5] = {false, false, true,  true, false};  poseMask(m, gStep); }
void thumbsUp()     { bool m[5] = {false, false, false, false, true};  poseMask(m, gStep); }
void countN(int n) {
  int order[5] = {3, 2, 1, 0, 4};
  bool m[5] = {false, false, false, false, false};
  for (int i = 0; i < n && i < 5; i++) m[order[i]] = true;
  poseMask(m, gStep);
}
int arg(String s, int idx) {
  int start = 0, count = 0;
  while (start < s.length()) {
    int sp = s.indexOf(' ', start);
    if (sp < 0) sp = s.length();
    if (count == idx) return s.substring(start, sp).toInt();
    start = sp + 1; count++;
  }
  return 0;
}
void handHandle(String line) {
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
    String m = l.substring(4); m.trim();
    if (m.length() >= 5) {
      int t[5];
      for (int i = 0; i < 5; i++) t[i] = (m[i] == '1') ? OPEN_US[i] : CLOSE_US[i];
      moveTo(t, 45);
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
void initHand() {
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);
  Wire.beginTransmission(PCA);
  bool ok = (Wire.endTransmission() == 0);
  setFreq(50);
  for (int i = 0; i < 5; i++) { curUS[i] = OPEN_US[i]; applyUS(i, OPEN_US[i]); }
  delay(400);
  fist();
  Serial.println();
  Serial.println(String("Stella hand ready. PCA9685: ") + (ok ? "OK" : "NOT FOUND") + ". Home = closed fist.");
}

// ===========================================================================
//  EYES  (two GC9D01 on shared SPI)
// ===========================================================================
#define CS_L 2
#define CS_R 6
#define BL_L 1
#define BL_R 3
#define EYE_SIZE   160
#define EYE_C      80
#define EYE_R      80
#define IRIS_R     42
#define PUPIL_R    18
#define MAX_OFFSET 26
#define COL_BG      0x0000
#define COL_SCLERA  0xFFFF
#define COL_IRIS    0x2D7F
#define COL_PUPIL   0x0000
#define COL_HILITE  0xFFFF

TFT_eSPI    tft = TFT_eSPI();
TFT_eSprite eye = TFT_eSprite(&tft);

float gx = 0, gy = 0, tgx = 0, tgy = 0;
int   emotion = 0;
float pupilScale = 1.0;
bool  blinking = false;
float lid = 0.0;
int   blinkDir = 0;
unsigned long lastCmd = 0, lastBlink = 0, lastIdleMove = 0;
const unsigned long IDLE_MS = 4000;

void selectEye(int cs) {
  digitalWrite(CS_L, cs == CS_L ? LOW : HIGH);
  digitalWrite(CS_R, cs == CS_R ? LOW : HIGH);
}
void startBlink() { if (!blinking) { blinking = true; blinkDir = 1; } }

void eyeParse(char *s) {
  lastCmd = millis();
  switch (s[0]) {
    case 'G': {
      int x = 0, y = 0;
      if (sscanf(s + 1, "%d %d", &x, &y) == 2) {
        tgx = constrain(x, -100, 100) / 100.0f;
        tgy = constrain(y, -100, 100) / 100.0f;
      }
      break;
    }
    case 'B': startBlink(); break;
    case 'C': tgx = 0; tgy = 0; break;
    case 'E': emotion = atoi(s + 1); break;
    case 'P': pupilScale = constrain(atoi(s + 1), 0, 100) / 50.0f; break;
  }
}
void drawEye(float ox, float oy) {
  eye.fillSprite(COL_BG);
  eye.fillCircle(EYE_C, EYE_C, EYE_R, COL_SCLERA);
  int ix = EYE_C + (int)(ox * MAX_OFFSET);
  int iy = EYE_C + (int)(oy * MAX_OFFSET);
  uint16_t iris = COL_IRIS;
  if      (emotion == 2) iris = 0xF800;
  else if (emotion == 1) iris = 0x07E0;
  else if (emotion == 4) iris = 0x7BEF;
  eye.fillCircle(ix, iy, IRIS_R, iris);
  int pr = (int)(PUPIL_R * pupilScale);
  if (emotion == 3) pr = PUPIL_R + 8;
  eye.fillCircle(ix, iy, pr, COL_PUPIL);
  eye.fillCircle(ix - 11, iy - 11, 5, COL_HILITE);
  int close = (int)(lid * EYE_R);
  if (emotion == 4) close = max(close, (int)(EYE_R * 0.45));
  if (close > 0) {
    eye.fillRect(0, 0, EYE_SIZE, close, COL_BG);
    eye.fillRect(0, EYE_SIZE - close, EYE_SIZE, close, COL_BG);
  }
}
void pushToBoth() {
  selectEye(CS_L); eye.pushSprite(0, 0);
  selectEye(CS_R); eye.pushSprite(0, 0);
}
void initEyes() {
  pinMode(CS_L, OUTPUT); pinMode(CS_R, OUTPUT);
  pinMode(BL_L, OUTPUT); pinMode(BL_R, OUTPUT);
  digitalWrite(BL_L, HIGH); digitalWrite(BL_R, HIGH);
  digitalWrite(CS_L, LOW); digitalWrite(CS_R, LOW);
  tft.init();
  digitalWrite(CS_L, HIGH); digitalWrite(CS_R, HIGH);
  selectEye(CS_L); tft.setRotation(0); tft.fillScreen(COL_BG);
  selectEye(CS_R); tft.setRotation(0); tft.fillScreen(COL_BG);
  selectEye(-1);
  eye.setColorDepth(16);
  eye.createSprite(EYE_SIZE, EYE_SIZE);
  lastCmd = lastBlink = lastIdleMove = millis();
  randomSeed(micros());
}
void animateEyes() {
  unsigned long now = millis();
  if (now - lastCmd > IDLE_MS && now - lastIdleMove > 1500) {
    tgx = random(-60, 61) / 100.0f;
    tgy = random(-40, 41) / 100.0f;
    lastIdleMove = now;
  }
  if (now - lastBlink > (unsigned long)random(3000, 6000)) { startBlink(); lastBlink = now; }
  gx += (tgx - gx) * 0.25f;
  gy += (tgy - gy) * 0.25f;
  if (blinking) {
    lid += 0.25f * blinkDir;
    if (lid >= 1.0f) { lid = 1.0f; blinkDir = -1; }
    else if (lid <= 0.0f && blinkDir == -1) { lid = 0.0f; blinking = false; }
  }
  drawEye(gx, gy);
  pushToBoth();
}

// ===========================================================================
//  SHARED serial: one line stream, routed by the first character
// ===========================================================================
void routeLine(char *s) {
  char *p = s;
  while (*p == ' ' || *p == '\t') p++;
  char c0 = *p;
  bool isEye = (c0 == 'G' || c0 == 'B' || c0 == 'C' || c0 == 'E' || c0 == 'P')
               && (p[1] == 0 || p[1] == ' ');
  if (isEye) eyeParse(p);
  else       handHandle(String(p));
}
void handleSerialRouted() {
  static char line[64];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n) { line[n] = 0; routeLine(line); n = 0; }
    } else if (n < sizeof(line) - 1) {
      line[n++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  initHand();   // hand FIRST, so any display trouble can never stop the servos
  initEyes();   // then the eye LCDs
  Serial.println("Stella S3 ready: hand + eyes on one link.");
}

void loop() {
  handleSerialRouted();
  animateEyes();
  delay(16);   // ~ up to 60 fps; real rate limited by the two SPI pushes
}
