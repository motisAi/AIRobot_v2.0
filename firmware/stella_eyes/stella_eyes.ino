/*
 * stella_eyes.ino  —  Stella's physical eyes
 * ---------------------------------------------------------------------------
 * Two GC9D01 160x160 round LCDs on ONE shared SPI bus, driven by an ESP32-C3.
 * Renders two eyes and points the gaze wherever the Raspberry Pi (Stella) says.
 *
 * The LCD module is just the two displays — it needs this ESP32-C3 to drive it.
 * The Pi talks to this board over USB serial and sends a gaze target; the board
 * does all the eye drawing so the Pi's CPU stays free for vision + the LLM.
 *
 * HARDWARE  (ESP32-C3; GPIO8/GPIO9 are RESERVED for the PCA9685 I2C bus) -------
 *   Shared SPI : SCLK = GPIO7   MOSI/DIN = GPIO4   DC = GPIO10
 *   Resets     : RST1 = GPIO0 (TFT_RST)   RST2 = GPIO5 (TFT2_RST)
 *                (one per display; TFT_eSPI pulses both together in init())
 *   Left  eye  : CS = GPIO2    BL = GPIO1
 *   Right eye  : CS = GPIO6    BL = GPIO3
 *   I2C (other): SDA = GPIO8   SCL = GPIO9   (16-ch PCA9685 servo driver)
 *   Power      : module VCC -> 3V3, GND -> GND
 *   USB        : ESP32-C3 -> Raspberry Pi (power + gaze serial @115200)
 *
 * REQUIRED LIBRARY: the vendor's bundled TFT_eSPI (GC9D01 driver, dual-reset
 *   build). In its User_Setup.h, ESP32-C3 block, change ONLY these two lines:
 *       #define TFT_DC   10     // was 9  (GPIO9 is the PCA9685 SCL)
 *       #define TFT_RST  0      // was 8  (GPIO8 is the PCA9685 SDA)
 *   Leave as they are:  TFT_MOSI 4, TFT_SCLK 7, TFT_CS -1, TFT2_RST 5.
 *   Board: "ESP32C3 Dev Module".
 *
 * INIT PATTERN (matches the vendor dual-eye demo): select BOTH eyes, call
 *   tft.init() ONCE — the library resets both panels (TFT_RST + TFT2_RST) and
 *   programs them in one pass — then deselect. Calling init() per eye would
 *   re-reset the first panel while initialising the second, so don't.
 *
 * SERIAL PROTOCOL  (Pi -> eyes, 115200 baud, newline-terminated ASCII) --------
 *   G <x> <y>   gaze target, x,y each -100..100   (0 0 = straight ahead)
 *               x<0 look left, x>0 right, y<0 up, y>0 down.  e.g.  "G -40 10"
 *   B           blink once
 *   C           recenter the gaze
 *   E <n>       emotion: 0 neutral, 1 happy, 2 angry, 3 surprised, 4 sleepy
 *   P <n>       pupil size 0..100 (50 = normal)
 * Unknown lines are ignored. If no command arrives for IDLE_MS the eyes drift
 * gently and auto-blink, so they look alive even before Stella is wired up.
 * ---------------------------------------------------------------------------
 */

#include <SPI.h>
#include <TFT_eSPI.h>

// ---- per-eye control pins (SPI/DC/RST come from TFT_eSPI User_Setup.h) ------
#define CS_L 2
#define CS_R 6
#define BL_L 1
#define BL_R 3

// ---- eye geometry (display is 160x160) --------------------------------------
#define EYE_SIZE   160
#define EYE_C      80      // center
#define EYE_R      80      // sclera radius
#define IRIS_R     42
#define PUPIL_R    18
#define MAX_OFFSET 26      // max iris slide from center, px

// ---- colors (RGB565) --------------------------------------------------------
#define COL_BG      0x0000
#define COL_SCLERA  0xFFFF
#define COL_IRIS    0x2D7F   // Stella blue
#define COL_PUPIL   0x0000
#define COL_HILITE  0xFFFF

TFT_eSPI    tft = TFT_eSPI();
TFT_eSprite eye = TFT_eSprite(&tft);   // one 160x160 sprite, reused for both eyes

// gaze (-1..1), eased toward target
float gx = 0, gy = 0, tgx = 0, tgy = 0;
int   emotion = 0;
float pupilScale = 1.0;

// blink
bool  blinking = false;
float lid = 0.0;        // 0 open .. 1 closed
int   blinkDir = 0;
unsigned long lastCmd = 0, lastBlink = 0, lastIdleMove = 0;
const unsigned long IDLE_MS = 4000;

// Assert exactly one eye's CS (pass -1 to deselect both).
void selectEye(int cs) {
  digitalWrite(CS_L, cs == CS_L ? LOW : HIGH);
  digitalWrite(CS_R, cs == CS_R ? LOW : HIGH);
}

void startBlink() { if (!blinking) { blinking = true; blinkDir = 1; } }

void parseCmd(char *s) {
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

void handleSerial() {
  static char buf[32];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n) { buf[n] = 0; parseCmd(buf); n = 0; }
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = c;
    }
  }
}

void drawEye(float ox, float oy) {
  eye.fillSprite(COL_BG);
  eye.fillCircle(EYE_C, EYE_C, EYE_R, COL_SCLERA);

  int ix = EYE_C + (int)(ox * MAX_OFFSET);
  int iy = EYE_C + (int)(oy * MAX_OFFSET);

  uint16_t iris = COL_IRIS;
  if      (emotion == 2) iris = 0xF800;   // angry  -> red
  else if (emotion == 1) iris = 0x07E0;   // happy  -> green
  else if (emotion == 4) iris = 0x7BEF;   // sleepy -> grey

  eye.fillCircle(ix, iy, IRIS_R, iris);
  int pr = (int)(PUPIL_R * pupilScale);
  if (emotion == 3) pr = PUPIL_R + 8;     // surprised -> dilated
  eye.fillCircle(ix, iy, pr, COL_PUPIL);
  eye.fillCircle(ix - 11, iy - 11, 5, COL_HILITE);  // catch-light

  // eyelids: blink + sleepy half-lid. Cover top & bottom with background.
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

void setup() {
  Serial.begin(115200);
  pinMode(CS_L, OUTPUT); pinMode(CS_R, OUTPUT);
  pinMode(BL_L, OUTPUT); pinMode(BL_R, OUTPUT);
  digitalWrite(BL_L, HIGH); digitalWrite(BL_R, HIGH);   // backlights on

  // Vendor init pattern: both eyes selected, ONE init() (resets + programs
  // both panels via TFT_RST/TFT2_RST), then deselect.
  digitalWrite(CS_L, LOW); digitalWrite(CS_R, LOW);
  tft.init();
  digitalWrite(CS_L, HIGH); digitalWrite(CS_R, HIGH);

  // Per-eye orientation + clear.
  selectEye(CS_L); tft.setRotation(0); tft.fillScreen(COL_BG);
  selectEye(CS_R); tft.setRotation(0); tft.fillScreen(COL_BG);
  selectEye(-1);

  eye.setColorDepth(16);
  eye.createSprite(EYE_SIZE, EYE_SIZE);

  lastCmd = lastBlink = lastIdleMove = millis();
  randomSeed(micros());
}

void loop() {
  handleSerial();
  unsigned long now = millis();

  // Idle autonomy when Stella isn't driving the gaze.
  if (now - lastCmd > IDLE_MS && now - lastIdleMove > 1500) {
    tgx = random(-60, 61) / 100.0f;
    tgy = random(-40, 41) / 100.0f;
    lastIdleMove = now;
  }
  if (now - lastBlink > (unsigned long)random(3000, 6000)) { startBlink(); lastBlink = now; }

  // Ease gaze toward the target for a natural move.
  gx += (tgx - gx) * 0.25f;
  gy += (tgy - gy) * 0.25f;

  // Blink animation.
  if (blinking) {
    lid += 0.25f * blinkDir;
    if (lid >= 1.0f) { lid = 1.0f; blinkDir = -1; }
    else if (lid <= 0.0f && blinkDir == -1) { lid = 0.0f; blinking = false; }
  }

  drawEye(gx, gy);
  pushToBoth();
  delay(16);   // ~ up to 60 fps; real rate is limited by the two SPI pushes
}
