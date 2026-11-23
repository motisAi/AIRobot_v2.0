/*
 * stella_s3.ino  -  Stella's single ESP32-S3: EYES + ARM + HAND
 * ===========================================================================
 * ONE ESP32-S3 drives:
 *   - two GC9D01 160x160 eye LCDs (custom hardware-SPI driver)       eyes.cpp
 *   - the 5-servo arm + the 5-finger hand on one PCA9685              servos.cpp
 * The Pi talks to it over ONE USB cable (the CH343 UART type-C, "usb-1a86_*") at 115200.
 *
 * CORES: the eyes render on their own task on core 0 (a frame takes ~41 ms to push to
 * both panels); the serial link + the 20 ms servo tick run here in loop() on core 1, so
 * the arm stays smooth whatever the eyes are doing.
 *
 * PINS ------------------------------------------------------------------------
 *   Eyes : SPI DIN=GPIO4 CLK=GPIO7 DC=GPIO10  RST1=GPIO11 RST2=GPIO5
 *          CS1=GPIO2 CS2=GPIO6  BL1=GPIO1 BL2=GPIO3   (module VCC 5V, common GND)
 *   (classic ESP32 build `pio run -e esp32`: other pins, listed in eyes.cpp / servos.cpp)
 *   Servos: I2C SDA=GPIO12 SCL=GPIO9 -> PCA9685 @0x40 (logic 3.3V, servo power external)
 *          ch0 pinky 1 ring 2 middle 3 index 4 thumb | 5 wrist_roll 6 wrist_pitch
 *          7 elbow 8 shoulder 9 base
 *
 * SERIAL PROTOCOL (one line per command; routed by the first character) -------
 *   EYES  = one UPPERCASE letter followed by a space or end of line:
 *     G <x> <y>  gaze -100..100 (x: + = her right, y: + = down)   B blink   C center
 *     E <n> emotion 0-4   P <n> pupil 0-100   T orientation marker   M <1|2> <hex> MADCTL
 *   Everything else (case-insensitive) goes to the servos:
 *     hand : fist|rest|lower open hello middle middlehold point peace thumbs
 *            count <0-5>  set <5 bits>  grip <0-100>  f <ch> open|close  m <ch> <us>  s <ch> <us>
 *            speed slow|med|fast|<n>        (the same words the old hand firmware took)
 *     arm  : pos  home  pose <roll> <pitch> <elbow> <shoulder> <base>  en <ch> <us>
 *            j <ch> <us>  n <ch> <+-us>  spd <1-20>  off <ch>  x | off all
 *            osc <ch> <us> <ms> <times>  motion smooth|classic
 *     cal  : lim <ch> <lo> <hi>  hm <ch> <us>  sethome  dump  clearcal   other: scan ver help
 *
 * POWER-UP SAFETY: servo power just switched on -> every servo LIMP until commanded.
 * ESP32 reset only (reflash, Pi opens the port) -> servos keep holding where they are.
 * Fingers no longer close into a fist at boot (the arm shares this board now).
 *
 * BUILD: PlatformIO, firmware/platformio.ini (espressif32@6.5.0 = arduino-esp32 2.0.x,
 *        -DARDUINO_USB_CDC_ON_BOOT=0 so Serial = the CH343 UART). Recipe: firmware/README.md
 * ===========================================================================
 */
#include "eyes.h"
#include "servos.h"

#define FW_NAME "Stella firmware [" CONFIG_IDF_TARGET "]: eyes + arm + hand (" __DATE__ " " __TIME__ ")"

const char HELP[] =
  "EYES  G <x> <y> look (-100..100, + = her right / down)  B blink  C center\n"
  "      E <0-4> emotion  P <0-100> pupil  T orientation marker  M <1|2> <hex> panel MADCTL\n"
  "HAND  fist open hello middle middlehold point peace thumbs  count <0-5>  set <5 bits>\n"
  "      f <ch> open|close  m <ch> <us>  s <ch> <us> (instant)  speed slow|med|fast|<5-120>\n"
  "      grip <0-100>  all fingers part-way (0 = open, 100 = closed)\n"
  "ARM   pos   home   pose <roll> <pitch> <elbow> <shoulder> <base>  (-1 = leave that joint)\n"
  "      en <ch> <us>  switch a LIMP servo on (it jumps there: hold the joint near it first)\n"
  "      j <ch> <us> move   n <ch> <+-us> nudge (10 us ~ 1 deg)   spd <1-20> arm speed\n"
  "      osc <ch> <us> <period ms> <times> swing in place   motion smooth|classic (profile)\n"
  "      seq on|off  pose/home one joint at a time (default on)   home = home grip (thumb open) + arm\n"
  "      blend <0-60>  next joint starts in the last N% of a move (default 30, 0 = strictly one by one)\n"
  "      off <ch>   x | off all  EVERYTHING limp now (the arm drops - support it)\n"
  "CAL   lim <ch> <lo> <hi>  hm <ch> <us>  sethome  (saved)   dump   clearcal (built-in values)\n"
  "      scan  ver  pcatest (simulate a servo-chip reset: the watchdog restores within 0.2 s)\n";

void route(char* s){
  while(*s==' '||*s=='\t') s++;
  if(!*s) return;
  if(isEyeCommand(s)) eyeCommand(s);
  else if(!strcmp(s,"help")) Serial.print(HELP);
  else if(!strcmp(s,"ver")) Serial.println(FW_NAME);
  else servoCommand(s);
}

void handleSerial(){
  static char line[64]; static uint8_t n=0;
  while(Serial.available()){ char c=Serial.read();
    if(c=='\n'||c=='\r'){ if(n){ line[n]=0; route(line); n=0; } }
    else if(n<sizeof(line)-1) line[n++]=c; }
}

void setup(){
  Serial.begin(115200); delay(200);
  Serial.println(); Serial.println(FW_NAME); Serial.println("'help' for commands");
  servosBegin();     // servos FIRST: hold or limp is decided before anything else runs
  eyesBegin();       // then the eye task on core 0
}

void loop(){
  handleSerial();
  servosLoop();
  delay(1);
}
