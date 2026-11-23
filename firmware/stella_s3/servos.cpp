/* servos.cpp - Stella's arm + 5-finger hand on one PCA9685.
 *
 * SAFE BY DESIGN:
 *   - PCA9685 freshly powered (COLD): every servo LIMP. Nothing moves until commanded.
 *   - ESP32 reset only (reflash, the Pi opening the serial port, reset button) (WARM): the
 *     PCA9685 keeps driving its last pulses on its own; we READ THEM BACK and keep holding,
 *     so the arm neither drops nor jumps.
 *   - The arm only moves on a smooth profile (minimum-jerk by default, see `motion`) toward a
 *     target inside its soft limits. Instant jumps (s) are for finger channels only.
 *   - Arm calibration (limits + home) is built in below; lim / hm / sethome save overrides
 *     in NVS (namespace "arm"), which survive resets and reflashes. clearcal drops them.
 *
 * Channels: 0 pinky 1 ring 2 middle 3 index 4 thumb | 5 wrist_roll 6 wrist_pitch 7 elbow
 *           8 shoulder 9 base.  PCA9685 @0x40 on SDA=GPIO12 SCL=GPIO9, 50 Hz, logic VCC 3.3V.
 * This PCA9685's clock runs ~10% fast, so every value here is COMMANDED us: the MG996R
 * centre is ~1660, not 1500. All calibration was measured in these units, so it's consistent.
 */
#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include "servos.h"

#if CONFIG_IDF_TARGET_ESP32
#define SDA_PIN 21                   // classic ESP32: its default I2C pins (12 is a boot strap there)
#define SCL_PIN 22
#else
#define SDA_PIN 12                   // ESP32-S3; was 8 until 2026-11-23 (SDA wouldn't read on the old S3)
#define SCL_PIN 9
#endif
#define PCA 0x40
#define MODE1 0x00
#define PRESCALE 0xFE
#define LED0_ON_L 0x06
#define ALL_OFF_H 0xFD
#define PRESCALE_50HZ 121            // round(25 MHz / (4096 * 50 Hz) - 1)
#define NCH 10
#define ARM0 5                       // first arm channel
#define TICK_MS 20
#define US_LO 200                    // widest range `lim` accepts
#define US_HI 3000
#define ARM_SPD 4                    // default arm speed, us per tick (~20 deg/s)
#define F_SLOW 11                    // finger speeds, us per tick (the old 8/18/45 per 15 ms)
#define F_MED 24
#define F_FAST 60

static const char* NAME[NCH] = {"pinky","ring","middle","index","thumb",
                                "wrist_roll","wrist_pitch","elbow","shoulder","base"};
static const int OPEN_US[5]  = {1500, 1500, 1500, 1500, 2000};
static const int CLOSE_US[5] = {2600, 2600, 2600, 2600,  500};
// Arm calibration, user-confirmed 2026-11-23 on the 6 V supply.  roll  pitch elbow shoulder base
static const int ARM_LO_DEF[5]   = { 760,  750,  780,  720,  610};
static const int ARM_HI_DEF[5]   = {2560, 2290, 2700, 2200, 2200};   // roll: 2560 is the servo's own end
static const int ARM_HOME_DEF[5] = {1660, 2119, 2650, 2200, 1516};   // folded rest pose (elbow 2650: 2700 pressed on its end)

static bool pcaOK=false, on[NCH], homing=false;
static int curUS[NCH], tgtUS[NCH], lo[NCH], hi[NCH], homeUS[5];
static float pos[NCH], vel[NCH], spd[NCH];   // exact position, speed, top speed (us per tick)
static int armSpd=ARM_SPD, fSpd=40;                     // fingers a bit quicker than "speed med" by default
// each arm joint's share of `spd`: the light wrist moves quicker, the heavily loaded shoulder slower
static const float JSCALE[5]={1.8f, 1.8f, 1.0f, 0.8f, 1.0f};   // roll, pitch, elbow, shoulder, base
static bool smooth=true;                                // motion profile: smooth (minimum-jerk) or classic
static bool tRun[NCH], oRun[NCH];                       // a smooth move / a swing is in progress
static float tP0[NCH], tV0[NCH], tT[NCH], tAt[NCH];     // smooth move: start, start speed, duration, elapsed (ticks; <0 = waiting)
static int tTo[NCH];                                    // smooth move: target
static float oA[NCH], oPer[NCH], oTot[NCH], oAt[NCH];   // swing: amplitude, period, total, elapsed (ticks)
static unsigned long lastTick=0, lastProbe=0;
static Preferences prefs;

// ---------------- PCA9685 ----------------
static void wr(uint8_t reg,uint8_t v){ Wire.beginTransmission(PCA); Wire.write(reg); Wire.write(v); Wire.endTransmission(); }
static uint8_t rd(uint8_t reg){ Wire.beginTransmission(PCA); Wire.write(reg);
  if(Wire.endTransmission()!=0) return 0xFF;
  Wire.requestFrom((uint8_t)PCA,(uint8_t)1); return Wire.available()?Wire.read():0xFF; }
static void setPWM(uint8_t ch,uint16_t onC,uint16_t offC){
  Wire.beginTransmission(PCA); Wire.write(LED0_ON_L+4*ch);
  Wire.write(onC&0xFF); Wire.write(onC>>8); Wire.write(offC&0xFF); Wire.write(offC>>8);
  Wire.endTransmission(); }
static void applyUS(int ch,int us){ setPWM(ch,0,(uint16_t)(((uint32_t)us*4096+10000)/20000)); }
static bool probe(){ Wire.beginTransmission(PCA); return Wire.endTransmission()==0; }

// freshly powered / reset PCA9685 -> 50 Hz, every output off, awake
static void pcaWake50Hz(){
  wr(MODE1,0x10); wr(PRESCALE,PRESCALE_50HZ);       // sleep to set 50 Hz
  wr(ALL_OFF_H,0x10);                               // all outputs limp before waking
  wr(MODE1,0x20); delay(5); wr(MODE1,0xA0); }       // wake + restart, auto-increment

// COLD (PCA just powered: asleep / not at 50 Hz): set 50 Hz with every output limp.
// WARM (only the ESP32 reset; PCA still running at 50 Hz): read back what each servo
// is being driven to and keep holding it.
static void pcaInit(){
  uint8_t m1=rd(MODE1), ps=rd(PRESCALE);
  if(m1!=0xFF && !(m1&0x10) && ps==PRESCALE_50HZ){
    wr(MODE1,(m1&0x6F)|0x20);                         // stay awake, ensure auto-increment
    int held=0;
    for(int ch=0;ch<NCH;ch++){ uint8_t b=LED0_ON_L+4*ch;
      uint8_t onL=rd(b),onH=rd(b+1),offL=rd(b+2),offH=rd(b+3);
      if(onH&0x10){ setPWM(ch,0,4096); on[ch]=false; continue; }   // full-ON is never valid for a servo
      if(offH&0x10){ on[ch]=false; continue; }        // already limp
      int cnt=((((offH&0x0F)<<8)|offL)-(((onH&0x0F)<<8)|onL))&0xFFF;
      curUS[ch]=tgtUS[ch]=(int)((cnt*20000L+2048)/4096); pos[ch]=curUS[ch]; vel[ch]=0; on[ch]=true; held++; }
    Serial.printf("PCA9685 OK - WARM start: holding %d servo(s) where they are\n",held);
  } else {
    pcaWake50Hz();
    for(int ch=0;ch<NCH;ch++) on[ch]=false;
    Serial.println("PCA9685 OK - COLD start: all servos LIMP");
  }
  pcaOK=true;
}

// ---------------- calibration (NVS overrides on top of the built-in values) ----------------
static void saveCal(){ prefs.begin("arm",false); char k[8];
  for(int c=ARM0;c<NCH;c++){
    sprintf(k,"lo%d",c); prefs.putShort(k,lo[c]); sprintf(k,"hi%d",c); prefs.putShort(k,hi[c]);
    sprintf(k,"hm%d",c); prefs.putShort(k,homeUS[c-ARM0]); }
  prefs.end(); }
static void loadCal(){ prefs.begin("arm",false); char k[8];
  for(int c=ARM0;c<NCH;c++){ int i=c-ARM0;
    sprintf(k,"lo%d",c); lo[c]=prefs.getShort(k,ARM_LO_DEF[i]);
    sprintf(k,"hi%d",c); hi[c]=prefs.getShort(k,ARM_HI_DEF[i]);
    sprintf(k,"hm%d",c); homeUS[i]=prefs.getShort(k,ARM_HOME_DEF[i]); }
  prefs.end(); }
static bool calIsBuiltIn(){
  for(int i=0;i<5;i++) if(lo[ARM0+i]!=ARM_LO_DEF[i]||hi[ARM0+i]!=ARM_HI_DEF[i]||homeUS[i]!=ARM_HOME_DEF[i]) return false;
  return true; }
static void dumpCal(){
  Serial.println("// ch5..ch9 = wrist_roll, wrist_pitch, elbow, shoulder, base");
  Serial.printf("ARM_LO  = {%d, %d, %d, %d, %d};\n",lo[5],lo[6],lo[7],lo[8],lo[9]);
  Serial.printf("ARM_HI  = {%d, %d, %d, %d, %d};\n",hi[5],hi[6],hi[7],hi[8],hi[9]);
  Serial.printf("HOME_US = {%d, %d, %d, %d, %d};\n",homeUS[0],homeUS[1],homeUS[2],homeUS[3],homeUS[4]);
  Serial.println(calIsBuiltIn()?"(= built-in calibration)":"(saved overrides differ from the built-in calibration)"); }

// ---------------- motion ----------------
// Two profiles (`motion smooth|classic`); no instant starts/stops in either - those make a
// loaded arm lurch.
//  smooth (default): minimum-jerk - the speed rises and falls as a bell curve, like a human
//    reach. Top speed grows with the square root of the distance, so long moves take longer
//    but not proportionally. In a pose the elbow and wrist start a beat after the shoulder
//    (follow-through). A new target mid-move blends in from the current speed.
//  classic: accelerate to the joint's top speed, cruise, brake in time to land on the target.
// `osc` swings one joint around where it stands (handshake, wave) in either profile.
// never yank: the limits always stretch to include where the joint already is
static void setTarget(int ch,int us){ tgtUS[ch]=constrain(us,min(lo[ch],curUS[ch]),max(hi[ch],curUS[ch])); }
static float accel(int ch){ return ch<ARM0 ? spd[ch]/3.0f : spd[ch]/20.0f; }   // classic arm: full speed in 0.4 s
static float jointSpd(int c){ return c<ARM0 ? fSpd : armSpd*JSCALE[c-ARM0]; }
static bool inHome(int c){ return on[c] && homeUS[c-ARM0]>=0; }
static int homeOf(int c){ return constrain(homeUS[c-ARM0],lo[c],hi[c]); }
static bool atHome(){ for(int c=ARM0;c<NCH;c++) if(inHome(c) && (curUS[c]!=homeOf(c) || vel[c]!=0)) return false; return true; }

// stand still at `us`, dropping any move or swing (the caller drives the servo)
static void standAt(int ch,int us){ curUS[ch]=tgtUS[ch]=us; pos[ch]=us; vel[ch]=0; tRun[ch]=oRun[ch]=false; }
// new exact position; the servo gets a new pulse only when the whole us changes
static void moveTo(int ch,float x){ int us=lroundf(x); pos[ch]=x; if(us!=curUS[ch]){ curUS[ch]=us; applyUS(ch,us); } }
static void arrived(int ch,int us){ moveTo(ch,us); vel[ch]=0;
  if(ch>=ARM0) Serial.printf("%s (ch%d) at %d\n",NAME[ch],ch,us); }

static void classicStep(int ch){
  float d=tgtUS[ch]-pos[ch], a=accel(ch);
  if(fabsf(d)<=a && fabsf(vel[ch])<=a){                      // arrived: within one accel step -> snap on
    if(vel[ch]!=0) arrived(ch,tgtUS[ch]);
    return; }
  // top speed that can still stop exactly on the target braking by `a` per 20 ms tick
  // (the continuous sqrt(2ad) overshoots in discrete steps and could hunt forever)
  float want=copysignf(fminf(spd[ch],a*(sqrtf(2*fabsf(d)/a+0.25f)-0.5f)),d);
  vel[ch]+=constrain(want-vel[ch],-a,a);
  moveTo(ch,pos[ch]+vel[ch]); }

// smooth move duration in ticks: top speed = spd x 0.6..1.6 (sqrt of distance / 500 us); never
// under 0.3 s for a finger or 0.6 s for an arm joint, so even a small arm move visibly eases in and out
static float moveTicks(float dist,float s,int ch){
  float g=constrain(sqrtf(dist/500.0f),0.6f,1.6f);
  return fmaxf(ch<ARM0?15.0f:30.0f,1.2f*dist/(fmaxf(s,0.3f)*g)); }
// start from where the joint is, keeping its current speed; wait `delayT` ticks first (only from standstill)
static void startMove(int ch,int to,float T,float delayT){
  if(vel[ch]!=0) delayT=0;
  tP0[ch]=pos[ch]; tV0[ch]=vel[ch]; tTo[ch]=to; tT[ch]=T; tAt[ch]=-delayT; tRun[ch]=true; }
// quintic from (start, start speed) to (target, at rest, no acceleration); u = 0..1.
// With a zero start speed it is the classic minimum-jerk curve 10u^3 - 15u^4 + 6u^5.
static float mjPos(int ch,float u){
  float T=tT[ch], V=tV0[ch]*T, D=tTo[ch]-tP0[ch];
  float a3=10*D-6*V, a4=-15*D+8*V, a5=6*D-3*V, u3=u*u*u;
  return tP0[ch]+V*u+u3*(a3+u*(a4+u*a5)); }

static void smoothStep(int ch){
  bool retarget=tRun[ch] ? tTo[ch]!=tgtUS[ch] : (fabsf(tgtUS[ch]-pos[ch])>=0.5f || vel[ch]!=0);
  if(retarget) startMove(ch,tgtUS[ch],moveTicks(fabsf(tgtUS[ch]-pos[ch]),spd[ch],ch),0);
  if(!tRun[ch]) return;
  if(++tAt[ch]<=0) return;                                   // staggered start: still waiting
  if(tAt[ch]>=tT[ch]){ tRun[ch]=false; arrived(ch,tTo[ch]); return; }
  float x=constrain(mjPos(ch,tAt[ch]/tT[ch]),(float)min(lo[ch],(int)tP0[ch]),(float)max(hi[ch],(int)tP0[ch]));
  vel[ch]=x-pos[ch]; moveTo(ch,x); }

// swing around the centre (tgtUS): the amplitude eases in over the first period and out over
// the last, so the joint starts and ends at rest exactly on the centre
static void oscStep(int ch){
  float t=++oAt[ch], P=oPer[ch];
  if(t>=oTot[ch]){ oRun[ch]=false; arrived(ch,tgtUS[ch]); return; }
  float e=fminf(1.0f,fminf(t/P,(oTot[ch]-t)/P)); e=e*e*(3-2*e);
  float x=constrain(tgtUS[ch]+oA[ch]*e*sinf(2*PI*t/P),(float)lo[ch],(float)hi[ch]);
  vel[ch]=x-pos[ch]; moveTo(ch,x); }

// ---------------- one joint at a time (`seq on`, the default) ----------------
// Several MG996Rs starting together pull the servo supply down (the PCA9685 even reset), so by
// default pose/home queue the joints and move them one after another. With `blend` the next
// joint already starts during the last part of the previous move, where that one is slowing
// down, so the motion flows without stops while only one joint accelerates at a time.
//  Leaving home: base -> elbow halfway -> shoulder -> elbow the rest -> wrist pitch -> roll
//    (the forearm lifts off first, then the arm raises and extends, like a person).
//  Going home: roll -> pitch -> shoulder -> elbow -> base (the shoulder leans back before the
//    elbow folds; an elbow much straighter than a right angle first bends to ELBOW_RIGHT, so an
//    outstretched arm never swings back into the desk).
#define ELBOW_CH 7
#define ELBOW_RIGHT 1700                                     // elbow at about 90 degrees
static bool seqMode=true, aqRun=false, aqEnd=false;
static int blendPct=30;
static int aqCh[8], aqTo[8], aqN=0, aqI=0;
static void aqAdd(int c,int to){ if(aqN<8){ aqCh[aqN]=c; aqTo[aqN]=to; aqN++; } }
static void aqBegin(int k){ int c=aqCh[k]; spd[c]=jointSpd(c); setTarget(c,aqTo[k]); }
static bool arrivedAt(int c){ return !on[c] || (curUS[c]==tgtUS[c] && vel[c]==0 && !tRun[c]); }
static void armSeqTick(){
  if(!aqRun) return;
  if(aqEnd){                                                 // all started: wait until every joint has landed
    for(int k=0;k<aqN;k++) if(!arrivedAt(aqCh[k])) return;
    aqRun=aqEnd=false; if(!homing) Serial.println("pose done"); return; }
  int c=aqCh[aqI];
  bool tail=smooth && tRun[c] && tAt[c]>=tT[c]*(1.0f-blendPct/100.0f);   // slowing down: the next may start
  if(!arrivedAt(c) && !tail) return;
  if(++aqI>=aqN){ aqEnd=true; return; }
  aqBegin(aqI); }

static void tick(){
  for(int ch=0;ch<NCH;ch++){ if(!on[ch]) continue;
    if(oRun[ch]) oscStep(ch); else if(smooth) smoothStep(ch); else classicStep(ch); }
  armSeqTick();
  if(homing && atHome()){ homing=false; Serial.println("home reached"); }
}

// Move several arm joints: one at a time (seq on), or all arriving TOGETHER (seq off):
//  smooth: one shared duration (the longest joint's); the elbow starts 60 ms and the wrist
//    120 ms after the shoulder/base, so the motion flows out along the arm.
//  classic: each joint's top speed (and so its acceleration) is scaled by its share of the longest move.
// t[] = targets for ch5..ch9; -1 leaves that joint alone. Returns how many joints move.
static const float STAGGER[5]={6,6,3,0,0};                   // ticks: roll, pitch, elbow, shoulder, base
static int poseTo(const int* t,bool towardHome=false){
  int tg[5]; float d[5], dmax=0, T=0; int n=0;
  aqRun=false;                                               // a new pose replaces any queued one
  for(int i=0;i<5;i++){ int c=ARM0+i; tg[i]=-1;
    if(t[i]<0 || !on[c]) continue;
    tg[i]=constrain(t[i],min(lo[c],curUS[c]),max(hi[c],curUS[c]));
    d[i]=fabsf(tg[i]-pos[c]); dmax=fmaxf(dmax,d[i]); T=fmaxf(T,moveTicks(d[i],jointSpd(c),c)); n++; }
  if(seqMode){                                               // i: 0 roll 1 pitch 2 elbow 3 shoulder 4 base
    aqN=0; aqI=0; aqEnd=false;
    bool el=tg[2]>=0 && d[2]>=0.5f, sh=tg[3]>=0 && d[3]>=0.5f;
    auto add=[&](int i){ if(tg[i]>=0 && d[i]>=0.5f) aqAdd(ARM0+i,tg[i]); };
    if(!towardHome){
      add(4);
      if(el && sh) aqAdd(ELBOW_CH,(curUS[ELBOW_CH]+tg[2])/2);                      // elbow halfway first
      add(3); add(2); add(1); add(0); }
    else {
      add(0); add(1);
      if(el && sh && curUS[ELBOW_CH]<ELBOW_RIGHT-100) aqAdd(ELBOW_CH,ELBOW_RIGHT); // straight arm: bend to ~90 first
      add(3); add(2); add(4); }
    aqRun=aqN>0; if(aqRun) aqBegin(0);
    return aqN; }
  for(int i=0;i<5;i++){ int c=ARM0+i; if(tg[i]<0) continue;
    tgtUS[c]=tg[i];
    if(smooth){ spd[c]=jointSpd(c); if(d[i]>=0.5f || vel[c]!=0) startMove(c,tg[i],T,STAGGER[i]); }
    else spd[c]=dmax>0 ? fmaxf(0.3f,armSpd*d[i]/dmax) : armSpd; }
  return n; }

static void goHome(){
  int t[5], k=0; for(int i=0;i<5;i++){ t[i]=inHome(ARM0+i)?homeOf(ARM0+i):-1; if(t[i]>=0) k++; }
  if(!k){ Serial.println("home: no ON arm joint has a home"); return; }
  int n=poseTo(t,true);
  if(atHome()) Serial.println("home reached");
  else { homing=true; Serial.printf("going home (%d step%s%s)\n",n,n==1?"":"s",seqMode?", one joint at a time":""); } }

static void enableAt(int ch,int us){
  if(on[ch]){ Serial.printf("%s is already ON\n",NAME[ch]); return; }
  us=constrain(us,lo[ch],hi[ch]); spd[ch]=jointSpd(ch);
  standAt(ch,us); applyUS(ch,us); on[ch]=true;
  Serial.printf("%s ON at %d\n",NAME[ch],us); }
static void limp(int ch){ setPWM(ch,0,4096); on[ch]=false; vel[ch]=0; tRun[ch]=oRun[ch]=false;   // full-OFF bit: no pulses -> limp
  if(ch>=ARM0) aqRun=false; }

// ---------------- finger gestures: a short pose list played by the tick (never blocks) ----------------
#define GMAX 10
struct FPose{ int us[5]; int step; int holdMs; };            // us -1 = leave that finger alone
static FPose gq[GMAX]; static int gN=0, gI=0, gPhase=0; static unsigned long gUntil=0;
static char gName[20]="";

static void gBegin(const char* name){ gN=gI=gPhase=0; strncpy(gName,name,sizeof(gName)-1); gName[sizeof(gName)-1]=0; }
static void gAdd(const int* us,int step,int holdMs=0){
  if(gN>=GMAX) return; FPose& p=gq[gN++];
  for(int i=0;i<5;i++) p.us[i]=us[i]; p.step=step; p.holdMs=holdMs; }
static void gStop(){ gN=gI=gPhase=0; }
// The thumb's servo screw catches on the arm if the thumb closes at home, so while the arm is
// at home - or limp, which means it is resting there - gestures never close the thumb.
static bool armAtRest(){ for(int c=ARM0;c<NCH;c++) if(on[c] && (curUS[c]!=homeOf(c) || vel[c]!=0)) return false; return true; }
static void fingersTo(const FPose& p){
  for(int i=0;i<5;i++){ if(p.us[i]<0) continue; int us=p.us[i]; spd[i]=p.step;
    if(i==4 && us<OPEN_US[4] && armAtRest()) us=OPEN_US[4];   // thumb stays open at home
    if(!on[i]){ us=constrain(us,lo[i],hi[i]);                 // light fingers: switching on with a jump is harmless
      standAt(i,us); applyUS(i,us); on[i]=true; }
    else setTarget(i,us); } }
static bool fingersArrived(const FPose& p){
  for(int i=0;i<5;i++) if(p.us[i]>=0 && on[i] && (curUS[i]!=tgtUS[i] || vel[i]!=0)) return false;
  return true; }
static void gestureTick(){
  if(gI>=gN) return;
  FPose& p=gq[gI];
  if(gPhase==0){ fingersTo(p); gPhase=1; }
  else if(gPhase==1){ if(fingersArrived(p)){ gUntil=millis()+p.holdMs; gPhase=2; } }
  else if((long)(millis()-gUntil)>=0){ gPhase=0; if(++gI>=gN) Serial.println(gName); }   // report when done
}

// mask: 5 chars, pinky ring middle index thumb, '1' = open (extended)
static void maskPose(int* t,const char* m){ for(int i=0;i<5;i++) t[i]=(m[i]=='1')?OPEN_US[i]:CLOSE_US[i]; }
static void playMask(const char* name,const char* m,int step){ int t[5]; maskPose(t,m); gBegin(name); gAdd(t,step); }
static void hello(){
  int th=(OPEN_US[4]+CLOSE_US[4])/2, flat[5], curl[5];
  for(int i=0;i<4;i++){ flat[i]=OPEN_US[i]; curl[i]=OPEN_US[i]+(int)(0.8f*(CLOSE_US[i]-OPEN_US[i])); }
  flat[4]=curl[4]=th;
  gBegin("hello"); gAdd(flat,53);
  for(int k=0;k<3;k++){ gAdd(curl,47); gAdd(flat,47); }
  gAdd(CLOSE_US,fSpd); }
static void middleFinger(bool hold){
  int t[5]; maskPose(t,"00100");
  if(hold){ gBegin("middlehold"); gAdd(t,60); }
  else { gBegin("middle"); gAdd(t,60,3000); gAdd(CLOSE_US,fSpd); } }
static void countN(int n){
  static const int order[5]={3,2,1,0,4};                    // index, middle, ring, pinky, thumb
  char m[6]="00000"; for(int i=0;i<n&&i<5;i++) m[order[i]]='1';
  playMask("count",m,fSpd); }
static void oneFinger(int ch,int us,const char* name){ int t[5]={-1,-1,-1,-1,-1}; t[ch]=us; gBegin(name); gAdd(t,fSpd); }

// ---------------- serial ----------------
static bool chOK(int ch){ if(ch>=0&&ch<NCH) return true; Serial.println("bad channel (0-9)"); return false; }
static bool fingerOK(int ch){ if(ch>=0&&ch<ARM0) return true; Serial.println("finger channel 0-4 only (arm: j / en)"); return false; }
static bool needOn(int ch){ if(on[ch]) return true; Serial.printf("%s (ch%d) is limp - use: en %d <us>\n",NAME[ch],ch,ch); return false; }

static void showPos(){
  for(int ch=0;ch<NCH;ch++){
    if(on[ch]) Serial.printf("ch%d %-11s ON   %4d -> %4d   lim %d..%d",ch,NAME[ch],curUS[ch],tgtUS[ch],lo[ch],hi[ch]);
    else       Serial.printf("ch%d %-11s limp                 lim %d..%d",ch,NAME[ch],lo[ch],hi[ch]);
    if(ch>=ARM0) Serial.printf("   home %d",homeUS[ch-ARM0]);
    Serial.println(); } }
static void scan(){ int n=0;
  for(uint8_t a=1;a<127;a++){ Wire.beginTransmission(a);
    if(Wire.endTransmission()==0){ Serial.printf("I2C device at 0x%02X\n",a); n++; } }
  if(n) return;
  Serial.printf("no I2C devices - check SDA=%d SCL=%d, VCC=3.3V, GND\n",SDA_PIN,SCL_PIN);
  // the servo board's own pull-ups hold idle SDA/SCL HIGH whenever its VCC is powered
  Wire.end(); pinMode(SDA_PIN,INPUT_PULLDOWN); pinMode(SCL_PIN,INPUT_PULLDOWN); delay(2);
  int sda=digitalRead(SDA_PIN), scl=digitalRead(SCL_PIN);
  if(!sda && scl){   // a chip cut off mid-message can hold SDA low: clock it free, then send a STOP
    pinMode(SCL_PIN,OUTPUT_OPEN_DRAIN); pinMode(SDA_PIN,OUTPUT_OPEN_DRAIN); digitalWrite(SDA_PIN,HIGH);
    for(int i=0;i<16;i++){ digitalWrite(SCL_PIN,LOW); delayMicroseconds(10); digitalWrite(SCL_PIN,HIGH); delayMicroseconds(10); }
    digitalWrite(SDA_PIN,LOW); delayMicroseconds(10); digitalWrite(SDA_PIN,HIGH); delayMicroseconds(10);   // STOP
    pinMode(SDA_PIN,INPUT_PULLDOWN); pinMode(SCL_PIN,INPUT_PULLDOWN); delay(2);
    int after=digitalRead(SDA_PIN);
    Serial.printf("  bus recovery (16 clocks + STOP): SDA now %s\n",after?"HIGH - released":"still low");
    sda=after; }
  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(100000);
  if(sda && probe() && !pcaOK){ Serial.println("  PCA9685 answers now"); pcaInit(); }
  Serial.printf("  idle lines (weakly pulled down here): SDA=%s SCL=%s -> %s\n",sda?"HIGH":"low",scl?"HIGH":"low",
    sda&&scl ? "the board's pull-ups are powered: check the SDA/SCL wires (swapped? wrong pins?)"
    : sda||scl ? "the servo board has power, but the LOW line is not reaching it (loose / wrong pin) or is held low"
               : "no pull-up voltage: the servo board's VCC (3.3 V) or GND is probably missing"); }

void servoCommand(char* s){
  for(char* p=s;*p;p++) *p=tolower(*p);
  int ch,a,b,n; char w[8]; int t[5];
  if(!strcmp(s,"x")||!strcmp(s,"off all")){ wr(ALL_OFF_H,0x10);   // one byte, works in any state
    for(int i=0;i<NCH;i++){ on[i]=false; vel[i]=0; tRun[i]=oRun[i]=false; } homing=aqRun=false; gStop(); Serial.println("ALL LIMP"); return; }
  if(!strcmp(s,"scan")){ scan(); return; }
  if(!strcmp(s,"pcatest")){   // I2C general-call software reset = what a power dip does; the watchdog must catch it
    Wire.beginTransmission(0x00); Wire.write(0x06); Wire.endTransmission();
    Serial.println("PCA9685 software reset sent - the watchdog should restore within 0.2 s"); return; }
  if(!strcmp(s,"dump")){ dumpCal(); return; }
  if(!strcmp(s,"clearcal")){ prefs.begin("arm",false); prefs.clear(); prefs.end(); loadCal();
    Serial.println("saved calibration cleared - using the built-in values"); return; }
  if(!pcaOK){ Serial.println("PCA9685 not found yet (try: scan)"); return; }
  // ---- hand (same words as the old hand firmware, which the Pi already sends) ----
  if(!strcmp(s,"fist")||!strcmp(s,"rest")||!strcmp(s,"lower")){ gBegin("fist"); gAdd(CLOSE_US,fSpd); }
  else if(!strcmp(s,"open")){ gBegin("open"); gAdd(OPEN_US,fSpd); }
  else if(!strcmp(s,"hello")) hello();
  else if(!strcmp(s,"middle")) middleFinger(false);
  else if(!strcmp(s,"middlehold")) middleFinger(true);
  else if(!strcmp(s,"point")) playMask("point","00010",fSpd);
  else if(!strcmp(s,"peace")) playMask("peace","00110",fSpd);
  else if(!strcmp(s,"thumbs")||!strcmp(s,"thumbsup")) playMask("thumbs","00001",fSpd);
  else if(sscanf(s,"count %d",&a)==1) countN(constrain(a,0,5));
  else if(!strncmp(s,"set ",4)){ char* m=s+4; while(*m==' ') m++;
    if(strlen(m)>=5){ char nm[12]; snprintf(nm,sizeof(nm),"set %.5s",m); playMask(nm,m,60); }
    else Serial.println("usage: set <5 bits: pinky ring middle index thumb, 1 = open>"); }
  else if(sscanf(s,"grip %d",&a)==1){ a=constrain(a,0,100); int g[5]; char nm[12];   // 0 = open .. 100 = closed, all fingers
    for(int i=0;i<5;i++) g[i]=OPEN_US[i]+(CLOSE_US[i]-OPEN_US[i])*a/100;
    snprintf(nm,sizeof(nm),"grip %d",a); gBegin(nm); gAdd(g,fSpd); }
  else if(sscanf(s,"speed %d",&a)==1){ fSpd=constrain(a,5,120); Serial.printf("speed step=%d\n",fSpd); }
  else if(!strncmp(s,"speed",5)){ fSpd=strstr(s,"slow")?F_SLOW:strstr(s,"fast")?F_FAST:F_MED;
    Serial.printf("speed step=%d\n",fSpd); }
  else if(sscanf(s,"f %d %7s",&ch,w)==2){ if(fingerOK(ch)){ bool op=!strcmp(w,"open"); char nm[16];
      snprintf(nm,sizeof(nm),"f%d %s",ch,op?"open":"close"); oneFinger(ch,op?OPEN_US[ch]:CLOSE_US[ch],nm); } }
  else if(sscanf(s,"m %d %d",&ch,&a)==2){ if(fingerOK(ch)){ char nm[16];
      snprintf(nm,sizeof(nm),"m%d %d",ch,constrain(a,lo[ch],hi[ch])); oneFinger(ch,a,nm); } }
  else if(sscanf(s,"s %d %d",&ch,&a)==2){ if(fingerOK(ch)){ gStop(); a=constrain(a,lo[ch],hi[ch]);
      standAt(ch,a); applyUS(ch,a); on[ch]=true; Serial.printf("s%d %d\n",ch,a); } }
  // ---- arm (+ any channel) ----
  else if(!strcmp(s,"pos")) showPos();
  else if(!strcmp(s,"home")){                                // hand to its home grip first: four fingers closed, thumb open
    int g[5]={CLOSE_US[0],CLOSE_US[1],CLOSE_US[2],CLOSE_US[3],OPEN_US[4]};
    gBegin("home grip"); gAdd(g,fSpd); goHome(); }
  else if(sscanf(s,"pose %d %d %d %d %d",&t[0],&t[1],&t[2],&t[3],&t[4])==5){ homing=false;
    int n=poseTo(t); Serial.printf("pose: %d %s\n",n,seqMode?"step(s), one joint at a time":"joint(s) moving together");
    if(seqMode && !n) Serial.println("pose done"); }
  else if(sscanf(s,"blend %d",&a)==1){ blendPct=constrain(a,0,60);
    Serial.printf("blend %d%%: the next joint starts during the last %d%% of a move\n",blendPct,blendPct); }
  else if(!strncmp(s,"seq",3)){ if(strstr(s,"off")) seqMode=false; else if(strstr(s,"on")) seqMode=true;
    Serial.printf("seq %s\n",seqMode?"on: pose/home move one joint at a time":"off: pose/home move the joints together"); }
  else if(sscanf(s,"en %d %d",&ch,&a)==2){ if(chOK(ch)) enableAt(ch,a); }
  else if(sscanf(s,"j %d %d",&ch,&a)==2){ if(chOK(ch)&&needOn(ch)){ spd[ch]=jointSpd(ch); homing=false; if(ch>=ARM0) aqRun=false;   // a manual move cancels a queued pose
      setTarget(ch,a); Serial.printf("%s -> %d\n",NAME[ch],tgtUS[ch]);
      if(ch>=ARM0 && tgtUS[ch]==curUS[ch] && vel[ch]==0 && !tRun[ch]) Serial.printf("%s (ch%d) at %d\n",NAME[ch],ch,curUS[ch]); } }   // already there
  else if(sscanf(s,"n %d %d",&ch,&a)==2){ if(chOK(ch)&&needOn(ch)){ spd[ch]=jointSpd(ch); homing=false; if(ch>=ARM0) aqRun=false;   // a manual move cancels a queued pose
      setTarget(ch,tgtUS[ch]+a); Serial.printf("%s -> %d\n",NAME[ch],tgtUS[ch]); } }
  else if(sscanf(s,"off %d",&ch)==1){ if(chOK(ch)){ if(ch<ARM0) gStop(); limp(ch); Serial.printf("%s (ch%d) limp\n",NAME[ch],ch); } }
  else if(sscanf(s,"spd %d",&a)==1){ armSpd=constrain(a,1,20); for(int c=ARM0;c<NCH;c++) spd[c]=jointSpd(c);
    Serial.printf("arm speed %d us per %d ms (wrist x1.8, shoulder x0.8)\n",armSpd,TICK_MS); }
  else if(!strncmp(s,"motion",6)){
    bool want=strstr(s,"classic")?false:strstr(s,"smooth")?true:smooth;
    if(want!=smooth){ smooth=want; for(int c=0;c<NCH;c++) tRun[c]=false; }   // picks up from the current position + speed
    Serial.printf("motion %s\n",smooth?"smooth (minimum-jerk)":"classic (accel/cruise/brake)"); }
  else if(sscanf(s,"osc %d %d %d %d",&ch,&a,&b,&n)==4){ if(chOK(ch)&&needOn(ch)){
      if(curUS[ch]!=tgtUS[ch] || vel[ch]!=0 || oRun[ch]) Serial.printf("%s is moving - osc once it stands still\n",NAME[ch]);
      else { if(ch<ARM0) gStop(); else aqRun=false; homing=false; tRun[ch]=false; n=constrain(n,1,20);
        oA[ch]=constrain(a,1,400); oPer[ch]=constrain(b,200,5000)/(float)TICK_MS; oTot[ch]=oPer[ch]*n; oAt[ch]=0; oRun[ch]=true;
        Serial.printf("%s swings +-%d us around %d, %d ms x %d\n",NAME[ch],(int)oA[ch],tgtUS[ch],(int)(oPer[ch]*TICK_MS),n); } } }
  // ---- calibration ----
  else if(sscanf(s,"lim %d %d %d",&ch,&a,&b)==3){ if(chOK(ch)){
      lo[ch]=constrain(min(a,b),US_LO,US_HI); hi[ch]=constrain(max(a,b),US_LO,US_HI); if(ch>=ARM0) saveCal();
      Serial.printf("%s limits %d..%d%s\n",NAME[ch],lo[ch],hi[ch],ch>=ARM0?" (saved)":""); } }
  else if(sscanf(s,"hm %d %d",&ch,&a)==2){ if(chOK(ch)&&ch>=ARM0){
      homeUS[ch-ARM0]=a<0?-1:constrain(a,US_LO,US_HI); saveCal();
      Serial.printf("%s home = %d (saved)\n",NAME[ch],homeUS[ch-ARM0]); } }
  else if(!strcmp(s,"sethome")){ int n=0;
    for(int c=ARM0;c<NCH;c++) if(on[c]){ homeUS[c-ARM0]=curUS[c]; n++; }
    if(n){ saveCal(); dumpCal(); } else Serial.println("no arm joint is ON"); }
  else Serial.printf("? %s  (help)\n",s);
}

void servosBegin(){
  for(int ch=0;ch<NCH;ch++){ on[ch]=false; curUS[ch]=tgtUS[ch]=1500; pos[ch]=1500; vel[ch]=0;
    if(ch<ARM0){ lo[ch]=min(OPEN_US[ch],CLOSE_US[ch]); hi[ch]=max(OPEN_US[ch],CLOSE_US[ch]); spd[ch]=fSpd; }
    else spd[ch]=jointSpd(ch); }
  loadCal();
  Serial.println(calIsBuiltIn()?"arm calibration: built-in":"arm calibration: saved overrides ('dump' shows them)");
  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(100000);
  if(probe()) pcaInit();
  else Serial.printf("PCA9685 NOT FOUND at 0x40 - retrying every 2 s (check SDA=%d SCL=%d, VCC=3.3V, GND)\n",SDA_PIN,SCL_PIN);
}

// ---------------- watchdog: the PCA9685 can reset itself in a power dip ----------------
// It then sleeps with every output off - the servos go limp while we still think they hold.
// Every 200 ms check it is awake at 50 Hz; if not, set it up again and re-send every ON
// servo's position at once, so the arm only sags for a moment. Three resets within a minute
// means the power is not coping: go limp on purpose and say so.
static unsigned long lastCheck=0, resetAt[3]={0,0,0};
static void pcaWatch(){
  uint8_t m1=rd(MODE1), ps=rd(PRESCALE);
  if(m1==0xFF) return;                              // no answer this time: check again next round
  if(!(m1&0x10) && ps==PRESCALE_50HZ) return;       // awake at 50 Hz: fine
  unsigned long now=millis();
  resetAt[0]=resetAt[1]; resetAt[1]=resetAt[2]; resetAt[2]=now;
  pcaWake50Hz();
  if(resetAt[0] && now-resetAt[0]<60000){
    for(int ch=0;ch<NCH;ch++){ on[ch]=false; vel[ch]=0; tRun[ch]=oRun[ch]=false; }
    homing=aqRun=false; gStop();
    Serial.println("WARNING: PCA9685 reset 3 times in a minute - the servo power is dipping. ALL LIMP for safety.");
    return; }
  int n=0; for(int ch=0;ch<NCH;ch++) if(on[ch]){ applyUS(ch,curUS[ch]); n++; }
  Serial.printf("WARNING: PCA9685 reset (power dip?) - restored %d servo(s)\n",n); }

void servosLoop(){
  unsigned long now=millis();
  if(!pcaOK && now-lastProbe>2000){ lastProbe=now; if(probe()) pcaInit(); }
  if(pcaOK && now-lastCheck>=200){ lastCheck=now; pcaWatch(); }
  if(pcaOK && now-lastTick>=TICK_MS){ lastTick=now; gestureTick(); tick(); }
}
