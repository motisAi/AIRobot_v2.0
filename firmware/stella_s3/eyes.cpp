/* eyes.cpp - REALISTIC animated eyes on two GC9D01 160x160 LCDs, hardware SPI, NO TFT_eSPI.
 * (TFT_eSPI's hw-SPI path garbles the GC9D01 on the S3 - horizontal stripes - so the panels
 * are driven directly: one 160x160 RAM framebuffer pushed over FSPI to both panels.)
 *
 * Look: beige skin + beige top-down eyelids, off-white sclera with thin branching veins,
 * detailed iris (limbal ring, radial fibers, collarette, pupil, corneal glints).
 * Fast blinks, ~35% of them doubles. With no command for 4 s the gaze drifts on its own.
 *
 * The two panels differ by a pure 180 deg rotation (MADCTL eye1 0xA0, eye2 0x60), so the SAME
 * framebuffer upright on both = gaze naturally synced. Don't render a mirrored copy per eye.
 *
 * Everything below runs on its own task on core 0 (init included): pushing a frame to both
 * panels takes ~41 ms at 20 MHz, which would stall the 20 ms servo tick on core 1.
 * Commands arrive from core 1 and only set "request" variables that the task picks up.
 *
 * Pins (module VCC = 5V):
 *   ESP32-S3:      DIN=4  CLK=7  DC=10  RST1=11 RST2=5   CS1=2  CS2=6   BL1=1  BL2=3
 *   classic ESP32: DIN=23 CLK=18 DC=19  RST1=27 RST2=32  CS1=25 CS2=26  BL1=33 BL2=4
 *     (GPIO6-11 are its flash, 1/3 its USB serial, 0/2/5/12/15 boot straps - all avoided)
 */
#include <Arduino.h>
#include <SPI.h>
#include <math.h>
#include "eyes.h"

#if CONFIG_IDF_TARGET_ESP32
#define PIN_DIN 23
#define PIN_CLK 18
#define PIN_DC 19
#define PIN_RST1 27
#define PIN_RST2 32
#define PIN_CS1 25
#define PIN_CS2 26
#define PIN_BL1 33
#define PIN_BL2 4
#define LCD_SPI_HOST VSPI            // FSPI is the flash bus on the classic ESP32
#else
#define PIN_DIN 4
#define PIN_CLK 7
#define PIN_DC 10
#define PIN_RST1 11
#define PIN_RST2 5
#define PIN_CS1 2
#define PIN_CS2 6
#define PIN_BL1 1
#define PIN_BL2 3
#define LCD_SPI_HOST FSPI
#endif
#define N 160

#define RGB(r,g,b) ((uint16_t)((((r)&0xF8)<<8)|(((g)&0xFC)<<3)|((b)>>3)))
#define SKIN        RGB(226,190,152)
#define SKIN_LID    RGB(202,166,132)
#define CREASE      RGB(150,112,88)
#define SCLERA      RGB(244,240,233)
#define VEIN        RGB(196,96,90)
#define VEIN2       RGB(216,138,132)   // finer, lighter capillaries
#define IRIS_DARK   RGB(16,36,64)
#define IRIS_MID    RGB(52,104,150)
#define IRIS_LIGHT  RGB(120,172,214)
#define PUPILC      RGB(0,0,0)
#define HILITE      RGB(255,255,255)
#define GLINT2      RGB(190,214,236)

#define EYE_C 80
#define SCL_A 78          // sclera half-width
#define SCL_B 58          // sclera half-height
#define IRIS_R 40
#define PUPIL_R 16
#define MAX_OFFX 24
#define MAX_OFFY 14

static SPIClass lcdspi(LCD_SPI_HOST);
static uint16_t fb[N*N];                   // stored byte-swapped, ready to send
static uint8_t madctl[2] = {0xA0, 0x60};   // eye1 (her left, CS1) / eye2 (her right, CS2) - both upright

// ---- written by core 1 (eyeCommand), read by the eye task ----
static volatile float tgx=0, tgy=0;        // gaze target -1..1
static volatile int emotion=0;             // 0 neutral 1 happy 2 sad/angry 3 surprised 4 sleepy
static volatile float pupilScale=1.0f;
static volatile int blinkReq=0;            // >0: blink this many times
static volatile int mode=0;                // 0 = eye, 1 = orientation marker
static volatile int madReq[2]={-1,-1};     // MADCTL change requests
static volatile unsigned long lastCmd=0;

static inline uint16_t sw(uint16_t c){ return (uint16_t)((c>>8)|(c<<8)); }
static inline void wcmd(uint8_t c){ digitalWrite(PIN_DC,LOW);  lcdspi.transfer(c); }
static inline void wdat(uint8_t d){ digitalWrite(PIN_DC,HIGH); lcdspi.transfer(d); }
static inline void selectEye(int cs){
  digitalWrite(PIN_CS1, cs==PIN_CS1?LOW:HIGH);
  digitalWrite(PIN_CS2, cs==PIN_CS2?LOW:HIGH);
}
static inline void px(int x,int y,uint16_t v){ if((unsigned)x<N&&(unsigned)y<N) fb[y*N+x]=v; }

static void fbFill(uint16_t c){ uint16_t v=sw(c); for(int i=0;i<N*N;i++) fb[i]=v; }
static void fbRect(int x,int y,int w,int h,uint16_t c){ uint16_t v=sw(c);
  int x0=max(0,x),y0=max(0,y),x1=min(N,x+w),y1=min(N,y+h);
  for(int yy=y0;yy<y1;yy++){ uint16_t* r=&fb[yy*N+x0]; for(int xx=x0;xx<x1;xx++)*r++=v; } }
static void fbDisk(int cx,int cy,int r,uint16_t c){ uint16_t v=sw(c); int r2=r*r;
  for(int y=max(0,cy-r);y<=min(N-1,cy+r);y++){ int dy=y-cy,dx2=r2-dy*dy; if(dx2<0)continue;
    int dx=(int)sqrtf((float)dx2); uint16_t* row=&fb[y*N+max(0,cx-dx)];
    for(int x=max(0,cx-dx);x<=min(N-1,cx+dx);x++)*row++=v; } }
static void fbEllipse(int cx,int cy,int a,int b,uint16_t c){ uint16_t v=sw(c);
  for(int y=max(0,cy-b);y<=min(N-1,cy+b);y++){ float t=(float)(y-cy)/b; float k=1.0f-t*t;
    if(k<0)continue; int dx=(int)(a*sqrtf(k)); uint16_t* row=&fb[y*N+max(0,cx-dx)];
    for(int x=max(0,cx-dx);x<=min(N-1,cx+dx);x++)*row++=v; } }
static void fbLine(int x0,int y0,int x1,int y1,uint16_t c){ uint16_t v=sw(c);
  int dx=abs(x1-x0),sx=x0<x1?1:-1,dy=-abs(y1-y0),sy=y0<y1?1:-1,e=dx+dy;
  for(;;){ px(x0,y0,v); if(x0==x1&&y0==y1)break; int e2=2*e;
    if(e2>=dy){e+=dy;x0+=sx;} if(e2<=dx){e+=dx;y0+=sy;} } }

static void setMADCTL(int cs,uint8_t v){ selectEye(cs); wcmd(0x36); wdat(v); selectEye(-1); }
static void pushTo(int cs){ selectEye(cs);
  wcmd(0x2A); wdat(0);wdat(0);wdat(0);wdat(0x9F);
  wcmd(0x2B); wdat(0);wdat(0);wdat(0);wdat(0x9F);
  wcmd(0x2C); digitalWrite(PIN_DC,HIGH);
  lcdspi.writeBytes((uint8_t*)fb,sizeof(fb)); selectEye(-1); }
static void refresh(){ pushTo(PIN_CS1); pushTo(PIN_CS2); }

// ---- scleral veins: fine branching vessels, generated once (stable, no per-frame flicker) ----
struct Seg{ int16_t x0,y0,x1,y1; uint16_t col; };
static Seg veinSeg[200]; static int veinCount=0;
static void addSeg(int x0,int y0,int x1,int y1,uint16_t col){
  if(veinCount<200){ veinSeg[veinCount++]={(int16_t)x0,(int16_t)y0,(int16_t)x1,(int16_t)y1,col}; } }
static void growVein(float x,float y,float a,float len,int depth){
  float x2=x+cosf(a)*len, y2=y+sinf(a)*len;
  addSeg((int)x,(int)y,(int)x2,(int)y2, depth>0?VEIN:VEIN2);
  if(depth<=0) return;
  float da=random(22,54)*0.01745f; int sign=random(0,2)?1:-1;
  growVein(x2,y2,a+sign*da,       len*0.62f, depth-1);
  if(random(0,2)) growVein(x2,y2,a-sign*da*0.6f, len*0.52f, depth-1);
}
static void genVeins(){                            // fixed seed -> identical veins every boot
  veinCount=0; randomSeed(0xBEEF);
  const int roots=10;
  for(int i=0;i<roots;i++){
    float ang=i*(2*M_PI/roots)+0.25f;
    float rx=EYE_C+cosf(ang)*(SCL_A-5), ry=EYE_C+sinf(ang)*(SCL_B-5);
    float toC=atan2f(EYE_C-ry,EYE_C-rx)+random(-16,16)*0.01745f;
    growVein(rx,ry,toC, random(15,24), 2);         // 2 branch levels -> fine lines toward center
  }
}
static void drawVeins(){ for(int i=0;i<veinCount;i++)
    fbLine(veinSeg[i].x0,veinSeg[i].y0,veinSeg[i].x1,veinSeg[i].y1,veinSeg[i].col); }

// ---- animation state (eye task only) ----
static float gx=0,gy=0;
static bool blinking=false; static float lid=0.0f; static int blinkDir=0, pendingBlinks=0;
static unsigned long lastBlink=0,lastIdle=0;
static const unsigned long IDLE_MS=4000;
static void startBlink(int times){ if(!blinking){ blinking=true; blinkDir=1; pendingBlinks=times; } }

static void drawEye(float ox,float oy){
  fbFill(SKIN);
  fbEllipse(EYE_C,EYE_C,SCL_A,SCL_B,SCLERA);
  drawVeins();
  int ix=EYE_C+(int)(ox*MAX_OFFX), iy=EYE_C+(int)(oy*MAX_OFFY);
  int emo=emotion;
  uint16_t body=IRIS_MID;
  if(emo==2) body=RGB(150,40,40); else if(emo==1) body=RGB(40,140,90);
  else if(emo==4) body=RGB(90,100,120);
  fbDisk(ix,iy, IRIS_R+3, IRIS_DARK);          // limbal ring + outer iris
  fbDisk(ix,iy, IRIS_R,   body);               // iris body
  for(int k=0;k<44;k++){ float a=k*(2*M_PI/44.0f),cs=cosf(a),sn=sinf(a);   // radial fibers
    uint16_t col=(k&1)?IRIS_LIGHT:IRIS_DARK;
    fbLine(ix+(int)(cs*(PUPIL_R+3)),iy+(int)(sn*(PUPIL_R+3)),
           ix+(int)(cs*(IRIS_R-2)), iy+(int)(sn*(IRIS_R-2)),col); }
  fbDisk(ix,iy, PUPIL_R+4, IRIS_DARK);         // collarette ring around the pupil
  int pr=(int)(PUPIL_R*pupilScale); if(emo==3) pr=PUPIL_R+6;
  fbDisk(ix,iy, pr, PUPILC);                   // pupil
  fbDisk(ix-9,iy-11, 6, HILITE);               // corneal glint
  fbDisk(ix+7,iy+8, 3, GLINT2);                // small secondary glint
  // eyelids (beige), top-down blink; crease line at the lid rim
  int upCy=-95+(int)(lid*135);                 // upper lid sweeps top -> all the way to bottom
  fbDisk(EYE_C,upCy,124,CREASE); fbDisk(EYE_C,upCy,120,SKIN_LID);
  int loCy=256;                                // lower lid static (almond rim); the blink is upper-lid only
  fbDisk(EYE_C,loCy,124,CREASE); fbDisk(EYE_C,loCy,120,SKIN_LID);
}

static void drawMarker(){ fbFill(0x0000);   // orientation check: black background
  fbRect(0,0,N,22,   0xF800);   // RED   bar = TOP
  fbRect(0,0,22,N,   0x07E0);   // GREEN bar = LEFT
  fbRect(0,138,N,22, 0x001F);   // BLUE  bar = BOTTOM
  fbDisk(80,80,10,   0xFFFF); } // white center dot

static void animate(){
  unsigned long now=millis();
  if(now-lastCmd>IDLE_MS && now-lastIdle>1500){
    tgx=random(-70,71)/100.0f; tgy=random(-60,61)/100.0f; lastIdle=now; }
  if(blinkReq){ startBlink(blinkReq); blinkReq=0; }
  if(now-lastBlink>(unsigned long)random(2500,5500)){
    startBlink(random(0,100)<35 ? 2 : 1); lastBlink=now; }   // ~35% of blinks are doubles
  gx+=(tgx-gx)*0.25f; gy+=(tgy-gy)*0.25f;
  if(blinking){ lid+=0.62f*blinkDir;
    if(lid>=1.0f){ lid=1.0f; blinkDir=-1; }
    else if(lid<=0.0f && blinkDir==-1){ lid=0.0f;
      if(--pendingBlinks>0){ blinkDir=1; }                    // immediately blink again (double)
      else blinking=false; } }
  drawEye(gx, gy); refresh();
}

static void initPanel(){               // GC9D01 init, copied from the Waveshare DualEye demo
  wcmd(0xFE); wcmd(0xEF);
  for(uint8_t c=0x80;c<=0x8F;c++){ wcmd(c); wdat(0xFF); }
  wcmd(0x3A); wdat(0x05); wcmd(0xEC); wdat(0x01);
  wcmd(0x74); wdat(0x02);wdat(0x0E);wdat(0x00);wdat(0x00);wdat(0x00);wdat(0x00);wdat(0x00);
  wcmd(0x98); wdat(0x3E); wcmd(0x99); wdat(0x3E);
  wcmd(0xB5); wdat(0x0D);wdat(0x0D);
  wcmd(0x60); wdat(0x38);wdat(0x0F);wdat(0x79);wdat(0x67);
  wcmd(0x61); wdat(0x38);wdat(0x11);wdat(0x79);wdat(0x67);
  wcmd(0x64); wdat(0x38);wdat(0x17);wdat(0x71);wdat(0x5F);wdat(0x79);wdat(0x67);
  wcmd(0x65); wdat(0x38);wdat(0x13);wdat(0x71);wdat(0x5B);wdat(0x79);wdat(0x67);
  wcmd(0x6A); wdat(0x00);wdat(0x00);
  wcmd(0x6C); wdat(0x22);wdat(0x02);wdat(0x22);wdat(0x02);wdat(0x22);wdat(0x22);wdat(0x50);
  wcmd(0x6E);
  { const uint8_t d[]={0x03,0x03,0x01,0x01,0x00,0x00,0x0f,0x0f,0x0d,0x0d,0x0b,0x0b,0x09,0x09,
                       0x00,0x00,0x00,0x00,0x0a,0x0a,0x0c,0x0c,0x0e,0x0e,0x10,0x10,0x00,0x00,
                       0x02,0x02,0x04,0x04}; for(uint8_t i=0;i<sizeof(d);i++) wdat(d[i]); }
  wcmd(0xBF); wdat(0x01); wcmd(0xF9); wdat(0x40);
  wcmd(0x9B); wdat(0x3B); wcmd(0x93); wdat(0x33);wdat(0x7F);wdat(0x00);
  wcmd(0x7E); wdat(0x30);
  wcmd(0x70); wdat(0x0D);wdat(0x02);wdat(0x08);wdat(0x0D);wdat(0x02);wdat(0x08);
  wcmd(0x71); wdat(0x0D);wdat(0x02);wdat(0x08);
  wcmd(0x91); wdat(0x0E);wdat(0x09);
  wcmd(0xC3); wdat(0x19); wcmd(0xC4); wdat(0x19); wcmd(0xC9); wdat(0x3C);
  wcmd(0xF0); wdat(0x53);wdat(0x15);wdat(0x0A);wdat(0x04);wdat(0x00);wdat(0x3E);
  wcmd(0xF2); wdat(0x53);wdat(0x15);wdat(0x0A);wdat(0x04);wdat(0x00);wdat(0x3A);
  wcmd(0xF1); wdat(0x56);wdat(0xA8);wdat(0x7F);wdat(0x33);wdat(0x34);wdat(0x5F);
  wcmd(0xF3); wdat(0x52);wdat(0xA4);wdat(0x7F);wdat(0x33);wdat(0x34);wdat(0xDF);
  wcmd(0x36); wdat(0x00);
  wcmd(0x11); delay(200); wcmd(0x29); wcmd(0x2C);
}

static void panelsBegin(){
  pinMode(PIN_DC,OUTPUT); pinMode(PIN_RST1,OUTPUT); pinMode(PIN_RST2,OUTPUT);
  pinMode(PIN_CS1,OUTPUT); pinMode(PIN_CS2,OUTPUT); pinMode(PIN_BL1,OUTPUT); pinMode(PIN_BL2,OUTPUT);
  digitalWrite(PIN_BL1,HIGH); digitalWrite(PIN_BL2,HIGH);
  lcdspi.begin(PIN_CLK,-1,PIN_DIN,-1);
  lcdspi.beginTransaction(SPISettings(20000000,MSBFIRST,SPI_MODE0));   // this task owns the bus for good
  digitalWrite(PIN_RST1,HIGH); digitalWrite(PIN_RST2,HIGH); delay(50);
  digitalWrite(PIN_RST1,LOW);  digitalWrite(PIN_RST2,LOW);  delay(50);
  digitalWrite(PIN_RST1,HIGH); digitalWrite(PIN_RST2,HIGH); delay(150);
  digitalWrite(PIN_CS1,LOW); digitalWrite(PIN_CS2,LOW);              // init both panels at once
  initPanel(); selectEye(-1);
  setMADCTL(PIN_CS1,madctl[0]); setMADCTL(PIN_CS2,madctl[1]);
  selectEye(PIN_CS1); wcmd(0x20); selectEye(PIN_CS2); wcmd(0x20); selectEye(-1);   // inversion OFF
}

static void eyeTask(void*){
  panelsBegin();
  genVeins();                      // random() state is per task: seed + use it here, in this task
  randomSeed(micros());
  lastCmd=lastBlink=lastIdle=millis();
  int shown=-1;
  for(;;){
    for(int e=0;e<2;e++){ int v=madReq[e];
      if(v>=0){ madReq[e]=-1; madctl[e]=(uint8_t)v; setMADCTL(e?PIN_CS2:PIN_CS1,(uint8_t)v); shown=-1; } }
    if(mode){ if(shown!=1){ drawMarker(); refresh(); shown=1; } }
    else { animate(); shown=0; }
    vTaskDelay(pdMS_TO_TICKS(8));
  }
}

void eyesBegin(){ xTaskCreatePinnedToCore(eyeTask,"eyes",6144,nullptr,1,nullptr,0); }

bool isEyeCommand(const char* s){
  return s[0] && strchr("GBCEPMT",s[0]) && (s[1]==0 || s[1]==' ');
}

void eyeCommand(const char* s){
  lastCmd=millis();
  switch(s[0]){
    case 'G':{ int x=0,y=0; if(sscanf(s+1,"%d %d",&x,&y)==2){
        tgx=constrain(x,-100,100)/100.0f; tgy=constrain(y,-100,100)/100.0f; } break; }
    case 'B': blinkReq=1; break;
    case 'C': tgx=0; tgy=0; break;
    case 'E': emotion=constrain(atoi(s+1),0,4); break;
    case 'P': pupilScale=constrain(atoi(s+1),0,100)/50.0f; break;
    case 'T': mode^=1; Serial.printf("eyes: %s\n",mode?"orientation marker (red=top, green=left)":"normal"); break;
    case 'M':{ int eye=0; unsigned v=0;
      if(sscanf(s+1," %d %x",&eye,&v)==2 && (eye==1||eye==2) && v<256){
        madReq[eye-1]=(int)v; Serial.printf("eye%d MADCTL=0x%02X\n",eye,v); }
      else Serial.println("usage: M <1|2> <hex>"); break; }
  }
}
