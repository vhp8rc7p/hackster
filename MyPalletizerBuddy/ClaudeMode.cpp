#include "ClaudeMode.h"
#include <M5Stack.h>
#include "ble_buddy.h"
#include "buddy_data.h"
#include "buddy_stats.h"
#include "buddy.h"
#include "buddy_common.h"


// Anthropic design palette (RGB565)
#define COL_BG       0x0000
#define COL_PANEL    0x18E3
#define COL_TEXT     0xFFF5
#define COL_DIM      0xB574
#define COL_BAR_BG   0x2945
#define COL_GREEN    0x7C6B
#define COL_AMBER    0xDBAB
#define COL_RED      0xC165
#define COL_ACCENT   0xDBAB
#define HOT          0xFA20

// Layout: 320x240, pet on left (0..159), info on right (160..319)
#define PET_W       160
#define INFO_X      160
#define INFO_W      160
#define HEADER_H    20
#define FOOTER_H    20
#define FOOTER_Y    220

enum PersonaState { P_SLEEP, P_IDLE, P_BUSY, P_ATTENTION, P_CELEBRATE, P_DIZZY, P_HEART };
static const char* stateNames[] = { "sleep", "idle", "busy", "attention", "celebrate", "dizzy", "heart" };

static TamaState tama;
static PersonaState baseState   = P_SLEEP;
static PersonaState activeState = P_SLEEP;
static uint32_t oneShotUntil    = 0;
static bool responseSent        = false;
static uint32_t promptArrivedMs = 0;
static char lastPromptId[40]    = "";
static uint16_t lastLineGen     = 0;
static uint8_t msgScroll        = 0;
static uint8_t petPage          = 0;
static const uint8_t PET_PAGES  = 2;
static uint32_t lastInteractMs  = 0;
static bool dimmed              = false;
static bool infoDirty           = true;
static bool infoFullClear       = true;
static const uint32_t DIM_MS    = 45000;
static const uint8_t BRIGHT_FULL = 200;
static const uint8_t BRIGHT_DIM  = 40;

enum DisplayMode { DISP_NORMAL, DISP_PET, DISP_INFO, DISP_COUNT };
static uint8_t displayMode = DISP_NORMAL;

static void beep(uint16_t freq, uint16_t dur) {
  M5.Speaker.tone(freq, dur);
}

static void wake() {
  lastInteractMs = millis();
  if (dimmed) { M5.Lcd.setBrightness(BRIGHT_FULL); dimmed = false; }
}

// Robot arm pose sets per state
// {J1, J2, J3, J4, speed}
static const float ARM_SLEEP[][5] = {
    { -90, 30, 50,   0, 20}, {-85, 32, 48,  10, 20},
    { -90, 30, 50,   0, 20}, {-95, 32, 48, -10, 20},
};
static const float ARM_IDLE[][5] = {
    { -90, 45, 30,   0, 30}, {-80, 48, 28,  20, 30},
    { -90, 45, 30,  40, 30}, {-100, 48, 28, -20, 30},
};
static const float ARM_BUSY[][5] = {
    { -90, 55, 20,   0, 40}, {-75, 60, 15,  40, 45},
    {-105, 60, 15, -40, 45}, {-90, 55, 20,  80, 40},
};
static const float ARM_ATTENTION[][5] = {
    { -90, 65,  5,   0, 20}, {-65, 55, 10,  50, 20},
    { -90, 65,  5,   0, 20}, {-115, 55, 10, -50, 20},
};
static const float ARM_CELEBRATE[][5] = {
    { -90, 70,  0,   0, 30}, {-90, 30, 50,  90, 30},
    { -90, 70,  0, -90, 30}, {-90, 30, 50,   0, 30},
};
static const float ARM_DIZZY[][5] = {
    { -70, 50, 25,  40, 20}, {-110, 55, 20, -40, 20},
    { -75, 45, 30,  60, 20}, {-105, 45, 30, -60, 20},
};
static const float ARM_HEART[][5] = {
    { -90, 35, 40,   0, 15}, {-80, 38, 37,  25, 15},
    { -90, 35, 40,  50, 15}, {-100, 38, 37,  25, 15},
};

static const float (*ARM_POSES[])[5] = {
    ARM_SLEEP, ARM_IDLE, ARM_BUSY, ARM_ATTENTION,
    ARM_CELEBRATE, ARM_DIZZY, ARM_HEART
};
static const uint8_t ARM_POSE_COUNT = 4;

static uint8_t armPoseIdx = 0;
static uint8_t lastArmBeat = 0xFF;

static const uint16_t ARM_BEAT_MS[] = {
  1000,  // SLEEP:     t/5 * 200ms = 1s
  1000,  // IDLE:      t/5 * 200ms = 1s
  1000,  // BUSY:      t/5 * 200ms = 1s
  1000,  // ATTENTION: t/5 * 200ms = 1s
   600,  // CELEBRATE: t/3 * 200ms = 600ms
   800,  // DIZZY:     t/4 * 200ms = 800ms
  1000,  // HEART:     t/5 * 200ms = 1s
};

static PersonaState derive(const TamaState& s) {
  if (!s.connected)            return P_IDLE;
  if (s.sessionsWaiting > 0)   return P_ATTENTION;
  if (s.recentlyCompleted)     return P_CELEBRATE;
  if (s.sessionsRunning >= 3)  return P_BUSY;
  return P_IDLE;
}

static void triggerOneShot(PersonaState s, uint32_t durMs) {
  activeState = s;
  oneShotUntil = millis() + durMs;
}

static void sendCmd(const char* json) {
  Serial.println(json);
  size_t n = strlen(json);
  bleBuddyWrite((const uint8_t*)json, n);
  bleBuddyWrite((const uint8_t*)"\n", 1);
}

static void drawHeader() {
  M5.Lcd.fillRect(0, 0, PET_W, HEADER_H, COL_BG);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(2, 2);
  M5.Lcd.print(buddySpeciesName());

  M5.Lcd.fillRect(INFO_X, 0, INFO_W, HEADER_H, COL_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(INFO_X + 4, 6);
  M5.Lcd.printf("[%s]", stateNames[activeState]);

  if (dataDemo()) {
    M5.Lcd.setTextColor(COL_AMBER, COL_BG);
    M5.Lcd.setCursor(240, 6);
    M5.Lcd.print("DEMO");
  }
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(262, 2);
  M5.Lcd.printf("L%u ", buddyStats().level);

  if (bleBuddyConnected()) {
    M5.Lcd.fillCircle(310, 10, 4, COL_GREEN);
  } else {
    M5.Lcd.fillCircle(310, 10, 4, COL_RED);
  }
}

static void drawFooter() {
  M5.Lcd.fillRect(0, FOOTER_Y, 320, FOOTER_H, COL_BG);
  M5.Lcd.drawFastHLine(0, FOOTER_Y, 320, COL_DIM);
  M5.Lcd.setTextSize(2);

  bool inPrompt = tama.promptId[0] && !responseSent;
  if (inPrompt) {
    M5.Lcd.setTextColor(HOT, COL_BG);
    M5.Lcd.setCursor(41, FOOTER_Y + 2);
    M5.Lcd.print("DENY");
    M5.Lcd.setTextColor(COL_GREEN, COL_BG);
    M5.Lcd.setCursor(243, FOOTER_Y + 2);
    M5.Lcd.print("OK");
  } else {
    M5.Lcd.setTextColor(COL_DIM, COL_BG);
    M5.Lcd.setCursor(41, FOOTER_Y + 2);
    M5.Lcd.print("View");
    M5.Lcd.setCursor(142, FOOTER_Y + 2);
    M5.Lcd.print("Pet");
    M5.Lcd.setCursor(231, FOOTER_Y + 2);
    M5.Lcd.print("Exit");
  }
}

static uint8_t wrapInto(const char* in, char out[][26], uint8_t maxRows, uint8_t width) {
  uint8_t row = 0, col = 0;
  const char* p = in;
  while (*p && row < maxRows) {
    while (*p == ' ') p++;
    const char* w = p;
    while (*p && *p != ' ') p++;
    uint8_t wlen = p - w;
    if (wlen == 0) break;
    uint8_t need = (col > 0 ? 1 : 0) + wlen;
    if (col + need > width) {
      out[row][col] = 0;
      if (++row >= maxRows) return row;
      col = 0;
    }
    if (col > 0) out[row][col++] = ' ';
    while (wlen > width - col) {
      uint8_t take = width - col;
      memcpy(&out[row][col], w, take); col += take; w += take; wlen -= take;
      out[row][col] = 0;
      if (++row >= maxRows) return row;
      col = 0;
    }
    memcpy(&out[row][col], w, wlen); col += wlen;
  }
  if (col > 0 && row < maxRows) { out[row][col] = 0; row++; }
  return row;
}

static void clearRow(int y, int h) {
  M5.Lcd.fillRect(INFO_X + 1, y, INFO_W - 1, h, COL_BG);
}

static void drawTranscript() {
  const int TOP = HEADER_H + 4;
  const int BOT = FOOTER_Y - 4;
  const int AREA_H = BOT - TOP;
  const int LH = 10;
  const int SHOW = AREA_H / LH;
  const int WIDTH = 24;

  bool inPrompt = tama.promptId[0] && !responseSent;
  if (inPrompt) {
    clearRow(TOP, 24);
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(HOT, COL_BG);
    M5.Lcd.setCursor(INFO_X + 4, TOP + 4);
    uint32_t waited = (millis() - promptArrivedMs) / 1000;
    M5.Lcd.printf("? %lus  ", (unsigned long)waited);

    clearRow(TOP + 28, 26);
    M5.Lcd.setTextSize(3);
    M5.Lcd.setTextColor(COL_TEXT, COL_BG);
    M5.Lcd.setCursor(INFO_X + 4, TOP + 30);
    M5.Lcd.print(tama.promptTool);

    clearRow(TOP + 58, 24);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(COL_DIM, COL_BG);
    M5.Lcd.setCursor(INFO_X + 4, TOP + 60);
    M5.Lcd.printf("%.24s", tama.promptHint);
    if (strlen(tama.promptHint) > 24) {
      M5.Lcd.setCursor(INFO_X + 4, TOP + 72);
      M5.Lcd.printf("%.24s", tama.promptHint + 24);
    }
    return;
  }

  if (tama.nLines == 0) {
    clearRow(TOP, AREA_H);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(COL_TEXT, COL_BG);
    M5.Lcd.setCursor(INFO_X + 4, TOP + AREA_H / 2);
    M5.Lcd.print(tama.msg);
    return;
  }

  static char disp[32][26];
  static uint8_t srcOf[32];
  uint8_t nDisp = 0;
  for (uint8_t i = 0; i < tama.nLines && nDisp < 32; i++) {
    uint8_t got = wrapInto(tama.lines[i], &disp[nDisp], 32 - nDisp, WIDTH);
    for (uint8_t j = 0; j < got; j++) srcOf[nDisp + j] = i;
    nDisp += got;
  }

  uint8_t maxBack = (nDisp > SHOW) ? (nDisp - SHOW) : 0;
  if (msgScroll > maxBack) msgScroll = maxBack;

  int end = (int)nDisp - msgScroll;
  int start = end - SHOW; if (start < 0) start = 0;
  uint8_t newest = tama.nLines - 1;
  M5.Lcd.setTextSize(1);
  int drawn = end - start;
  for (int i = 0; i < SHOW; i++) {
    int py = TOP + i * LH;
    if (i < drawn) {
      uint8_t row = start + i;
      bool fresh = (srcOf[row] == newest) && (msgScroll == 0);
      clearRow(py, LH);
      M5.Lcd.setTextColor(fresh ? COL_TEXT : COL_DIM, COL_BG);
      M5.Lcd.setCursor(INFO_X + 4, py);
      M5.Lcd.print(disp[row]);
    } else {
      clearRow(py, LH);
    }
  }
  clearRow(BOT - 12, 12);
  if (msgScroll > 0) {
    M5.Lcd.setTextColor(COL_ACCENT, COL_BG);
    M5.Lcd.setCursor(INFO_X + INFO_W - 24, BOT - 10);
    M5.Lcd.printf("-%u", msgScroll);
  }
}

static void drawPetStats() {
  const int TOP = HEADER_H + 4;
  int y = TOP + 2;
  int x = INFO_X + 4;

  clearRow(y, 20);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("mood");
  uint8_t mood = buddyStatsMoodTier();
  uint16_t moodCol = (mood >= 3) ? COL_RED : (mood >= 2) ? HOT : COL_DIM;
  for (int i = 0; i < 4; i++) {
    int hx = x + 64 + i * 18;
    if (i < (int)mood) M5.Lcd.fillCircle(hx, y + 7, 5, moodCol);
    else M5.Lcd.drawCircle(hx, y + 7, 5, COL_DIM);
  }
  y += 24;

  clearRow(y, 20);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("fed");
  uint8_t fed = buddyStatsFedProgress();
  for (int i = 0; i < 8; i++) {
    int px = x + 48 + i * 12;
    if (i < (int)fed) M5.Lcd.fillCircle(px, y + 7, 4, buddySpeciesColor());
    else M5.Lcd.drawCircle(px, y + 7, 4, COL_DIM);
  }
  y += 24;

  clearRow(y, 20);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("nrg");
  uint8_t en = buddyStatsEnergyTier();
  uint16_t enCol = (en >= 4) ? BUDDY_CYAN : (en >= 2) ? BUDDY_YEL : HOT;
  for (int i = 0; i < 5; i++) {
    int px = x + 48 + i * 18;
    if (i < (int)en) M5.Lcd.fillRect(px, y + 2, 14, 12, enCol);
    else M5.Lcd.drawRect(px, y + 2, 14, 12, COL_DIM);
  }
  y += 28;

  clearRow(y, 18);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(x, y);
  M5.Lcd.printf("approved %u  ", buddyStats().approvals);
  clearRow(y + 20, 18);
  M5.Lcd.setCursor(x, y + 20);
  M5.Lcd.printf("denied   %u  ", buddyStats().denials);
  clearRow(y + 40, 18);
  M5.Lcd.setCursor(x, y + 40);
  M5.Lcd.print("tokens ");
  uint32_t tok = buddyStats().tokens;
  if (tok >= 1000000) M5.Lcd.printf("%lu.%luM ", tok/1000000, (tok/100000)%10);
  else if (tok >= 1000) M5.Lcd.printf("%lu.%luK ", tok/1000, (tok/100)%10);
  else M5.Lcd.printf("%lu ", tok);
  y += 64;

  clearRow(y, 20);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(COL_TEXT, COL_BG);
  M5.Lcd.setCursor(x, y);
  M5.Lcd.printf("%s", buddySpeciesName());
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(INFO_X + INFO_W - 30, y + 4);
  M5.Lcd.printf("%u/%u", buddySpeciesIdx() + 1, buddySpeciesCount());
  M5.Lcd.fillRect(INFO_X + INFO_W - 30, TOP + 2, 30, 12, COL_BG);
  M5.Lcd.setCursor(INFO_X + INFO_W - 24, TOP + 4);
  M5.Lcd.printf("%u/%u", petPage + 1, PET_PAGES);
}

static void drawPetHowTo() {
  const int TOP = HEADER_H + 4;
  const int BOT = FOOTER_Y - 4;
  int y = TOP + 2;
  int x = INFO_X + 4;

  clearRow(y, 18);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("HOW");
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(INFO_X + INFO_W - 24, y + 4);
  M5.Lcd.printf("%u/%u", petPage + 1, PET_PAGES);
  y += 22;

  clearRow(y, 32);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("mood");
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(x, y + 18); M5.Lcd.print("fast approve = up");
  y += 34;

  clearRow(y, 32);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("fed");
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(x, y + 18); M5.Lcd.print("50K tokens = lvl up");
  y += 34;

  clearRow(y, 32);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("energy");
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(x, y + 18); M5.Lcd.print("drains, reboots fix");
}

static void drawInfo() {
  const int TOP = HEADER_H + 4;
  int y = TOP + 2;
  int x = INFO_X + 4;

  M5.Lcd.setTextSize(2);
  clearRow(y, 18);
  M5.Lcd.setTextColor(COL_TEXT, COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("CLAUDE");
  y += 20;
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" ses %u  ", tama.sessionsTotal);
  y += 18;
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" run %u  ", tama.sessionsRunning);
  y += 18;
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" wait %u  ", tama.sessionsWaiting);
  y += 24;

  clearRow(y, 18);
  M5.Lcd.setTextColor(COL_TEXT, COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("LINK");
  y += 20;
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" %s  ", dataScenarioName());
  y += 18;
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y);
  M5.Lcd.printf(" %s  ", !bleBuddyConnected() ? "ble -" : bleBuddySecure() ? "ble enc" : "ble OPEN");
  y += 18;
  clearRow(y, 18);
  if (tama.lastUpdated) {
    uint32_t age = (millis() - tama.lastUpdated) / 1000;
    M5.Lcd.setCursor(x, y); M5.Lcd.printf(" %lus ago  ", (unsigned long)age);
  }
  y += 24;

  clearRow(y, 18);
  M5.Lcd.setTextColor(COL_TEXT, COL_BG);
  M5.Lcd.setCursor(x, y); M5.Lcd.print("DEVICE");
  y += 20;
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  uint32_t up = millis() / 1000;
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" %luh%02lum  ", up/3600, (up/60)%60);
  y += 18;
  clearRow(y, 18);
  M5.Lcd.setCursor(x, y); M5.Lcd.printf(" %uKB free  ", ESP.getFreeHeap()/1024);
}

static void drawPasskey() {
  M5.Lcd.fillRect(INFO_X + 1, 40, INFO_W - 1, 160, COL_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(INFO_X + 10, 60);
  M5.Lcd.print("BLUETOOTH PAIRING");
  M5.Lcd.setCursor(INFO_X + 10, 140);
  M5.Lcd.print("enter on desktop:");
  M5.Lcd.setTextSize(3);
  M5.Lcd.setTextColor(COL_TEXT, COL_BG);
  char b[8]; snprintf(b, sizeof(b), "%06lu", (unsigned long)bleBuddyPasskey());
  M5.Lcd.setCursor(INFO_X + 20, 90);
  M5.Lcd.print(b);
}

void ClaudeMode::run(MyPalletizerBasic &myCobot) {
  EXIT = false;
  memset(&tama, 0, sizeof(tama));
  strncpy(tama.msg, "No Claude connected", sizeof(tama.msg)-1);
  baseState = P_SLEEP;
  activeState = P_SLEEP;
  oneShotUntil = 0;
  responseSent = false;
  lastPromptId[0] = 0;
  lastLineGen = 0;
  msgScroll = 0;
  displayMode = DISP_NORMAL;
  armPoseIdx = 0;
  lastArmBeat = 0xFF;

  buddyStatsLoad();
  buddyStatsFinishLoad();
  buddyInit();
  petPage = 0;
  lastInteractMs = millis();
  dimmed = false;

  static bool ble_inited = false;
  if (!ble_inited) {
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_BT);
    char btName[20];
    snprintf(btName, sizeof(btName), "Claude-%02X%02X", mac[4], mac[5]);
    bleBuddyInit(btName);
    ble_inited = true;
  }

  M5.Lcd.setBrightness(BRIGHT_FULL);

  const float *p0 = ARM_IDLE[0];
  MyPalletizerAngles home = {p0[0], p0[1], p0[2], p0[3]};
  myCobot.writeAngles(home, (int)p0[4]);

  M5.Lcd.clear(COL_BG);
  M5.Lcd.drawFastVLine(PET_W, HEADER_H, FOOTER_Y - HEADER_H, COL_DIM);

  M5.Lcd.setTextSize(3);
  M5.Lcd.setTextColor(buddySpeciesColor(), COL_BG);
  M5.Lcd.setCursor(20, 80);
  M5.Lcd.print(buddySpeciesName());
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(COL_DIM, COL_BG);
  M5.Lcd.setCursor(170, 120);
  M5.Lcd.print("appears!");
  delay(1200);
  M5.Lcd.clear(COL_BG);
  M5.Lcd.drawFastVLine(PET_W, HEADER_H, FOOTER_Y - HEADER_H, COL_DIM);

  drawHeader();
  drawFooter();
  buddyInvalidate();

  uint32_t lastHeaderMs = 0;
  uint32_t lastInfoMs = 0;
  PersonaState lastDrawnActive = (PersonaState)0xFF;
  uint32_t lastPasskey = 0;

  while (!EXIT) {
    M5.update();
    uint32_t now = millis();

    uint16_t prevLineGen = tama.lineGen;
    uint32_t prevTokens = tama.tokensToday;
    dataPoll(&tama);
    if (tama.lineGen != prevLineGen || tama.tokensToday != prevTokens) infoDirty = true;
    buddyStatsOnBridgeTokens(tama.tokensToday);
    if (buddyStatsPollLevelUp()) { triggerOneShot(P_CELEBRATE, 3000); beep(2400, 120); }
    baseState = derive(tama);
    if ((int32_t)(now - oneShotUntil) >= 0) activeState = baseState;

    if (strcmp(tama.promptId, lastPromptId) != 0) {
      strncpy(lastPromptId, tama.promptId, sizeof(lastPromptId)-1);
      lastPromptId[sizeof(lastPromptId)-1] = 0;
      responseSent = false;
      infoDirty = true;
      infoFullClear = true;
      if (tama.promptId[0]) {
        promptArrivedMs = now;
        displayMode = DISP_NORMAL;
        wake();
        beep(1200, 80);
      }
    }

    bool inPrompt = tama.promptId[0] && !responseSent;

    if (M5.BtnA.wasReleased() || M5.BtnB.wasReleased() || M5.BtnC.wasReleased()) wake();

    static bool btnALong = false;
    if (M5.BtnA.pressedFor(800) && !btnALong) {
      btnALong = true;
      dataSetDemo(!dataDemo());
      beep(dataDemo() ? 2000 : 800, 100);
    }
    if (M5.BtnA.wasReleased() && btnALong) {
      btnALong = false;
    } else if (M5.BtnA.wasReleased()) {
      beep(1800, 30);
      displayMode = (displayMode + 1) % DISP_COUNT;
      petPage = 0;
      M5.Lcd.fillRect(INFO_X + 1, HEADER_H, INFO_W - 1, FOOTER_Y - HEADER_H, COL_BG);
      infoDirty = true;
    }

    if (M5.BtnB.wasReleased()) {
      if (inPrompt) {
        char cmd[96];
        snprintf(cmd, sizeof(cmd), "{\"cmd\":\"permission\",\"id\":\"%s\",\"decision\":\"deny\"}", tama.promptId);
        sendCmd(cmd);
        responseSent = true;
        buddyStatsOnDenial();
        infoDirty = true;
        infoFullClear = true;
        beep(600, 60);
      } else if (displayMode == DISP_PET) {
        if (petPage == 0) {
          buddyNextSpecies();
          buddyInvalidate();
          M5.Lcd.fillRect(0, 0, PET_W, FOOTER_Y, COL_BG);
        }
        petPage = (petPage + 1) % PET_PAGES;
        infoDirty = true;
        infoFullClear = true;
        beep(1800, 30);
      } else if (displayMode == DISP_INFO) {
        beep(1800, 30);
        infoDirty = true;
      } else {
        msgScroll = (msgScroll >= 30) ? 0 : msgScroll + 1;
        infoDirty = true;
      }
    }

    if (M5.BtnC.wasReleased()) {
      if (inPrompt) {
        char cmd[96];
        snprintf(cmd, sizeof(cmd), "{\"cmd\":\"permission\",\"id\":\"%s\",\"decision\":\"once\"}", tama.promptId);
        sendCmd(cmd);
        responseSent = true;
        uint32_t tookS = (now - promptArrivedMs) / 1000;
        buddyStatsOnApproval(tookS);
        infoDirty = true;
        infoFullClear = true;
        beep(2400, 60);
        if (tookS < 5) triggerOneShot(P_HEART, 2000);
      } else {
        EXIT = true;
      }
    }

    if (tama.lineGen != lastLineGen) { msgScroll = 0; lastLineGen = tama.lineGen; infoDirty = true; }

    buddyTick(activeState);

    bool promptActive = tama.promptId[0] && !responseSent;
    if (promptActive) infoDirty = true;

    if (infoDirty && now - lastInfoMs >= 250) {
      lastInfoMs = now;
      infoDirty = false;
      if (infoFullClear) {
        M5.Lcd.fillRect(INFO_X + 1, HEADER_H, INFO_W - 1, FOOTER_Y - HEADER_H, COL_BG);
        infoFullClear = false;
      }
      uint32_t pk = bleBuddyPasskey();
      if (pk) {
        drawPasskey();
      } else if (displayMode == DISP_PET) {
        if (petPage == 0) drawPetStats();
        else drawPetHowTo();
      } else if (displayMode == DISP_INFO) {
        drawInfo();
      } else {
        drawTranscript();
      }
      lastPasskey = pk;
    }

    if (now - lastHeaderMs >= 1000 || activeState != lastDrawnActive) {
      lastHeaderMs = now;
      lastDrawnActive = activeState;
      drawHeader();
      drawFooter();
    }

    {
      uint8_t st = (uint8_t)activeState;
      if (st > 6) st = 1;
      uint8_t beat = (uint8_t)((now / ARM_BEAT_MS[st]) % ARM_POSE_COUNT);
      if (beat != lastArmBeat) {
        lastArmBeat = beat;
        const float *p = ARM_POSES[st][beat];
        MyPalletizerAngles angles = {p[0], p[1], p[2], p[3]};
        myCobot.writeAngles(angles, (int)p[4]);
      }
    }

    if (!dimmed && !inPrompt && (now - lastInteractMs) > DIM_MS) {
      M5.Lcd.setBrightness(BRIGHT_DIM);
      dimmed = true;
    }

    delay(16);
  }

  M5.Lcd.setBrightness(BRIGHT_FULL);
  MyPalletizerAngles zero = {-90, 0, 0, 0};
  myCobot.writeAngles(zero, 30);
  M5.Lcd.clear(COL_BG);
}
