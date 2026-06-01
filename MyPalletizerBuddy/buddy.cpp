#include "buddy.h"
#include "buddy_common.h"
#include <M5Stack.h>
#include <string.h>

enum { B_SLEEP, B_IDLE, B_BUSY, B_ATTENTION, B_CELEBRATE, B_DIZZY, B_HEART };

static uint8_t _scale = 2;

void buddyPrintLine(const char* line, int yPx, uint16_t color, int xOff) {
  int len = strlen(line);
  if (_scale > 1) {
    while (len && line[len-1] == ' ') len--;
    while (len && *line == ' ')       { line++; len--; }
  }
  int w = len * BUDDY_CHAR_W * _scale;
  int x = BUDDY_X_CENTER - w / 2 + xOff * _scale;
  M5.Lcd.setTextColor(color, BUDDY_BG);
  M5.Lcd.setCursor(x, yPx);
  for (int i = 0; i < len; i++) M5.Lcd.print(line[i]);
}

void buddyPrintSprite(const char* const* lines, uint8_t nLines, int yOffset, uint16_t color, int xOff) {
  M5.Lcd.setTextSize(_scale);
  int yBase = BUDDY_Y_BASE * _scale - (_scale - 1) * 14;
  for (uint8_t i = 0; i < nLines; i++) {
    buddyPrintLine(lines[i], yBase + (yOffset + i * BUDDY_CHAR_H) * _scale, color, xOff);
  }
}

void buddySetCursor(int x, int y) {
  M5.Lcd.setCursor(BUDDY_X_CENTER + (x - BUDDY_X_CENTER) * _scale, y * _scale);
}
void buddySetColor(uint16_t fg)   { M5.Lcd.setTextColor(fg, BUDDY_BG); }
void buddyPrint(const char* s)    { M5.Lcd.setTextSize(_scale); M5.Lcd.print(s); }

// Species registry
extern const Species CAPYBARA_SPECIES;
extern const Species DUCK_SPECIES;
extern const Species GOOSE_SPECIES;
extern const Species BLOB_SPECIES;
extern const Species CAT_SPECIES;
extern const Species DRAGON_SPECIES;
extern const Species OCTOPUS_SPECIES;
extern const Species OWL_SPECIES;
extern const Species PENGUIN_SPECIES;
extern const Species TURTLE_SPECIES;
extern const Species SNAIL_SPECIES;
extern const Species GHOST_SPECIES;
extern const Species AXOLOTL_SPECIES;
extern const Species CACTUS_SPECIES;
extern const Species ROBOT_SPECIES;
extern const Species RABBIT_SPECIES;
extern const Species MUSHROOM_SPECIES;
extern const Species CHONK_SPECIES;

static const Species* SPECIES_TABLE[] = {
  &CAPYBARA_SPECIES, &DUCK_SPECIES, &GOOSE_SPECIES, &BLOB_SPECIES,
  &CAT_SPECIES, &DRAGON_SPECIES, &OCTOPUS_SPECIES, &OWL_SPECIES,
  &PENGUIN_SPECIES, &TURTLE_SPECIES, &SNAIL_SPECIES, &GHOST_SPECIES,
  &AXOLOTL_SPECIES, &CACTUS_SPECIES, &ROBOT_SPECIES, &RABBIT_SPECIES,
  &MUSHROOM_SPECIES, &CHONK_SPECIES,
};
static const uint8_t N_SPECIES = sizeof(SPECIES_TABLE) / sizeof(SPECIES_TABLE[0]);
static uint8_t currentSpeciesIdx = 0;

static uint32_t tickCount  = 0;
static uint32_t nextTickAt = 0;
static const uint32_t TICK_MS = 200;

#include "buddy_stats.h"

void buddyInit() {
  tickCount = 0;
  nextTickAt = 0;
  uint8_t saved = buddyStatsSpeciesLoad();
  if (saved < N_SPECIES) currentSpeciesIdx = saved;
}

void buddySetSpeciesIdx(uint8_t idx) {
  if (idx < N_SPECIES) currentSpeciesIdx = idx;
}

const char* buddySpeciesName() { return SPECIES_TABLE[currentSpeciesIdx]->name; }
uint16_t buddySpeciesColor() { return SPECIES_TABLE[currentSpeciesIdx]->bodyColor; }
uint8_t buddySpeciesCount() { return N_SPECIES; }
uint8_t buddySpeciesIdx() { return currentSpeciesIdx; }

void buddyNextSpecies() {
  currentSpeciesIdx = (currentSpeciesIdx + 1) % N_SPECIES;
  buddyStatsSpeciesSave(currentSpeciesIdx);
}

static uint8_t lastDrawnState = 0xFF;
static uint8_t lastDrawnSpecies = 0xFF;
void buddyInvalidate() { lastDrawnState = 0xFF; }

void buddyTick(uint8_t personaState) {
  uint32_t now = millis();
  bool ticked = false;
  if ((int32_t)(now - nextTickAt) >= 0) {
    nextTickAt = now + TICK_MS;
    tickCount++;
    ticked = true;
  }

  if (personaState >= 7) personaState = B_IDLE;
  if (!ticked && personaState == lastDrawnState
              && currentSpeciesIdx == lastDrawnSpecies) {
    return;
  }
  lastDrawnState = personaState;
  lastDrawnSpecies = currentSpeciesIdx;

  M5.Lcd.fillRect(0, 0, BUDDY_CANVAS_W,
               (BUDDY_Y_BASE + 5 * BUDDY_CHAR_H + 12) * _scale, BUDDY_BG);

  const Species* sp = SPECIES_TABLE[currentSpeciesIdx];
  if (sp->states[personaState]) sp->states[personaState](tickCount);
}
