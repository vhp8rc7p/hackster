#pragma once
#include <stdint.h>

void buddyInit();
void buddyTick(uint8_t personaState);
void buddyInvalidate();
void buddySetSpeciesIdx(uint8_t idx);
void buddyNextSpecies();
uint8_t buddySpeciesIdx();
uint8_t buddySpeciesCount();
const char* buddySpeciesName();
uint16_t buddySpeciesColor();

typedef void (*StateFn)(uint32_t t);

struct Species {
  const char* name;
  uint16_t bodyColor;
  StateFn states[7];
};
