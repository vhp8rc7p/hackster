#pragma once
#include <stdint.h>

// Shared geometry for 320x240 landscape M5Stack Basic.
// Pet renders in the left half (0..159), info in the right half (160..319).
#define BUDDY_X_CENTER  80
#define BUDDY_CANVAS_W  160
#define BUDDY_Y_BASE    44
#define BUDDY_Y_OVERLAY 16
#define BUDDY_CHAR_W    6
#define BUDDY_CHAR_H    8

// Shared colors
#define BUDDY_BG     0x0000
#define BUDDY_HEART  0xF810
#define BUDDY_DIM    0x8410
#define BUDDY_YEL    0xFFE0
#define BUDDY_WHITE  0xFFFF
#define BUDDY_CYAN   0x07FF
#define BUDDY_GREEN  0x07E0
#define BUDDY_PURPLE 0xA01F
#define BUDDY_RED    0xF800
#define BUDDY_BLUE   0x041F

void buddyPrintLine(const char* line, int yPx, uint16_t color, int xOff = 0);
void buddyPrintSprite(const char* const* lines, uint8_t nLines, int yOffset, uint16_t color, int xOff = 0);
void buddySetCursor(int x, int y);
void buddySetColor(uint16_t fg);
void buddyPrint(const char* s);
