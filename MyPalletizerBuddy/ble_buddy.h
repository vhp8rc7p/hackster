#pragma once
#include <stdint.h>
#include <stddef.h>

void bleBuddyInit(const char* deviceName);
bool bleBuddyConnected();
bool bleBuddySecure();
uint32_t bleBuddyPasskey();
void bleBuddyClearBonds();
size_t bleBuddyAvailable();
int bleBuddyRead();
size_t bleBuddyWrite(const uint8_t* data, size_t len);
