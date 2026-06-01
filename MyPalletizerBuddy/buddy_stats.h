#pragma once
#include <Arduino.h>
#include <Preferences.h>

static const uint32_t TOKENS_PER_LEVEL = 50000;

struct BuddyStats {
  uint32_t napSeconds;
  uint16_t approvals;
  uint16_t denials;
  uint16_t velocity[8];
  uint8_t  velIdx;
  uint8_t  velCount;
  uint8_t  level;
  uint32_t tokens;
};

static BuddyStats _bstats;
static Preferences _bprefs;
static bool _bdirty = false;

static const uint8_t NVS_MAGIC = 0xB5;
static bool _needsInit = false;

inline void buddyStatsLoad() {
  _bprefs.begin("buddy", true);
  uint8_t magic = _bprefs.getUChar("mag", 0);
  if (magic != NVS_MAGIC) {
    _bprefs.end();
    memset(&_bstats, 0, sizeof(_bstats));
    _needsInit = true;
    return;
  }
  _bstats.napSeconds = _bprefs.getUInt("nap", 0);
  _bstats.approvals  = _bprefs.getUShort("appr", 0);
  _bstats.denials    = _bprefs.getUShort("deny", 0);
  _bstats.velIdx     = _bprefs.getUChar("vidx", 0);
  _bstats.velCount   = _bprefs.getUChar("vcnt", 0);
  _bstats.level      = _bprefs.getUChar("lvl", 0);
  _bstats.tokens     = _bprefs.getUInt("tok", 0);
  size_t got = _bprefs.getBytes("vel", _bstats.velocity, sizeof(_bstats.velocity));
  if (got != sizeof(_bstats.velocity)) memset(_bstats.velocity, 0, sizeof(_bstats.velocity));
  _bprefs.end();
  if (_bstats.tokens == 0 && _bstats.level > 0)
    _bstats.tokens = (uint32_t)_bstats.level * TOKENS_PER_LEVEL;
}

inline void buddyStatsSave() {
  if (!_bdirty) return;
  _bprefs.begin("buddy", false);
  _bprefs.putUChar("mag", NVS_MAGIC);
  _bprefs.putUInt("nap", _bstats.napSeconds);
  _bprefs.putUShort("appr", _bstats.approvals);
  _bprefs.putUShort("deny", _bstats.denials);
  _bprefs.putUChar("vidx", _bstats.velIdx);
  _bprefs.putUChar("vcnt", _bstats.velCount);
  _bprefs.putUChar("lvl", _bstats.level);
  _bprefs.putUInt("tok", _bstats.tokens);
  _bprefs.putBytes("vel", _bstats.velocity, sizeof(_bstats.velocity));
  _bprefs.end();
  _bdirty = false;
}

inline void buddyStatsFinishLoad() {
  if (_needsInit) {
    _needsInit = false;
    _bdirty = true;
    buddyStatsSave();
  }
}

inline void buddyStatsOnApproval(uint32_t secondsToRespond) {
  _bstats.approvals++;
  _bstats.velocity[_bstats.velIdx] = (uint16_t)min(secondsToRespond, (uint32_t)65535);
  _bstats.velIdx = (_bstats.velIdx + 1) % 8;
  if (_bstats.velCount < 8) _bstats.velCount++;
  _bdirty = true; buddyStatsSave();
}

static uint32_t _lastBridgeTokens = 0;
static bool _tokensSynced = false;
static bool _levelUpPending = false;

inline void buddyStatsOnBridgeTokens(uint32_t bridgeTotal) {
  if (!_tokensSynced) {
    _lastBridgeTokens = bridgeTotal;
    _tokensSynced = true;
    return;
  }
  if (bridgeTotal < _lastBridgeTokens) {
    _lastBridgeTokens = bridgeTotal;
    return;
  }
  uint32_t delta = bridgeTotal - _lastBridgeTokens;
  _lastBridgeTokens = bridgeTotal;
  if (delta == 0) return;
  uint8_t lvlBefore = (uint8_t)(_bstats.tokens / TOKENS_PER_LEVEL);
  _bstats.tokens += delta;
  uint8_t lvlAfter = (uint8_t)(_bstats.tokens / TOKENS_PER_LEVEL);
  if (lvlAfter > lvlBefore) {
    _bstats.level = lvlAfter;
    _levelUpPending = true;
    _bdirty = true; buddyStatsSave();
  }
}

inline bool buddyStatsPollLevelUp() {
  bool r = _levelUpPending;
  _levelUpPending = false;
  return r;
}

inline void buddyStatsOnDenial() { _bstats.denials++; _bdirty = true; buddyStatsSave(); }

inline void buddyStatsOnNapEnd(uint32_t seconds) {
  _bstats.napSeconds += seconds;
  _bdirty = true; buddyStatsSave();
}

inline uint16_t buddyStatsMedianVelocity() {
  if (_bstats.velCount == 0) return 0;
  uint16_t tmp[8];
  memcpy(tmp, _bstats.velocity, sizeof(tmp));
  uint8_t n = _bstats.velCount;
  for (uint8_t i = 1; i < n; i++) {
    uint16_t k = tmp[i]; int8_t j = i - 1;
    while (j >= 0 && tmp[j] > k) { tmp[j+1] = tmp[j]; j--; }
    tmp[j+1] = k;
  }
  return tmp[n/2];
}

inline uint8_t buddyStatsMoodTier() {
  uint16_t vel = buddyStatsMedianVelocity();
  int8_t tier;
  if (vel == 0) tier = 2;
  else if (vel < 15) tier = 4;
  else if (vel < 30) tier = 3;
  else if (vel < 60) tier = 2;
  else if (vel < 120) tier = 1;
  else tier = 0;
  uint16_t a = _bstats.approvals, d = _bstats.denials;
  if (a + d >= 3) {
    if (d > a) tier -= 2;
    else if (d * 2 > a) tier -= 1;
  }
  if (tier < 0) tier = 0;
  return (uint8_t)tier;
}

static uint32_t _lastNapEndMs = 0;
static uint8_t  _energyAtNap  = 3;

inline void buddyStatsOnWake() { _lastNapEndMs = millis(); _energyAtNap = 5; }

inline uint8_t buddyStatsEnergyTier() {
  uint32_t hoursSince = (millis() - _lastNapEndMs) / 3600000;
  int8_t e = (int8_t)_energyAtNap - (int8_t)(hoursSince / 2);
  if (e < 0) e = 0; if (e > 5) e = 5;
  return (uint8_t)e;
}

inline uint8_t buddyStatsFedProgress() {
  return (uint8_t)((_bstats.tokens % TOKENS_PER_LEVEL) / (TOKENS_PER_LEVEL / 10));
}

inline uint8_t buddyStatsSpeciesLoad() {
  _bprefs.begin("buddy", true);
  uint8_t v = _bprefs.getUChar("species", 0);
  _bprefs.end();
  return v;
}

inline void buddyStatsSpeciesSave(uint8_t idx) {
  _bprefs.begin("buddy", false);
  _bprefs.putUChar("species", idx);
  _bprefs.end();
}

inline const BuddyStats& buddyStats() { return _bstats; }
