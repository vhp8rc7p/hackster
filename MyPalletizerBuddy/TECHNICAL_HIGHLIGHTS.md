# Claude Buddy: Technical Highlights

## 1. Claude Code Hook System as Hardware Bridge

PreToolUse/PostToolUse hooks — originally designed for CI/linting — repurposed as a physical device permission gateway. The architecture uses Unix domain socket IPC between hook scripts and a long-running BLE daemon. Hooks are stateless short-lived processes invoked by Claude Code on every tool call; the daemon holds the persistent BLE connection to the M5Stack device.

```
Claude Code ──PreToolUse hook──> ble-permission-hook.py
                                      │ (Unix socket)
                                      v
                               ble-permission-daemon.py
                                      │ (BLE NUS)
                                      v
                                M5Stack Basic
                                  [OK] [DENY]
```

The hook output format was a non-trivial discovery — Claude Code requires a nested structure:
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
```
A flat `{"permissionDecision": "allow"}` is silently ignored and falls through to the terminal prompt.

## 2. BLE Nordic UART Service (NUS) for Bidirectional JSON

Full JSON protocol over BLE's 20-byte MTU. Both sides implement chunked writes with newline-delimited framing and a reassembly buffer:

```python
# Daemon side: reassemble chunked BLE notifications
def on_ble_notify(data):
    rx_buffer.extend(data)
    while b'\n' in rx_buffer:
        line, rx_buffer = rx_buffer.split(b'\n', 1)
        msg = json.loads(line)
```

The permission flow is inherently async — the daemon sends a prompt JSON to the device, then blocks on an `asyncio.Future`. The device responds whenever the user physically presses a button, seconds or minutes later. A 120-second timeout auto-denies if no response comes.

## 3. Real Token Counting from Session Transcript

Claude Code's session transcript (a JSONL file at `transcript_path`) contains real `output_tokens` usage data inside `assistant` type messages. The PostToolUse hook reads this incrementally:

```python
def get_session_tokens(transcript_path, session_id):
    # Resume from last byte offset (cached in /tmp/claude-buddy-tokens.json)
    f.seek(cached_offset)
    for line in f:
        obj = json.loads(line)
        if obj["type"] == "assistant":
            total += obj["message"]["usage"]["output_tokens"]
    # Cache new offset + total for next call
```

Key design decisions:
- **Byte-offset caching** avoids re-reading multi-hundred-MB transcript files on every tool call
- **Session ID tracking** resets the cache when a new session starts
- **Cumulative counting** — the daemon receives the total and computes deltas, so restarts don't lose progress

## 4. Tamagotchi Gamification Mapped to Real Developer Behavior

Each pet stat maps to a real developer interaction metric:

| Stat | Source | Formula |
|------|--------|---------|
| **Mood** (0-4) | Approval response time | Median of last 8 approvals: <15s=4, <30s=3, <60s=2, <120s=1, else=0. Penalized if denials > approvals |
| **Fed** (0-9) | Token consumption | `(tokens % 50000) / 5000` — XP progress within current level |
| **Level** | Total tokens | `tokens / 50000` — each 50K output tokens = 1 level |
| **Energy** (0-5) | Uptime | Starts at 3, drains 1 bar every 2 hours |

This creates a genuine incentive loop: approve permissions quickly (mood goes up), keep using Claude (tokens feed the pet and level it up), and the physical robot arm celebrates with a fist-pump animation on level-up.

## 5. Flicker-Free Partial Screen Updates on ESP32

The M5Stack's 320x240 LCD has no hardware double-buffering. Naive full-panel clears (`fillRect` every 250ms) caused visible flicker. The solution was a two-layer dirty flag system:

```cpp
static bool infoDirty     = true;   // any data changed?
static bool infoFullClear = true;   // layout changed? (mode switch, prompt transition)

// Main loop:
if (infoDirty && now - lastInfoMs >= 250) {
    if (infoFullClear) {
        M5.Lcd.fillRect(...);        // one-time full clear
        infoFullClear = false;
    }
    drawCurrentScreen();             // per-row clears only
    infoDirty = false;
}
```

**Per-row clearing** (`clearRow`) replaces full-panel clearing in all draw functions. Each text line clears only its own height before redrawing.

**Full clear on transitions only**: switching between screens (transcript -> pet stats -> info) or entering/leaving a permission prompt sets `infoFullClear = true` for a one-time wipe, preventing leftover text from the previous layout.

A subtle bug: the page indicator (`1/2`) used `clearRow(TOP + 2, 12)` which wiped the entire top row — erasing the "mood" label just drawn above it. Fixed by clearing only the small rectangle where the page number renders:
```cpp
// Before (bug): clears entire row including mood
clearRow(TOP + 2, 12);
// After (fix): clears only the page number corner
M5.Lcd.fillRect(INFO_X + INFO_W - 30, TOP + 2, 30, 12, COL_BG);
```

## 6. Robot Arm Animation Synced to Pet Sprite

The ASCII pet sprite and the physical robot arm share the same beat timing. Each persona state defines a tempo matching the pet's animation frame rate:

```cpp
static const uint16_t ARM_BEAT_MS[] = {
    1000,  // SLEEP:     pet uses t/5 at 200ms ticks = 1s
    1000,  // IDLE:      1s
    1000,  // BUSY:      1s
    1000,  // ATTENTION: 1s
     600,  // CELEBRATE: pet uses t/3 = 600ms (faster, excited)
     800,  // DIZZY:     pet uses t/4 = 800ms
    1000,  // HEART:     1s
};

// Arm only moves on beat change — no duplicate commands
uint8_t beat = (millis() / ARM_BEAT_MS[state]) % 4;
if (beat != lastArmBeat) {
    myCobot.writeAngles(poses[state][beat]);
    lastArmBeat = beat;
}
```

Each state has a distinct motion vocabulary:
- **Idle**: gentle left-right sway
- **Attention**: wide beckoning wave
- **Celebrate**: vertical fist pump
- **Heart**: slow graceful sway

Earlier iterations used fixed timers (1200ms) and `checkRunning()` polling, both causing servo jitter from overlapping commands. The beat-sync approach sends exactly one `writeAngles` per beat transition.

## 7. NVS Magic Byte Validation

On first flash, the ESP32's NVS (Non-Volatile Storage) contains garbage data. The Preferences API reads these as valid values — resulting in nonsensical stats like "Level 36" on a fresh device.

The fix adds a magic byte sentinel:

```cpp
static const uint8_t NVS_MAGIC = 0xB5;

inline void buddyStatsLoad() {
    uint8_t mag = _bprefs.getUChar("mag", 0);
    if (mag != NVS_MAGIC) {
        memset(&_bstats, 0, sizeof(_bstats));
        _needsInit = true;
        return;
    }
    // Valid data: load normally
}
```

A deferred-init pattern handles the circular dependency where `buddyStatsLoad()` needs to call `buddyStatsSave()` (defined later in the file):

```cpp
buddyStatsLoad();        // detects missing magic, sets flag
buddyStatsFinishLoad();  // now buddyStatsSave() is defined, write initial data
```

## 8. Graceful Degradation

The system fails silently at every layer:

- **Daemon not running**: hooks return `"ask"` — normal Claude Code terminal prompts
- **BLE disconnects mid-prompt**: pending Futures resolve as `"disconnect"` -> deny
- **Token cache mismatch**: session ID change resets cache cleanly
- **No extra dependencies**: only `bleak` (pip) on Mac; hooks use Python stdlib only
