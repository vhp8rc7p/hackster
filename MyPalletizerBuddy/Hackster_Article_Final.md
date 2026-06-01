# MyPalletizer Buddy: A Physical Tamagotchi for Claude Code


Usually, when you run an AI coding agent like Anthropic’s Claude Code CLI on your local machine and give it access to tools—like modifying a script or running a bash command—you are glued to a dry, scrolling terminal window. When the AI hits a risky tool call, it pauses and waits for you to type 'y' or 'n'. 

That workflow felt entirely too sterile. I wanted to see if I could take a standard 4-axis industrial robotic arm and turn it into something entirely different: a physical, emotive desktop companion that acts as a secure Human-in-the-Loop (HITL) gateway. 

Enter the **MyPalletizer Buddy**. By pairing an Elephant Robotics MyPalletizer 260 with an M5Stack, I built a physical Tamagotchi for Claude. When Claude needs permission to use a tool, it sends a payload over Bluetooth Low Energy (BLE) to the M5Stack. The Buddy displays the tool request on its screen, waiting for you to physically press a hardware button to approve or deny it. Meanwhile, the robotic arm physically animates based on the AI's state—slumping over when sleeping, celebrating when a task succeeds, or moving frantically when 'busy'. 

[INSERT HERO GIF OF THE ARM REACTING TO A CLAUDE PROMPT HERE]


## Environment Setup & Gotchas

Getting the ESP32 environment right is critical. Open the Arduino IDE, install the ESP32 packages via the Board Manager, and grab your libraries. But watch out for these two massive roadblocks:

### Gotcha 1: The "Sketch Too Large" Error
When you combine the BLE stack, the M5Stack graphics library (which holds the byte arrays for our pet sprites), and the arm control logic, the compiled binary gets massive. If you use the default partition scheme, the Arduino IDE will throw a "Sketch too large" error during compilation.
**The Fix:** Go to `Tools > Partition Scheme` and change it to **Huge APP (3MB No OTA/1MB SPIFFS)**. This allocates enough memory for the BLE stack and all the virtual pet graphics.

### Gotcha 2: NVS Garbage Data Initialization
On a first flash, the ESP32's Non-Volatile Storage (NVS) contains garbage data. If you read this via the Preferences API, you get nonsensical stats—your fresh device might boot up thinking it is "Level 36" with a negative mood.
**The Fix:** I implemented a magic byte sentinel (`0xB5`) to validate initial reads. If the firmware boots and does not see `0xB5` at the target memory address, it wipes the memory cleanly and starts the Tamagotchi at Level 1.

## The Code & Logic: Technical Highlights

The firmware and host scripts are where this project shines. Here are the core architectural choices that make it work.

### 1. Claude Code Hooks & Unix Socket IPC
To intercept Claude Code tool executions without modifying its core binary, I repurposed its PreToolUse and PostToolUse hooks. These are stateless, short-lived Python scripts invoked by Claude. 

To bridge these short-lived scripts to a persistent BLE connection, I built a long-running BLE daemon. The hook scripts use Unix domain socket IPC to send messages to the daemon, which in turn talks to the M5Stack. 

*A fun discovery:* Claude Code silently ignores flat JSON returns from hooks. If you just return `{"permissionDecision": "allow"}`, it falls through to the terminal prompt. You must nest the response specifically like this to bypass the terminal:
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
```

### 2. BLE NUS Bidirectional JSON
Bluetooth Low Energy limits payloads to a tiny 20-byte MTU. To send full JSON payloads containing tool names and hints, both the Python daemon and the ESP32 implement chunked writes with newline-delimited framing and a reassembly buffer. 

The daemon sends the prompt JSON, then blocks on an `asyncio.Future`. The M5Stack responds whenever you physically press a button. A 120-second timeout on the daemon auto-denies the prompt if you walk away from your desk.

### 3. Real Token Counting & Gamification
What is a Tamagotchi without stats? The PostToolUse Python hook incrementally reads Claude Code's session transcript (`.jsonl` file). To prevent re-reading massive files on every hook execution, it uses byte-offset caching.

These token counts feed the gamification engine. Every 50,000 output tokens equals 1 Level. The pet's 'Fed' metric tracks token consumption, while its 'Mood' is driven by the median response time of your last 8 approvals (e.g., approving in under 15 seconds yields max mood). This creates a genuine incentive loop to code faster and interact with the AI!

### 4. Flicker-Free UI
The M5Stack's 320x240 LCD lacks hardware double-buffering. Naive `fillRect` clears caused awful flickering every time the pet blinked. 

The solution was a two-layer dirty flag system (`infoDirty` and `infoFullClear`) combined with strict per-row clearing. A full screen wipe only occurs on complete view transitions (e.g., moving from the main menu into ClaudeMode). Otherwise, the code only overwrites the exact pixels that changed.

### 5. Animation Synchronization
Initially, I used fixed timers for the robotic arm movements. This caused terrible servo jitter due to overlapping commands. 

I refactored this to tie the physical arm's servo movements directly to the ASCII pet's animation beat timing. The arm only receives a `writeAngles` command exactly on a beat transition. Here is the logic:

```cpp
// Robot arm pose sets per state: {J1, J2, J3, J4, speed}
static const float ARM_CELEBRATE[][5] = {
    {  0, 70,  0,   0, 30}, {  0, 30, 50,  90, 30},
    {  0, 70,  0, -90, 30}, {  0, 30, 50,   0, 30},
};

static const uint16_t ARM_BEAT_MS[] = {
    1000,  // SLEEP: 1s ticks
     600,  // CELEBRATE: 600ms (faster, excited)
};

// Arm only moves on beat change
uint8_t beat = (millis() / ARM_BEAT_MS[state]) % 4;
if (beat != lastArmBeat) {
    myCobot.writeAngles(poses[state][beat]);
    lastArmBeat = beat;
}
```

## Testing & Calibration

Because the MyPalletizer uses stepper motors, they do not inherently know their absolute position on startup. If your arm boots up and the poses look mangled, do not panic. 

After flashing, use the physical buttons to navigate to the 'Calibration' routine in the main menu. You will physically move the arm to its zero position (straight up, aligned joints) and hit confirm. The M5Stack saves these offsets to the EEPROM, ensuring that when the arm goes to 'celebrate', it doesn't accidentally smash into your monitor.

## Next Steps

This project turns an AI coding workflow into a tangible, interactive experience. The codebase is fully open-source, and there is plenty of room for community upgrades. 

Some potential next steps:
*   Add an I2C Time-of-Flight (TOF) sensor to the end effector so the pet wakes up when your hand gets close.
*   Integrate a microphone module directly to the ESP32 for voice-to-text approval.
*   Design and submit new digital species for the virtual zoo!

If you build your own Claude Buddy, drop a link in the comments. I'd love to see what virtual species you come up with. Happy hacking!
