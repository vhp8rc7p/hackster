#ifndef ClaudeMode_h
#define ClaudeMode_h

#include <MyPalletizerBasic.h>
#include "ServerBase.h"
#include "SoundUtil.h"
#include "config.h"

class ClaudeMode : public ServerBase {
public:
    void run(MyPalletizerBasic &myCobot);
    static ServerBase *createInstance() {
        return new ClaudeMode();
    }

private:
    void drawSplash();
    void drawUsageScreen();
    void drawUsageBars();
    void drawStatusLine();
    void handleCommunication(MyPalletizerBasic &myCobot);

    bool EXIT = false;
    uint8_t current_frame = 0;
    uint32_t last_frame_ms = 0;
};

#endif
