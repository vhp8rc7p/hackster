#ifndef ClaudeMode_h
#define ClaudeMode_h

#include <MyPalletizerBasic.h>
#include "ServerBase.h"
#include "config.h"

class ClaudeMode : public ServerBase {
public:
    void run(MyPalletizerBasic &myCobot);
    static ServerBase *createInstance() {
        return new ClaudeMode();
    }

private:
    bool EXIT = false;
};

#endif
