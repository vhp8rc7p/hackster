#ifndef SoundUtil_h
#define SoundUtil_h

#include <M5Stack.h>
#include <SD.h>

class SoundUtil {
public:
    static bool sd_ok;

    static void init() {
        M5.Speaker.begin();
        M5.Speaker.setVolume(8);
        dacWrite(25, 0);
        SPI.begin();
        sd_ok = SD.begin(4, SPI, 20000000);
        if (!sd_ok) sd_ok = SD.begin(4);
    }

    static void play(const char* path) {
        if (!sd_ok) return;

        File file = SD.open(path);
        if (!file) return;

        size_t fileSize = file.size();
        if (fileSize <= 44) { file.close(); return; }
        file.seek(44);
        size_t dataSize = fileSize - 44;

        uint8_t *buf = (uint8_t *)malloc(dataSize);
        if (!buf) { file.close(); return; }

        file.read(buf, dataSize);
        file.close();

        for (size_t i = 0; i < dataSize; i++) {
            dacWrite(25, buf[i] / 3);
            delayMicroseconds(125);
        }
        dacWrite(25, 0);
        free(buf);
    }

    static void beep() {
        M5.Speaker.beep();
    }
};

inline bool SoundUtil::sd_ok = false;

#endif
