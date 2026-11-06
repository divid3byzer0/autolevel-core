// Mimics exactly what the daemon does:
//   audio thread  -> AudioEngine::process()      (JackAudioEngine::processCallback / Alsa / Dummy)
//   web thread    -> AudioEngine::reset()        (WebServer POST /api/reset handler)
#include "AudioEngine.h"
#include <thread>
#include <memory>
#include <atomic>
#include <vector>
#include <cstdio>
#include <cmath>

int main() {
    auto engPtr = std::make_unique<autolevel::AudioEngine>(); auto& eng = *engPtr;
    eng.prepare(48000.0);

    std::atomic<bool> run{true};

    // The audio thread
    std::thread audio([&]{
        std::vector<float> inL(256), inR(256), outL(256), outR(256);
        double ph = 0.0;
        while (run.load()) {
            for (size_t i = 0; i < 256; ++i) {
                float s = (float)(0.2 * std::sin(ph));
                inL[i] = inR[i] = s;
                ph += 2.0 * 3.14159265358979 * 440.0 / 48000.0;
            }
            eng.process(inL.data(), inR.data(), outL.data(), outR.data(), 256);
        }
    });

    // The web thread: POST /api/reset
    std::thread web([&]{
        for (int i = 0; i < 300 && run.load(); ++i) {
            eng.reset();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    web.join();
    run.store(false);
    audio.join();
    printf("done\n");
    return 0;
}
