// Render audio through the engine exactly as autolevel-box runs it: defaults (or a preset),
// 48 kHz, 256-frame blocks. Used to check that another host of the engine (the plugin) sounds
// the same as the box, built with the box's own compiler flags.
//
//   box_render --make-input in.f32 [seconds]       write a test program (interleaved stereo f32)
//   box_render in.f32 out.f32 [preset.json]        render it
//   box_render --compare a.f32 b.f32               report how far apart two renders are
//
// Build like the box:
//   c++ -std=gnu++17 -O3 -ffast-math -fno-finite-math-only -I src/dsp -I src/engine \
//       tools/box_render.cpp -o /tmp/box_render

#include "AudioEngine.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

static std::vector<float> readF32(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
    return out;
}

static bool writeF32(const char* path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<bool>(f);
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--make-input") {
        // Loud and quiet sections (5 s each) so the Leveler, breakdown freeze and limiter all
        // work; a 55 Hz tone for the Sub band and hiss for Air.
        const double sr = 48000.0;
        const size_t n = static_cast<size_t>(sr * (argc >= 4 ? std::atof(argv[3]) : 60.0));
        std::vector<float> v(n * 2);
        std::mt19937 rng(99);
        std::normal_distribution<double> g(0.0, 1.0);
        double bl = 0.0, br = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / sr;
            const double lvl = (static_cast<int>(t / 5.0) % 2) ? 0.04 : 0.45;
            bl = 0.96 * bl + 0.04 * g(rng);
            br = 0.96 * br + 0.04 * g(rng);
            const double sub = 0.3 * std::sin(2.0 * 3.141592653589793 * 55.0 * t);
            const double hiss = 0.02 * g(rng);
            v[i * 2] = static_cast<float>(lvl * (3.0 * bl + sub) + hiss);
            v[i * 2 + 1] = static_cast<float>(lvl * (3.0 * br + sub) - hiss);
        }
        return writeF32(argv[2], v) ? 0 : 1;
    }

    if (argc == 4 && std::string(argv[1]) == "--compare") {
        const auto a = readF32(argv[2]);
        const auto b = readF32(argv[3]);
        const size_t n = std::min(a.size(), b.size());
        size_t differ = 0;
        double maxd = 0.0, diffsq = 0.0, sigsq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - b[i];
            if (a[i] != b[i]) ++differ;
            maxd = std::max(maxd, std::abs(d));
            diffsq += d * d;
            sigsq += static_cast<double>(a[i]) * a[i];
        }
        std::printf("%zu samples compared (%zu vs %zu), %zu differ\n", n, a.size(), b.size(), differ);
        if (differ) {
            std::printf("largest difference %.2f dBFS, difference energy %.1f dB below the signal\n",
                        20.0 * std::log10(maxd), 10.0 * std::log10(sigsq / diffsq));
        }
        return 0;
    }

    if (argc < 3) {
        std::fprintf(stderr, "usage: see the comment at the top of box_render.cpp\n");
        return 2;
    }
    auto v = readF32(argv[1]);
    autolevel::AudioEngine engine;
    if (argc >= 4) {
        std::ifstream f(argv[3]);
        engine.loadStateFromJson(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    }
    engine.prepare(48000.0);
    const size_t frames = v.size() / 2;
    constexpr size_t BLOCK = 256; // the box's JACK period
    std::vector<float> l(BLOCK), r(BLOCK), out;
    out.reserve(v.size());
    for (size_t pos = 0; pos + BLOCK <= frames; pos += BLOCK) {
        for (size_t i = 0; i < BLOCK; ++i) { l[i] = v[(pos + i) * 2]; r[i] = v[(pos + i) * 2 + 1]; }
        engine.process(l.data(), r.data(), l.data(), r.data(), BLOCK);
        for (size_t i = 0; i < BLOCK; ++i) { out.push_back(l[i]); out.push_back(r[i]); }
    }
    return writeF32(argv[2], out) ? 0 : 1;
}
