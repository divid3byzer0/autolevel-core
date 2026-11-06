# Diagnostic tools

Throwaway-looking programs that are worth keeping: each one proved a specific
defect during the 2026-09-10 bug hunt and is the cheapest way to prove that
defect has not come back. None are built by CMake; compile them by hand.

## tsan_reset_race.cpp

Drives `AudioEngine::process()` on one thread and `AudioEngine::reset()` on
another, exactly as the JACK callback and the `POST /api/reset` handler do.
Before the fix this reported **155 data races** across every DSP stage; it must
report **0**.

    c++ -std=gnu++17 -g -O0 -fno-inline -fsanitize=thread \
        -I src/dsp -I src/engine tools/tsan_reset_race.cpp -o /tmp/race
    /tmp/race 2>&1 | grep -c "WARNING: ThreadSanitizer: data race"   # expect 0

Run this after ANY change to reset/prepare paths or to how state is shared
between the audio thread and the web server.

## dsp_benchmark.cpp

Processes 60 s of stereo through the full chain at the shipped defaults and
reports the realtime factor. Use it before/after DSP changes to catch a
regression in CPU cost on the Pi (a Pi 4 has far less headroom than a laptop).

    g++ -std=gnu++17 -O3 -ffast-math -fno-finite-math-only \
        -I src/dsp tools/dsp_benchmark.cpp -o /tmp/bench && /tmp/bench

Reference figures measured on the Pi 4 at 44.1 kHz:
  before the crossover allpass fix : 2.417 s (4.03% of one core)
  after                            : 2.715 s (4.53% of one core), 22x realtime

## crossover_probe.cpp

Measures the summed magnitude response of the 6-band filterbank, and compares it
against a plain 2-way LR4 split as a control. This is what showed the tree had a
-5.15 dB notch at 1200 Hz while the biquads themselves were exact.

    c++ -std=gnu++17 -O2 -I src/dsp tools/crossover_probe.cpp -o /tmp/xp && /tmp/xp

`testLR4CrossoverSummation` in tests/dsp_test.cpp now covers this automatically.
Keep this probe for when you need the per-frequency detail rather than a
pass/fail, e.g. if you retune the crossover frequencies.

IMPORTANT when measuring the crossover: `compressionAmount` must be at least
`Bands::MIN_COMPRESSION` (0.02). Below that `MultibandCompressor::process()`
returns without touching the buffer, and any measurement becomes a comparison of
the input against itself. That is exactly how the original test passed while
measuring nothing.

## box_render.cpp

Renders interleaved stereo float32 through the engine exactly as the box runs it (48 kHz,
256-frame blocks, defaults or a preset), and compares two renders. This is how the plugin's
output is checked against the box built with the box's own flags.

    c++ -std=gnu++17 -O3 -ffast-math -fno-finite-math-only -I src/dsp -I src/engine \
        tools/box_render.cpp -o /tmp/box_render
    /tmp/box_render --make-input /tmp/in.f32 60
    /tmp/box_render /tmp/in.f32 /tmp/box.f32
    # plugin side (autolevel-dj-vst): build-dir/plugin_test --render /tmp/in.f32 /tmp/plugin.f32
    /tmp/box_render --compare /tmp/box.f32 /tmp/plugin.f32

Result on 2026-11-06 (Apple Silicon, 60 s program): identical to the plugin when both are
built without fast-math; with the box's -ffast-math the largest difference is -138 dBFS,
below 24-bit resolution.
