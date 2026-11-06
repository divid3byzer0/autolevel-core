# autolevel-core

The AutoLevel audio engine, shared by:

- **autolevel-box**: the Raspberry Pi daemon (JACK/ALSA, web UI)
- **autolevel-dj-vst**: the VST3 / AU / Standalone plugin (JUCE)

Both pull this repo in as a git submodule, so they run the same processing and a fix here
reaches both. Header-only C++17, no dependencies.

## Contents

| Path | What it is |
|---|---|
| `src/dsp/AutoLevelEngine.h` | The full chain: AGC → Air Lift → Pre-MBC EQ → 6-band MBC → Post Gain → Low Cut → true-peak limiter |
| `src/dsp/*.h` | The individual stages |
| `src/engine/AudioEngine.h` | Thread-safe parameter store over the chain: string parameter IDs with their ranges, input sanitising, and the preset JSON format (`getStateJson` / `loadStateFromJson`, including older-preset migration) |
| `tests/dsp_test.cpp` | Unit tests for the chain |
| `tools/` | Diagnostic programs: reset race (TSan), benchmark, crossover probe, and `box_render` (renders audio exactly as the box does, to compare another host against it); see `tools/README.md` |

## Using it

```cmake
add_subdirectory(external/autolevel-core)
target_link_libraries(my_target PRIVATE autolevel::core)
```

```cpp
#include "AudioEngine.h"
autolevel::AudioEngine engine;
engine.prepare(48000.0);
engine.setParam("target_lufs", -12.0f);
engine.process(inL, inR, outL, outR, 256);   // in-place (in == out) is fine
```

The box always calls `process()` with 256 frames (its JACK period). The Leveler and meters
update once per call, so a host that wants the box's exact output should call it in 256-frame
chunks too, as the plugin does.

## Tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/autolevel_core_dsp_test
```

## History

Started 2026-11-06 from autolevel-box `35a1dfe` (`src/dsp/`, `src/audio/AudioEngine.h`,
`tests/dsp_test.cpp`, `tools/`). Earlier history of these files is in autolevel-box.
