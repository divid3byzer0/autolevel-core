#pragma once

#include <array>
#include <cmath>
#include <string_view>
#include <numeric>

namespace autolevel::dsp {

class Bands {
public:
    static constexpr size_t COUNT = 6;

    static constexpr std::array<float, 5> CROSSOVERS = {
        120.0f, 400.0f, 1200.0f, 3500.0f, 8000.0f
    };

    static constexpr std::array<float, COUNT> LOWER_EDGES = {
        20.0f, 120.0f, 400.0f, 1200.0f, 3500.0f, 8000.0f
    };

    static constexpr std::array<float, COUNT> UPPER_EDGES = {
        120.0f, 400.0f, 1200.0f, 3500.0f, 8000.0f, 20000.0f
    };

    static constexpr std::array<std::string_view, COUNT> NAMES = {
        "Sub", "Bass", "Low-Mid", "High-Mid", "Presence", "Air"
    };

    /** Geometric centre frequency of each band */
    static inline float getCenterFreq(size_t band) noexcept {
        return std::sqrt(LOWER_EDGES[band] * UPPER_EDGES[band]);
    }

    /** Width of each band in octaves */
    static inline float getOctaveSpan(size_t band) noexcept {
        return static_cast<float>(std::log2(UPPER_EDGES[band] / LOWER_EDGES[band]));
    }

    /** log2 of the centre frequency */
    static inline float getOctave(size_t band) noexcept {
        return static_cast<float>(std::log2(getCenterFreq(band)));
    }

    // Precomputed octaves and spans matching Android exactly:
    // centres = [48.9898, 219.089, 692.82, 2049.39, 5291.5, 12649.1]
    // octaves = [5.61438, 7.77537, 9.43632, 11.0009, 12.3694, 13.6267]
    // spans   = [2.58496, 1.73696, 1.58496, 1.54432, 1.19264, 1.32192]
    static constexpr std::array<float, COUNT> OCTAVES = {
        5.614384f, 7.775374f, 9.436320f, 11.000922f, 12.369400f, 13.626772f
    };

    static constexpr std::array<float, COUNT> OCTAVE_SPANS = {
        2.584963f, 1.736966f, 1.584963f, 1.544321f, 1.192645f, 1.321928f
    };

    static constexpr float MEAN_OCTAVE = 9.970562f;

    // Not referenced anywhere else in the codebase (the actual UI-exposed range/default
    // live on the "tone_slope" AudioEngine param and the web UI's #fader-tone-slope slider).
    // Kept here, matching that parameter, purely for documentation.
    static constexpr float MIN_TONE_SLOPE = -3.0f;
    static constexpr float MAX_TONE_SLOPE = 0.0f;
    // Default is flat (0 dB/oct) as of 2026-09-11 - the app now always starts flat/neutral
    // regardless of any saved preset (see PresetStore.h); tilt and per-band contour are both
    // purely manual, opt-in shaping from here.
    static constexpr float DEFAULT_TONE_SLOPE = 0.0f;

    static constexpr float MBC_THRESHOLD_DB = -24.0f;
    static constexpr float MAX_RATIO = 5.0f;            // 4:1 until 2026-10-03
    // The MBC's base threshold sits this far below Target LUFS (15 dB until 2026-10-03):
    // -30 dBFS at a -12 LUFS target, -32 at the default -14.
    static constexpr float MBC_THRESHOLD_BELOW_TARGET_DB = 18.0f;
    static constexpr float MIN_COMPRESSION = 0.02f;

    /**
     * Per-band compressor thresholds in dBFS, exactly matching Android Shaper.kt.
     * Tilted puts each threshold on the tone curve, converted from power-per-octave
     * into the share-of-total-power that band levels are measured in (bandwidth term),
     * then re-centred so the average threshold is unchanged (-24 dB). `customOffsetsDb` is
     * the sole per-band contour source - a flat/all-zero array (the default) reproduces the
     * old "Pink Noise/Linear" tone-only curve exactly, since a zero contour is mathematically
     * a no-op on top of the tilt. The separate MODERN_MIX contour and the profile switch that
     * used to choose between it and a custom one were removed 2026-09-11, at the project
     * owner's request, in favor of a single always-flat-by-default, always-customizable curve.
     */
    static inline std::array<float, COUNT> thresholdsFor(
        bool tilted,
        float toneSlopeDbPerOctave,
        const std::array<float, COUNT>& customOffsetsDb = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        float baseThresholdDb = MBC_THRESHOLD_DB
    ) {
        std::array<float, COUNT> out{};
        if (!tilted) {
            out.fill(baseThresholdDb);
            return out;
        }

        float mean = 0.0f;
        for (size_t b = 0; b < COUNT; ++b) {
            float density = toneSlopeDbPerOctave * (OCTAVES[b] - MEAN_OCTAVE);
            float bandwidth = static_cast<float>(10.0 * std::log10(OCTAVE_SPANS[b]));
            out[b] = density + bandwidth + customOffsetsDb[b];
            mean += out[b];
        }
        mean /= static_cast<float>(COUNT);
        for (size_t b = 0; b < COUNT; ++b) {
            out[b] = baseThresholdDb + (out[b] - mean);
        }
        return out;
    }
};

} // namespace autolevel::dsp
