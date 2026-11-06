#pragma once

#include "Bands.h"
#include "Biquad.h"
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace autolevel::dsp {

/**
 * Pre-MBC EQ (2026-09-11): a plain 6-band parametric EQ, applied before the Multiband
 * Compressor so the project owner can shape tone by hand and let the MBC's own dynamics
 * reign that shaping back in afterward - replaces the adaptive Bass Lift stage entirely
 * (removed the same day; see git log / docs changelog), which is no longer part of the
 * signal chain in any form.
 *
 * Deliberately reuses the same 6 bands as the MultibandCompressor (Bands.h: Sub, Bass,
 * Low-Mid, High-Mid, Presence, Air) rather than inventing a separate band layout - a boost
 * here and the corresponding MBC threshold/meter are then talking about the same slice of
 * spectrum. Each band is one RBJ cookbook peaking (bell) filter centered at that band's
 * geometric center frequency (`Bands::getCenterFreq`), with Q derived from the band's own
 * octave span (`Q = sqrt(2^n) / (2^n - 1)`, the standard octave-bandwidth-to-Q conversion)
 * so each bell's width roughly matches the MBC band it precedes - no free Q parameter to
 * expose or reason about. Gain is a plain per-band dB value, no gating or adaptivity: this
 * is a manual tone-shaping tool, not another auto-detector.
 *
 * Bands are cascaded in series, one Biquad per band per channel (12 total). Coefficients
 * are only recomputed when a band's gain actually changes (same pattern as HighPassFilter's
 * setCutoff), not every block.
 */
class PreMbcEQ {
public:
    PreMbcEQ() = default;

    void prepare(double sampleRate) noexcept {
        m_sampleRate = (sampleRate > 1000.0) ? sampleRate : 44100.0;
        for (size_t b = 0; b < Bands::COUNT; ++b) {
            m_lastGainsDb[b] = 0.0f;
            updateBand(b, 0.0f);
        }
        reset();
    }

    void reset() noexcept {
        for (auto& f : m_filtersL) f.reset();
        for (auto& f : m_filtersR) f.reset();
    }

    void process(float* left, float* right, size_t numSamples,
                 const std::array<float, Bands::COUNT>& gainsDb) noexcept {
        if (numSamples == 0) return;

        bool anyEngaged = false;
        for (size_t b = 0; b < Bands::COUNT; ++b) {
            if (std::abs(gainsDb[b] - m_lastGainsDb[b]) > 0.02f) {
                updateBand(b, gainsDb[b]);
                m_lastGainsDb[b] = gainsDb[b];
            }
            if (std::abs(gainsDb[b]) > 0.02f) anyEngaged = true;
        }

        if (!anyEngaged) {
            return; // All bands flat: exact bypass, no filtering cost.
        }

        for (size_t i = 0; i < numSamples; ++i) {
            double l = static_cast<double>(left[i]);
            double r = static_cast<double>(right[i]);
            for (size_t b = 0; b < Bands::COUNT; ++b) {
                l = m_filtersL[b].process(l);
                r = m_filtersR[b].process(r);
            }
            left[i] = static_cast<float>(l);
            right[i] = static_cast<float>(r);
        }
    }

private:
    void updateBand(size_t b, float gainDb) noexcept {
        double centerHz = static_cast<double>(Bands::getCenterFreq(b));
        double octaves = static_cast<double>(Bands::getOctaveSpan(b));
        double pow2n = std::pow(2.0, octaves);
        double q = std::sqrt(pow2n) / (pow2n - 1.0);
        m_filtersL[b].setupPeaking(centerHz, static_cast<double>(gainDb), q, m_sampleRate);
        m_filtersR[b].setupPeaking(centerHz, static_cast<double>(gainDb), q, m_sampleRate);
    }

    double m_sampleRate = 44100.0;
    std::array<Biquad, Bands::COUNT> m_filtersL{};
    std::array<Biquad, Bands::COUNT> m_filtersR{};
    std::array<float, Bands::COUNT> m_lastGainsDb{};
};

} // namespace autolevel::dsp
