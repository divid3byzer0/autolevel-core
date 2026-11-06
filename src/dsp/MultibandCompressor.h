#pragma once

#include "Bands.h"
#include "Biquad.h"
#include <array>
#include <cmath>
#include <algorithm>

namespace autolevel::dsp {

struct MBCParams {
    bool enabled = true;
    float compressionAmount = 0.5f;     // 0.0 (bypass) to 1.0 (heavy)
    float toneSlopeDbPerOctave = -2.0f; // Tonal target tilt (-3.0 to 0.0 dB/oct, default -2.0)
    std::array<float, Bands::COUNT> customOffsetsDb = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float baseThresholdDb = Bands::MBC_THRESHOLD_DB;
    /**
     * Ballistics for bands 1-5, in ms; the Sub band always runs at twice both. The defaults are
     * the old "Normal" speed (Sub 30/400), which these replaced along with Slow and Fast.
     */
    float attackMs = DEFAULT_ATTACK_MS;
    float releaseMs = DEFAULT_RELEASE_MS;
    /**
     * Detector character, 0..1: 0 is the stereo peak detector (the long-standing behaviour),
     * 1 is RMS, between is a linear blend of the two levels before attack/release. RMS is scaled
     * by sqrt(2) so a steady sine reads about the same either way (within ~1 dB: the peak path,
     * smoothed by the attack, never quite reaches a sine's crest) - the blend changes how
     * transients are treated, not how deep sustained material is compressed.
     *
     * The RMS window is long (150 ms; 300 ms on Sub) on purpose. The square root of a
     * part-filled average rises like sqrt(t), so a short window reacts to a burst *faster* than
     * a peak path smoothed by a 15 ms attack - the opposite of what RMS is for. Measured in the
     * Zeroplayer engine this was ported from: with 20 ms, RMS clamped 4 ms bursts harder than
     * peak at the default attack.
     */
    float detectorRms = 0.0f;
    /**
     * Program-dependent release. A second, slow detector path (attack 10x, at least 100 ms;
     * release 5x) runs alongside and the band follows whichever is higher: a short hit still
     * recovers at the set release, a sustained loud passage builds up the slow path and holds
     * its reduction, so the gain does not pump back up between beats. It adds no latency - the
     * audio is never delayed, only the gain's recovery changes.
     */
    bool autoRelease = false;

    static constexpr float DEFAULT_ATTACK_MS = 15.0f;
    static constexpr float MIN_ATTACK_MS = 1.0f;
    static constexpr float MAX_ATTACK_MS = 100.0f;
    static constexpr float DEFAULT_RELEASE_MS = 200.0f;
    static constexpr float MIN_RELEASE_MS = 20.0f;
    static constexpr float MAX_RELEASE_MS = 1000.0f;
};

class BandCompressor {
public:
    BandCompressor() = default;

    void setup(double sampleRate, float attackMs, float releaseMs, float rmsWindowMs) {
        m_sampleRate = sampleRate;
        updateBallistics(attackMs, releaseMs);
        m_rmsCoeff = coeff(rmsWindowMs);
        reset();
    }

    void updateBallistics(float attackMs, float releaseMs) noexcept {
        m_attackCoeff = coeff(attackMs);
        m_releaseCoeff = coeff(releaseMs);
        // Auto release's slow path, see MBCParams::autoRelease.
        m_slowAttackCoeff = coeff(std::max(attackMs * 10.0f, 100.0f));
        m_slowReleaseCoeff = coeff(releaseMs * 5.0f);
    }

    void reset() {
        m_envelope = 0.0;
        m_slowEnvelope = 0.0;
        m_meanSquare = 0.0;
        m_gainReductionDb = 0.0f;
    }

    /**
     * Exact 6 dB soft-knee dynamics compression matching Android DynamicsProcessing MBC.
     * One detector on the louder channel, the same gain applied to both.
     */
    inline void processStereo(float& left, float& right, float thresholdDb, float ratio, float amount,
                              float detectorRms, bool autoRelease) noexcept {
        if (amount < Bands::MIN_COMPRESSION || ratio <= 1.001f) {
            m_gainReductionDb = 0.0f;
            return;
        }

        // Level: the stereo peak, blended towards RMS. The mean square is tracked even at a
        // blend of 0, so moving the control never starts from a cold average.
        const double peak = std::max(std::abs(left), std::abs(right));
        const double sq = std::max(static_cast<double>(left) * left,
                                   static_cast<double>(right) * right);
        m_meanSquare = sq + m_rmsCoeff * (m_meanSquare - sq);
        double level = peak;
        if (detectorRms > 0.0f) {
            constexpr double SQRT2 = 1.4142135623730951; // sine calibration, see MBCParams
            const double rms = std::sqrt(m_meanSquare) * SQRT2;
            const double b = std::min(static_cast<double>(detectorRms), 1.0);
            level = (1.0 - b) * level + b * rms;
        }

        m_envelope = level + (level > m_envelope ? m_attackCoeff : m_releaseCoeff) * (m_envelope - level);
        // The slow path is tracked always, so switching auto release on does not start cold.
        m_slowEnvelope = level + (level > m_slowEnvelope ? m_slowAttackCoeff : m_slowReleaseCoeff)
                                     * (m_slowEnvelope - level);
        const double env = autoRelease ? std::max(m_envelope, m_slowEnvelope) : m_envelope;

        double envDb = (env > 1e-6) ? 20.0 * std::log10(env) : -120.0;
        double overDb = envDb - thresholdDb;

        // Soft-knee calculation (knee width = 6.0 dB, half-knee = 3.0 dB)
        constexpr double halfKnee = 3.0;
        double grDb = 0.0;
        double slope = 1.0 - 1.0 / static_cast<double>(ratio);

        if (overDb <= -halfKnee) {
            grDb = 0.0;
        } else if (overDb < halfKnee) {
            // Inside the 6 dB soft-knee transition
            double x = overDb + halfKnee;
            grDb = -(slope * x * x) / (4.0 * halfKnee);
        } else {
            // Fully above the knee
            grDb = -slope * overDb;
        }

        m_gainReductionDb = static_cast<float>(grDb);
        if (grDb < 0.0) {
            float gainLin = static_cast<float>(std::pow(10.0, grDb / 20.0));
            left *= gainLin;
            right *= gainLin;
        }
    }

    float getGainReductionDb() const noexcept {
        return m_gainReductionDb;
    }

private:
    double coeff(float ms) const noexcept {
        return std::exp(-1.0 / (m_sampleRate * (static_cast<double>(ms) * 0.001)));
    }

    double m_sampleRate = 48000.0;
    double m_attackCoeff = 0.0;
    double m_releaseCoeff = 0.0;
    double m_rmsCoeff = 0.0;
    double m_slowAttackCoeff = 0.0;
    double m_slowReleaseCoeff = 0.0;
    double m_envelope = 0.0;
    double m_slowEnvelope = 0.0;
    double m_meanSquare = 0.0;
    float m_gainReductionDb = 0.0f;
};

/**
 * 6-band Linkwitz-Riley (LR4) crossover filterbank and dynamic compressor.
 * Thresholds mirror Android Shaper; attack/release are set by MBCParams (default 15/200 ms,
 * the Sub band always at twice both, i.e. 30/400 ms). Knee width: 6 dB.
 */
class MultibandCompressor {
public:
    MultibandCompressor() = default;

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;

        // Setup crossovers at: 120, 400, 1200, 3500, 8000 Hz
        for (size_t ch = 0; ch < 2; ++ch) {
            for (size_t i = 0; i < 5; ++i) {
                m_lp[ch][i].setup(Biquad::Type::LowPass, Bands::CROSSOVERS[i], sampleRate);
                m_hp[ch][i].setup(Biquad::Type::HighPass, Bands::CROSSOVERS[i], sampleRate);
            }
        }

        // Phase-compensation allpasses (see AP_COMPENSATION below)
        for (size_t ch = 0; ch < 2; ++ch) {
            for (size_t b = 0; b < Bands::COUNT; ++b) {
                for (size_t k = 0; k < AP_MAX; ++k) {
                    int xover = AP_COMPENSATION[b][k];
                    if (xover >= 0) {
                        m_ap[ch][b][k].setup(Bands::CROSSOVERS[static_cast<size_t>(xover)], sampleRate);
                    }
                }
            }
        }

        // Default ballistics (Sub at twice the others, slower in bass as broadcast chains do).
        // RMS windows: long enough that RMS never out-reacts the attack-smoothed peak path on a
        // transient (see MBCParams::detectorRms); Sub twice as long, as with its ballistics.
        m_bands[0].setup(sampleRate, MBCParams::DEFAULT_ATTACK_MS * 2.0f,
                         MBCParams::DEFAULT_RELEASE_MS * 2.0f, 300.0f);
        for (size_t b = 1; b < Bands::COUNT; ++b) {
            m_bands[b].setup(sampleRate, MBCParams::DEFAULT_ATTACK_MS, MBCParams::DEFAULT_RELEASE_MS, 150.0f);
        }
        // setup() above reinstalls the default ballistics, so forget the cached times;
        // otherwise updateTimes() early-returns and user-set times silently revert to the
        // defaults after any prepare() (e.g. a JACK sample-rate change).
        m_attackMs = MBCParams::DEFAULT_ATTACK_MS;
        m_releaseMs = MBCParams::DEFAULT_RELEASE_MS;

        reset();
    }

    void reset() {
        for (size_t ch = 0; ch < 2; ++ch) {
            for (size_t i = 0; i < 5; ++i) {
                m_lp[ch][i].reset();
                m_hp[ch][i].reset();
            }
            for (size_t b = 0; b < Bands::COUNT; ++b) {
                for (size_t k = 0; k < AP_MAX; ++k) m_ap[ch][b][k].reset();
            }
        }
        for (size_t b = 0; b < Bands::COUNT; ++b) {
            m_bands[b].reset();
        }
    }

    /**
     * Bands 1-5 at attackMs/releaseMs, Sub at twice both. Cached, so the exp() calls in
     * updateBallistics run only when a value actually changes.
     */
    void updateTimes(float attackMs, float releaseMs) {
        if (attackMs == m_attackMs && releaseMs == m_releaseMs) return;
        m_attackMs = attackMs;
        m_releaseMs = releaseMs;
        m_bands[0].updateBallistics(attackMs * 2.0f, releaseMs * 2.0f);
        for (size_t b = 1; b < Bands::COUNT; ++b) {
            m_bands[b].updateBallistics(attackMs, releaseMs);
        }
    }

    void process(float* left, float* right, size_t numSamples, const MBCParams& params) {
        if (!params.enabled || params.compressionAmount < Bands::MIN_COMPRESSION) {
            for (size_t b = 0; b < Bands::COUNT; ++b) {
                m_bands[b].reset();
            }
            return;
        }

        updateTimes(std::clamp(params.attackMs, MBCParams::MIN_ATTACK_MS, MBCParams::MAX_ATTACK_MS),
                    std::clamp(params.releaseMs, MBCParams::MIN_RELEASE_MS, MBCParams::MAX_RELEASE_MS));
        const float detectorRms = std::clamp(params.detectorRms, 0.0f, 1.0f);

        // Calculate thresholds per band using exact Android Shaper formula
        std::array<float, Bands::COUNT> thresholds = Bands::thresholdsFor(
            true, params.toneSlopeDbPerOctave, params.customOffsetsDb, params.baseThresholdDb
        );

        // Compression ratio: 1:1 at 0 to 5:1 at 1.0 (was 4:1, matching Android Shaper.kt)
        float ratio = 1.0f + params.compressionAmount * (Bands::MAX_RATIO - 1.0f);

        // No makeup gain here: the MBC's output level is set by hand with the Post-MBC Gain
        // stage that follows it (auto-makeup was removed 2026-09-28 at the owner's request).
        for (size_t s = 0; s < numSamples; ++s) {
            std::array<float, Bands::COUNT> bandL;
            std::array<float, Bands::COUNT> bandR;

            split6Bands(0, left[s], bandL);
            split6Bands(1, right[s], bandR);

            float outL = 0.0f;
            float outR = 0.0f;

            for (size_t b = 0; b < Bands::COUNT; ++b) {
                m_bands[b].processStereo(bandL[b], bandR[b], thresholds[b], ratio, params.compressionAmount,
                                         detectorRms, params.autoRelease);
                outL += bandL[b];
                outR += bandR[b];
            }

            left[s] = outL;
            right[s] = outR;
        }
    }

    std::array<float, Bands::COUNT> getGainReductionsDb() const noexcept {
        std::array<float, Bands::COUNT> gr;
        for (size_t b = 0; b < Bands::COUNT; ++b) {
            gr[b] = m_bands[b].getGainReductionDb();
        }
        return gr;
    }

private:
    inline void split6Bands(size_t ch, float in, std::array<float, Bands::COUNT>& outBands) noexcept {
        // Crossover 2 (1200 Hz): Split into Low (< 1200 Hz) and High (> 1200 Hz)
        double low1200 = m_lp[ch][2].process(in);
        double high1200 = m_hp[ch][2].process(in);

        // Low branch: Split at Crossover 1 (400 Hz)
        double low400 = m_lp[ch][1].process(low1200);
        double high400 = m_hp[ch][1].process(low1200);

        // Sub branch: Split low400 at Crossover 0 (120 Hz)
        outBands[0] = static_cast<float>(m_lp[ch][0].process(low400)); // Sub (< 120)
        outBands[1] = static_cast<float>(m_hp[ch][0].process(low400)); // Bass (120 - 400)
        outBands[2] = static_cast<float>(high400);                     // Low-Mid (400 - 1200)

        // High branch: Split at Crossover 3 (3500 Hz)
        double low3500 = m_lp[ch][3].process(high1200);
        double high3500 = m_hp[ch][3].process(high1200);

        outBands[3] = static_cast<float>(low3500);                     // High-Mid (1200 - 3500)

        // HighHigh branch: Split at Crossover 4 (8000 Hz)
        outBands[4] = static_cast<float>(m_lp[ch][4].process(high3500)); // Presence (3500 - 8000)
        outBands[5] = static_cast<float>(m_hp[ch][4].process(high3500)); // Air (> 8000)

        // Phase-align the bands so they reconstruct flat when summed. Each band
        // is passed through the allpasses of the splits its own path skipped;
        // this leaves every band's magnitude response untouched (so the per-band
        // thresholds keep their calibration) and only corrects the summation.
        for (size_t b = 0; b < Bands::COUNT; ++b) {
            double v = static_cast<double>(outBands[b]);
            for (size_t k = 0; k < AP_MAX; ++k) {
                if (AP_COMPENSATION[b][k] < 0) break;
                v = m_ap[ch][b][k].process(v);
            }
            outBands[b] = static_cast<float>(v);
        }
    }

    /**
     * Which crossovers each band must be allpass-corrected by, as indices into
     * Bands::CROSSOVERS = {120, 400, 1200, 3500, 8000}; -1 terminates the list.
     *
     * The tree splits at 1200 first, then 400 and 120 down the low branch and
     * 3500 and 8000 down the high branch. Writing L for the 1200 low branch and
     * H for the high one, and using LP+HP = AP at each split:
     *
     *   bands 0+1 already sum to LP1200*LP400*AP120, so band 2 (LP1200*HP400)
     *   needs AP120 to let the 400 split close:  L = LP1200*AP120*AP400
     *   bands 4+5 already sum to HP1200*HP3500*AP8000, so band 3 needs AP8000:
     *                                            H = HP1200*AP8000*AP3500
     *   L and H now carry different allpasses, so the 1200 split cannot close.
     *   Give L the high branch's pair and H the low branch's pair, and the whole
     *   sum collapses to AP120*AP400*AP3500*AP8000*AP1200 - flat magnitude.
     */
    static constexpr size_t AP_MAX = 3;
    static constexpr int AP_COMPENSATION[Bands::COUNT][AP_MAX] = {
        { 3,  4, -1 },   // Sub       : cross-branch AP3500, AP8000
        { 3,  4, -1 },   // Bass      : cross-branch AP3500, AP8000
        { 0,  3,  4 },   // Low-Mid   : intra AP120 + cross-branch AP3500, AP8000
        { 4,  0,  1 },   // High-Mid  : intra AP8000 + cross-branch AP120, AP400
        { 0,  1, -1 },   // Presence  : cross-branch AP120, AP400
        { 0,  1, -1 }    // Air       : cross-branch AP120, AP400
    };

    double m_sampleRate = 48000.0;

    std::array<std::array<LR4Filter, 5>, 2> m_lp;
    std::array<std::array<LR4Filter, 5>, 2> m_hp;
    std::array<std::array<std::array<AllpassLR4, AP_MAX>, Bands::COUNT>, 2> m_ap;

    std::array<BandCompressor, Bands::COUNT> m_bands;
    float m_attackMs = MBCParams::DEFAULT_ATTACK_MS;
    float m_releaseMs = MBCParams::DEFAULT_RELEASE_MS;
};

} // namespace autolevel::dsp
