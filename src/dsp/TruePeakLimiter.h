#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace autolevel::dsp {

/**
 * Lookahead true-peak brickwall. Transparent unless something would go over the ceiling.
 *
 * Ported from the Zeroplayer engine (2026-10-03). Replaces SafetyLimiter, a 20:1 compressor
 * with a 1 ms attack followed by a hard clamp: anything held over the ceiling came out
 * clipped (1.8-3.5% THD on a steady tone 1-6 dB over, ~12 ms of every kick 6 dB over on the
 * clamp), and it only saw sample peaks, so inter-sample peaks went straight to the DAC.
 *
 * How this one works, per sample:
 *
 *  1. **Detect the true peak.** An 8x windowed-sinc interpolator estimates the waveform
 *     between samples as well as at them. Detection only: the audio itself is never
 *     resampled or filtered. When a window's sample peak times the interpolator's worst-case
 *     gain cannot reach the ceiling, the interpolation is skipped, so quiet audio costs a
 *     few compares.
 *  2. **Required gain** r = ceiling / truePeak, or exactly 1 when under the ceiling.
 *  3. **Look ahead.** The audio is delayed by [latencyFrames]. A sliding minimum of r over
 *     the lookahead, then a moving average over the same span, brings the gain down in a
 *     smooth ~2 ms ramp that is complete by the time the peak reaches the output. Nothing
 *     has to be clipped.
 *  4. **Hold, then release.** The reduction holds for 25 ms, longer than half a cycle of
 *     20 Hz, so the gain cannot ripple within a bass cycle (the old limiter's bass
 *     distortion). Then it eases back over ~80 ms. A peak restarts the hold when it needs
 *     at least the current reduction *give or take HOLD_TOLERANCE_DB* - see there.
 *  5. **Pass-through is exact.** While the whole averaging window is at unity the gain is
 *     exactly 1.0 and the sample is copied untouched, so audio that never reaches the
 *     ceiling comes out bit-identical, only delayed.
 *
 * A clamp at the ceiling stays as a last resort. It should not engage, and in testing it
 * never has.
 *
 * Real-time safe: every buffer is sized in [prepare]; [process] does not allocate.
 *
 * Bypass: process(..., apply = false) keeps everything running - delay line, detector, gain
 * state - but outputs the delayed audio untouched. The box's timing then never changes when
 * bypass is toggled (no 2 ms jump, no stale audio replayed), and switching back on starts from
 * a gain that already matches the signal.
 */
class TruePeakLimiter {
public:
    static constexpr int OVERSAMPLE = 8;
    static constexpr int TAPS = 16;
    static constexpr double LOOKAHEAD_MS = 2.0;
    static constexpr double HOLD_MS = 25.0;
    static constexpr double RELEASE_MS = 80.0;
    /**
     * Box addition (2026-10-03). When a bass cycle is not a whole number of samples (40 Hz at
     * 44.1 kHz is 1102.5), its peaks land differently on the samples from cycle to cycle and
     * the detector's estimate wobbles by a few thousandths of a dB. With an exact "<=" a
     * slightly smaller peak did not restart the hold, the gain released between peaks and
     * was pulled back at the next one: 0.14-0.75% THD on 40 Hz, 1-12 dB over, at 44.1 kHz
     * (clean at 48 kHz, where the original was measured). 0.01 dB already removes it from 30
     * to 100 Hz at both rates; 0.05 dB leaves margin, and at most holds 0.05 dB more
     * reduction than strictly needed.
     */
    static constexpr double HOLD_TOLERANCE_DB = 0.05;

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;
        m_lookahead = std::max(1, static_cast<int>(std::lround(sampleRate * LOOKAHEAD_MS * 1e-3)));
        m_holdFrames = static_cast<int>(std::lround(sampleRate * HOLD_MS * 1e-3));
        m_releaseCoeff = std::exp(-1.0 / (sampleRate * RELEASE_MS * 1e-3));
        m_holdTolerance = std::pow(10.0, HOLD_TOLERANCE_DB / 20.0);
        buildKernel();

        // Detection at window position CENTRE covers the input sample (TAPS - 1 - CENTRE)
        // frames back and the stretch just before it; the lookahead then has to cover that too.
        m_latency = m_lookahead - 1 + (TAPS - 1 - CENTRE);
        for (auto& line : m_delay) line.assign(static_cast<size_t>(m_latency) + 1, 0.0f);
        m_box.assign(static_cast<size_t>(m_lookahead), 1.0);
        // The minimum spans lookahead + 1 values: see process().
        m_minCapacity = static_cast<size_t>(m_lookahead) + 2;
        m_minIndex.assign(m_minCapacity, 0);
        m_minValue.assign(m_minCapacity, 1.0);
        reset();
    }

    void reset() {
        m_histL.fill(0.0f);
        m_histR.fill(0.0f);
        m_histPos = 0;
        for (auto& line : m_delay) std::fill(line.begin(), line.end(), 0.0f);
        m_delayPos = 0;
        std::fill(m_box.begin(), m_box.end(), 1.0);
        m_boxPos = 0;
        m_boxSum = static_cast<double>(m_lookahead);
        m_boxBelowUnity = 0;
        m_minHead = 0;
        m_minCount = 0;
        m_frame = 0;
        m_envelope = 1.0;
        m_holdLeft = 0;
        m_gainReductionDb = 0.0f;
    }

    void setCeilingDb(float ceilingDb) {
        m_ceiling = std::pow(10.0, static_cast<double>(ceilingDb) / 20.0);
    }

    void process(float* left, float* right, size_t numSamples, bool apply = true) {
        const auto ceilingF = static_cast<float>(m_ceiling);
        double blockMinGain = 1.0;

        for (size_t i = 0; i < numSamples; ++i, ++m_frame) {
            // 1-2. Detector history, written twice so the window is always contiguous.
            m_histL[m_histPos] = m_histL[m_histPos + TAPS] = left[i];
            m_histR[m_histPos] = m_histR[m_histPos + TAPS] = right[i];
            m_histPos = (m_histPos + 1) % TAPS;
            const float* wl = &m_histL[m_histPos];
            const float* wr = &m_histR[m_histPos];
            const double required = requiredGain(wl, wr);

            // 3a. Sliding minimum over the last lookahead + 1 values. One more than the
            // averaging span, so the average at any frame is bounded by the requirement of the
            // sample it is applied to *and* of the stretch just before it.
            pushMin(m_frame, required);
            const long oldest = m_frame - m_lookahead;
            while (m_minCount > 0 && m_minIndex[m_minHead] < oldest) popFrontMin();
            const double minimum = m_minValue[m_minHead];

            // 4. Hold, then release towards whatever is required now.
            // The tolerance only applies while a reduction is needed: at minimum == 1 it would
            // keep restarting the hold and the gain would never get back to exactly unity.
            if (minimum <= m_envelope ||
                (minimum < 1.0 && minimum <= m_envelope * m_holdTolerance)) {
                m_envelope = std::min(m_envelope, minimum);
                m_holdLeft = m_holdFrames;
            } else if (m_holdLeft > 0) {
                --m_holdLeft;
            } else {
                m_envelope = minimum + m_releaseCoeff * (m_envelope - minimum);
                // Within 0.0001 dB of unity: done, so pass-through becomes exact again.
                if (minimum == 1.0 && 1.0 - m_envelope < 1e-5) m_envelope = 1.0;
            }

            // 3b. Moving average over the lookahead. Exactly 1 while every value in it is 1;
            // the running sum is re-anchored then, so it cannot drift.
            const double leaving = m_box[m_boxPos];
            if (leaving < 1.0) --m_boxBelowUnity;
            m_box[m_boxPos] = m_envelope;
            if (m_envelope < 1.0) ++m_boxBelowUnity;
            m_boxPos = (m_boxPos + 1) % static_cast<size_t>(m_lookahead);
            double gain = 1.0;
            if (m_boxBelowUnity == 0) {
                m_boxSum = static_cast<double>(m_lookahead);
            } else {
                m_boxSum += m_envelope - leaving;
                gain = std::min(1.0, m_boxSum / m_lookahead);
            }

            // 5. Delayed audio out, touched only when there is reduction to apply.
            m_delay[0][m_delayPos] = left[i];
            m_delay[1][m_delayPos] = right[i];
            m_delayPos = (m_delayPos + 1) % m_delay[0].size();
            float outL = m_delay[0][m_delayPos];
            float outR = m_delay[1][m_delayPos];
            if (!apply) {
                left[i] = outL;
                right[i] = outR;
                continue;
            }
            if (gain < 1.0) {
                outL = static_cast<float>(outL * gain);
                outR = static_cast<float>(outR * gain);
                blockMinGain = std::min(blockMinGain, gain);
            }
            // Last resort only.
            left[i] = std::clamp(outL, -ceilingF, ceilingF);
            right[i] = std::clamp(outR, -ceilingF, ceilingF);
        }

        // Meter: instant attack, ~16 dB/s fall, as before.
        const auto blockGrDb = static_cast<float>(20.0 * std::log10(blockMinGain));
        if (blockGrDb < m_gainReductionDb) {
            m_gainReductionDb = blockGrDb;
        } else {
            const float seconds = static_cast<float>(numSamples) / static_cast<float>(m_sampleRate);
            m_gainReductionDb = std::min(0.0f, m_gainReductionDb + seconds * 16.0f);
        }
    }

    /** Gain reduction for the meter, dB (<= 0). */
    float getGainReductionDb() const noexcept { return m_gainReductionDb; }

    /** How far the audio is delayed, in frames. About 2.1 ms at 48 kHz. */
    int latencyFrames() const noexcept { return m_latency; }

private:
    static constexpr int CENTRE = TAPS / 2;
    static constexpr double PI = 3.14159265358979323846;

    /**
     * Phase p estimates the waveform p/OVERSAMPLE of a sample before window[CENTRE]; phase 0
     * is window[CENTRE] itself (the kernel is an exact impulse there). Across successive
     * frames every sample and every eighth of a sample between them gets checked.
     */
    void buildKernel() {
        const double half = TAPS / 2.0;
        m_kernelGain = 0.0;
        for (int p = 0; p < OVERSAMPLE; ++p) {
            double sum = 0.0;
            for (int t = 0; t < TAPS; ++t) {
                const double x = t - (CENTRE - static_cast<double>(p) / OVERSAMPLE);
                const double sinc = std::abs(x) < 1e-12 ? 1.0 : std::sin(PI * x) / (PI * x);
                const double window = std::abs(x) >= half ? 0.0 : 0.5 + 0.5 * std::cos(PI * x / half);
                m_kernel[p][t] = sinc * window;
                sum += m_kernel[p][t];
            }
            double absSum = 0.0;
            for (int t = 0; t < TAPS; ++t) {
                m_kernel[p][t] /= sum;
                absSum += std::abs(m_kernel[p][t]);
            }
            m_kernelGain = std::max(m_kernelGain, absSum);
        }
    }

    /** ceiling / true peak around the window's centre, or exactly 1 when under. */
    double requiredGain(const float* wl, const float* wr) const noexcept {
        float windowPeak = 0.0f;
        for (int t = 0; t < TAPS; ++t) {
            windowPeak = std::max(windowPeak, std::max(std::abs(wl[t]), std::abs(wr[t])));
        }
        // No interpolated value can exceed the window peak times the kernel's absolute sum.
        if (windowPeak * m_kernelGain <= m_ceiling) return 1.0;

        double peak = 0.0;
        for (int p = 0; p < OVERSAMPLE; ++p) {
            double yl = 0.0;
            double yr = 0.0;
            for (int t = 0; t < TAPS; ++t) {
                yl += m_kernel[p][t] * wl[t];
                yr += m_kernel[p][t] * wr[t];
            }
            peak = std::max(peak, std::max(std::abs(yl), std::abs(yr)));
        }
        return peak > m_ceiling ? m_ceiling / peak : 1.0;
    }

    // Monotonic queue for the sliding minimum, on fixed arrays.
    void pushMin(long index, double value) noexcept {
        while (m_minCount > 0) {
            const size_t back = (m_minHead + m_minCount - 1) % m_minCapacity;
            if (m_minValue[back] < value) break;
            --m_minCount;
        }
        const size_t slot = (m_minHead + m_minCount) % m_minCapacity;
        m_minIndex[slot] = index;
        m_minValue[slot] = value;
        ++m_minCount;
    }

    void popFrontMin() noexcept {
        m_minHead = (m_minHead + 1) % m_minCapacity;
        --m_minCount;
    }

    double m_sampleRate = 48000.0;
    double m_ceiling = 0.9660508789898133;  // -0.3 dBTP, the box's default ceiling
    int m_lookahead = 96;
    int m_holdFrames = 1200;
    int m_latency = 102;
    double m_releaseCoeff = 0.0;
    double m_holdTolerance = 1.0;

    std::array<std::array<double, TAPS>, OVERSAMPLE> m_kernel{};
    double m_kernelGain = 1.0;
    std::array<float, TAPS * 2> m_histL{};
    std::array<float, TAPS * 2> m_histR{};
    int m_histPos = 0;

    std::array<std::vector<float>, 2> m_delay;
    size_t m_delayPos = 0;

    std::vector<double> m_box;
    size_t m_boxPos = 0;
    double m_boxSum = 0.0;
    int m_boxBelowUnity = 0;

    std::vector<long> m_minIndex;
    std::vector<double> m_minValue;
    size_t m_minCapacity = 0;
    size_t m_minHead = 0;
    size_t m_minCount = 0;
    long m_frame = 0;

    double m_envelope = 1.0;
    int m_holdLeft = 0;
    float m_gainReductionDb = 0.0f;
};

}  // namespace autolevel::dsp
