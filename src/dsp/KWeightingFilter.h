#pragma once

#include <cmath>
#include <array>
#include <algorithm>

namespace autolevel::dsp {

inline constexpr double PI_CONST = 3.14159265358979323846;

/**
 * ITU-R BS.1770-4 K-weighting pre-filter.
 * Cascades:
 * 1. High-shelf filter (head-shadow / acoustic simulation)
 * 2. High-pass filter (RLB weighting, subjective bass curve)
 */
class KWeightingFilter {
public:
    KWeightingFilter() = default;

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;
        reset();
        calculateCoefficients(sampleRate);
    }

    void reset() {
        for (auto& s : m_stage1L) s = 0.0;
        for (auto& s : m_stage1R) s = 0.0;
        for (auto& s : m_stage2L) s = 0.0;
        for (auto& s : m_stage2R) s = 0.0;
    }

    /**
     * Process a single stereo sample pair in-place or into output.
     */
    inline void processSample(double inL, double inR, double& outL, double& outR) noexcept {
        // Stage 1: High shelf filter (Direct Form II Transposed)
        double s1_L = m_b1[0] * inL + m_stage1L[0];
        m_stage1L[0] = m_b1[1] * inL - m_a1[1] * s1_L + m_stage1L[1];
        m_stage1L[1] = m_b1[2] * inL - m_a1[2] * s1_L;

        double s1_R = m_b1[0] * inR + m_stage1R[0];
        m_stage1R[0] = m_b1[1] * inR - m_a1[1] * s1_R + m_stage1R[1];
        m_stage1R[1] = m_b1[2] * inR - m_a1[2] * s1_R;

        // Stage 2: High pass filter (Direct Form II Transposed)
        outL = m_b2[0] * s1_L + m_stage2L[0];
        m_stage2L[0] = m_b2[1] * s1_L - m_a2[1] * outL + m_stage2L[1];
        m_stage2L[1] = m_b2[2] * s1_L - m_a2[2] * outL;

        outR = m_b2[0] * s1_R + m_stage2R[0];
        m_stage2R[0] = m_b2[1] * s1_R - m_a2[1] * outR + m_stage2R[1];
        m_stage2R[1] = m_b2[2] * s1_R - m_a2[2] * outR;
    }

private:
    void calculateCoefficients(double fs) {
        // Stage 1: High shelf (+3.9998 dB around 1681.97 Hz)
        const double dbGain = 3.999843853973347;
        const double f0_1 = 1681.974450955533;
        const double Q1 = 0.707175236927419;
        const double K1 = std::tan(PI_CONST * f0_1 / fs);
        const double Vh = std::pow(10.0, dbGain / 20.0);
        const double Vb = std::pow(Vh, 0.4996667741545416);

        const double a0_1 = 1.0 + K1 / Q1 + K1 * K1;
        m_b1[0] = (Vh + Vb * K1 / Q1 + K1 * K1) / a0_1;
        m_b1[1] = 2.0 * (K1 * K1 - Vh) / a0_1;
        m_b1[2] = (Vh - Vb * K1 / Q1 + K1 * K1) / a0_1;
        m_a1[0] = 1.0;
        m_a1[1] = 2.0 * (K1 * K1 - 1.0) / a0_1;
        m_a1[2] = (1.0 - K1 / Q1 + K1 * K1) / a0_1;

        // Stage 2: High pass (RLB weighting, f0 ~ 38.135 Hz, Q ~ 0.5003)
        const double f0_2 = 38.13547087602444;
        const double Q2 = 0.5003270373238773;
        const double K2 = std::tan(PI_CONST * f0_2 / fs);
        const double a0_2 = 1.0 + K2 / Q2 + K2 * K2;

        m_b2[0] = 1.0 / a0_2;
        m_b2[1] = -2.0 / a0_2;
        m_b2[2] = 1.0 / a0_2;
        m_a2[0] = 1.0;
        m_a2[1] = 2.0 * (K2 * K2 - 1.0) / a0_2;
        m_a2[2] = (1.0 - K2 / Q2 + K2 * K2) / a0_2;
    }

    double m_sampleRate = 48000.0;

    // Filter coefficients [b0, b1, b2] and [a0, a1, a2]
    std::array<double, 3> m_b1{1.0, 0.0, 0.0};
    std::array<double, 3> m_a1{1.0, 0.0, 0.0};
    std::array<double, 3> m_b2{1.0, 0.0, 0.0};
    std::array<double, 3> m_a2{1.0, 0.0, 0.0};

    // State history for stereo channels
    std::array<double, 2> m_stage1L{0.0, 0.0};
    std::array<double, 2> m_stage1R{0.0, 0.0};
    std::array<double, 2> m_stage2L{0.0, 0.0};
    std::array<double, 2> m_stage2R{0.0, 0.0};
};

} // namespace autolevel::dsp
