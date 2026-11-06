#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace autolevel::dsp {

/**
 * 4th-Order Butterworth High-Pass Filter (24 dB/octave)
 *
 * Implemented as two cascaded 2nd-order sections (Biquad Direct Form II Transposed)
 * with double-precision state variables for optimal numerical precision at low frequencies (20-50 Hz).
 *
 * Pole Q values for 4th-order Butterworth:
 *   Stage 1: Q1 = 1 / (2 * cos(pi / 8))  ~= 0.54119610
 *   Stage 2: Q2 = 1 / (2 * cos(3pi / 8)) ~= 1.30656296
 *
 * Algorithmic Latency: Exactly 0 samples (causal IIR).
 */
class HighPassFilter {
public:
    HighPassFilter() = default;

    void prepare(double sampleRate) noexcept {
        m_sampleRate = (sampleRate > 1000.0) ? sampleRate : 44100.0;
        reset();
        updateCoefficients();
    }

    void reset() noexcept {
        m_s1L_z1 = m_s1L_z2 = 0.0;
        m_s1R_z1 = m_s1R_z2 = 0.0;
        m_s2L_z1 = m_s2L_z2 = 0.0;
        m_s2R_z1 = m_s2R_z2 = 0.0;
    }

    /**
     * Set cutoff frequency in Hz (20.0 to 50.0 Hz).
     * Set enabled to false (or freq < 18 Hz) to bypass.
     */
    void setCutoff(float cutoffHz, bool enabled = true) noexcept {
        m_enabled = enabled && (cutoffHz >= 18.0f);
        cutoffHz = std::clamp(cutoffHz, 20.0f, 50.0f);
        if (std::abs(cutoffHz - m_cutoffHz) > 0.1f) {
            m_cutoffHz = cutoffHz;
            updateCoefficients();
        }
    }

    bool isEnabled() const noexcept { return m_enabled; }
    float getCutoffHz() const noexcept { return m_cutoffHz; }

    void process(float* left, float* right, size_t numSamples) noexcept {
        if (!m_enabled || numSamples == 0) {
            return;
        }

        const double b0_1 = m_b0_1, b1_1 = m_b1_1, b2_1 = m_b2_1, a1_1 = m_a1_1, a2_1 = m_a2_1;
        const double b0_2 = m_b0_2, b1_2 = m_b1_2, b2_2 = m_b2_2, a1_2 = m_a1_2, a2_2 = m_a2_2;

        double s1L_1 = m_s1L_z1, s1L_2 = m_s1L_z2;
        double s1R_1 = m_s1R_z1, s1R_2 = m_s1R_z2;
        double s2L_1 = m_s2L_z1, s2L_2 = m_s2L_z2;
        double s2R_1 = m_s2R_z1, s2R_2 = m_s2R_z2;

        for (size_t i = 0; i < numSamples; ++i) {
            // Stage 1 - Left
            double inL = static_cast<double>(left[i]);
            double out1L = b0_1 * inL + s1L_1;
            s1L_1 = b1_1 * inL - a1_1 * out1L + s1L_2;
            s1L_2 = b2_1 * inL - a2_1 * out1L;

            // Stage 2 - Left
            double out2L = b0_2 * out1L + s2L_1;
            s2L_1 = b1_2 * out1L - a1_2 * out2L + s2L_2;
            s2L_2 = b2_2 * out1L - a2_2 * out2L;

            left[i] = static_cast<float>(out2L);

            // Stage 1 - Right
            double inR = static_cast<double>(right[i]);
            double out1R = b0_1 * inR + s1R_1;
            s1R_1 = b1_1 * inR - a1_1 * out1R + s1R_2;
            s1R_2 = b2_1 * inR - a2_1 * out1R;

            // Stage 2 - Right
            double out2R = b0_2 * out1R + s2R_1;
            s2R_1 = b1_2 * out1R - a1_2 * out2R + s2R_2;
            s2R_2 = b2_2 * out1R - a2_2 * out2R;

            right[i] = static_cast<float>(out2R);
        }

        m_s1L_z1 = s1L_1; m_s1L_z2 = s1L_2;
        m_s1R_z1 = s1R_1; m_s1R_z2 = s1R_2;
        m_s2L_z1 = s2L_1; m_s2L_z2 = s2L_2;
        m_s2R_z1 = s2R_1; m_s2R_z2 = s2R_2;
    }

private:
    void updateCoefficients() noexcept {
        constexpr double PI = 3.14159265358979323846;
        const double w0 = 2.0 * PI * static_cast<double>(m_cutoffHz) / m_sampleRate;
        const double cosW0 = std::cos(w0);
        const double sinW0 = std::sin(w0);

        // Q factors for 4th-order Butterworth
        constexpr double Q1 = 0.541196100146197; // 1 / (2 * cos(pi/8))
        constexpr double Q2 = 1.306562964876376; // 1 / (2 * cos(3*pi/8))

        auto calcBiquadHP = [&](double Q, double& b0, double& b1, double& b2, double& a1, double& a2) {
            double alpha = sinW0 / (2.0 * Q);
            double a0 = 1.0 + alpha;
            double num = (1.0 + cosW0) / 2.0;
            b0 = num / a0;
            b1 = -(1.0 + cosW0) / a0;
            b2 = num / a0;
            a1 = (-2.0 * cosW0) / a0;
            a2 = (1.0 - alpha) / a0;
        };

        calcBiquadHP(Q1, m_b0_1, m_b1_1, m_b2_1, m_a1_1, m_a2_1);
        calcBiquadHP(Q2, m_b0_2, m_b1_2, m_b2_2, m_a1_2, m_a2_2);
    }

    double m_sampleRate = 44100.0;
    float m_cutoffHz = 30.0f;
    bool m_enabled = true;

    // Stage 1 coefficients
    double m_b0_1 = 1.0, m_b1_1 = 0.0, m_b2_1 = 0.0, m_a1_1 = 0.0, m_a2_1 = 0.0;
    // Stage 2 coefficients
    double m_b0_2 = 1.0, m_b1_2 = 0.0, m_b2_2 = 0.0, m_a1_2 = 0.0, m_a2_2 = 0.0;

    // Transposed Direct Form II state variables (double precision)
    double m_s1L_z1 = 0.0, m_s1L_z2 = 0.0;
    double m_s1R_z1 = 0.0, m_s1R_z2 = 0.0;
    double m_s2L_z1 = 0.0, m_s2L_z2 = 0.0;
    double m_s2R_z1 = 0.0, m_s2R_z2 = 0.0;
};

} // namespace autolevel::dsp
