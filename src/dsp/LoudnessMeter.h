#pragma once

#include "KWeightingFilter.h"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <atomic>

namespace autolevel::dsp {

constexpr float SILENCE_LUFS = -120.0f;
constexpr float ABSOLUTE_GATE_LUFS = -70.0f; // EBU R128 absolute gate Gamma_a
constexpr float RELATIVE_GATE_LU = 10.0f;     // EBU R128 relative gate Gamma_r

// EBU R128 / BS.1770-4 high-precision histogram: -100.0 LUFS to +15.0 LUFS (0.1 LUFS resolution)
constexpr float HISTOGRAM_MIN_LUFS = -100.0f;
constexpr float HISTOGRAM_MAX_LUFS = 15.0f;
constexpr float HISTOGRAM_BIN_WIDTH = 0.1f;
constexpr int HISTOGRAM_BINS = static_cast<int>((HISTOGRAM_MAX_LUFS - HISTOGRAM_MIN_LUFS) / HISTOGRAM_BIN_WIDTH) + 1; // 1151 bins

struct LoudnessReadings {
    float momentaryLUFS = SILENCE_LUFS;
    float shortTermLUFS = SILENCE_LUFS;
    float integratedLUFS = SILENCE_LUFS;
    float peakDbfs = SILENCE_LUFS;
};

class LoudnessMeter {
public:
    LoudnessMeter() = default;

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;
        m_kFilter.prepare(sampleRate);

        // 100 ms step (400 ms block with 75% overlap)
        m_stepSamples = static_cast<size_t>(sampleRate * 0.1);
        m_blockSamples = static_cast<size_t>(sampleRate * 0.4);

        // 3-second buffer for short-term (30 steps of 100ms)
        m_shortTermSteps = 30;
        m_stepEnergies.assign(m_shortTermSteps, 0.0);
        m_stepIndex = 0;

        // 4 steps in 400ms momentary block
        m_momentarySteps = 4;

        reset();
    }

    /**
     * Clears all measurement state. Must only be called from the audio thread
     * (AutoLevelEngine defers reset requests from other threads into process()).
     */
    void reset() noexcept {
        doReset();
    }

    /**
     * Maps response slider (0..1) to memory half-life in seconds.
     * Exact logarithmic mapping from Android Model.kt:
     * - 0.0: infinite (whole-track average, no decay)
     * - 1.0: 4 seconds
     */
    static inline float halfLifeForResponse(float amount) noexcept {
        constexpr float MIN_HALF_LIFE_S = 4.0f;
        constexpr float MAX_HALF_LIFE_S = 120.0f;
        if (amount < 0.01f) return 0.0f;
        return MAX_HALF_LIFE_S * std::pow(MIN_HALF_LIFE_S / MAX_HALF_LIFE_S, std::clamp(amount, 0.0f, 1.0f));
    }

    void setLevelResponse(float amount) {
        m_halfLifeSeconds = halfLifeForResponse(amount);
    }

    void process(const float* left, const float* right, size_t numSamples) noexcept {
        for (size_t i = 0; i < numSamples; ++i) {
            double l = left[i];
            double r = right[i];

            // Peak tracking with ~3 dB/s decay
            float absMax = static_cast<float>(std::max(std::abs(l), std::abs(r)));
            float peakDb = (absMax > 1e-6f) ? 20.0f * std::log10(absMax) : SILENCE_LUFS;
            if (peakDb > m_peakDbfs) {
                m_peakDbfs = peakDb;
            } else {
                m_peakDbfs -= static_cast<float>(3.0 / m_sampleRate);
                if (m_peakDbfs < SILENCE_LUFS) m_peakDbfs = SILENCE_LUFS;
            }

            // ITU-R BS.1770 K-weighting pre-filter
            double fL = 0.0;
            double fR = 0.0;
            m_kFilter.processSample(l, r, fL, fR);

            // Channel power sum
            double sampleEnergy = fL * fL + fR * fR;
            m_currentStepEnergySum += sampleEnergy;
            m_currentStepSampleCount++;

            if (m_currentStepSampleCount >= m_stepSamples) {
                finalizeStep();
            }
        }
    }

    LoudnessReadings getReadings() const noexcept {
        return {
            m_momentaryLUFS,
            m_shortTermLUFS,
            m_integratedLUFS,
            m_peakDbfs
        };
    }

    size_t getBlocksIntegrated() const {
        return m_blocksIntegrated;
    }

    float getHalfLifeSeconds() const noexcept {
        return m_halfLifeSeconds;
    }

private:
    void finalizeStep() {
        if (m_currentStepSampleCount == 0) return;

        double stepMeanPower = m_currentStepEnergySum / static_cast<double>(m_currentStepSampleCount);
        m_currentStepSampleCount = 0;
        m_currentStepEnergySum = 0.0;

        m_stepEnergies[m_stepIndex] = stepMeanPower;
        m_stepIndex = (m_stepIndex + 1) % m_shortTermSteps;

        // 400ms momentary loudness
        double momentarySum = 0.0;
        for (size_t i = 0; i < m_momentarySteps; ++i) {
            size_t idx = (m_stepIndex + m_shortTermSteps - 1 - i) % m_shortTermSteps;
            momentarySum += m_stepEnergies[idx];
        }
        double momentaryPower = momentarySum / static_cast<double>(m_momentarySteps);
        m_momentaryLUFS = toLUFS(momentaryPower);

        // 3s short-term loudness
        double shortTermSum = 0.0;
        for (double p : m_stepEnergies) shortTermSum += p;
        double shortTermPower = shortTermSum / static_cast<double>(m_shortTermSteps);
        m_shortTermLUFS = toLUFS(shortTermPower);

        // 100ms step to histogram
        updateHistogram(m_momentaryLUFS, 100);
    }

    void updateHistogram(float blockLUFS, long dtMs) {
        if (blockLUFS <= ABSOLUTE_GATE_LUFS) return;

        // Half-life decay from Android Leveler.kt
        if (m_halfLifeSeconds > 0.0f) {
            double decay = std::exp(-0.6931471805599453 * (static_cast<double>(dtMs) / 1000.0) / static_cast<double>(m_halfLifeSeconds));
            for (auto& bin : m_histogram) {
                bin *= decay;
            }
        }

        int bin = static_cast<int>(std::round((blockLUFS - HISTOGRAM_MIN_LUFS) / HISTOGRAM_BIN_WIDTH));
        bin = std::clamp(bin, 0, HISTOGRAM_BINS - 1);
        m_histogram[static_cast<size_t>(bin)] += 1.0;
        m_blocksIntegrated++;
        m_gatedMs += dtMs;

        m_integratedLUFS = computeGatedMean();
    }

    float computeGatedMean() {
        float ungated = meanAbove(ABSOLUTE_GATE_LUFS);
        if (ungated <= ABSOLUTE_GATE_LUFS) return SILENCE_LUFS;

        float relativeThreshold = ungated - RELATIVE_GATE_LU;
        return meanAbove(relativeThreshold);
    }

    float meanAbove(float thresholdLUFS) {
        double powerSum = 0.0;
        double weightSum = 0.0;

        for (int i = 0; i < HISTOGRAM_BINS; ++i) {
            double n = m_histogram[static_cast<size_t>(i)];
            if (n <= 0.0) continue;

            float centerLUFS = HISTOGRAM_MIN_LUFS + static_cast<float>(i) * HISTOGRAM_BIN_WIDTH;
            if (centerLUFS < thresholdLUFS) continue;

            powerSum += n * std::pow(10.0, (centerLUFS + 0.691) / 10.0);
            weightSum += n;
        }

        if (weightSum <= 0.0) return SILENCE_LUFS;
        return static_cast<float>(-0.691 + 10.0 * std::log10(powerSum / weightSum));
    }

    static inline float toLUFS(double meanPower) {
        if (meanPower <= 1e-12) return SILENCE_LUFS;
        return static_cast<float>(-0.691 + 10.0 * std::log10(meanPower));
    }

    double m_sampleRate = 48000.0;
    KWeightingFilter m_kFilter;

    size_t m_stepSamples = 4800;
    size_t m_blockSamples = 19200;
    size_t m_shortTermSteps = 30;
    size_t m_momentarySteps = 4;

    size_t m_currentStepSampleCount = 0;
    double m_currentStepEnergySum = 0.0;
    std::vector<double> m_stepEnergies;
    size_t m_stepIndex = 0;

    std::array<double, HISTOGRAM_BINS> m_histogram{};
    size_t m_blocksIntegrated = 0;
    long m_gatedMs = 0;
    float m_halfLifeSeconds = 0.0f; // 0 = whole track by default

    void doReset() noexcept {
        m_kFilter.reset();
        m_currentStepSampleCount = 0;
        m_currentStepEnergySum = 0.0;
        std::fill(m_stepEnergies.begin(), m_stepEnergies.end(), 0.0);
        m_stepIndex = 0;

        std::fill(m_histogram.begin(), m_histogram.end(), 0.0);
        m_blocksIntegrated = 0;
        m_gatedMs = 0;

        m_momentaryLUFS = SILENCE_LUFS;
        m_shortTermLUFS = SILENCE_LUFS;
        m_integratedLUFS = SILENCE_LUFS;
        m_peakDbfs = SILENCE_LUFS;
    }

    float m_momentaryLUFS = SILENCE_LUFS;
    float m_shortTermLUFS = SILENCE_LUFS;
    float m_integratedLUFS = SILENCE_LUFS;
    float m_peakDbfs = SILENCE_LUFS;
};

} // namespace autolevel::dsp
