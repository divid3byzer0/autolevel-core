#pragma once

#include "LoudnessMeter.h"
#include "Leveler.h"
#include "DynamicAirLift.h"
#include "PreMbcEQ.h"
#include "HighPassFilter.h"
#include "MultibandCompressor.h"
#include "TruePeakLimiter.h"
#include <atomic>
#include <array>

namespace autolevel::dsp {

// Backward-compatible type alias for existing codebase & tests
using AirWeight = AirLiftMode;

struct EngineParameters {
    float targetLUFS = -14.0f;
    float maxBoostDb = 12.0f;
    float maxCutDb = 12.0f;
    float levelResponse = 0.85f; // 0..1 slider, maps to memory half-life (LoudnessMeter side)
    LevelerSpeed slewSpeed = LevelerSpeed::NORMAL; // Steady-state gain slew rate (Leveler side)
    bool freezeBreakdowns = true;
    float breakdownThresholdLU = 7.0f;
    float toneSlopeDbPerOctave = Bands::DEFAULT_TONE_SLOPE; // -3.0 to 0.0 dB/oct, default flat (0)
    std::array<float, Bands::COUNT> customOffsetsDb{}; // per-band MBC contour, dB, default flat
    float mbcAttackMs = MBCParams::DEFAULT_ATTACK_MS;   // bands 1-5; Sub runs at twice
    float mbcReleaseMs = MBCParams::DEFAULT_RELEASE_MS;
    float mbcDetectorRms = 0.0f;  // 0 = peak .. 1 = RMS
    bool mbcAutoRelease = false;
    AirLiftMode airLift = AirLiftMode::OFF;
    // Backward compatibility alias
    AirWeight airWeight = AirWeight::OFF;
    std::array<float, Bands::COUNT> preEqGainsDb{}; // Pre-MBC 6-band EQ, dB, default flat
    float compressionAmount = 0.5f; // 0..1 slider
    float postMbcGainDb = 0.0f;     // -24 to +24 dB
    float hpfCutoffHz = 30.0f;      // 20 to 50 Hz low-cut filter
    bool hpfEnabled = true;
    float ceilingDb = -0.3f;        // -0.3 dBFS default master ceiling
    bool bypass = false;
};

struct EngineVisualState {
    LoudnessReadings loudness;
    float appliedGainDb = 0.0f;
    float targetGainDb = 0.0f;
    bool isFrozen = false;
    std::array<float, Bands::COUNT> mbcGainReductionsDb{};
    std::array<float, Bands::COUNT> mbcThresholdsDb{};
    std::array<float, Bands::COUNT> customOffsetsDb{};
    float limiterGainReductionDb = 0.0f;
    float outputPeakDbL = -60.0f;
    float outputPeakDbR = -60.0f;
    float activeHalfLifeSeconds = 0.0f;
    LevelerSpeed activeSlewSpeed = LevelerSpeed::NORMAL;
    AirLiftMode activeAirLift = AirLiftMode::OFF;
    float airLiftDb = 0.0f;
    // Compatibility field for UI meters
    AirWeight activeAirWeight = AirWeight::OFF;
    float airInjectedLevel = 0.0f;
    std::array<float, Bands::COUNT> preEqGainsDb{};
    float activeToneSlope = -2.0f;
    float postMbcGainDb = 0.0f;
    float hpfCutoffHz = 30.0f;
    bool hpfEnabled = true;
};

class AutoLevelEngine {
public:
    AutoLevelEngine() = default;

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;
        m_loudnessMeter.prepare(sampleRate);
        m_leveler.prepare(sampleRate);
        m_airLift.prepare(sampleRate);
        m_preEq.prepare(sampleRate);
        m_mbc.prepare(sampleRate);
        m_hpf.prepare(sampleRate);
        m_limiter.prepare(sampleRate);
        resetImmediate();
    }

    /**
     * Request a reset of all measurement and filter state.
     *
     * Safe to call from any thread (the Web server calls it from POST /api/reset).
     * The work itself is deferred to the next process() call so that filter state
     * is only ever touched by the audio thread - doing it inline from the caller
     * raced with the audio callback across every stage of the chain.
     */
    void reset() {
        m_resetRequested.store(true, std::memory_order_release);
    }

    /** How far the chain delays the audio, in frames: the limiter's lookahead (~2.1 ms). */
    int latencyFrames() const noexcept { return m_limiter.latencyFrames(); }

    /**
     * Perform the reset synchronously. Only safe while the audio thread is not
     * running (construction / prepare()).
     */
    void resetImmediate() {
        doReset();
    }

    /**
     * Exact chain: AGC -> Air Lift -> Pre-MBC EQ -> Multiband Compressor (MBC) -> Post-Gain -> HPF (Low Cut) -> Limiter
     */
    void process(float* left, float* right, size_t numSamples, const EngineParameters& params) {
        // Serviced before the bypass early-out so a reset requested while bypassed
        // is not left pending indefinitely.
        if (m_resetRequested.exchange(false, std::memory_order_acquire)) {
            doReset();
        }

        if (numSamples == 0) return;
        if (params.bypass) {
            // Untouched, but through the limiter's delay line, so toggling bypass never shifts
            // the timing (see TruePeakLimiter).
            m_limiter.setCeilingDb(params.ceilingDb);
            m_limiter.process(left, right, numSamples, false);
            return;
        }

        // 1. Measure incoming raw perceived loudness (ITU-R BS.1770-4 K-weighting + dual-gating)
        m_loudnessMeter.setLevelResponse(params.levelResponse);
        m_loudnessMeter.process(left, right, numSamples);
        LoudnessReadings readings = m_loudnessMeter.getReadings();

        // 2. Compute Leveler / AGC target gain & slew
        LevelerParams levelerParams;
        levelerParams.targetLUFS = params.targetLUFS;
        levelerParams.maxBoostDb = params.maxBoostDb;
        levelerParams.maxCutDb = params.maxCutDb;
        levelerParams.freezeBreakdowns = params.freezeBreakdowns;
        levelerParams.breakdownThresholdLU = params.breakdownThresholdLU;
        levelerParams.speed = params.slewSpeed;

        float dtSeconds = static_cast<float>(numSamples) / static_cast<float>(m_sampleRate);
        m_leveler.update(levelerParams, readings, m_loudnessMeter.getBlocksIntegrated(), dtSeconds);

        // 3. Stage 1: AGC Makeup Gain applied to audio
        m_leveler.processBlock(left, right, numSamples);

        // 3.5. Stage 1.5: Dynamic Air Lift (> 9.5 kHz, 100% distortion-free)
        AirLiftMode effAir = (params.airLift != AirLiftMode::OFF) ? params.airLift : params.airWeight;
        m_airLift.process(left, right, numSamples, effAir);

        // 3.6. Stage 1.6: Pre-MBC EQ (manual 6-band tone shaping, ahead of the compressor)
        m_preEq.process(left, right, numSamples, params.preEqGainsDb);

        // 4. Stage 2: 6-band Multiband Dynamic Tone Shaper (thresholds linked to tone curve & profile)
        MBCParams mbcParams;
        mbcParams.enabled = (params.compressionAmount >= Bands::MIN_COMPRESSION);
        mbcParams.compressionAmount = params.compressionAmount;
        mbcParams.toneSlopeDbPerOctave = params.toneSlopeDbPerOctave;
        mbcParams.customOffsetsDb = params.customOffsetsDb;
        // Target LUFS minus 18 dB: -30 dBFS at a -12 LUFS target, -32 at the default -14
        // (set by the owner on 2026-10-03; it was minus 15). Bands::MBC_THRESHOLD_DB (-24,
        // "exactly matching Android Shaper.kt") is no longer reached at any Target LUFS the
        // UI allows (-20 to -6).
        mbcParams.baseThresholdDb = params.targetLUFS - Bands::MBC_THRESHOLD_BELOW_TARGET_DB;
        mbcParams.attackMs = params.mbcAttackMs;
        mbcParams.releaseMs = params.mbcReleaseMs;
        mbcParams.detectorRms = params.mbcDetectorRms;
        mbcParams.autoRelease = params.mbcAutoRelease;
        m_mbc.process(left, right, numSamples, mbcParams);

        // 5. Stage 3: Post-MBC Gain stage (makeup/trim before safety limiter)
        if (std::abs(params.postMbcGainDb) > 0.01f) {
            float postGainLin = std::pow(10.0f, params.postMbcGainDb / 20.0f);
            for (size_t s = 0; s < numSamples; ++s) {
                left[s] *= postGainLin;
                right[s] *= postGainLin;
            }
        }

        // 5.5. Stage 3.5: 4th-Order Butterworth High-Pass Filter (Low Cut 20-50 Hz)
        // Cleanly strips inaudible subsonic rumble right before the safety limiter
        m_hpf.setCutoff(params.hpfCutoffHz, params.hpfEnabled);
        m_hpf.process(left, right, numSamples);

        // 6. Stage 4: lookahead true-peak limiter (2 ms lookahead, 25 ms hold, 80 ms release)
        m_limiter.setCeilingDb(params.ceilingDb);
        m_limiter.process(left, right, numSamples);

        // 6.5. Track master output peak (linear with decay for smooth visual metering)
        float blockPeakL = 0.0f;
        float blockPeakR = 0.0f;
        for (size_t s = 0; s < numSamples; ++s) {
            blockPeakL = std::max(blockPeakL, std::abs(left[s]));
            blockPeakR = std::max(blockPeakR, std::abs(right[s]));
        }
        float decayLin = dtSeconds * 2.0f; // ~20 dB/sec decay
        m_outPeakLinL = (blockPeakL > m_outPeakLinL) ? blockPeakL : std::max(0.0f, m_outPeakLinL - decayLin);
        m_outPeakLinR = (blockPeakR > m_outPeakLinR) ? blockPeakR : std::max(0.0f, m_outPeakLinR - decayLin);

        // 7. Cache state for UI into a buffer no reader can currently be holding.
        // With only two buffers the writer reclaims the published buffer on the
        // very next block, so a reader preempted mid-copy could be overwritten.
        // Three buffers plus a cursor that never targets the published one give
        // the reader a full extra block of grace.
        size_t published = m_activeVisualIndex.load(std::memory_order_relaxed);
        size_t writeIdx = m_nextVisualIndex;
        if (writeIdx == published) writeIdx = (writeIdx + 1) % VISUAL_BUFFERS;
        auto& vs = m_visualStates[writeIdx];

        vs.loudness = readings;
        vs.appliedGainDb = m_leveler.getCurrentGainDb();
        vs.targetGainDb = m_leveler.getTargetGainDb();
        vs.isFrozen = params.freezeBreakdowns && m_leveler.isFrozen();
        vs.mbcGainReductionsDb = m_mbc.getGainReductionsDb();
        vs.customOffsetsDb = params.customOffsetsDb;
        vs.mbcThresholdsDb = Bands::thresholdsFor(
            true, params.toneSlopeDbPerOctave, params.customOffsetsDb, mbcParams.baseThresholdDb
        );
        vs.limiterGainReductionDb = m_limiter.getGainReductionDb();
        vs.outputPeakDbL = (m_outPeakLinL > 1e-4f) ? (20.0f * std::log10(m_outPeakLinL)) : -60.0f;
        vs.outputPeakDbR = (m_outPeakLinR > 1e-4f) ? (20.0f * std::log10(m_outPeakLinR)) : -60.0f;
        vs.activeHalfLifeSeconds = m_loudnessMeter.getHalfLifeSeconds();
        vs.activeSlewSpeed = params.slewSpeed;
        vs.activeAirLift = effAir;
        vs.airLiftDb = m_airLift.getLiftDb();
        // UI compatibility
        vs.activeAirWeight = effAir;
        vs.airInjectedLevel = m_airLift.getLiftDb() / 6.5f * 0.25f; // scale to 0..0.25 for LED rack
        vs.preEqGainsDb = params.preEqGainsDb;
        vs.activeToneSlope = params.toneSlopeDbPerOctave;
        vs.postMbcGainDb = params.postMbcGainDb;
        vs.hpfCutoffHz = m_hpf.getCutoffHz();
        vs.hpfEnabled = m_hpf.isEnabled();

        // Atomically publish new visual state buffer
        m_activeVisualIndex.store(writeIdx, std::memory_order_release);
        m_nextVisualIndex = (writeIdx + 1) % VISUAL_BUFFERS;
    }

    EngineVisualState getVisualState() const noexcept {
        size_t readIdx = m_activeVisualIndex.load(std::memory_order_acquire);
        return m_visualStates[readIdx];
    }

private:
    void doReset() {
        m_loudnessMeter.reset();
        m_leveler.reset();
        m_airLift.reset();
        m_preEq.reset();
        m_mbc.reset();
        m_hpf.reset();
        // Not the limiter: it holds ~2 ms of audio in its delay line, and clearing that on a
        // loudness reset mid-set would cut a hole in the music. prepare() still resets it.
        m_outPeakLinL = 0.0f;
        m_outPeakLinR = 0.0f;
    }

    std::atomic<bool> m_resetRequested{false};
    double m_sampleRate = 48000.0;
    LoudnessMeter m_loudnessMeter;
    Leveler m_leveler;
    DynamicAirLift m_airLift;
    PreMbcEQ m_preEq;
    MultibandCompressor m_mbc;
    HighPassFilter m_hpf;
    TruePeakLimiter m_limiter;

    float m_outPeakLinL = 0.0f;
    float m_outPeakLinR = 0.0f;

    static constexpr size_t VISUAL_BUFFERS = 3;
    std::array<EngineVisualState, VISUAL_BUFFERS> m_visualStates{};
    std::atomic<size_t> m_activeVisualIndex{0};
    size_t m_nextVisualIndex{1};   // audio thread only
};

} // namespace autolevel::dsp
