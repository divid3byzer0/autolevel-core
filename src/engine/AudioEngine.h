#pragma once

#include "../dsp/AutoLevelEngine.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <regex>
#include <optional>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace autolevel {

inline void setDenormalsFlushToZero() noexcept {
#if defined(__aarch64__)
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24); // Bit 24: FZ (Flush-to-zero mode)
    asm volatile("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    _mm_setcsr(_mm_getcsr() | 0x8040); // FTZ (bit 15) and DAZ (bit 6)
    #endif
#endif
}

inline bool isFiniteSample(float f) noexcept {
    static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x7f800000U) != 0x7f800000U;
}

class AudioEngine {
public:
    AudioEngine() {
        // Initialize default parameters
        m_targetLufs.store(-14.0f);
        m_maxBoost.store(12.0f);
        m_maxCut.store(12.0f);
        m_levelResponse.store(0.85f);
        m_slewSpeed.store(1);     // Normal (0.75 dB/s up / 1.5 dB/s down)
        m_compressionAmount.store(0.50f);
        m_toneSlope.store(dsp::Bands::DEFAULT_TONE_SLOPE); // 0.0 - flat by default
        m_mbcAttackMs.store(dsp::MBCParams::DEFAULT_ATTACK_MS);
        m_mbcReleaseMs.store(dsp::MBCParams::DEFAULT_RELEASE_MS);
        m_mbcDetectorRms.store(0.0f); // Peak
        m_mbcAutoRelease.store(false);
        m_airExciter.store(0);    // Off
        m_postMbcGain.store(0.0f);
        m_hpfFreq.store(30.0f);   // 30 Hz
        m_ceilingDb.store(-0.3f);
        m_freezeBreakdowns.store(true);
        m_breakdownThresholdLU.store(7.0f);
        m_bypass.store(false);
        for (size_t i = 0; i < 6; ++i) {
            m_customOffsets[i].store(0.0f); // flat by default (Modern Mix contour removed 2026-09-11)
            m_preEqGains[i].store(0.0f);
        }
    }

    void prepare(double sampleRate) {
        m_sampleRate = sampleRate;
        m_engine.prepare(sampleRate);
    }

    void reset() {
        m_engine.reset();
    }

    // Audio processing callback - thread-safe, called from high-priority audio thread
    void process(const float* inL, const float* inR, float* outL, float* outR, size_t numSamples) {
        if (numSamples == 0) return;

        // Ensure Denormals-Are-Zero and Flush-to-Zero on audio thread
        thread_local bool s_denormalsSet = false;
        if (!s_denormalsSet) {
            setDenormalsFlushToZero();
            s_denormalsSet = true;
        }

        // Copy and sanitize input audio: guarantee clean finite samples (discard any NaN / Inf)
        for (size_t s = 0; s < numSamples; ++s) {
            float l = inL[s];
            float r = inR[s];
            outL[s] = isFiniteSample(l) ? std::clamp(l, -8.0f, 8.0f) : 0.0f;
            outR[s] = isFiniteSample(r) ? std::clamp(r, -8.0f, 8.0f) : 0.0f;
        }

        // Read active parameters atomically
        dsp::EngineParameters params;
        params.targetLUFS = m_targetLufs.load(std::memory_order_relaxed);
        params.maxBoostDb = m_maxBoost.load(std::memory_order_relaxed);
        params.maxCutDb = m_maxCut.load(std::memory_order_relaxed);
        params.levelResponse = m_levelResponse.load(std::memory_order_relaxed);
        int slew = m_slewSpeed.load(std::memory_order_relaxed);
        params.slewSpeed = (slew == 0) ? dsp::LevelerSpeed::SLOW
                          : (slew == 2) ? dsp::LevelerSpeed::FAST
                                        : dsp::LevelerSpeed::NORMAL;
        params.compressionAmount = m_compressionAmount.load(std::memory_order_relaxed);
        params.toneSlopeDbPerOctave = m_toneSlope.load(std::memory_order_relaxed);
        for (size_t b = 0; b < 6; ++b) {
            params.customOffsetsDb[b] = m_customOffsets[b].load(std::memory_order_relaxed);
            params.preEqGainsDb[b] = m_preEqGains[b].load(std::memory_order_relaxed);
        }

        params.mbcAttackMs = m_mbcAttackMs.load(std::memory_order_relaxed);
        params.mbcReleaseMs = m_mbcReleaseMs.load(std::memory_order_relaxed);
        params.mbcDetectorRms = m_mbcDetectorRms.load(std::memory_order_relaxed);
        params.mbcAutoRelease = m_mbcAutoRelease.load(std::memory_order_relaxed);

        int air = m_airExciter.load(std::memory_order_relaxed);
        params.airWeight = (air == 1) ? dsp::AirWeight::LOW
                         : (air == 2) ? dsp::AirWeight::MED
                         : (air == 3) ? dsp::AirWeight::HIGH
                                      : dsp::AirWeight::OFF;

        params.postMbcGainDb = m_postMbcGain.load(std::memory_order_relaxed);
        params.hpfCutoffHz = m_hpfFreq.load(std::memory_order_relaxed);
        params.hpfEnabled = (params.hpfCutoffHz >= 20.0f);
        params.ceilingDb = m_ceilingDb.load(std::memory_order_relaxed);
        params.freezeBreakdowns = m_freezeBreakdowns.load(std::memory_order_relaxed);
        params.breakdownThresholdLU = m_breakdownThresholdLU.load(std::memory_order_relaxed);
        params.bypass = m_bypass.load(std::memory_order_relaxed);

        m_engine.process(outL, outR, numSamples, params);
    }

    dsp::EngineVisualState getVisualState() const noexcept {
        return m_engine.getVisualState();
    }

    void setParam(const std::string& id, float value) {
        if (id == "target_lufs") m_targetLufs.store(std::clamp(value, -20.0f, -6.0f));
        else if (id == "max_boost") m_maxBoost.store(std::clamp(value, 0.0f, 18.0f));
        else if (id == "max_cut") m_maxCut.store(std::clamp(value, 0.0f, 18.0f));
        else if (id == "level_response") m_levelResponse.store(std::clamp(value, 0.0f, 1.0f));
        else if (id == "slew_speed") m_slewSpeed.store(static_cast<int>(value));
        else if (id == "compression_amount") m_compressionAmount.store(std::clamp(value, 0.0f, 1.0f));
        else if (id == "tone_slope") m_toneSlope.store(std::clamp(value, -3.0f, 0.0f));
        else if (id == "mbc_attack_ms") m_mbcAttackMs.store(std::clamp(value,
            dsp::MBCParams::MIN_ATTACK_MS, dsp::MBCParams::MAX_ATTACK_MS));
        else if (id == "mbc_release_ms") m_mbcReleaseMs.store(std::clamp(value,
            dsp::MBCParams::MIN_RELEASE_MS, dsp::MBCParams::MAX_RELEASE_MS));
        else if (id == "mbc_detector_rms") m_mbcDetectorRms.store(std::clamp(value, 0.0f, 1.0f));
        else if (id == "mbc_auto_release") m_mbcAutoRelease.store(value > 0.5f);
        else if (id == "air_exciter") m_airExciter.store(static_cast<int>(value));
        else if (id == "post_mbc_gain") m_postMbcGain.store(std::clamp(value, -24.0f, 24.0f));
        else if (id == "hpf_freq") {
            if (value <= 0.0f) m_hpfFreq.store(0.0f);
            else m_hpfFreq.store(std::clamp(value, 20.0f, 50.0f));
        }
        else if (id == "ceiling_db") m_ceilingDb.store(std::clamp(value, -3.0f, 0.0f));
        else if (id == "freeze_breakdowns") m_freezeBreakdowns.store(value > 0.5f);
        else if (id == "breakdown_threshold_lu") m_breakdownThresholdLU.store(std::clamp(value, 3.0f, 15.0f));
        else if (id == "bypass") m_bypass.store(value > 0.5f);
        else if (id.rfind("custom_band_", 0) == 0) {
            try {
                int b = std::stoi(id.substr(12));
                if (b >= 0 && b < 6) m_customOffsets[b].store(std::clamp(value, -12.0f, 12.0f));
            } catch (...) {}
        }
        else if (id.rfind("pre_eq_band_", 0) == 0) {
            try {
                int b = std::stoi(id.substr(12));
                if (b >= 0 && b < 6) m_preEqGains[b].store(std::clamp(value, -12.0f, 12.0f));
            } catch (...) {}
        }
    }

    float getParam(const std::string& id) const {
        if (id == "target_lufs") return m_targetLufs.load();
        if (id == "max_boost") return m_maxBoost.load();
        if (id == "max_cut") return m_maxCut.load();
        if (id == "level_response") return m_levelResponse.load();
        if (id == "slew_speed") return static_cast<float>(m_slewSpeed.load());
        if (id == "compression_amount") return m_compressionAmount.load();
        if (id == "tone_slope") return m_toneSlope.load();
        if (id == "mbc_attack_ms") return m_mbcAttackMs.load();
        if (id == "mbc_release_ms") return m_mbcReleaseMs.load();
        if (id == "mbc_detector_rms") return m_mbcDetectorRms.load();
        if (id == "mbc_auto_release") return m_mbcAutoRelease.load() ? 1.0f : 0.0f;
        if (id == "air_exciter") return static_cast<float>(m_airExciter.load());
        if (id == "post_mbc_gain") return m_postMbcGain.load();
        if (id == "hpf_freq") return m_hpfFreq.load();
        if (id == "ceiling_db") return m_ceilingDb.load();
        if (id == "freeze_breakdowns") return m_freezeBreakdowns.load() ? 1.0f : 0.0f;
        if (id == "breakdown_threshold_lu") return m_breakdownThresholdLU.load();
        if (id == "bypass") return m_bypass.load() ? 1.0f : 0.0f;
        if (id.rfind("custom_band_", 0) == 0) {
            try {
                int b = std::stoi(id.substr(12));
                if (b >= 0 && b < 6) return m_customOffsets[b].load();
            } catch (...) {}
        }
        if (id.rfind("pre_eq_band_", 0) == 0) {
            try {
                int b = std::stoi(id.substr(12));
                if (b >= 0 && b < 6) return m_preEqGains[b].load();
            } catch (...) {}
        }
        return 0.0f;
    }

    void toggleBypass() {
        bool current = m_bypass.load();
        m_bypass.store(!current);
    }

    /**
     * Apply a full parameter snapshot - the same flat shape getStateJson() produces (a saved
     * preset is literally that string, written to disk verbatim). Every recognized field is
     * dispatched through setParam(), so the same clamping/validation as a live /api/param call
     * applies here too - a preset can never push a value outside its normal range. Unrecognized
     * or missing fields are silently skipped (whatever the engine's current value is for that
     * field stays as-is), so a preset saved before a parameter existed still loads cleanly.
     *
     * This is presets' only entry point into the engine - it is never called automatically
     * (not at startup, not on a timer); only an explicit POST /api/presets/load reaches it.
     */
    void loadStateFromJson(const std::string& json) {
        static const char* kScalarIds[] = {
            "target_lufs", "max_boost", "max_cut", "level_response", "slew_speed",
            "compression_amount", "tone_slope", "mbc_attack_ms", "mbc_release_ms",
            "mbc_detector_rms", "air_exciter", "post_mbc_gain", "hpf_freq", "ceiling_db",
            "breakdown_threshold_lu"
        };
        for (const char* id : kScalarIds) {
            std::regex re(std::string("\"") + id + "\"\\s*:\\s*(-?[0-9]*\\.?[0-9]+)");
            std::smatch m;
            if (std::regex_search(json, m, re)) {
                setParam(id, std::stof(m[1].str()));
            }
        }

        // Presets saved before 2026-10-03 carry "mbc_speed" (0/1/2 = Slow/Normal/Fast) instead
        // of attack/release times. Translate it to the times that preset actually ran at, and
        // to the detector and release it ran with: peak, no auto release - they did not exist.
        if (json.find("\"mbc_attack_ms\"") == std::string::npos) {
            std::smatch m;
            if (std::regex_search(json, m, std::regex("\"mbc_speed\"\\s*:\\s*(-?[0-9]+)"))) {
                const int speed = std::stoi(m[1].str());
                setParam("mbc_attack_ms", speed == 0 ? 30.0f : speed == 2 ? 7.5f : 15.0f);
                setParam("mbc_release_ms", speed == 0 ? 400.0f : speed == 2 ? 100.0f : 200.0f);
                setParam("mbc_detector_rms", 0.0f);
                setParam("mbc_auto_release", 0.0f);
            }
        }

        // "mbc_auto_makeup" (removed 2026-09-28) still appears in older presets and is
        // skipped like any other unrecognized field.
        static const char* kBoolIds[] = { "freeze_breakdowns", "bypass", "mbc_auto_release" };
        for (const char* id : kBoolIds) {
            std::regex re(std::string("\"") + id + "\"\\s*:\\s*(true|false)");
            std::smatch m;
            if (std::regex_search(json, m, re)) {
                setParam(id, (m[1].str() == "true") ? 1.0f : 0.0f);
            }
        }

        auto applyArray6 = [&](const char* key, const char* paramPrefix) {
            std::regex re(std::string("\"") + key +
                "\"\\s*:\\s*\\[\\s*(-?[0-9.]+)\\s*,\\s*(-?[0-9.]+)\\s*,\\s*(-?[0-9.]+)\\s*,"
                "\\s*(-?[0-9.]+)\\s*,\\s*(-?[0-9.]+)\\s*,\\s*(-?[0-9.]+)\\s*\\]");
            std::smatch m;
            if (std::regex_search(json, m, re)) {
                for (int b = 0; b < 6; ++b) {
                    setParam(paramPrefix + std::to_string(b), std::stof(m[b + 1].str()));
                }
            }
        };
        applyArray6("custom_offsets", "custom_band_");
        applyArray6("pre_eq_gains", "pre_eq_band_");
    }

    void setDeviceInfo(const std::string& info) {
        m_deviceInfo = info;
    }

    std::string getDeviceInfo() const {
        return m_deviceInfo;
    }

    // System-health telemetry, written by whichever audio backend is active (JACK, ALSA, or the
    // dummy simulator all hold a reference to this same AudioEngine, same pattern as
    // setDeviceInfo above) and read by WebServer for the header display. An xrun means the audio
    // thread missed its deadline - a real, audible glitch - so the count is a running total since
    // start, never reset, deliberately visible rather than swallowed the way it was before this.
    void incrementXrunCount() noexcept { m_xrunCount.fetch_add(1, std::memory_order_relaxed); }
    uint64_t getXrunCount() const noexcept { return m_xrunCount.load(std::memory_order_relaxed); }

    // How much of each audio buffer's time budget the DSP chain is actually using, 0-100.
    // JACK reports this itself (jack_cpu_load); ALSA/dummy backends time their own process()
    // call against the buffer's real-time duration. Not the process's overall CPU% - that stays
    // near-flat and low value on this workload - this is the number that predicts xruns.
    void setDspLoadPercent(float pct) noexcept { m_dspLoadPercent.store(pct, std::memory_order_relaxed); }
    float getDspLoadPercent() const noexcept { return m_dspLoadPercent.load(std::memory_order_relaxed); }

    std::string getStateJson() const {
        char buf[1400];
        snprintf(buf, sizeof(buf),
            "{"
            "\"target_lufs\":%.1f,"
            "\"max_boost\":%.1f,"
            "\"max_cut\":%.1f,"
            "\"level_response\":%.2f,"
            "\"slew_speed\":%d,"
            "\"compression_amount\":%.2f,"
            "\"tone_slope\":%.1f,"
            "\"custom_offsets\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
            "\"pre_eq_gains\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
            "\"mbc_attack_ms\":%.1f,"
            "\"mbc_release_ms\":%.0f,"
            "\"mbc_detector_rms\":%.2f,"
            "\"mbc_auto_release\":%s,"
            "\"air_exciter\":%d,"
            "\"ceiling_db\":%.1f,"
            "\"post_mbc_gain\":%.1f,"
            "\"hpf_freq\":%.1f,"
            "\"freeze_breakdowns\":%s,"
            "\"breakdown_threshold_lu\":%.1f,"
            "\"bypass\":%s"
            "}",
            m_targetLufs.load(),
            m_maxBoost.load(),
            m_maxCut.load(),
            m_levelResponse.load(),
            m_slewSpeed.load(),
            m_compressionAmount.load(),
            m_toneSlope.load(),
            m_customOffsets[0].load(), m_customOffsets[1].load(), m_customOffsets[2].load(),
            m_customOffsets[3].load(), m_customOffsets[4].load(), m_customOffsets[5].load(),
            m_preEqGains[0].load(), m_preEqGains[1].load(), m_preEqGains[2].load(),
            m_preEqGains[3].load(), m_preEqGains[4].load(), m_preEqGains[5].load(),
            m_mbcAttackMs.load(),
            m_mbcReleaseMs.load(),
            m_mbcDetectorRms.load(),
            m_mbcAutoRelease.load() ? "true" : "false",
            m_airExciter.load(),
            m_ceilingDb.load(),
            m_postMbcGain.load(),
            m_hpfFreq.load(),
            m_freezeBreakdowns.load() ? "true" : "false",
            m_breakdownThresholdLU.load(),
            m_bypass.load() ? "true" : "false"
        );
        return std::string(buf);
    }

    std::string getVisualStateJson() const {
        auto vs = m_engine.getVisualState();
        char buf[1250];
        snprintf(buf, sizeof(buf),
            "{"
            "\"momentaryLufs\":%.1f,"
            "\"integratedLufs\":%.1f,"
            "\"appliedGainDb\":%.2f,"
            "\"limiterGrDb\":%.2f,"
            "\"peakL\":%.1f,"
            "\"peakR\":%.1f,"
            "\"isFrozen\":%s,"
            "\"mbcGrDb\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
            "\"customOffsets\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
            "\"preEqGains\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
            "\"airLiftDb\":%.1f,"
            "\"deviceInfo\":\"%s\","
            "\"xrunCount\":%llu,"
            "\"dspLoadPct\":%.1f"
            "}",
            vs.loudness.momentaryLUFS,
            vs.loudness.integratedLUFS,
            vs.appliedGainDb,
            vs.limiterGainReductionDb,
            vs.outputPeakDbL,
            vs.outputPeakDbR,
            vs.isFrozen ? "true" : "false",
            vs.mbcGainReductionsDb[0], vs.mbcGainReductionsDb[1], vs.mbcGainReductionsDb[2],
            vs.mbcGainReductionsDb[3], vs.mbcGainReductionsDb[4], vs.mbcGainReductionsDb[5],
            vs.customOffsetsDb[0], vs.customOffsetsDb[1], vs.customOffsetsDb[2],
            vs.customOffsetsDb[3], vs.customOffsetsDb[4], vs.customOffsetsDb[5],
            vs.preEqGainsDb[0], vs.preEqGainsDb[1], vs.preEqGainsDb[2],
            vs.preEqGainsDb[3], vs.preEqGainsDb[4], vs.preEqGainsDb[5],
            vs.airLiftDb,
            m_deviceInfo.c_str(),
            static_cast<unsigned long long>(m_xrunCount.load(std::memory_order_relaxed)),
            m_dspLoadPercent.load(std::memory_order_relaxed)
        );
        return std::string(buf);
    }

private:
    dsp::AutoLevelEngine m_engine;
    double m_sampleRate = 48000.0;
    std::string m_deviceInfo = "JACK • 48kHz";
    std::atomic<uint64_t> m_xrunCount{0};
    std::atomic<float> m_dspLoadPercent{0.0f};

    // Parameters
    std::atomic<float> m_targetLufs{-14.0f};
    std::atomic<float> m_maxBoost{12.0f};
    std::atomic<float> m_maxCut{12.0f};
    std::atomic<float> m_levelResponse{0.85f};
    std::atomic<int> m_slewSpeed{1}; // 0=Slow, 1=Normal, 2=Fast
    std::atomic<float> m_compressionAmount{0.50f};
    std::atomic<float> m_toneSlope{0.0f};
    std::atomic<float> m_mbcAttackMs{dsp::MBCParams::DEFAULT_ATTACK_MS};
    std::atomic<float> m_mbcReleaseMs{dsp::MBCParams::DEFAULT_RELEASE_MS};
    std::atomic<float> m_mbcDetectorRms{0.0f};
    std::atomic<bool> m_mbcAutoRelease{false};
    std::atomic<int> m_airExciter{0};
    std::atomic<float> m_postMbcGain{0.0f};
    std::atomic<float> m_hpfFreq{30.0f};
    std::atomic<float> m_ceilingDb{-0.3f};
    std::atomic<bool> m_freezeBreakdowns{true};
    std::atomic<float> m_breakdownThresholdLU{7.0f};
    std::atomic<bool> m_bypass{false};
    std::array<std::atomic<float>, 6> m_customOffsets;
    std::array<std::atomic<float>, 6> m_preEqGains;
};

} // namespace autolevel
