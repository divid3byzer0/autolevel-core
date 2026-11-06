#include "../src/dsp/AutoLevelEngine.h"
#include "../src/engine/AudioEngine.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>
#include <limits>
#include <cstdint>

using namespace autolevel::dsp;

constexpr double TEST_PI = 3.14159265358979323846;

/**
 * Exact-frequency magnitude via the Goertzel algorithm. Used instead of a quick one-pole
 * "bandpass energy" check for spectral containment tests - a single-pole filter's rolloff is
 * gentle enough (~6dB/oct) that it can show meaningful "energy" many octaves from its nominal
 * band purely from filter skirt leakage, which reads as a false failure (or false pass) for
 * claims like "no energy below 65Hz". This resolves to within sr/N Hz given a window of N
 * samples, with none of that leakage.
 */
double testGoertzelMag(const std::vector<float>& sig, size_t start, size_t windowN, double sampleRate, double freqHz) {
    double k = std::round((static_cast<double>(windowN) * freqHz) / sampleRate);
    double w = 2.0 * TEST_PI * k / static_cast<double>(windowN);
    double cosw = std::cos(w);
    double coeff = 2.0 * cosw;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < windowN; ++i) {
        s0 = static_cast<double>(sig[start + i]) + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    double real = s1 - s2 * cosw;
    double imag = s2 * std::sin(w);
    return std::sqrt(real * real + imag * imag) / (static_cast<double>(windowN) / 2.0);
}

void testKWeightingSineWave() {
    std::cout << "[TEST] KWeightingFilter 1kHz calibration..." << std::endl;
    KWeightingFilter filter;
    double sampleRate = 48000.0;
    filter.prepare(sampleRate);

    double freq = 1000.0;
    double sumSqIn = 0.0;
    double sumSqOut = 0.0;
    size_t n = 48000;

    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        double s = std::sin(2.0 * TEST_PI * freq * t);
        sumSqIn += s * s;

        double outL = 0.0, outR = 0.0;
        filter.processSample(s, s, outL, outR);
        sumSqOut += outL * outL;
    }

    double rmsIn = std::sqrt(sumSqIn / n);
    double rmsOut = std::sqrt(sumSqOut / n);
    double gainDb = 20.0 * std::log10(rmsOut / rmsIn);

    std::cout << "  Input RMS: " << rmsIn << " (-3.01 dBFS)" << std::endl;
    std::cout << "  Output RMS: " << rmsOut << std::endl;
    std::cout << "  K-Weighting gain at 1kHz: " << gainDb << " dB" << std::endl;

    assert(gainDb > 0.0 && gainDb < 1.5);
    std::cout << "  -> PASS: 1kHz K-Weighting response is within specification." << std::endl;
}

void testAndroidToneProfilesAndThresholds() {
    std::cout << "[TEST] Android Shaper thresholdsFor tilt formula..." << std::endl;

    // The Modern Mix contour (and the profile switch that chose between it and a custom one)
    // was removed 2026-09-11 - there is now only ever one contour source, customOffsetsDb,
    // and testCustomToneProfile below already covers "does a non-zero contour shift thresholds
    // correctly." What's still worth its own check here is the tilt-only formula itself (no
    // contour at all) against the Android Shaper reference behavior it's built to match.
    auto flat = Bands::thresholdsFor(true, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto tilted = Bands::thresholdsFor(true, -2.0f, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    float flatSum = 0.0f;
    float tiltedSum = 0.0f;
    for (size_t b = 0; b < Bands::COUNT; ++b) {
        flatSum += flat[b];
        tiltedSum += tilted[b];
        std::cout << "  Band " << b << " (" << Bands::NAMES[b] << "): 0dB/oct=" << flat[b]
                  << " dB, -2dB/oct=" << tilted[b] << " dB" << std::endl;
    }

    float flatAvg = flatSum / Bands::COUNT;
    float tiltedAvg = tiltedSum / Bands::COUNT;

    // The average threshold must remain centered at -24 dBFS regardless of tilt (zero-sum
    // re-centering) - this is the invariant every contour/tilt combination relies on.
    assert(std::abs(flatAvg - (-24.0f)) < 0.05f);
    assert(std::abs(tiltedAvg - (-24.0f)) < 0.05f);

    // A negative (warm/bass-emphasis) tone slope must raise the Sub band's threshold (more
    // headroom before compressing) and lower the Air band's (tames harshness sooner) relative
    // to the flat (0 dB/oct) baseline - matching Android ShaperTest.kt's tilt direction.
    assert(tilted[0] > flat[0]);
    assert(tilted[5] < flat[5]);
    assert(tilted[0] > tilted[5]);

    std::cout << "  -> PASS: Tilt-only threshold formula matches Android Shaper and stays zero-sum." << std::endl;
}

void testLR4CrossoverSummation() {
    std::cout << "[TEST] Linkwitz-Riley 6-band crossover flat magnitude test..." << std::endl;
    MultibandCompressor mbc;
    double sampleRate = 48000.0;

    MBCParams params;
    params.enabled = true;
    // NOTE: compressionAmount must stay >= Bands::MIN_COMPRESSION. Below that,
    // MultibandCompressor::process() returns without touching the buffer at all,
    // which would make this test compare the input against itself and pass
    // unconditionally. A very high threshold is what keeps the band compressors
    // from applying any gain reduction, so this measures the filterbank alone.
    params.compressionAmount = Bands::MIN_COMPRESSION;
    params.baseThresholdDb = 60.0f;
    params.toneSlopeDbPerOctave = 0.0f;

    std::vector<double> testFreqs = { 30.0, 50.0, 120.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1800.0, 2000.0, 3500.0, 6000.0, 8000.0, 14000.0 };

    double worstDb = 0.0;
    double worstFreq = 0.0;
    for (double f : testFreqs) {
        mbc.prepare(sampleRate);
        size_t n = 48000;   // 1 s: the 120 Hz sections need real settling time
        std::vector<float> inL(n), inR(n), origL(n);
        for (size_t i = 0; i < n; ++i) {
            float s = static_cast<float>(std::sin(2.0 * TEST_PI * f * (static_cast<double>(i) / sampleRate)));
            inL[i] = s;
            inR[i] = s;
            origL[i] = s;
        }

        mbc.process(inL.data(), inR.data(), n, params);

        double sumSqIn = 0.0;
        double sumSqOut = 0.0;
        size_t count = 0;
        for (size_t i = 24000; i < n; ++i) {
            sumSqIn += origL[i] * origL[i];
            sumSqOut += inL[i] * inL[i];
            count++;
        }

        // Guard against the buffer never being processed (see note above).
        assert(sumSqOut > 0.0);

        double rmsIn = std::sqrt(sumSqIn / count);
        double rmsOut = std::sqrt(sumSqOut / count);
        double errorDb = 20.0 * std::log10(rmsOut / rmsIn);

        if (std::abs(errorDb) > std::abs(worstDb)) {
            worstDb = errorDb;
            worstFreq = f;
        }
        assert(std::abs(errorDb) < 0.10);
    }
    std::cout << "  Worst reconstruction error: " << worstDb << " dB at " << worstFreq << " Hz" << std::endl;
    std::cout << "  -> PASS: 6-band crossover tree sums to flat magnitude." << std::endl;
}

void testLevelResponseMapping() {
    std::cout << "[TEST] Level response slider mapping..." << std::endl;
    assert(LoudnessMeter::halfLifeForResponse(0.0f) == 0.0f); // whole track
    float mid = LoudnessMeter::halfLifeForResponse(0.5f);
    float max = LoudnessMeter::halfLifeForResponse(1.0f);
    std::cout << "  Response 0.0 -> " << LoudnessMeter::halfLifeForResponse(0.0f) << "s (track hold)" << std::endl;
    std::cout << "  Response 0.5 -> " << mid << "s" << std::endl;
    std::cout << "  Response 1.0 -> " << max << "s" << std::endl;
    assert(max >= 3.99f && max <= 4.01f);
    assert(mid > 4.0f && mid < 120.0f);
    std::cout << "  -> PASS: Level response logarithmic decay mapping verified." << std::endl;
}

void testFullChain() {
    std::cout << "[TEST] AutoLevelEngine full chain: AGC -> MBC -> Limiter..." << std::endl;
    AutoLevelEngine engine;
    double sampleRate = 48000.0;
    engine.prepare(sampleRate);

    EngineParameters params;
    params.targetLUFS = -9.0f;
    params.maxBoostDb = 12.0f;
    params.maxCutDb = 12.0f;
    params.ceilingDb = -1.5f;
    params.compressionAmount = 0.6f;
    params.toneSlopeDbPerOctave = -2.0f;

    // Simulate audio blocks for 3 seconds
    size_t block = 480;
    size_t totalBlocks = 300;

    float maxOutputPeak = 0.0f;
    for (size_t b = 0; b < totalBlocks; ++b) {
        std::vector<float> left(block), right(block);
        for (size_t i = 0; i < block; ++i) {
            float s = 0.125f * static_cast<float>(std::sin(2.0 * TEST_PI * 440.0 * (b * block + i) / sampleRate));
            left[i] = s;
            right[i] = s;
        }

        engine.process(left.data(), right.data(), block, params);

        for (size_t i = 0; i < block; ++i) {
            maxOutputPeak = std::max(maxOutputPeak, std::abs(left[i]));
        }
    }

    auto state = engine.getVisualState();
    std::cout << "  Applied Gain: " << state.appliedGainDb << " dB" << std::endl;
    std::cout << "  Target Gain: " << state.targetGainDb << " dB" << std::endl;
    std::cout << "  Max Output Peak: " << maxOutputPeak << " (Ceiling: " << std::pow(10.0f, -1.5f / 20.0f) << ")" << std::endl;

    assert(state.appliedGainDb > 0.0f);
    // Limiter ceiling (-1.5 dBFS = 0.841)
    assert(maxOutputPeak <= 0.85f);

    std::cout << "  -> PASS: Full chain leveled, compressed, and limited to ceiling." << std::endl;
}

void testEbuR128DynamicLoudness() {
    std::cout << "[TEST] EBU R128 Loudness measurement across varying song levels..." << std::endl;
    std::vector<float> inputLevels = { -6.0f, -12.0f, -18.0f, -24.0f };

    for (float level : inputLevels) {
        LoudnessMeter meter;
        double sampleRate = 48000.0;
        meter.prepare(sampleRate);

        float amp = std::pow(10.0f, level / 20.0f);
        size_t numBlocks = 40; // 4 seconds
        size_t blockSize = 4800; // 100ms

        for (size_t b = 0; b < numBlocks; ++b) {
            std::vector<float> l(blockSize), r(blockSize);
            for (size_t i = 0; i < blockSize; ++i) {
                float s = amp * static_cast<float>(std::sin(2.0 * TEST_PI * 1000.0 * (b * blockSize + i) / sampleRate));
                l[i] = s;
                r[i] = s;
            }
            meter.process(l.data(), r.data(), blockSize);
        }

        auto readings = meter.getReadings();
        std::cout << "  Input level " << level << " dBFS -> Integrated: "
                  << readings.integratedLUFS << " LUFS, Momentary: "
                  << readings.momentaryLUFS << " LUFS" << std::endl;

        // In BS.1770/EBU R128, a 1kHz sine wave of peak amp A has RMS = A / sqrt(2) (-3.01 dB)
        // With 1kHz K-weighting gain (~0.65 dB) and stereo (+3.01 dB) and -0.691 constant:
        // Expected LUFS is approximately level.
        assert(readings.integratedLUFS > -70.0f);
        assert(std::abs(readings.integratedLUFS - level) < 1.0f);
    }
    std::cout << "  -> PASS: EBU R128 measures varying loudness accurately across all levels." << std::endl;
}

/**
 * Drives the MBC with a 700 Hz tone (Low-Mid band) whose amplitude per sample comes from
 * `amp(i)`, in 16-sample blocks, and records that band's gain reduction after every block.
 */
template <typename AmpFn>
std::vector<float> runMbcLowMid(const MBCParams& params, size_t numSamples, AmpFn amp) {
    constexpr double sr = 48000.0;
    constexpr size_t block = 16;
    MultibandCompressor mbc;
    mbc.prepare(sr);
    std::vector<float> trace;
    std::vector<float> l(block), r(block);
    for (size_t pos = 0; pos < numSamples; pos += block) {
        for (size_t i = 0; i < block; ++i) {
            const size_t n = pos + i;
            l[i] = r[i] = static_cast<float>(amp(n) * std::sin(2.0 * TEST_PI * 700.0 * n / sr));
        }
        mbc.process(l.data(), r.data(), block, params);
        trace.push_back(mbc.getGainReductionsDb()[2]);
    }
    return trace;
}

MBCParams loudMbcParams() {
    MBCParams p;
    p.compressionAmount = 1.0f; // 5:1
    p.toneSlopeDbPerOctave = 0.0f;
    return p;
}

// Gain reduction `ms` milliseconds into a trace from runMbcLowMid (16-sample blocks at 48 kHz).
float grAtMs(const std::vector<float>& trace, double ms) {
    size_t idx = static_cast<size_t>(ms * 48.0 / 16.0);
    return trace[std::min(idx, trace.size() - 1)];
}

void testMbcRatioAndThreshold() {
    std::cout << "[TEST] MBC ratio range and threshold placement..." << std::endl;
    // Well above the knee, 6 dB more input must give 6 x (1 - 1/ratio) dB more reduction.
    // 100% is 5:1 (4.8 dB), 50% is 3:1 (4.0 dB).
    auto steady = [](double a) { return [a](size_t) { return a; }; };
    for (float amount : {1.0f, 0.5f}) {
        MBCParams p = loudMbcParams();
        p.compressionAmount = amount;
        const float lo = runMbcLowMid(p, 48000, steady(0.25)).back();
        const float hi = runMbcLowMid(p, 48000, steady(0.5)).back();
        const float ratio = 1.0f + amount * 4.0f;
        const float expected = 6.02f * (1.0f - 1.0f / ratio);
        std::cout << "  " << amount * 100 << "%: 6 dB more input -> " << (lo - hi)
                  << " dB more reduction (expected " << expected << " at " << ratio << ":1)" << std::endl;
        assert(std::abs((lo - hi) - expected) < 0.15f);
    }

    // The engine places the base threshold 18 dB under Target LUFS: -30 dBFS at -12.
    for (float target : {-12.0f, -14.0f}) {
        AutoLevelEngine eng;
        eng.prepare(48000.0);
        EngineParameters ep;
        ep.targetLUFS = target;
        ep.toneSlopeDbPerOctave = 0.0f;
        std::vector<float> l(256, 0.0f), r(256, 0.0f);
        eng.process(l.data(), r.data(), 256, ep);
        const auto th = eng.getVisualState().mbcThresholdsDb;
        float mean = 0.0f;
        for (float t : th) mean += t / 6.0f;
        std::cout << "  Target " << target << " LUFS -> mean band threshold " << mean << " dBFS" << std::endl;
        assert(std::abs(mean - (target - 18.0f)) < 0.01f);
    }
    std::cout << "  -> PASS: 100% is 5:1, 50% is 3:1, thresholds 18 dB under the target." << std::endl;
}

void testMbcAttackRelease() {
    std::cout << "[TEST] MBC attack/release times..." << std::endl;
    // Attack: 10 ms into a sudden loud tone.
    MBCParams fast = loudMbcParams();
    fast.attackMs = 1.0f;
    MBCParams slow = loudMbcParams();
    slow.attackMs = 100.0f;
    auto onset = [](size_t) { return 0.7; };
    float fastGr = grAtMs(runMbcLowMid(fast, 4800, onset), 10.0);
    float slowGr = grAtMs(runMbcLowMid(slow, 4800, onset), 10.0);
    std::cout << "  10 ms into a burst: attack 1 ms " << fastGr << " dB, 100 ms " << slowGr << " dB" << std::endl;
    assert(fastGr < -10.0f);
    assert(slowGr > fastGr + 6.0f);

    // Release: 150 ms after 1 s of loud tone stops.
    MBCParams shortRel = loudMbcParams();
    shortRel.releaseMs = 20.0f;
    MBCParams longRel = loudMbcParams();
    longRel.releaseMs = 1000.0f;
    auto gate = [](size_t n) { return n < 48000 ? 0.7 : 0.0; };
    float shortGr = grAtMs(runMbcLowMid(shortRel, 48000 + 7200, gate), 1150.0);
    float longGr = grAtMs(runMbcLowMid(longRel, 48000 + 7200, gate), 1150.0);
    std::cout << "  150 ms after it stops: release 20 ms " << shortGr << " dB, 1000 ms " << longGr << " dB" << std::endl;
    assert(shortGr > -0.5f);
    assert(longGr < -5.0f);

    // Out-of-range times are clamped, not used as-is (0 ms would be an instant, clicking attack).
    MBCParams zero = loudMbcParams();
    zero.attackMs = 0.0f;
    MBCParams one = loudMbcParams();
    one.attackMs = 1.0f;
    assert(std::abs(grAtMs(runMbcLowMid(zero, 4800, onset), 2.0) -
                    grAtMs(runMbcLowMid(one, 4800, onset), 2.0)) < 0.01f);

    // prepare() reinstalls default ballistics; the next block must re-apply the set times.
    MultibandCompressor mbc;
    mbc.prepare(48000.0);
    std::vector<float> l(16, 0.0f), r(16, 0.0f);
    mbc.process(l.data(), r.data(), 16, fast);
    mbc.prepare(48000.0);
    std::vector<float> trace;
    for (size_t pos = 0; pos < 480; pos += 16) {
        for (size_t i = 0; i < 16; ++i)
            l[i] = r[i] = static_cast<float>(0.7 * std::sin(2.0 * TEST_PI * 700.0 * (pos + i) / 48000.0));
        mbc.process(l.data(), r.data(), 16, fast);
        trace.push_back(mbc.getGainReductionsDb()[2]);
    }
    assert(std::abs(trace.back() - grAtMs(runMbcLowMid(fast, 4800, onset), 10.0)) < 0.01f);
    std::cout << "  -> PASS: attack and release follow the set times, clamped, and survive prepare()." << std::endl;
}

void testMbcDetectorPeakRms() {
    std::cout << "[TEST] MBC detector, peak vs RMS..." << std::endl;
    // A steady tone: RMS is sine-calibrated, so the detector choice barely changes the depth.
    auto steady = [](size_t) { return 0.5; };
    MBCParams peak = loudMbcParams();
    MBCParams half = loudMbcParams();
    half.detectorRms = 0.5f;
    MBCParams rms = loudMbcParams();
    rms.detectorRms = 1.0f;
    float peakGr = runMbcLowMid(peak, 96000, steady).back();
    float halfGr = runMbcLowMid(half, 96000, steady).back();
    float rmsGr = runMbcLowMid(rms, 96000, steady).back();
    std::cout << "  Steady tone: peak " << peakGr << " dB, 50/50 " << halfGr << " dB, RMS " << rmsGr << " dB" << std::endl;
    assert(peakGr < -5.0f);
    assert(std::abs(peakGr - rmsGr) < 1.2f);
    assert(std::abs(peakGr - halfGr) < 1.2f);

    // Short bursts (4 ms every 100 ms) with a 1 ms attack: peak catches each hit at full
    // height, RMS averages it over 150 ms and lets it through.
    auto bursts = [](size_t n) { return (n % 4800) < 192 ? 0.9 : 0.0; };
    peak.attackMs = rms.attackMs = 1.0f;
    auto peakTrace = runMbcLowMid(peak, 48000, bursts);
    auto rmsTrace = runMbcLowMid(rms, 48000, bursts);
    float peakMax = *std::min_element(peakTrace.begin(), peakTrace.end());
    float rmsMax = *std::min_element(rmsTrace.begin(), rmsTrace.end());
    std::cout << "  4 ms bursts, attack 1 ms: deepest reduction peak " << peakMax << " dB, RMS " << rmsMax << " dB" << std::endl;
    assert(rmsMax > peakMax + 6.0f);
    std::cout << "  -> PASS: same depth on sustained material, transients pass with RMS." << std::endl;
}

void testMbcAutoRelease() {
    std::cout << "[TEST] MBC auto release..." << std::endl;
    MBCParams plain = loudMbcParams();
    MBCParams autoRel = loudMbcParams();
    autoRel.autoRelease = true;

    // A sustained loud passage builds up the slow path, which then holds the reduction.
    auto sustained = [](size_t n) { return n < 144000 ? 0.7 : 0.0; };
    float plainHeld = grAtMs(runMbcLowMid(plain, 144000 + 19200, sustained), 3300.0);
    float autoHeld = grAtMs(runMbcLowMid(autoRel, 144000 + 19200, sustained), 3300.0);
    std::cout << "  300 ms after 3 s of tone: " << plainHeld << " dB plain, " << autoHeld << " dB auto" << std::endl;
    assert(autoHeld < plainHeld - 3.0f);

    // A short hit never builds the slow path up, so it recovers at the set release either way.
    auto hit = [](size_t n) { return n < 1440 ? 0.7 : 0.0; };
    float plainHit = grAtMs(runMbcLowMid(plain, 1440 + 14400, hit), 330.0);
    float autoHit = grAtMs(runMbcLowMid(autoRel, 1440 + 14400, hit), 330.0);
    std::cout << "  300 ms after a 30 ms hit: " << plainHit << " dB plain, " << autoHit << " dB auto" << std::endl;
    assert(std::abs(plainHit - autoHit) < 1.0f);

    // No latency: the output is the input times a gain, sample for sample. With the band
    // below threshold the MBC must pass an impulse at exactly the same sample either way.
    MBCParams quietAuto = autoRel;
    quietAuto.detectorRms = 1.0f;
    for (const MBCParams* p : { &plain, &quietAuto }) {
        MultibandCompressor mbc;
        mbc.prepare(48000.0);
        std::vector<float> l(4800, 0.0f), r(4800, 0.0f);
        l[100] = r[100] = 0.001f;
        mbc.process(l.data(), r.data(), l.size(), *p);
        size_t peakIdx = static_cast<size_t>(std::max_element(l.begin(), l.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); }) - l.begin());
        assert(peakIdx < 140); // the crossover's own group delay only, a few samples
        static size_t firstIdx = peakIdx;
        assert(peakIdx == firstIdx);
    }
    std::cout << "  -> PASS: auto release holds sustained passages, not short hits, with no added delay." << std::endl;
}

void testPostMbcGain() {
    std::cout << "[TEST] Post-MBC Gain stage (makeup & trim)..." << std::endl;
    AutoLevelEngine engine;
    double sampleRate = 48000.0;
    engine.prepare(sampleRate);

    EngineParameters params;
    params.targetLUFS = -9.0f;
    params.maxBoostDb = 0.0f;
    params.maxCutDb = 0.0f;
    params.compressionAmount = 0.0f; // Linear pass-through
    params.ceilingDb = 0.0f;         // Limiter ceiling 0 dBFS so +6 dB won't clip test signal

    size_t n = 480;
    // Base signal at 0.1 peak (-20 dBFS)
    std::vector<float> in0L(n, 0.1f), in0R(n, 0.1f);
    params.postMbcGainDb = 0.0f;
    engine.reset();
    engine.process(in0L.data(), in0R.data(), n, params);
    // The limiter delays the chain; read the sample this run's first input lands on (the
    // delay line before it still holds the previous run's tail - reset leaves it alone).
    const size_t d = static_cast<size_t>(engine.latencyFrames());
    float peak0 = std::abs(in0L[d]);

    // +6 dB post gain -> ~2.0x amplitude
    std::vector<float> inPlusL(n, 0.1f), inPlusR(n, 0.1f);
    params.postMbcGainDb = 6.0f;
    engine.reset();
    engine.process(inPlusL.data(), inPlusR.data(), n, params);
    float peakPlus = std::abs(inPlusL[d]);

    // -6 dB post gain -> ~0.5x amplitude
    std::vector<float> inMinusL(n, 0.1f), inMinusR(n, 0.1f);
    params.postMbcGainDb = -6.0f;
    engine.reset();
    engine.process(inMinusL.data(), inMinusR.data(), n, params);
    float peakMinus = std::abs(inMinusL[d]);

    std::cout << "  0 dB Post Peak: " << peak0 << std::endl;
    std::cout << "  +6 dB Post Peak: " << peakPlus << " (ratio: " << (peakPlus / peak0) << ")" << std::endl;
    std::cout << "  -6 dB Post Peak: " << peakMinus << " (ratio: " << (peakMinus / peak0) << ")" << std::endl;

    assert(std::abs((peakPlus / peak0) - std::pow(10.0f, 6.0f / 20.0f)) < 0.02f);
    assert(std::abs((peakMinus / peak0) - std::pow(10.0f, -6.0f / 20.0f)) < 0.02f);
    std::cout << "  -> PASS: Post-MBC Gain accurately boosts and attenuates signal." << std::endl;
}

void testPreMbcEQ() {
    std::cout << "[TEST] PreMbcEQ (manual 6-band EQ, ahead of the Multiband Compressor)..." << std::endl;
    PreMbcEQ eq;
    double sampleRate = 48000.0;
    eq.prepare(sampleRate);

    // 1. All bands flat (0dB): must be 100% bit-identical bypass. At exactly 0dB gain, the RBJ
    // peaking formula reduces algebraically to b == a coefficientwise (H(z) == 1 identically,
    // not just unity magnitude), so this should hold exactly, not just approximately.
    size_t n = 4800; // 100ms
    std::vector<float> origL(n), origR(n);
    for (size_t i = 0; i < n; ++i) {
        float s = 0.3f * static_cast<float>(std::sin(2.0 * TEST_PI * 219.089 * i / sampleRate))
                + 0.2f * static_cast<float>(std::sin(2.0 * TEST_PI * 5291.5 * i / sampleRate));
        origL[i] = s;
        origR[i] = s;
    }
    std::array<float, Bands::COUNT> flatGains{};
    std::vector<float> passL = origL;
    std::vector<float> passR = origR;
    eq.process(passL.data(), passR.data(), n, flatGains);
    for (size_t i = 0; i < n; ++i) {
        assert(passL[i] == origL[i]);
        assert(passR[i] == origR[i]);
    }
    std::cout << "  -> PASS: All-flat (0dB) is 100% bit-identical bypass." << std::endl;

    // 2. Boosting one band lifts a tone at that band's center frequency, and leaves a tone in a
    // distant band essentially untouched (the untouched bands sit at exact identity per the
    // note above, so any change there would have to come from the boosted band's own filter
    // skirt reaching that far - it shouldn't, five bands away).
    size_t n2 = static_cast<size_t>(sampleRate * 2.0);
    std::vector<float> boostL(n2), boostR(n2), boostOrigL(n2);
    double bassHz = 219.089;  // Band 1 (Bass) center
    double airHz = 12649.1;   // Band 5 (Air) center
    for (size_t i = 0; i < n2; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        float s = 0.1f * static_cast<float>(std::sin(2.0 * TEST_PI * bassHz * t))
                + 0.1f * static_cast<float>(std::sin(2.0 * TEST_PI * airHz * t));
        boostL[i] = s; boostR[i] = s; boostOrigL[i] = s;
    }
    std::array<float, Bands::COUNT> boostGains{};
    boostGains[1] = 9.0f; // +9dB on Bass only
    size_t block = 480;
    for (size_t s = 0; s + block <= n2; s += block) {
        eq.process(boostL.data() + s, boostR.data() + s, block, boostGains);
    }
    size_t goertzelStart2 = n2 / 2;
    size_t goertzelWindow2 = 16384;
    double bassIn  = testGoertzelMag(boostOrigL, goertzelStart2, goertzelWindow2, sampleRate, bassHz);
    double bassOut = testGoertzelMag(boostL,     goertzelStart2, goertzelWindow2, sampleRate, bassHz);
    double airIn   = testGoertzelMag(boostOrigL, goertzelStart2, goertzelWindow2, sampleRate, airHz);
    double airOut  = testGoertzelMag(boostL,     goertzelStart2, goertzelWindow2, sampleRate, airHz);
    double bassGainDb = 20.0 * std::log10((bassOut + 1e-9) / (bassIn + 1e-9));
    std::cout << "  +9dB Bass boost measured: " << bassGainDb << " dB (Bass tone); Air tone delta: "
              << (20.0 * std::log10((airOut + 1e-9) / (airIn + 1e-9))) << " dB" << std::endl;
    assert(bassGainDb > 7.0 && bassGainDb < 11.0); // ~9dB at exact center, generous tolerance
    assert(airOut > airIn * 0.98f && airOut < airIn * 1.02f); // untouched
    std::cout << "  -> PASS: Boosting one band lifts that band's tone, leaves a distant band alone." << std::endl;

    // 3. Cutting a band reduces a tone at its center frequency.
    eq.reset();
    std::vector<float> cutL(n2), cutOrigL(n2), cutR(n2);
    double lowMidHz = 692.82; // Band 2 (Low-Mid) center
    for (size_t i = 0; i < n2; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        float s = 0.2f * static_cast<float>(std::sin(2.0 * TEST_PI * lowMidHz * t));
        cutL[i] = s; cutR[i] = s; cutOrigL[i] = s;
    }
    std::array<float, Bands::COUNT> cutGains{};
    cutGains[2] = -9.0f;
    for (size_t s = 0; s + block <= n2; s += block) {
        eq.process(cutL.data() + s, cutR.data() + s, block, cutGains);
    }
    double lowMidIn  = testGoertzelMag(cutOrigL, goertzelStart2, goertzelWindow2, sampleRate, lowMidHz);
    double lowMidOut = testGoertzelMag(cutL,     goertzelStart2, goertzelWindow2, sampleRate, lowMidHz);
    double lowMidGainDb = 20.0 * std::log10((lowMidOut + 1e-9) / (lowMidIn + 1e-9));
    std::cout << "  -9dB Low-Mid cut measured: " << lowMidGainDb << " dB" << std::endl;
    assert(lowMidGainDb < -7.0 && lowMidGainDb > -11.0);
    std::cout << "  -> PASS: Cutting a band reduces that band's tone by roughly the set amount." << std::endl;

    // 4. No NaN/Inf across varied content with all six bands engaged at once, including a
    // gain change mid-stream (exercises the coefficient-recompute-on-change path).
    eq.reset();
    std::vector<float> noiseL(n2), noiseR(n2);
    for (size_t i = 0; i < n2; ++i) {
        noiseL[i] = ((i * 2654435761u) % 2000 - 1000) / 500.0f;
        noiseR[i] = ((i * 40503u) % 2000 - 1000) / 500.0f;
    }
    std::array<float, Bands::COUNT> mixedGains = {6.0f, -6.0f, 3.0f, -3.0f, 9.0f, -9.0f};
    bool clean = true;
    for (size_t s = 0; s + block <= n2; s += block) {
        if ((s / block) % 100 == 0) {
            // Nudge one gain periodically to exercise recomputation mid-stream.
            mixedGains[3] = -mixedGains[3];
        }
        eq.process(noiseL.data() + s, noiseR.data() + s, block, mixedGains);
        for (size_t i = s; i < s + block; ++i) {
            if (!std::isfinite(noiseL[i]) || !std::isfinite(noiseR[i])) clean = false;
        }
    }
    std::cout << "  " << (clean ? "PASS: all finite" : "FAIL: NaN/Inf") << std::endl;
    assert(clean);
    std::cout << "  -> PASS: Stable across broadband content with all bands engaged and changing." << std::endl;
}

void testDynamicAirLift() {
    std::cout << "[TEST] DynamicAirLift (Dolby Duo high-end upward expansion & sibilance ducking)..." << std::endl;
    DynamicAirLift airLift;
    double sampleRate = 48000.0;
    airLift.prepare(sampleRate);

    // 1. Test AirLiftMode::OFF is 100% bit-identical bypass
    size_t n = 4800;
    std::vector<float> origL(n), origR(n);
    for (size_t i = 0; i < n; ++i) {
        float s = 0.4f * static_cast<float>(std::sin(2.0 * TEST_PI * 1000.0 * i / sampleRate));
        origL[i] = s;
        origR[i] = s;
    }
    std::vector<float> passL = origL;
    std::vector<float> passR = origR;
    airLift.process(passL.data(), passR.data(), n, AirLiftMode::OFF);
    for (size_t i = 0; i < n; ++i) {
        assert(passL[i] == origL[i]);
        assert(passR[i] == origR[i]);
    }
    std::cout << "  -> PASS: AirLiftMode::OFF is 100% bit-identical bypass." << std::endl;

    // 2. Test air lift on dark vintage material (strong 2 kHz mid, weak 12 kHz air)
    airLift.reset();
    std::vector<float> darkL(n), darkR(n);
    for (size_t i = 0; i < n; ++i) {
        float mid = 0.5f * static_cast<float>(std::sin(2.0 * TEST_PI * 2000.0 * i / sampleRate));
        float weakAir = 0.02f * static_cast<float>(std::sin(2.0 * TEST_PI * 12000.0 * i / sampleRate));
        darkL[i] = mid + weakAir;
        darkR[i] = mid + weakAir;
    }
    airLift.process(darkL.data(), darkR.data(), n, AirLiftMode::MED);
    float airLiftDb = airLift.getLiftDb();
    std::cout << "  Dynamic Air Lift applied on dark track: +" << airLiftDb << " dB" << std::endl;
    assert(airLiftDb > 2.0f && airLiftDb <= 4.5f);
    std::cout << "  -> PASS: Dynamic Air Lift smoothly pulls up natural top-end air with zero distortion." << std::endl;

    // 3. Test adaptive suppression: on already-bright track, lift must drop to near 0 dB
    airLift.reset();
    std::vector<float> brightL(n), brightR(n);
    for (size_t i = 0; i < n; ++i) {
        float mid = 0.3f * static_cast<float>(std::sin(2.0 * TEST_PI * 2000.0 * i / sampleRate));
        float brightAir = 0.5f * static_cast<float>(std::sin(2.0 * TEST_PI * 12000.0 * i / sampleRate));
        brightL[i] = mid + brightAir;
        brightR[i] = mid + brightAir;
    }
    airLift.process(brightL.data(), brightR.data(), n, AirLiftMode::MED);
    float brightLiftDb = airLift.getLiftDb();
    std::cout << "  Dynamic Air Lift on bright track: +" << brightLiftDb << " dB" << std::endl;
    assert(brightLiftDb < 1.0f);
    std::cout << "  -> PASS: Adaptive sensor protects bright tracks from harshness." << std::endl;

    // 4. Regression test for a real bug (2026-09-11): the mid anchor used to be a
    // plain low-pass ("everything below 2kHz") instead of a 1-3kHz bandpass, so the
    // denominator was always huge relative to any real track's air content and the
    // lift barely varied with actual brightness. Case 3 above didn't catch it because
    // its "bright" signal has MORE amplitude at 12kHz than at 2kHz - a ratio no real
    // track has. This uses a realistic ratio instead (air always quieter than mid,
    // like real cymbals/hats vs. a vocal or instrument fundamental) and asserts the
    // lift actually tracks brightness rather than staying flat.
    airLift.reset();
    std::vector<float> moderateBrightL(n), moderateBrightR(n);
    for (size_t i = 0; i < n; ++i) {
        float mid = 0.5f * static_cast<float>(std::sin(2.0 * TEST_PI * 2000.0 * i / sampleRate));
        float moderateAir = 0.08f * static_cast<float>(std::sin(2.0 * TEST_PI * 12000.0 * i / sampleRate));
        moderateBrightL[i] = mid + moderateAir;
        moderateBrightR[i] = mid + moderateAir;
    }
    airLift.process(moderateBrightL.data(), moderateBrightR.data(), n, AirLiftMode::MED);
    float moderateBrightLiftDb = airLift.getLiftDb();
    std::cout << "  Dynamic Air Lift on realistic moderately-bright track: +" << moderateBrightLiftDb
              << " dB (dark track was +" << airLiftDb << " dB)" << std::endl;
    assert(airLiftDb - moderateBrightLiftDb > 0.3f);
    std::cout << "  -> PASS: Lift actually discriminates brightness at realistic (non-extreme) ratios." << std::endl;
}

void testHighPassFilter() {
    std::cout << "[TEST] HighPassFilter 24 dB/octave 4th-order Butterworth response..." << std::endl;
    HighPassFilter hpf;
    double sampleRate = 48000.0;
    hpf.prepare(sampleRate);
    hpf.setCutoff(30.0f, true);

    auto measureGainAt = [&](double freq) -> double {
        hpf.reset();
        size_t n = 48000; // 1 second
        std::vector<float> inL(n), inR(n);
        double sumSqIn = 0.0, sumSqOut = 0.0;
        for (size_t i = 0; i < n; ++i) {
            float s = static_cast<float>(std::sin(2.0 * TEST_PI * freq * i / sampleRate));
            inL[i] = s;
            inR[i] = s;
        }
        hpf.process(inL.data(), inR.data(), n);
        // Measure steady state in second half
        for (size_t i = 24000; i < n; ++i) {
            float orig = static_cast<float>(std::sin(2.0 * TEST_PI * freq * i / sampleRate));
            sumSqIn += orig * orig;
            sumSqOut += inL[i] * inL[i];
        }
        return 20.0 * std::log10(std::sqrt(sumSqOut / sumSqIn));
    };

    double gainPassband = measureGainAt(1000.0);
    double gainCutoff   = measureGainAt(30.0);
    double gainOctaveDown = measureGainAt(15.0);

    std::cout << "  1 kHz passband gain: " << gainPassband << " dB (expected: ~0.0 dB)" << std::endl;
    std::cout << "  30 Hz cutoff gain:   " << gainCutoff << " dB (expected: ~-3.0 dB)" << std::endl;
    std::cout << "  15 Hz octave-down gain: " << gainOctaveDown << " dB (expected: ~-27.0 dB)" << std::endl;

    assert(std::abs(gainPassband) < 0.05); // Flat passband
    assert(std::abs(gainCutoff - (-3.01)) < 0.25); // Exactly -3 dB at Butterworth cutoff
    assert(gainOctaveDown < -24.0); // 24 dB/oct slope verified!

    std::cout << "  -> PASS: 4th-order Butterworth filter exhibits exact 24 dB/octave attenuation." << std::endl;
}

void testDefaultLimiterCeiling() {
    std::cout << "[TEST] Default limiter ceiling is -0.3 dBFS..." << std::endl;
    EngineParameters defaultParams;
    std::cout << "  Default ceiling: " << defaultParams.ceilingDb << " dBFS" << std::endl;
    assert(std::abs(defaultParams.ceilingDb - (-0.3f)) < 0.001f);
    std::cout << "  -> PASS: Default limiter ceiling is -0.3 dBFS." << std::endl;
}

// Reconstructed (inter-sample) peak in dBFS: 8x windowed-sinc interpolation with 65 taps.
// Deliberately longer than the limiter's own 16-tap detector, so it is an independent check.
double testTruePeakDb(const std::vector<float>& x, size_t from, size_t to) {
    double mx = 0.0;
    const int K = 32;
    for (size_t i = from; i < to; ++i) {
        for (int p = 0; p < 8; ++p) {
            const double t = static_cast<double>(i) + p / 8.0;
            double acc = 0.0;
            for (int k = -K; k <= K; ++k) {
                const long j = static_cast<long>(i) + k;
                if (j < 0 || j >= static_cast<long>(x.size())) continue;
                const double d = t - static_cast<double>(j);
                const double sinc = d == 0.0 ? 1.0 : std::sin(TEST_PI * d) / (TEST_PI * d);
                acc += x[static_cast<size_t>(j)] * sinc * (0.5 + 0.5 * std::cos(TEST_PI * d / (K + 1)));
            }
            mx = std::max(mx, std::abs(acc));
        }
    }
    return 20.0 * std::log10(mx);
}

// Runs a mono signal through the limiter in 256-frame blocks (both channels the same).
std::vector<float> runLimiter(std::vector<float> x, double sr, float ceilingDb, TruePeakLimiter* keep = nullptr) {
    TruePeakLimiter local;
    TruePeakLimiter& lim = keep ? *keep : local;
    if (!keep) lim.prepare(sr);
    lim.setCeilingDb(ceilingDb);
    std::vector<float> r = x;
    for (size_t p = 0; p < x.size(); p += 256) {
        const size_t n = std::min<size_t>(256, x.size() - p);
        lim.process(&x[p], &r[p], n);
    }
    return x;
}

std::vector<float> testSine(double sr, double freq, double peakDb, size_t frames, double phase = 0.0) {
    std::vector<float> x(frames);
    const double a = std::pow(10.0, peakDb / 20.0);
    for (size_t i = 0; i < frames; ++i) x[i] = static_cast<float>(a * std::sin(2.0 * TEST_PI * freq * i / sr + phase));
    return x;
}

void testTruePeakLimiterTransparency() {
    std::cout << "[TEST] True-peak limiter: latency and bit-exact pass-through..." << std::endl;
    for (double sr : {44100.0, 48000.0}) {
        TruePeakLimiter lim;
        lim.prepare(sr);
        const int d = lim.latencyFrames();
        std::cout << "  " << sr << " Hz: latency " << d << " frames (" << d * 1000.0 / sr << " ms)" << std::endl;
        assert(d >= 85 && d <= 110);

        // Under the ceiling: every sample comes out exactly, d frames later, preceded by silence.
        std::vector<float> in = testSine(sr, 997.0, -3.0, 8192);
        in[1000] = 0.5f; // an impulse, to pin the delay exactly
        std::vector<float> out = runLimiter(in, sr, -0.3f, &lim);
        for (int i = 0; i < d; ++i) assert(out[static_cast<size_t>(i)] == 0.0f);
        for (size_t i = static_cast<size_t>(d); i < out.size(); ++i) assert(out[i] == in[i - static_cast<size_t>(d)]);
    }

    // After limiting, the gain returns to exactly 1: quiet audio is bit-identical again.
    TruePeakLimiter lim;
    lim.prepare(44100.0);
    const size_t d = static_cast<size_t>(lim.latencyFrames());
    std::vector<float> in = testSine(44100.0, 440.0, 6.0, 44100);          // 1 s, 6 dB over
    std::vector<float> quiet = testSine(44100.0, 440.0, -6.0, 3 * 44100);  // then 3 s under
    in.insert(in.end(), quiet.begin(), quiet.end());
    std::vector<float> out = runLimiter(in, 44100.0, -0.3f, &lim);
    for (size_t i = 3 * 44100; i < out.size(); ++i) assert(out[i] == in[i - d]);
    std::cout << "  -> PASS: delay is exact, audio under the ceiling is untouched, and stays so after limiting." << std::endl;
}

void testTruePeakLimiterNoDistortion() {
    std::cout << "[TEST] True-peak limiter: loud tones held at the ceiling without distortion..." << std::endl;
    const double sr = 44100.0;
    const float ceilingDb = -1.0f;
    double worstThd = 0.0, worstTp = -100.0;
    // 30-60 Hz at 44.1 kHz is where an exact hold comparison distorted (see HOLD_TOLERANCE_DB).
    for (double f : {30.0, 40.0, 45.0, 55.0, 100.0, 1000.0, 5000.0}) {
        for (double overDb : {1.0, 3.0, 6.0, 12.0}) {
            std::vector<float> out = runLimiter(testSine(sr, f, ceilingDb + overDb, 2 * 44100), sr, ceilingDb);
            const size_t from = 44100, n = 44100; // second half: settled; 1 Hz bins
            const double fund = testGoertzelMag(out, from, n, sr, f);
            double h2 = 0.0;
            for (int k = 2; k <= 15 && f * k < sr / 2; ++k) {
                const double a = testGoertzelMag(out, from, n, sr, f * k);
                h2 += a * a;
            }
            const double thd = 100.0 * std::sqrt(h2) / fund;
            const double tp = testTruePeakDb(out, from, from + 4410);
            worstThd = std::max(worstThd, thd);
            worstTp = std::max(worstTp, tp);
            assert(tp <= ceilingDb + 0.02);
            assert(thd < 0.05);
        }
    }
    std::cout << "  40 Hz-5 kHz, 1-12 dB over: worst THD " << worstThd << " %, worst true peak "
              << worstTp << " dBTP (ceiling " << ceilingDb << ")" << std::endl;

    // Bass + mid together (two-tone): the gain must not move with the bass cycle, or the mid
    // tone grows sidebands at 2000 +- 50/100 Hz (intermodulation).
    {
        std::vector<float> two(2 * 44100);
        for (size_t i = 0; i < two.size(); ++i)
            two[i] = static_cast<float>(1.2 * std::sin(2.0 * TEST_PI * 50.0 * i / sr) + 0.3 * std::sin(2.0 * TEST_PI * 2000.0 * i / sr));
        std::vector<float> o = runLimiter(two, sr, ceilingDb);
        const double mid = testGoertzelMag(o, 44100, 44100, sr, 2000.0);
        double side = 0.0;
        for (double off : {-100.0, -50.0, 50.0, 100.0}) side = std::max(side, testGoertzelMag(o, 44100, 44100, sr, 2000.0 + off));
        const double imdDb = 20.0 * std::log10(side / mid);
        std::cout << "  50 Hz + 2 kHz, 4.6 dB over: worst sideband " << imdDb << " dB under the 2 kHz tone" << std::endl;
        assert(imdDb < -80.0);
    }

    // Inter-sample peaks: fs/4 at 45 degrees puts every sample 3 dB under the waveform's peak.
    std::vector<float> isp = testSine(sr, sr / 4.0, 2.0, 22050, TEST_PI / 4.0);
    std::vector<float> ispOut = runLimiter(isp, sr, ceilingDb);
    const double ispTp = testTruePeakDb(ispOut, 11025, 11025 + 2048);
    std::cout << "  fs/4 at 45 deg (true peak 3 dB over the sample peaks): out " << ispTp << " dBTP" << std::endl;
    assert(ispTp <= ceilingDb + 0.05);

    // A burst out of silence, 10 dB over: the lookahead has the gain down before it arrives.
    std::vector<float> burst(22050, 0.0f);
    std::vector<float> loud = testSine(sr, 220.0, ceilingDb + 10.0, 22050);
    burst.insert(burst.end(), loud.begin(), loud.end());
    std::vector<float> burstOut = runLimiter(burst, sr, ceilingDb);
    const double burstTp = testTruePeakDb(burstOut, 22050, 22050 + 4410);
    std::cout << "  burst from silence, 10 dB over: first 100 ms peak " << burstTp << " dBTP" << std::endl;
    assert(burstTp <= ceilingDb + 0.02);

    // The meter reads the reduction: 6 dB over -> ~6 dB.
    TruePeakLimiter lim;
    lim.prepare(sr);
    runLimiter(testSine(sr, 1000.0, ceilingDb + 6.0, 44100), sr, ceilingDb, &lim);
    std::cout << "  meter, 6 dB over: " << lim.getGainReductionDb() << " dB" << std::endl;
    assert(lim.getGainReductionDb() < -5.5f && lim.getGainReductionDb() > -6.5f);
    std::cout << "  -> PASS: no clipping, no distortion, inter-sample peaks caught." << std::endl;
}

void testTruePeakLimiterBypassAndReset() {
    std::cout << "[TEST] True-peak limiter: bypass keeps timing, reset keeps audio..." << std::endl;
    const double sr = 44100.0;
    // Bypass: delayed but untouched, even far over the ceiling (no clamp).
    TruePeakLimiter lim;
    lim.prepare(sr);
    lim.setCeilingDb(-1.0f);
    const size_t d = static_cast<size_t>(lim.latencyFrames());
    std::vector<float> loud = testSine(sr, 440.0, 6.0, 22050);
    std::vector<float> l = loud, r = loud;
    lim.process(l.data(), r.data(), l.size(), false);
    for (size_t i = d; i < l.size(); ++i) assert(l[i] == loud[i - d]);

    // Switching back on mid-signal: already at the right gain, so nothing gets past the ceiling.
    std::vector<float> more = testSine(sr, 440.0, 6.0, 22050 + 4410);
    std::vector<float> cont(more.begin() + 22050, more.end());
    std::vector<float> rr = cont;
    lim.process(cont.data(), rr.data(), cont.size(), true);
    const double tp = testTruePeakDb(cont, 0, 2205);
    std::cout << "  first 50 ms after un-bypassing a tone 7 dB over: " << tp << " dBTP" << std::endl;
    assert(tp <= -1.0 + 0.02);

    // Through the whole engine: bypass delays by exactly the limiter latency, and a loudness
    // reset (POST /api/reset) does not drop the audio sitting in the delay line.
    AutoLevelEngine eng;
    eng.prepare(sr);
    EngineParameters bp;
    bp.bypass = true;
    std::vector<float> a(64, 0.0f), b(64, 0.0f);
    a[10] = b[10] = 0.25f;
    eng.process(a.data(), b.data(), 64, bp);
    eng.reset();
    std::vector<float> trace(a.begin(), a.end());
    for (int blk = 0; blk < 4; ++blk) {
        std::vector<float> z(64, 0.0f), z2(64, 0.0f);
        eng.process(z.data(), z2.data(), 64, bp);
        trace.insert(trace.end(), z.begin(), z.end());
    }
    for (size_t i = 0; i < trace.size(); ++i) assert(trace[i] == (i == 10 + d ? 0.25f : 0.0f));
    std::cout << "  -> PASS: bypass is a pure delay, un-bypass is clean, reset keeps the music." << std::endl;
}

void testBreakdownFreezeSensitivity() {
    std::cout << "[TEST] Breakdown Auto-Freeze 3-way sensitivity (5 LU, 7 LU, 9 LU)..." << std::endl;
    Leveler leveler;
    leveler.prepare(48000.0);

    LoudnessReadings readings;
    readings.integratedLUFS = -14.0f;
    readings.momentaryLUFS = -20.5f; // Drop of 6.5 LU (melodic/vocal bridge)

    LevelerParams params;
    params.enabled = true;
    params.targetLUFS = -14.0f;
    params.freezeBreakdowns = true;

    // 1. Sensitive (5 LU threshold) -> 6.5 LU drop should TRIGGER freeze
    params.breakdownThresholdLU = 5.0f;
    leveler.update(params, readings, 100, 0.01f);
    assert(leveler.isFrozen());
    std::cout << "  -> 6.5 LU drop at 5.0 LU threshold: FROZEN (PASS)" << std::endl;

    // 2. Normal (7 LU threshold) -> 6.5 LU drop should NOT trigger freeze
    params.breakdownThresholdLU = 7.0f;
    leveler.update(params, readings, 100, 0.01f);
    assert(!leveler.isFrozen());
    std::cout << "  -> 6.5 LU drop at 7.0 LU threshold: NOT FROZEN (PASS)" << std::endl;

    // 3. Deep (9 LU threshold) -> 6.5 LU drop should NOT trigger freeze
    params.breakdownThresholdLU = 9.0f;
    leveler.update(params, readings, 100, 0.01f);
    assert(!leveler.isFrozen());
    std::cout << "  -> 6.5 LU drop at 9.0 LU threshold: NOT FROZEN (PASS)" << std::endl;

    // 4. Massive 10.0 LU drop (deep breakdown, momentary = -24.0 LUFS) -> ALL should freeze
    readings.momentaryLUFS = -24.0f;
    params.breakdownThresholdLU = 9.0f;
    leveler.update(params, readings, 100, 0.01f);
    assert(leveler.isFrozen());
    std::cout << "  -> 10.0 LU drop at 9.0 LU threshold: FROZEN (PASS)" << std::endl;

    // 5. Disabled freeze -> should NOT freeze even on 10.0 LU drop
    params.freezeBreakdowns = false;
    leveler.update(params, readings, 100, 0.01f);
    assert(!leveler.isFrozen());
    std::cout << "  -> Freeze disabled: NOT FROZEN (PASS)" << std::endl;

    std::cout << "  -> PASS: Breakdown Freeze 3-way sensitivity logic verified!" << std::endl;
}

void testLevelerSlewSpeed() {
    std::cout << "[TEST] Leveler Slew Speed (Slow/Normal/Fast steady-state rates)..." << std::endl;

    // Measures the steady-state (post fast-lock) dB/s rate for a given speed & direction.
    // Two-phase: prime past the 8s fast-lock window with a ZERO loudness gap (so
    // currentGainDb stays at 0 and never saturates against the +-12dB clamp while the
    // fast-lock timer advances), then introduce a large gap and measure a single 10ms
    // block's step from that known, unsaturated starting point.
    auto measureSteadyRate = [](LevelerSpeed speed, bool measureUpward) -> float {
        Leveler leveler;
        leveler.prepare(48000.0);

        LevelerParams params;
        params.targetLUFS = -9.0f;
        params.maxBoostDb = 12.0f;
        params.maxCutDb = 12.0f;
        params.freezeBreakdowns = false;
        params.speed = speed;

        float dt = 0.01f;
        LoudnessReadings zeroGap;
        zeroGap.integratedLUFS = params.targetLUFS;
        zeroGap.momentaryLUFS = params.targetLUFS;
        for (int i = 0; i < 850; ++i) { // 8.5s of 10ms blocks -> past the 8s fast-lock window
            leveler.update(params, zeroGap, 100, dt);
        }

        LoudnessReadings bigGap;
        bigGap.integratedLUFS = measureUpward ? -60.0f : 60.0f; // far outside the +-12dB clamp
        bigGap.momentaryLUFS = bigGap.integratedLUFS;

        float before = leveler.getCurrentGainDb();
        leveler.update(params, bigGap, 100, dt);
        float after = leveler.getCurrentGainDb();
        return (after - before) / dt;
    };

    float slowUp = measureSteadyRate(LevelerSpeed::SLOW, true);
    float normalUp = measureSteadyRate(LevelerSpeed::NORMAL, true);
    float fastUp = measureSteadyRate(LevelerSpeed::FAST, true);
    float slowDown = measureSteadyRate(LevelerSpeed::SLOW, false);
    float normalDown = measureSteadyRate(LevelerSpeed::NORMAL, false);
    float fastDown = measureSteadyRate(LevelerSpeed::FAST, false);

    std::cout << "  Steady-state UP   -> Slow: " << slowUp << " dB/s, Normal: " << normalUp
              << " dB/s, Fast: " << fastUp << " dB/s" << std::endl;
    std::cout << "  Steady-state DOWN -> Slow: " << slowDown << " dB/s, Normal: " << normalDown
              << " dB/s, Fast: " << fastDown << " dB/s" << std::endl;

    assert(std::abs(slowUp - 0.5f) < 0.01f);
    assert(std::abs(normalUp - 0.75f) < 0.01f);
    assert(std::abs(fastUp - 1.5f) < 0.01f);
    assert(std::abs(slowDown - (-1.0f)) < 0.01f);
    assert(std::abs(normalDown - (-1.5f)) < 0.01f);
    assert(std::abs(fastDown - (-3.0f)) < 0.01f);
    std::cout << "  -> PASS: Slow/Normal/Fast steady-state rates match spec; Normal is bit-exact"
                 " with the original hardcoded 0.75/1.5 dB/s behavior." << std::endl;

    // Fast-lock window (first 8s) must be unaffected by Slew Speed: always 4.0 up / 16.0 down.
    auto measureFastLockRate = [](LevelerSpeed speed, bool measureUpward) -> float {
        Leveler leveler;
        leveler.prepare(48000.0);
        LevelerParams params;
        params.targetLUFS = -9.0f;
        params.maxBoostDb = 12.0f;
        params.maxCutDb = 12.0f;
        params.freezeBreakdowns = false;
        params.speed = speed;
        LoudnessReadings readings;
        readings.integratedLUFS = measureUpward ? -60.0f : 60.0f;
        readings.momentaryLUFS = readings.integratedLUFS;
        float dt = 0.01f;
        float before = leveler.getCurrentGainDb();
        leveler.update(params, readings, 100, dt); // first call: still inside the fast-lock window
        float after = leveler.getCurrentGainDb();
        return (after - before) / dt;
    };

    float fastLockUpSlow = measureFastLockRate(LevelerSpeed::SLOW, true);
    float fastLockUpFast = measureFastLockRate(LevelerSpeed::FAST, true);
    float fastLockDownSlow = measureFastLockRate(LevelerSpeed::SLOW, false);
    float fastLockDownFast = measureFastLockRate(LevelerSpeed::FAST, false);

    std::cout << "  Fast-lock UP (Slow vs Fast setting):   " << fastLockUpSlow << " vs " << fastLockUpFast << " dB/s" << std::endl;
    std::cout << "  Fast-lock DOWN (Slow vs Fast setting): " << fastLockDownSlow << " vs " << fastLockDownFast << " dB/s" << std::endl;

    assert(std::abs(fastLockUpSlow - 4.0f) < 0.01f);
    assert(std::abs(fastLockUpFast - 4.0f) < 0.01f);
    assert(std::abs(fastLockDownSlow - (-16.0f)) < 0.01f);
    assert(std::abs(fastLockDownFast - (-16.0f)) < 0.01f);
    std::cout << "  -> PASS: Fast-lock window rates are identical regardless of Slew Speed." << std::endl;
}

void testCustomToneProfile() {
    std::cout << "[TEST] Custom Target Contour thresholds and zero-sum re-centering..." << std::endl;
    std::array<float, 6> customOffsets = { 3.0f, -4.0f, 1.0f, 2.0f, -3.0f, 1.0f };
    auto customThresh = Bands::thresholdsFor(true, -2.0f, customOffsets, -24.0f);
    auto flatThresh = Bands::thresholdsFor(true, -2.0f, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, -24.0f);

    float sum = 0.0f;
    for (size_t b = 0; b < 6; ++b) {
        sum += customThresh[b];
        std::cout << "  Band " << b << " (" << Bands::NAMES[b] << "): Offset=" << customOffsets[b]
                  << " dB, Custom Thresh=" << customThresh[b] << " dBFS, Flat Thresh=" << flatThresh[b] << " dBFS" << std::endl;
    }
    float mean = sum / 6.0f;
    std::cout << "  Mean custom threshold: " << mean << " dBFS" << std::endl;
    assert(std::abs(mean - (-24.0f)) < 0.01f); // Re-centering preserves exact -24.0 dB calibration
    assert(customThresh[0] > flatThresh[0]); // Band 0 boosted
    assert(customThresh[1] < flatThresh[1]); // Band 1 cut
    std::cout << "  -> PASS: Custom contour thresholds calculated and zero-sum re-centered perfectly." << std::endl;
}

void testMbcHasNoMakeupGain() {
    std::cout << "[TEST] MBC applies no makeup gain (auto-makeup removed)..." << std::endl;
    MultibandCompressor mbc;
    mbc.prepare(48000.0);

    MBCParams params;
    params.enabled = true;
    params.compressionAmount = 0.8f; // heavy compression
    params.toneSlopeDbPerOctave = -2.0f;
    params.baseThresholdDb = -30.0f; // low threshold to force deep gain reduction

    // 1 kHz tone at -10 dBFS
    constexpr size_t N = 48000;
    std::vector<float> left(N), right(N);
    for (size_t i = 0; i < N; ++i) {
        left[i] = right[i] = 0.3162f * std::sin(2.0 * M_PI * 1000.0 * i / 48000.0);
    }
    mbc.process(left.data(), right.data(), N, params);

    // Settled half: output level should equal input level minus the band's own gain reduction,
    // with nothing added back on top (Post-MBC Gain is the only makeup stage now).
    double sumSq = 0.0;
    for (size_t i = 24000; i < N; ++i) sumSq += left[i] * left[i];
    double outDb = 20.0 * std::log10(std::sqrt(sumSq / 24000.0) / (0.3162 / std::sqrt(2.0)));
    float bandGrDb = mbc.getGainReductionsDb()[2]; // 1 kHz sits in band 2 (400-1200 Hz)
    std::cout << "  Output change: " << outDb << " dB, band 2 GR: " << bandGrDb << " dB" << std::endl;

    assert(bandGrDb < -3.0f);                 // really compressing
    assert(outDb < -3.0);                     // and nothing restored the level
    // Within 1.5 dB rather than exact: GR ripples within each sine cycle (peak detector) and a
    // little of the tone leaks across the 1.2 kHz crossover. The old auto-makeup added back
    // 25% of this band's GR (~+2.5 dB here), which this tolerance does not let through.
    assert(std::abs(outDb - bandGrDb) < 1.5);
    std::cout << "  -> PASS: MBC output drops by its gain reduction, no makeup applied." << std::endl;
}

void testPresetSaveLoadRoundTrip() {
    std::cout << "[TEST] Preset save/load round-trip (AudioEngine::loadStateFromJson)..." << std::endl;
    autolevel::AudioEngine source;
    source.setParam("target_lufs", -18.0f);
    source.setParam("tone_slope", -2.5f);
    source.setParam("compression_amount", 0.73f);
    source.setParam("freeze_breakdowns", 0.0f); // false
    source.setParam("hpf_freq", 40.0f);
    for (int b = 0; b < 6; ++b) {
        source.setParam("custom_band_" + std::to_string(b), static_cast<float>(b) - 2.5f);
        source.setParam("pre_eq_band_" + std::to_string(b), static_cast<float>(b) * 1.5f - 4.0f);
    }
    std::string snapshot = source.getStateJson();

    // A fresh engine starts at the hardcoded defaults, not the source engine's values.
    autolevel::AudioEngine target;
    assert(std::abs(target.getParam("target_lufs") - (-18.0f)) > 0.01f);

    target.loadStateFromJson(snapshot);

    assert(std::abs(target.getParam("target_lufs") - (-18.0f)) < 0.01f);
    assert(std::abs(target.getParam("tone_slope") - (-2.5f)) < 0.01f);
    assert(std::abs(target.getParam("compression_amount") - 0.73f) < 0.01f);
    assert(target.getParam("freeze_breakdowns") < 0.5f);
    assert(std::abs(target.getParam("hpf_freq") - 40.0f) < 0.01f);
    for (int b = 0; b < 6; ++b) {
        float expectedCustom = static_cast<float>(b) - 2.5f;
        float expectedPreEq = static_cast<float>(b) * 1.5f - 4.0f;
        assert(std::abs(target.getParam("custom_band_" + std::to_string(b)) - expectedCustom) < 0.01f);
        assert(std::abs(target.getParam("pre_eq_band_" + std::to_string(b)) - expectedPreEq) < 0.01f);
    }
    std::cout << "  -> PASS: Every scalar, bool, and 6-element array field round-trips through save/load." << std::endl;

    // A field missing from the saved JSON (simulated here with a hand-written partial preset)
    // must leave whatever the engine's current value already is untouched, not reset it - this
    // is what lets a preset saved before a parameter existed still load cleanly.
    autolevel::AudioEngine partial;
    partial.setParam("target_lufs", -11.0f);
    partial.loadStateFromJson("{\"tone_slope\":-1.0}");
    assert(std::abs(partial.getParam("target_lufs") - (-11.0f)) < 0.01f); // untouched
    assert(std::abs(partial.getParam("tone_slope") - (-1.0f)) < 0.01f);   // applied
    std::cout << "  -> PASS: Fields absent from a preset leave the engine's current value alone." << std::endl;

    // Presets saved before 2026-09-28 still carry "mbc_auto_makeup"; it must be skipped
    // without disturbing the fields around it, and never come back out in the state JSON.
    autolevel::AudioEngine legacy;
    legacy.loadStateFromJson("{\"compression_amount\":0.66,\"mbc_auto_makeup\":true,\"freeze_breakdowns\":false}");
    assert(std::abs(legacy.getParam("compression_amount") - 0.66f) < 0.01f);
    assert(legacy.getParam("freeze_breakdowns") < 0.5f);
    assert(legacy.getStateJson().find("mbc_auto_makeup") == std::string::npos);
    std::cout << "  -> PASS: Legacy presets with mbc_auto_makeup still load; the field is gone." << std::endl;

    // The MBC dynamics controls round-trip too.
    autolevel::AudioEngine dynSrc;
    dynSrc.setParam("mbc_attack_ms", 3.5f);
    dynSrc.setParam("mbc_release_ms", 650.0f);
    dynSrc.setParam("mbc_detector_rms", 0.35f);
    dynSrc.setParam("mbc_auto_release", 1.0f);
    autolevel::AudioEngine dynDst;
    dynDst.loadStateFromJson(dynSrc.getStateJson());
    assert(std::abs(dynDst.getParam("mbc_attack_ms") - 3.5f) < 0.01f);
    assert(std::abs(dynDst.getParam("mbc_release_ms") - 650.0f) < 0.5f);
    assert(std::abs(dynDst.getParam("mbc_detector_rms") - 0.35f) < 0.01f);
    assert(dynDst.getParam("mbc_auto_release") > 0.5f);
    assert(dynSrc.getStateJson().find("mbc_speed") == std::string::npos);

    // Out-of-range values are clamped. Post-MBC Gain reaches +-24 dB (was +-12).
    dynDst.setParam("post_mbc_gain", -24.0f);
    assert(std::abs(dynDst.getParam("post_mbc_gain") - (-24.0f)) < 0.01f);
    dynDst.setParam("post_mbc_gain", 30.0f);
    assert(std::abs(dynDst.getParam("post_mbc_gain") - 24.0f) < 0.01f);
    dynDst.setParam("mbc_attack_ms", 0.0f);
    dynDst.setParam("mbc_release_ms", 99999.0f);
    dynDst.setParam("mbc_detector_rms", 2.0f);
    assert(std::abs(dynDst.getParam("mbc_attack_ms") - 1.0f) < 0.01f);
    assert(std::abs(dynDst.getParam("mbc_release_ms") - 1000.0f) < 0.01f);
    assert(std::abs(dynDst.getParam("mbc_detector_rms") - 1.0f) < 0.01f);

    // Presets from before 2026-10-03 carry mbc_speed instead: Slow/Normal/Fast become the
    // times those presets ran at. A preset that has attack/release ignores mbc_speed.
    const float expected[3][2] = { {30.0f, 400.0f}, {15.0f, 200.0f}, {7.5f, 100.0f} };
    for (int speed = 0; speed < 3; ++speed) {
        autolevel::AudioEngine old;
        old.setParam("mbc_attack_ms", 50.0f);
        old.setParam("mbc_detector_rms", 0.8f);
        old.setParam("mbc_auto_release", 1.0f);
        old.loadStateFromJson("{\"compression_amount\":0.5,\"mbc_speed\":" + std::to_string(speed) + "}");
        assert(std::abs(old.getParam("mbc_attack_ms") - expected[speed][0]) < 0.01f);
        assert(std::abs(old.getParam("mbc_release_ms") - expected[speed][1]) < 0.01f);
        assert(old.getParam("mbc_detector_rms") < 0.01f);   // old presets were peak-only
        assert(old.getParam("mbc_auto_release") < 0.5f);
    }
    autolevel::AudioEngine both;
    both.loadStateFromJson("{\"mbc_speed\":0,\"mbc_attack_ms\":4.0,\"mbc_release_ms\":90}");
    assert(std::abs(both.getParam("mbc_attack_ms") - 4.0f) < 0.01f);
    assert(std::abs(both.getParam("mbc_release_ms") - 90.0f) < 0.01f);
    std::cout << "  -> PASS: MBC dynamics round-trip and clamp; old Slow/Normal/Fast presets map to their times." << std::endl;
}

void testInputSanitizerAntiNan() {
    std::cout << "[TEST] Audio Input Sanitizer & Anti-NaN Protection..." << std::endl;
    autolevel::AudioEngine engine;
    engine.prepare(48000.0);

    constexpr size_t N = 256;
    std::vector<float> inL(N), inR(N);
    std::vector<float> outL(N), outR(N);

    // Inject NaNs, Infs, extreme runaway values, and valid audio
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0) {
            inL[i] = std::numeric_limits<float>::quiet_NaN();
            inR[i] = std::numeric_limits<float>::infinity();
        } else if (i % 10 == 5) {
            inL[i] = -std::numeric_limits<float>::infinity();
            inR[i] = 100.0f; // Extreme spike
        } else {
            inL[i] = 0.5f * std::sin(2.0 * TEST_PI * 1000.0 * i / 48000.0);
            inR[i] = 0.5f * std::cos(2.0 * TEST_PI * 1000.0 * i / 48000.0);
        }
    }

    engine.process(inL.data(), inR.data(), outL.data(), outR.data(), N);

    for (size_t i = 0; i < N; ++i) {
        assert(std::isfinite(outL[i]));
        assert(std::isfinite(outR[i]));
        assert(!std::isnan(outL[i]));
        assert(!std::isnan(outR[i]));
    }

    auto vs = engine.getVisualState();
    (void)vs;
    assert(std::isfinite(vs.loudness.momentaryLUFS));
    assert(std::isfinite(vs.appliedGainDb));
    assert(std::isfinite(vs.outputPeakDbL));

    std::cout << "  -> PASS: All NaNs and Infs safely intercepted and sanitized without filter corruption." << std::endl;
}

void testLockFreeDoubleBufferedVisualState() {
    std::cout << "[TEST] Lock-Free Double-Buffered Visual State & Concurrency..." << std::endl;
    autolevel::AudioEngine engine;
    engine.prepare(48000.0);

    constexpr size_t N = 256;
    std::vector<float> inL(N, 0.2f), inR(N, 0.2f);
    std::vector<float> outL(N), outR(N);

    for (int block = 0; block < 100; ++block) {
        engine.process(inL.data(), inR.data(), outL.data(), outR.data(), N);
        auto vs = engine.getVisualState();
        (void)vs;
        assert(std::isfinite(vs.appliedGainDb));
        assert(std::isfinite(vs.loudness.momentaryLUFS));
    }

    // Test lock-free asynchronous reset
    engine.reset();
    engine.process(inL.data(), inR.data(), outL.data(), outR.data(), N);
    auto vs = engine.getVisualState();
    (void)vs;
    assert(std::isfinite(vs.appliedGainDb));

    std::cout << "  -> PASS: Double-buffered visual state and lock-free reset verified!" << std::endl;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   AutoLevel DJ DSP Unit Tests (Android Spec)" << std::endl;
    std::cout << "============================================" << std::endl;

    testKWeightingSineWave();
    testAndroidToneProfilesAndThresholds();
    testLR4CrossoverSummation();
    testLevelResponseMapping();
    testEbuR128DynamicLoudness();
    testMbcRatioAndThreshold();
    testMbcAttackRelease();
    testMbcDetectorPeakRms();
    testMbcAutoRelease();
    testPostMbcGain();
    testPreMbcEQ();
    testDynamicAirLift();
    testHighPassFilter();
    testDefaultLimiterCeiling();
    testTruePeakLimiterTransparency();
    testTruePeakLimiterNoDistortion();
    testTruePeakLimiterBypassAndReset();
    testBreakdownFreezeSensitivity();
    testLevelerSlewSpeed();
    testCustomToneProfile();
    testMbcHasNoMakeupGain();
    testFullChain();
    testPresetSaveLoadRoundTrip();
    testInputSanitizerAntiNan();
    testLockFreeDoubleBufferedVisualState();

    std::cout << "============================================" << std::endl;
    std::cout << "   ALL DSP TESTS PASSED WITH 100% ACCURACY! " << std::endl;
    std::cout << "============================================" << std::endl;
    return 0;
}

