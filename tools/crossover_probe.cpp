// Root-cause: is it the biquad math, or the TREE structure (missing allpass compensation)?
#include "Biquad.h"
#include "MultibandCompressor.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace autolevel::dsp;
static const double TPI = 3.14159265358979323846;

// Control: a PLAIN 2-way LR4 split at 1200 Hz, summed back.
static double twoWay(double f) {
    double fs = 48000.0;
    LR4Filter lp, hp;
    lp.setup(Biquad::Type::LowPass, 1200.0, fs);
    hp.setup(Biquad::Type::HighPass, 1200.0, fs);
    size_t n = 4800; double si=0, so=0; size_t c=0;
    for (size_t i = 0; i < n; ++i) {
        double s = std::sin(2.0*TPI*f*((double)i/fs));
        double out = lp.process(s) + hp.process(s);
        if (i >= 2800) { si += s*s; so += out*out; ++c; }
    }
    return 20.0*std::log10(std::sqrt(so/c)/std::sqrt(si/c));
}

static double sixBand(double f, float amount) {
    MultibandCompressor mbc; double fs = 48000.0; mbc.prepare(fs);
    MBCParams p; p.enabled=true; p.compressionAmount=amount; p.baseThresholdDb=60.0f;
    p.toneSlopeDbPerOctave=0.0f;
    size_t n=4800; std::vector<float> a(n), b(n), o(n);
    for (size_t i=0;i<n;++i){ float s=(float)std::sin(2.0*TPI*f*((double)i/fs)); a[i]=b[i]=o[i]=s; }
    mbc.process(a.data(), b.data(), n, p);
    double si=0,so=0; size_t c=0;
    for (size_t i=2800;i<n;++i){ si+=o[i]*o[i]; so+=a[i]*a[i]; ++c; }
    return 20.0*std::log10(std::sqrt(so/c)/std::sqrt(si/c));
}

int main() {
    printf("CONTROL: plain 2-way LR4 split at 1200 Hz, LP+HP summed\n");
    printf("  (LR4 theory: LP+HP sums to unity magnitude -> should be ~0.00 dB)\n");
    for (double f : {600.0, 1200.0, 2400.0})
        printf("    %6.0f Hz : %+7.3f dB\n", f, twoWay(f));

    printf("\nSAME 1200 Hz crossover, but inside the 6-band TREE:\n");
    for (double f : {600.0, 1200.0, 2400.0})
        printf("    %6.0f Hz : %+7.3f dB\n", f, sixBand(f, 0.02f));

    printf("\nAt the SHIPPED DEFAULT compressionAmount = 0.50:\n");
    for (double f : {400.0, 800.0, 1200.0, 1800.0, 3500.0})
        printf("    %6.0f Hz : %+7.3f dB\n", f, sixBand(f, 0.50f));
    return 0;
}
