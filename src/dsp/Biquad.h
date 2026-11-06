#pragma once

#include <cmath>
#include <array>

namespace autolevel::dsp {

inline constexpr double PI = 3.14159265358979323846;

/**
 * 2nd-order Biquad filter (Direct Form II Transposed)
 * Provides Butterworth Low-pass and High-pass filters with Q = 0.70710678 (1/sqrt(2)).
 * Cascading two identical Butterworth filters yields a 4th-order Linkwitz-Riley (LR4) filter.
 */
class Biquad {
public:
    enum class Type { LowPass, HighPass, Allpass };

    Biquad() = default;

    void setup(Type type, double cutoffHz, double sampleRate) {
        double omega = 2.0 * PI * cutoffHz / sampleRate;
        double sn = std::sin(omega);
        double cs = std::cos(omega);
        double alpha = sn / (2.0 * 0.7071067811865475); // Q = 1/sqrt(2)

        double a0 = 1.0 + alpha;

        if (type == Type::LowPass) {
            m_b[0] = ((1.0 - cs) / 2.0) / a0;
            m_b[1] = (1.0 - cs) / a0;
            m_b[2] = ((1.0 - cs) / 2.0) / a0;
        } else if (type == Type::HighPass) {
            m_b[0] = ((1.0 + cs) / 2.0) / a0;
            m_b[1] = -(1.0 + cs) / a0;
            m_b[2] = ((1.0 + cs) / 2.0) / a0;
        } else { // Allpass
            // Same pole pair as the Butterworth sections above, so this is exactly
            // the response an LR4 low/high pair sums to (see AllpassLR4 below).
            m_b[0] = (1.0 - alpha) / a0;
            m_b[1] = (-2.0 * cs) / a0;
            m_b[2] = 1.0;
        }

        m_a[1] = (-2.0 * cs) / a0;
        m_a[2] = (1.0 - alpha) / a0;

        reset();
    }

    /**
     * RBJ cookbook peaking (bell) EQ: unity gain outside the band, +-gainDb at centerHz,
     * shaped by q (higher q = narrower bell). Unlike setup() above, this needs a gain
     * parameter, so it is a separate entry point rather than a third Type case.
     */
    void setupPeaking(double centerHz, double gainDb, double q, double sampleRate) {
        double A = std::pow(10.0, gainDb / 40.0);
        double omega = 2.0 * PI * centerHz / sampleRate;
        double sn = std::sin(omega);
        double cs = std::cos(omega);
        double alpha = sn / (2.0 * q);

        double a0 = 1.0 + alpha / A;
        m_b[0] = (1.0 + alpha * A) / a0;
        m_b[1] = (-2.0 * cs) / a0;
        m_b[2] = (1.0 - alpha * A) / a0;
        m_a[1] = (-2.0 * cs) / a0;
        m_a[2] = (1.0 - alpha / A) / a0;

        reset();
    }

    void reset() {
        m_z1 = 0.0;
        m_z2 = 0.0;
    }

    inline double process(double in) noexcept {
        double out = m_b[0] * in + m_z1;
        m_z1 = m_b[1] * in - m_a[1] * out + m_z2;
        m_z2 = m_b[2] * in - m_a[2] * out;
        return out;
    }

private:
    std::array<double, 3> m_b{1.0, 0.0, 0.0};
    std::array<double, 3> m_a{1.0, 0.0, 0.0};
    double m_z1 = 0.0;
    double m_z2 = 0.0;
};

/**
 * The allpass an LR4 low/high pair sums to.
 *
 * With LP = B^2 and HP = (1-B)^2 built from the same 2nd-order Butterworth
 * section, LP + HP is not unity - it is a 2nd-order allpass at the same cutoff:
 *
 *     1/(s^2+r2*s+1)^2 + s^4/(s^2+r2*s+1)^2
 *       = (1 + s^4)/(s^2+r2*s+1)^2
 *       = (s^2-r2*s+1)/(s^2+r2*s+1)        [since 1+s^4 factors into the two]
 *
 * A tree of LR4 splits therefore only reconstructs flat if every band is
 * corrected by the allpasses of the splits its own path did not pass through.
 */
class AllpassLR4 {
public:
    AllpassLR4() = default;
    void setup(double cutoffHz, double sampleRate) { m_biquad.setup(Biquad::Type::Allpass, cutoffHz, sampleRate); }
    void reset() { m_biquad.reset(); }
    inline double process(double in) noexcept { return m_biquad.process(in); }
private:
    Biquad m_biquad;
};

/**
 * 4th-order Linkwitz-Riley filter (LR4)
 * Cascades two 2nd-order Butterworth filters.
 */
class LR4Filter {
public:
    LR4Filter() = default;

    void setup(Biquad::Type type, double cutoffHz, double sampleRate) {
        m_biquad1.setup(type, cutoffHz, sampleRate);
        m_biquad2.setup(type, cutoffHz, sampleRate);
    }

    void reset() {
        m_biquad1.reset();
        m_biquad2.reset();
    }

    inline double process(double in) noexcept {
        return m_biquad2.process(m_biquad1.process(in));
    }

private:
    Biquad m_biquad1;
    Biquad m_biquad2;
};

} // namespace autolevel::dsp
