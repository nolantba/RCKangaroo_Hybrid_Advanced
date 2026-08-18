// =============================================================================
// lissajous_validate.cpp  —  empirical comparison of jump magnitude generators
//
// Builds NO production code paths.  Standalone CPU tool to compare what a
// kangaroo would actually see when masking `X & (JMP_CNT - 1)` across:
//   - Teske uniform 512  (hypothetical: what you'd have if defs.h was fixed)
//   - Teske bimodal 768  (what your CURRENT build actually does)
//   - Lissajous variants in 512 uniform mode
//   - Lissajous in BIMODAL_768 mode (the interesting candidate)
//   - Lissajous in LEVY_TAIL mode (experimental)
//
// Build:
//   g++ -O3 -std=c++17 lissajous_validate.cpp -o lissajous_validate
//
// Run:
//   ./lissajous_validate              # default range_bits=85, n=200000
//   ./lissajous_validate 135 1000000
//
// Read the output:
//   - lag1_autocorr near 0   = no sequence memory (good)
//   - large |lag1_autocorr|  = autocorrelated => collides slower (BAD)
//   - chi2 small             = uniform coverage of magnitude range
//   - chi2 large             = clustering (only OK if intentional bimodal)
//   - bimodality_score       = how far below uniform the middle bin is.
//                              0 = uniform.  >0.3 = clearly bimodal.
//
// What "winning" looks like:
//   A Lissajous variant should match or beat Teske bimodal-768 on chi2 AND
//   stay near 0 on lag1.  If Lissajous shows |lag1| > 0.05 OR materially
//   higher chi2, do NOT wire it into a kangaroo build — it will collide
//   slower than the current Teske bimodal-768.
// =============================================================================

#include "LissajousJumps.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

// Bimodality detector: compares middle bins to outer bins.
// In a bimodal distribution, the middle of the magnitude range has
// noticeably lower density than the extremes.
static double bimodality_score(const std::vector<uint64_t>& seq, int n_bins = 64) {
    if (seq.empty()) return 0.0;
    long double mn = seq.front(), mx = mn;
    for (uint64_t v : seq) {
        if ((long double)v < mn) mn = v;
        if ((long double)v > mx) mx = v;
    }
    long double range = mx - mn;
    if (range <= 0) return 0.0;

    std::vector<size_t> bins(n_bins, 0);
    for (uint64_t v : seq) {
        int b = (int)(((long double)v - mn) * n_bins / range);
        if (b < 0) b = 0;
        if (b >= n_bins) b = n_bins - 1;
        bins[b]++;
    }

    // Average outer-quartile density vs middle-half density.
    size_t outer = 0, middle = 0;
    int q = n_bins / 4;
    for (int i = 0; i < q; i++)              outer  += bins[i];
    for (int i = n_bins - q; i < n_bins; i++) outer += bins[i];
    for (int i = q; i < n_bins - q; i++)     middle += bins[i];

    double outer_density  = (double)outer  / (2.0 * q);
    double middle_density = (double)middle / (n_bins - 2.0 * q);
    if (outer_density <= 0.0) return 0.0;
    double score = (outer_density - middle_density) / outer_density;
    if (score < 0.0) score = 0.0;
    return score;
}

static void print_stats(const char* label, const LissajousJumpGenerator::Stats& s,
                        double bm_score) {
    printf("  %-30s\n", label);
    printf("      mean    = %.6e   std_dev = %.6e\n", s.mean, s.std_dev);
    printf("      min     = %.6e   max     = %.6e\n", s.min,  s.max);
    printf("      lag1    = %+8.5f   chi2    = %10.3f   bimodality = %.3f\n",
           s.lag1_autocorr, s.partition_uniformity_chi2, bm_score);
}

static void run_lissajous(const char* label,
                          LissajousJumpGenerator::Config c,
                          size_t n_samples,
                          uint64_t seed) {
    LissajousJumpGenerator gen(c);
    if (!gen.is_initialized()) {
        printf("  [%s] init failed\n", label);
        return;
    }
    auto seq    = gen.materialize_kangaroo_view(n_samples, seed);
    auto stats  = LissajousJumpGenerator::compute_stats(seq);
    double bm   = bimodality_score(seq);
    print_stats(label, stats, bm);
}

int main(int argc, char** argv) {
    int    range_bits = (argc > 1) ? std::atoi(argv[1]) : 85;
    size_t n_samples  = (argc > 2) ? (size_t)std::atoll(argv[2]) : 200000;
    const uint64_t SEED = 0xC0FFEEFEEDFACEEDULL;

    printf("\n======================================================================\n");
    printf(" Magnitude generator comparison\n");
    printf("   range_bits = %d   samples = %zu\n", range_bits, n_samples);
    printf("   Sampling = kangaroo-realistic: random X[0], jmp_ind = X & (size-1)\n");
    printf("   Seed     = 0x%016llx (deterministic)\n", (unsigned long long)SEED);
    printf("======================================================================\n\n");

    // References
    printf("  -- References (Teske random) --\n");
    {
        auto seq   = LissajousJumpGenerator::teske_reference(n_samples, range_bits, SEED);
        auto stats = LissajousJumpGenerator::compute_stats(seq);
        double bm  = bimodality_score(seq);
        print_stats("Teske uniform-512 (hypothetical)", stats, bm);
    }
    printf("\n");
    {
        auto seq   = LissajousJumpGenerator::teske_bimodal_768(n_samples, range_bits, SEED);
        auto stats = LissajousJumpGenerator::compute_stats(seq);
        double bm  = bimodality_score(seq);
        print_stats("Teske bimodal-768 (CURRENT build)", stats, bm);
    }
    printf("\n");

    // Lissajous uniform 512
    printf("  -- Lissajous UNIFORM-512 (vs Teske-512) --\n");
    {
        auto c = LissajousJumpGenerator::classic_for_range(range_bits);
        run_lissajous("Lissajous Classic 512", c, n_samples, SEED);
    }
    {
        auto c = LissajousJumpGenerator::harmonograph_for_range(range_bits);
        run_lissajous("Lissajous Damped 512", c, n_samples, SEED);
    }
    {
        auto c = LissajousJumpGenerator::chaotic_for_range(range_bits, SEED);
        run_lissajous("Lissajous Chaotic 512", c, n_samples, SEED);
    }
    printf("\n");

    // Lissajous bimodal 768 - the real fight
    printf("  -- Lissajous BIMODAL-768 (vs Teske bimodal-768) --\n");
    {
        auto c = LissajousJumpGenerator::bimodal_768_for_range(range_bits);
        c.pattern_type = LissajousJumpGenerator::Config::CLASSIC_LISSAJOUS;
        run_lissajous("Lissajous Classic Bimodal-768", c, n_samples, SEED);
    }
    {
        auto c = LissajousJumpGenerator::bimodal_768_for_range(range_bits);
        c.pattern_type = LissajousJumpGenerator::Config::DAMPED_HARMONOGRAPH;
        c.damping_x = 1e-4; c.damping_y = 1.5e-4; c.damping_z = 2e-4;
        run_lissajous("Lissajous Damped Bimodal-768", c, n_samples, SEED);
    }
    {
        auto c = LissajousJumpGenerator::bimodal_768_for_range(range_bits);
        c.pattern_type = LissajousJumpGenerator::Config::CHAOTIC_MIX;
        run_lissajous("Lissajous Chaotic Bimodal-768", c, n_samples, SEED);
    }
    printf("\n");

    // Lissajous Levy
    printf("  -- Lissajous LEVY-TAIL 512 (experimental) --\n");
    {
        auto c = LissajousJumpGenerator::levy_for_range(range_bits);
        run_lissajous("Lissajous Classic Levy-512", c, n_samples, SEED);
    }
    printf("\n");

    printf("======================================================================\n");
    printf(" Verdict guide:\n");
    printf("   1. Compare each Lissajous Bimodal-768 against Teske bimodal-768.\n");
    printf("      Need lag1 <= Teske AND chi2 <= Teske to be a candidate.\n");
    printf("   2. Bimodality scores between 768-variants should be similar.\n");
    printf("   3. Stats win is gating only.  Real test = 200-trial K-factor on puzzle 65.\n");
    printf("======================================================================\n\n");
    return 0;
}
