// ============================================================================
// R2_JumpTable.h  —  Option A: drop-in R2 quasi-random jump table builder
// No kernel changes required. Replaces random distribution for J1/J2/J3
// with R2-sequence (Roberts sequence) ordered magnitudes.
//
// R2 star-discrepancy ~0.21 vs uniform-random ~O(1/sqrt(N))
// — better coverage of the magnitude range with the same 512 entries.
//
// Usage (in RCKangaroo.cpp, replace J2/J3 loops and optionally J1):
//   BuildR2JumpTable(EcJumps2, JMP_CNT, minjump, ec, 1);
//   BuildR2JumpTable(EcJumps3, JMP_CNT, minjump, ec, 2);
// ============================================================================

#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <ctime>

// R2 / golden ratio constants
static constexpr double R2_ALPHA = 0.6180339887498948482045868343656; // (sqrt(5)-1)/2 = 1/phi

// ============================================================================
// Option B: 4-variant R2 jump table for J1
// Layout: jumps[jmp_ind * 4 + variant]  (interleaved — 2048 total entries)
// GPU selects variant = (x[0] >> 9) & 3 at each jump (zero-cost, bits free in JMP_MASK)
// All variants share the same Teske-optimal mean mu = 2^(Range/2)
// ============================================================================

// Halton sequence: radical-inverse base `base`, n-th term in (0,1)
static inline double R2_Halton(int n, int base)
{
    double f = 1.0, r = 0.0;
    int i = n;
    while (i > 0) {
        f /= base;
        r += f * (i % base);
        i /= base;
    }
    return r;
}

// Scrambled Halton: Owen-style digit permutation per depth level.
// The same permutation is applied to all n at the same digit depth
// (salt advances once per depth) — preserves the (t,s)-net property.
// Scramble: cyclic digit shift  digit' = (digit + shift_d) mod base
static inline double R2_Halton_Scrambled(int n, int base, uint32_t salt)
{
    double   f   = 1.0, r = 0.0;
    int      i   = n;
    uint32_t lcg = salt;
    while (i > 0) {
        f   /= base;
        lcg  = lcg * 1664525u + 1013904223u;          // one step per digit depth
        int shift = (int)(lcg % (uint32_t)base);
        int digit = ((i % base) + shift) % base;      // cyclic shift in {0..base-1}
        r  += f * digit;
        i  /= base;
    }
    return r;
}

// Fill cnt entries for one variant into jumps[i*4+variant]
// mu     : Teske-optimal mean jump (2^(Range/2)), treated as scale reference
// lo, hi : fractional bounds relative to mu — jump in [lo*mu, hi*mu)
// fracs  : cnt values in [0,1) (low-discrepancy sequence)
static inline void R2_FillVariant(EcJMP* jumps, int cnt, int variant,
                                   EcInt& mu, double lo, double hi,
                                   const std::vector<double>& fracs, Ec& ec)
{
    // Find highest non-zero limb of mu for scaling
    int top = 3;
    while (top > 0 && mu.data[top] == 0) top--;

    double mu_hi = (top > 0) ? (double)mu.data[top] : (double)mu.data[0];
    double mu_lo = (top > 1 && mu.data[top - 1] != 0) ? (double)mu.data[top - 1] : 0.0;

    for (int i = 0; i < cnt; i++) {
        double t = lo + (hi - lo) * fracs[i];  // t in [lo, hi)

        EcInt dist;
        dist.SetZero();

        if (top == 0) {
            dist.data[0] = (uint64_t)(t * (double)mu.data[0]);
        } else {
            double scaled_hi = t * mu_hi;
            uint64_t hi_part = (uint64_t)scaled_hi;
            double   rem     = scaled_hi - (double)hi_part;
            uint64_t lo_part = (uint64_t)(rem * 18446744073709551616.0);
            if (mu_lo > 0.0)
                lo_part += (uint64_t)(t * mu_lo);
            dist.data[top]     = hi_part;
            dist.data[top - 1] = lo_part;
        }
        dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL; // must be even

        int entry = i * 4 + variant;
        jumps[entry].dist = dist;
        jumps[entry].p    = ec.MultiplyG_Lambda(jumps[entry].dist);
    }
}

// Build 4-variant R2 jump table — 4*cnt entries, layout jumps[jmp_ind*4+variant]
// mu = 2^(Range/2) = Teske-optimal mean (same as minjump = 2^jump1_exp)
//
// Per-run scrambling ensures each launch explores an independent sub-region:
//   V0  phi-R2    [0.50mu, 1.50mu)  random-shift-mod-1  (RQMC standard)
//   V1  psi-R2    [0.33mu, 1.67mu)  random-shift-mod-1
//   V2  H(5) scr  [0.25mu, 1.75mu)  Owen cyclic-digit scramble
//   V3  H(7) scr  [0.40mu, 1.60mu)  Owen cyclic-digit scramble
//
// All variants: mean=mu, all net-structure and low-discrepancy properties preserved.
static inline void BuildR2JumpTable4V(EcJMP* jumps, int cnt, EcInt& mu, Ec& ec)
{
    // ── Per-run salt: time (seconds) XOR clock ticks (sub-second variation) ──
    // clock() varies even across runs in the same second, giving unique seeds
    // for puzzle-75 batch loops where runs complete in <10 s each.
    uint32_t salt = (uint32_t)time(nullptr)
                  ^ (uint32_t)((uintptr_t)jumps >> 3)
                  ^ (uint32_t)clock();

    // Derive four independent per-variant values via LCG cascade
    uint32_t s0 = salt  * 1664525u + 1013904223u;
    uint32_t s1 = s0    * 1664525u + 1013904223u;
    uint32_t s2 = s1    * 1664525u + 1013904223u;
    uint32_t s3 = s2    * 1664525u + 1013904223u;

    // Random offsets in [0,1) for R2 shift scrambling (V0, V1)
    double off0 = (double)s0 * (1.0 / 4294967296.0);
    double off1 = (double)s1 * (1.0 / 4294967296.0);

    // Export to globals so GpuMonitor can display them live
    extern uint32_t g_r2_salt;
    extern double   g_r2_off0, g_r2_off1;
    g_r2_salt = salt;
    g_r2_off0 = off0;
    g_r2_off1 = off1;

    std::vector<double> fracs(cnt);

    // V0: golden-ratio R2 (phi) + random shift mod 1
    static constexpr double R2_PSI = 0.56984029099805326591;
    for (int i = 0; i < cnt; i++)
        fracs[i] = fmod((i + 1) * R2_ALPHA + off0, 1.0);
    R2_FillVariant(jumps, cnt, 0, mu, 0.50, 1.50, fracs, ec);

    // V1: super-golden R2 (psi) + random shift mod 1
    for (int i = 0; i < cnt; i++)
        fracs[i] = fmod((i + 1) * R2_PSI + off1, 1.0);
    R2_FillVariant(jumps, cnt, 1, mu, 1.0/3.0, 5.0/3.0, fracs, ec);

    // V2: Scrambled Halton base-5 (Owen cyclic-digit per depth)
    for (int i = 0; i < cnt; i++)
        fracs[i] = R2_Halton_Scrambled(i + 1, 5, s2);
    R2_FillVariant(jumps, cnt, 2, mu, 0.25, 1.75, fracs, ec);

    // V3: Scrambled Halton base-7 (Owen cyclic-digit per depth)
    for (int i = 0; i < cnt; i++)
        fracs[i] = R2_Halton_Scrambled(i + 1, 7, s3);
    R2_FillVariant(jumps, cnt, 3, mu, 0.40, 1.60, fracs, ec);
}

// ── Build a 512-entry jump table using R2-ordered magnitudes ─────────────────
// minjump : base magnitude (2^k), same as existing code sets it
// ec      : Secp256K1& for MultiplyG_Lambda
// table_id: 0=J1, 1=J2, 2=J3 — decorrelates tables from each other
// ─────────────────────────────────────────────────────────────────────────────
static inline void BuildR2JumpTable(EcJMP* jumps, int cnt,
                                    EcInt& minjump, Ec& ec,
                                    int table_id)
{
    // Step 1: generate R2 fractions for all slots
    // frac[i] = fractional part of (i+1+table_id*137) * R2_ALPHA
    // table_id offset decorrelates J1/J2/J3 from each other
    struct Slot { int idx; double frac; };
    std::vector<Slot> slots(cnt);
    for (int i = 0; i < cnt; i++) {
        double seq = (double)(i + 1 + table_id * 137);
        slots[i] = { i, fmod(seq * R2_ALPHA, 1.0) };
    }

    // Step 2: sort by R2 value — this gives low-discrepancy slot ordering
    std::sort(slots.begin(), slots.end(),
              [](const Slot& a, const Slot& b){ return a.frac < b.frac; });

    // Step 3: assign magnitudes — evenly spaced across [minjump, 2*minjump)
    // in R2-sorted order so adjacent slots get well-spread magnitudes
    for (int i = 0; i < cnt; i++) {
        int s = slots[i].idx;

        // Evenly divide [0, minjump) into cnt steps, use step i
        // offset = minjump * i / cnt  (integer arithmetic via shift)
        // We approximate: offset64 = (uint64_t)((double)i/cnt * 2^63) then
        // inject into data[0] of a scaled-down copy of minjump.
        // Simpler: use the fraction directly on data[0] of minjump.

        EcInt offset;
        offset.SetZero();

        // Get minjump.data[0] as the scaling word (the lowest non-zero limb)
        // Find the highest set limb of minjump
        int top_limb = 3;
        while (top_limb > 0 && minjump.data[top_limb] == 0)
            top_limb--;

        if (top_limb == 0) {
            // minjump fits in data[0] — scale directly
            double span = (double)minjump.data[0];
            offset.data[0] = (uint64_t)(span * (double)i / (double)cnt);
        } else {
            // Use top two limbs for precision
            double span_hi = (double)minjump.data[top_limb];
            double span_lo = (double)minjump.data[top_limb - 1];
            double frac_i  = (double)i / (double)cnt;
            double scaled  = frac_i * span_hi;
            uint64_t hi_part = (uint64_t)scaled;
            double   remainder = scaled - (double)hi_part;
            uint64_t lo_part = (uint64_t)(remainder * 18446744073709551616.0
                               + frac_i * span_lo);

            offset.data[top_limb]     = hi_part;
            offset.data[top_limb - 1] = lo_part;
        }
        offset.data[0] &= 0xFFFFFFFFFFFFFFFEULL; // keep even

        jumps[s].dist = minjump;
        jumps[s].dist.Add(offset);
        jumps[s].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL; // must be even
        jumps[s].p = ec.MultiplyG_Lambda(jumps[s].dist);
    }
}
