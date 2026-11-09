// zone_registry.h
// Dynamic zone registry — works for ANY Bitcoin puzzle
// Include after defs.h

#pragma once
#include "defs.h"
#include "Ec.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define ZONE_COUNT    100
#define ZONE_MAGIC    0x5A4F4E45u
#define ZONE_VERSION  3
#define ZONE_REG_FILE "zone_registry.bin"
#define ZONE_REG_TXT  "zone_registry.txt"

#pragma pack(push,1)
struct ZoneEntry {
    u64  range_start[4];    // zone start, little-endian u64[4]
    u64  range_end[4];      // zone end
    u64  dps_collected;     // total DPs accumulated
    u64  total_secs;        // total GPU seconds spent here
    u32  sessions;          // times searched
    u64  last_searched;     // unix timestamp
    char workfile[64];      // "zone_082.work"
    u8   pad[20];           // → 176 bytes (was pad[4]/160B -- fixed, see static_assert below)
};

struct ZoneRegistryHeader {
    u32  magic;
    u32  version;
    u32  puzzle_num;        // 80, 90, 100, 140 etc
    u32  range_bits;        // puzzle range (e.g. 140)
    u32  zone_count;        // 100
    u64  total_dps;
    u64  created;
    u64  last_updated;
    u8   pubkey[33];
    u8   solved;
    u8   solved_zone;
    u64  solved_key[4];
    u8   pad[145];          // → 256 bytes (was pad[138]/249B -- fixed, see static_assert below)
};
#pragma pack(pop)

static_assert(sizeof(ZoneEntry)          == 176, "ZoneEntry 176 bytes");
static_assert(sizeof(ZoneRegistryHeader) == 256, "Header 256 bytes");

// ============================================================
// Minimal unsigned 256-bit helpers (4x u64, little-endian limbs).
// Used only for zone boundary math in ZoneRegistry::Init() below —
// deliberately small/explicit rather than relying on unsigned __int128,
// whose shift-by->=128 behaviour is undefined and previously caused
// zone boundaries to silently come out as zero for puzzles >= 129 bits.
// ============================================================
static inline void u256_zero(u64* v) { v[0] = v[1] = v[2] = v[3] = 0; }

static inline void u256_set_bit(u64* v, int bit) {
    u256_zero(v);
    if (bit < 0 || bit > 255) return;
    v[bit / 64] |= (1ULL << (bit % 64));
}

// v *= m  (m assumed small, e.g. <= ZONE_COUNT; overflow beyond 256 bits
// is not possible for our use since range_start < 2^180 and m <= 100)
static inline void u256_mul_small(u64* v, u32 m) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 p = (unsigned __int128)v[i] * (unsigned __int128)m + carry;
        v[i]  = (u64)p;
        carry = p >> 64;
    }
}

// v /= d (integer division, small divisor, e.g. ZONE_COUNT)
static inline void u256_div_small(u64* v, u32 d) {
    unsigned __int128 rem = 0;
    for (int i = 3; i >= 0; i--) {
        unsigned __int128 cur = (rem << 64) | v[i];
        v[i] = (u64)(cur / d);
        rem  = cur % d;
    }
}

// out = a + b
static inline void u256_add(u64* out, const u64* a, const u64* b) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)a[i] + b[i] + carry;
        out[i] = (u64)s;
        carry  = s >> 64;
    }
}

// out = a - 1  (a must be >= 1)
static inline void u256_sub1(u64* out, const u64* a) {
    u64 borrow = 1;
    for (int i = 0; i < 4; i++) {
        u64 ai = a[i];
        out[i] = ai - borrow;
        borrow = (ai < borrow) ? 1 : 0;
    }
}

class ZoneRegistry {
public:
    ZoneRegistryHeader hdr;
    ZoneEntry          zones[ZONE_COUNT];
    bool               loaded;

    ZoneRegistry() : loaded(false) {
        memset(&hdr,  0, sizeof(hdr));
        memset(zones, 0, sizeof(zones));
    }

    // Load from disk
    bool Load(const char* path = ZONE_REG_FILE) {
        FILE* f = fopen(path, "rb");
        if (!f) { loaded = false; return false; }
        ZoneRegistryHeader h;
        if (fread(&h, sizeof(h), 1, f) != 1 ||
            h.magic   != ZONE_MAGIC   ||
            h.version != ZONE_VERSION ||
            fread(zones, sizeof(ZoneEntry), ZONE_COUNT, f) != ZONE_COUNT) {
            fclose(f);
            printf("[ZoneReg] Load failed — creating new\n");
            loaded = false; return false;
        }
        fclose(f);
        memcpy(&hdr, &h, sizeof(h));
        loaded = true;
        printf("[ZoneReg] Loaded Puzzle %u (%u-bit) | DPs: %llu\n",
               hdr.puzzle_num, hdr.range_bits,
               (unsigned long long)hdr.total_dps);
        return true;
    }

    // Save atomically
    bool Save(const char* path = ZONE_REG_FILE) {
        hdr.last_updated = (u64)time(nullptr);
        hdr.total_dps    = 0;
        for (int i = 0; i < ZONE_COUNT; i++)
            hdr.total_dps += zones[i].dps_collected;
        char tmp[280];
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        if (!f) { printf("[ZoneReg] Cannot write %s\n", tmp); return false; }
        fwrite(&hdr,  sizeof(hdr),       1,          f);
        fwrite(zones, sizeof(ZoneEntry), ZONE_COUNT, f);
        fclose(f);
        remove(path); rename(tmp, path);
        return true;
    }

    // Init for any puzzle — uses range_bits to compute zone boundaries
    //
    // NOTE: the original version of this function used unsigned __int128
    // for the >128-bit puzzle path (range_bits > 128, e.g. Puzzle 135/140)
    // and its high-word extraction (`zs >> 128`, `zs >> 192`) was undefined
    // behaviour (shift >= operand width) that in practice discarded the
    // puzzle's real magnitude and produced all-zero zone boundaries for
    // any puzzle >= 129 bits. Replaced with explicit 256-bit (u64[4])
    // arithmetic below, which is correct uniformly for all supported
    // range_bits (up to 256).
    void Init(u32 puzzle_num, u32 range_bits, const u8* pubkey) {
        memset(&hdr,  0, sizeof(hdr));
        memset(zones, 0, sizeof(zones));
        hdr.magic      = ZONE_MAGIC;
        hdr.version    = ZONE_VERSION;
        hdr.puzzle_num = puzzle_num;
        hdr.range_bits = range_bits;
        hdr.zone_count = ZONE_COUNT;
        hdr.created    = (u64)time(nullptr);
        if (pubkey) memcpy(hdr.pubkey, pubkey, 33);

        // Puzzle range: [2^(range_bits-1), 2^range_bits)
        // Zone width: 2^(range_bits-1) / 100
        int hi_bit = (int)range_bits - 1;  // 2^hi_bit = range start
        if (hi_bit < 0) hi_bit = 0;

        u64 range_start[4], zone_size[4];
        u256_set_bit(range_start, hi_bit);
        memcpy(zone_size, range_start, sizeof(zone_size));
        u256_div_small(zone_size, ZONE_COUNT);

        for (int z = 0; z < ZONE_COUNT; z++) {
            u64 zs[4], ze[4], t[4];

            memcpy(t, zone_size, sizeof(t));
            u256_mul_small(t, (u32)z);
            u256_add(zs, range_start, t);

            if (z == ZONE_COUNT - 1) {
                u64 twice[4];
                memcpy(twice, range_start, sizeof(twice));
                u256_mul_small(twice, 2);
                u256_sub1(ze, twice);
            } else {
                memcpy(t, zone_size, sizeof(t));
                u256_mul_small(t, (u32)(z + 1));
                u64 sum[4];
                u256_add(sum, range_start, t);
                u256_sub1(ze, sum);
            }

            memcpy(zones[z].range_start, zs, sizeof(zs));
            memcpy(zones[z].range_end,   ze, sizeof(ze));
            snprintf(zones[z].workfile, 64, "zone_%03d.work", z);
        }

        loaded = true;
        printf("[ZoneReg] Init Puzzle %u (%u-bit) — 100 zones\n",
               puzzle_num, range_bits);
        Save();
    }

    // Pick zone with lowest DP count
    int GetLowestDPZone() const {
        int best = 0;
        u64 min_dps = zones[0].dps_collected;
        for (int z = 1; z < ZONE_COUNT; z++) {
            if (zones[z].dps_collected < min_dps) {
                min_dps = zones[z].dps_collected;
                best    = z;
            }
        }
        return best;
    }

    // Update zone after a session.
    // total_dps_now = the CURRENT true cumulative DP count for this zone
    // (loaded-from-file + everything found this session combined), NOT a
    // per-session delta -- now that zone DP data actually persists across
    // visits (see g_save_tames_on_exit in RCKangaroo.cpp), the caller
    // always knows the real running total directly, so this is a set, not
    // an accumulate. Older behaviour (before persistence existed) added a
    // fresh delta each time since every visit started from zero.
    void UpdateZone(int z, u64 total_dps_now, u64 elapsed_secs) {
        if (z < 0 || z >= ZONE_COUNT) return;
        u64 prev = zones[z].dps_collected;
        zones[z].dps_collected  = total_dps_now;
        zones[z].total_secs    += elapsed_secs;
        zones[z].sessions++;
        zones[z].last_searched  = (u64)time(nullptr);
        Save();
        printf("[ZoneReg] Zone %d: %llu -> %llu DPs (persisted) | %llus\n",
               z, (unsigned long long)prev,
               (unsigned long long)zones[z].dps_collected,
               (unsigned long long)elapsed_secs);
    }

    // Record solution
    void RecordSolution(int z, const u64* key) {
        hdr.solved = 1; hdr.solved_zone = (u8)z;
        if (key) memcpy(hdr.solved_key, key, 32);
        Save(); SaveTextReport();
        printf("[ZoneReg] *** SOLVED in Zone %d ***\n", z);
    }

    // Hex display of zone boundaries (up to 256-bit, big-endian display,
    // leading all-zero words skipped). The previous version only ever
    // looked at words [0] and [1] (128 bits), so any puzzle >= 129 bits
    // (e.g. 135, 140) displayed as "0x0" here even when the underlying
    // value was correct -- fixed to walk all 4 words.
    static void FormatHex256(const u64* v, char* out, int len) {
        int top = 3;
        while (top > 0 && v[top] == 0) top--;
        int n = snprintf(out, len, "%llX", (unsigned long long)v[top]);
        for (int i = top - 1; i >= 0 && n < len; i--)
            n += snprintf(out + n, len - n, "%016llX", (unsigned long long)v[i]);
    }

    void GetZoneHex(int z, char* s_hex, char* e_hex, int len) const {
        if (z < 0 || z >= ZONE_COUNT) return;
        FormatHex256(zones[z].range_start, s_hex, len);
        FormatHex256(zones[z].range_end,   e_hex, len);
    }

    // Status display
    void PrintStatus() const {
        int searched = 0;
        u64 total_dps = 0, total_secs = 0;
        for (int i = 0; i < ZONE_COUNT; i++) {
            if (zones[i].dps_collected > 0) searched++;
            total_dps  += zones[i].dps_collected;
            total_secs += zones[i].total_secs;
        }
        printf("\n══════════════════════════════════════════\n");
        printf("  ZONE REGISTRY — Puzzle %u (%u-bit)\n",
               hdr.puzzle_num, hdr.range_bits);
        if (hdr.solved)
            printf("  Solved:         YES -- Zone %u/100\n", (unsigned)hdr.solved_zone);
        printf("  Zones with DPs: %d/100  (coverage across the whole puzzle, not \"which zone solved\")\n", searched);
        printf("  Total DPs:      %llu\n", (unsigned long long)total_dps);
        printf("  GPU hours:      %.1f\n", total_secs / 3600.0);
        printf("\n  Map: [.]=no DPs  [#]=has DPs%s\n  ",
               hdr.solved ? "  [S]=solved zone" : "");
        for (int i = 0; i < ZONE_COUNT; i++) {
            char c = zones[i].dps_collected ? '#' : '.';
            if (hdr.solved && i == hdr.solved_zone) c = 'S';
            printf("%c", c);
            if ((i+1) % 50 == 0) {
                printf("  %d%%\n", i+1);
                if (i < 99) printf("  ");
            }
        }
        printf("══════════════════════════════════════════\n\n");
    }

    void SaveTextReport(const char* path = ZONE_REG_TXT) const {
        FILE* f = fopen(path, "w");
        if (!f) return;
        time_t t = (time_t)hdr.last_updated;
        fprintf(f, "# Puzzle %u (%u-bit) Zone Coverage — %s",
                hdr.puzzle_num, hdr.range_bits, ctime(&t));
        fprintf(f, "# Zone | DPs | GPU-Hours | Sessions\n");
        for (int i = 0; i < ZONE_COUNT; i++)
            fprintf(f, "  %3d%% | %llu | %.2f | %u\n", i,
                    (unsigned long long)zones[i].dps_collected,
                    zones[i].total_secs / 3600.0,
                    zones[i].sessions);
        fclose(f);
    }
};

extern ZoneRegistry g_zone_registry;

// ============================================================
// Zone-mode CLI config and entry point (defined in zone_mode.cpp,
// set by ParseCommandLine()/main() in RCKangaroo.cpp)
// ============================================================
extern bool g_zone_mode;          // -zone-mode
extern int  g_zone_duration_mins; // -zone-duration N (minutes, approx budget per zone)
// -zone-force N (0-99): pin every zone visit for this whole run to zone N,
// bypassing BOTH GetKnownZone()'s first-attempt logic AND GetLowestDPZone()
// round-robin entirely. For controlled A/B benchmarking of any jump-
// selection or search-strategy variant (built and torn down for the
// Mandelbrot LUT experiment -- see archive/ARCHIVE_NOTE_mandelbrot_jump.md
// -- but independent of it and worth keeping for the next one) so results
// can never be affected by leftover zone_registry.bin state biasing which
// zone gets tried next -- see SelectNextZone() in zone_mode.cpp. -1 = not
// forced (normal known-zone / round-robin behavior).
extern int  g_zone_force_id;

// Drop-in-ish entry point called from main() instead of SolvePoint() when
// g_zone_mode is set. Unlike SolvePoint(), it owns its own retry/zone loop
// internally and reports success via out params so main()'s existing
// KEY FOUND / WIF / address / RESULTS.TXT reporting code can run unchanged
// for both zone-mode and normal runs.
void RunZoneModeV2(EcPoint pubkey, int puzzle_num, int range_bits, EcInt* pk_res, bool* out_solved);
