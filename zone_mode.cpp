// zone_mode.cpp
// Dynamic zone-based systematic search — works for ANY puzzle
// Wraps existing SolvePoint() unchanged
//
// USAGE:
//   ./rckangaroo -pubkey <key> -range 80 -start <hex> -gpu 012 -zone-mode
//   ./rckangaroo -pubkey <key> -range 140 -start <hex> -gpu 012 -zone-mode
//   ./rckangaroo -pubkey <key> -range 90 -start <hex> -gpu 012 -zone-mode -zone-duration 60
//
// NOTE: -start is still required by ParseCommandLine() like any other
// -pubkey run (it must be the known lower bound of the puzzle's interval,
// e.g. 2^(N-1) for puzzle N). Zone mode does NOT need -start to point at a
// specific zone — it only needs the true lower bound of the *whole*
// interval so puzzle_num/range_bits can be resolved consistently; the
// zone registry (below) computes each zone's own absolute sub-offset
// from range_bits and searches relative to that, independent of gStart.
//
// TESTING (use solved puzzles to verify the offset math + DP bookkeeping,
// NOT to prove anything about unsolved puzzles — see note below):
//   Puzzle 80:  key at zone 82%  (verified against btcpuzzle.info, Aug 2026)
//     ./rckangaroo -pubkey 037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc -range 80 -start 80000000000000000000 -zone-mode -zone-duration 5 -gpu 012
//   Puzzle 85:  key at zone 9%
//   Puzzle 90:  key at zone 40%
//
// WHAT ZONE MODE ACTUALLY IS — AND ISN'T:
//   Zone mode divides the puzzle's [2^(N-1), 2^N) interval into 100 equal
//   sub-ranges ("zones") and runs a fresh, self-contained SolvePoint() call
//   against whichever zone currently has the fewest accumulated DPs,
//   checkpointing per-zone stats (DPs/GPU-seconds/sessions — NOT the live
//   DP database itself; SolvePoint() clears its DP table on every return,
//   so each zone visit restarts that zone's kangaroo herd from scratch) to
//   zone_registry.bin so a long search can be resumed across sessions or
//   coordinated across multiple machines without re-covering the same zone.
//
//   This is an operational/checkpointing convenience, NOT an algorithmic
//   speedup. Kangaroo already solves the *whole* configured range in
//   O(sqrt(range)) expected group operations by construction — it does not
//   need to exhaustively cover the interval. Chopping that same interval
//   into 100 disjoint zones and fully searching each one costs roughly
//   O(sqrt(range) * sqrt(100)) in total, i.e. about 10x MORE work than one
//   pass over the full range. Use -zone-mode when you have an operational
//   reason (bounded per-session runtime, coordinating independent
//   machines/sessions, visibility into progress) — not as a way to make an
//   otherwise-infeasible range feasible.
//
//   For an ALREADY-SOLVED puzzle (used above only to verify the zone
//   offset math and bookkeeping are correct) SelectNextZone() cheats and
//   jumps straight to the known zone on the first run — that's why the
//   test above finds the key quickly despite the 10x overhead. For a real
//   unsolved target (e.g. Puzzle 140, still open as of Aug 2026 with a
//   ~14 BTC reward) GetKnownZone() returns -1 and there is no such
//   shortcut: zone mode provides no way around Puzzle 140's underlying
//   ~2^69-operation difficulty, which is far beyond a single machine's
//   reach in any realistic amount of time.

#include "zone_registry.h"
#include "defs.h"
#include "utils.h"
#include "Ec.h"
#include <string>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <csignal>

// ============================================================
// RCKangaroo.cpp globals — already defined there
// ============================================================
extern EcInt          gStart;
extern u32            gRange;
extern u32            gDP;
extern bool           gDP_manual;
extern EcPoint        gPubKey;
extern volatile bool  gSolved;
extern double         gMax;
extern std::string    g_work_filename;
extern TFastBase      db;
extern volatile u64   TotalOps;
extern u64            PntTotalOps;
extern bool           gGenMode;
extern Ec             ec;
extern EcInt          gPrivKey;
// Captured by SolvePoint() itself right before it clears its DP database
// on every return path. db.GetBlockCnt() is always 0 by the time
// SolvePoint() returns to us, so this is the only reliable way to see how
// many DPs a given zone attempt actually collected.
extern u64             g_last_solve_dp_count;
// Opt-in flag + the file it saves to (gTamesFileName, declared below) --
// see comment at its definition in RCKangaroo.cpp.
extern bool            g_save_tames_on_exit;
// Set only by the SIGINT/SIGTERM handler (real Ctrl+C) -- distinct from
// gIsOpsLimit, which is what a normal per-zone -zone-duration timeout sets.
extern volatile sig_atomic_t g_shutdown_requested;
extern char            gTamesFileName[1024];

// Declared in RCKangaroo.cpp
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res);

// ============================================================
// Globals
// ============================================================
ZoneRegistry g_zone_registry;
bool         g_zone_mode          = false; // -zone-mode
int          g_zone_duration_mins = 30;    // -zone-duration N
int          g_zone_force_id      = -1;    // -zone-force N (-1 = not forced)

// ============================================================
// Zone width for a given puzzle range
// 1% of range [2^(range_bits-1), 2^range_bits)
// = 2^(range_bits-1) / 100
// In bits: (range_bits-1) - log2(100) ≈ range_bits - 7.64
// Round UP to avoid coverage gaps
// ============================================================
static int CalcZoneRangeBits(int range_bits) {
    // zone_size = 2^(range_bits-1) / 100
    // log2(zone_size) = (range_bits-1) - log2(100) = range_bits - 7.644
    // Round UP: range_bits - 6 (conservative — slight overlap is fine)
    int zone_bits = range_bits - 6;
    if (zone_bits < 10)  zone_bits = 10;
    if (zone_bits > 180) zone_bits = 180;
    return zone_bits;
}

// ============================================================
// Set gStart from zone boundary
// EcInt is a 320-bit (40 byte) big integer
// Zone boundaries are stored as little-endian u64[4]
// ============================================================
static void SetStartFromZone(const ZoneEntry& z) {
    memset(gStart.data, 0, sizeof(gStart.data));
    // Copy 256-bit zone start into EcInt (which is >= 256 bits)
    memcpy(gStart.data, z.range_start, 32);
}

// ============================================================
// Get current DB DP count
// ============================================================
static u64 GetCurrentDPs() {
    return db.GetBlockCnt();
}

// ============================================================
// Per-zone DP checkpoint filename. One file per (puzzle, zone) pair so
// visiting zone 9 of Puzzle 85 never mixes DP data with zone 9 of Puzzle
// 140 -- distinguished points are only meaningful within the exact same
// PntToSolve reference frame they were collected against.
// ============================================================
static std::string ZoneDPFilename(int puzzle_num, int zone_id) {
    char buf[64];
    snprintf(buf, sizeof(buf), "zone_dps_p%d_z%02d.kangs", puzzle_num, zone_id);
    return std::string(buf);
}

// ============================================================
// Point kangaroo must actually solve for a given zone, and the
// inverse (recover the absolute private key from a zone-local
// SolvePoint() result). This mirrors exactly what main() does
// with gStart/gPubKey for a plain (non-zone) run — SolvePoint()
// itself knows nothing about start offsets, the caller has to
// apply/undo them. zone_start here is the zone's own absolute
// offset (as set into gStart by SetStartFromZone), not the CLI
// -start value.
// ============================================================
static EcPoint PointForZoneStart(EcPoint pubkey, EcInt zone_start) {
    if (zone_start.IsZero())
        return pubkey;
    EcPoint ofs = ec.MultiplyG_Lambda(zone_start);
    ofs.y.NegModP();
    return ec.AddPoints(pubkey, ofs);
}

// ============================================================
// Register zone_registry.bin is for the SAME puzzle/range/pubkey
// If different, reinitialize
// ============================================================
static void EnsureRegistry(u32 puzzle_num, u32 range_bits, EcPoint& pubkey) {
    // Internal fingerprint only (not a standard SEC1 compressed encoding) —
    // just enough to detect "this registry was built for a different
    // target" and avoid silently reusing stale zone bookkeeping.
    u8 fp[33] = {0};
    memcpy(fp, pubkey.x.data, sizeof(fp) < sizeof(pubkey.x.data) ? sizeof(fp) : sizeof(pubkey.x.data));

    if (g_zone_registry.Load()) {
        bool same_target = (g_zone_registry.hdr.puzzle_num == puzzle_num) &&
                            (g_zone_registry.hdr.range_bits == range_bits) &&
                            (memcmp(g_zone_registry.hdr.pubkey, fp, sizeof(fp)) == 0);
        if (same_target)
            return; // Already correct
        printf("[ZoneMode] Registry is for a different puzzle/range/pubkey, "
               "reinitializing for Puzzle %u/%u-bit\n",
               puzzle_num, range_bits);
    }
    g_zone_registry.Init(puzzle_num, range_bits, fp);
}

// ============================================================
// Known solved puzzle locations (verified against btcpuzzle.info, Aug 2026)
// Used to verify zone mode works — start at known zone first
// ============================================================
struct SolvedPuzzleInfo {
    int puzzle_num;
    int zone_pct;    // which 1% zone the key is in
    const char* pubkey_hex;
};

static const SolvedPuzzleInfo SOLVED_PUZZLES[] = {
    { 80,  82, "037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc" },
    { 85,   9, "0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a" },
    { 90,  40, "035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5" },
    { 95,  28, "02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045" },
    {100,  36, "03d2063d40402f030d4cc71331468827aa41a8a09bd6fd801ba77fb64f8e67e617" },
    {105,  43, "03bcf7ce887ffca5e62c9cabbdb7ffa71dc183c52c04ff4ee5ee82e0c55c39d77b" },
    {110,  67, "0309976ba5570966bf889196b7fdf5a0f9a1e9ab340556ec29f8bb60599616167d" },
    {115,  51, "0248d313b0398d4923cdca73b8cfa6532b91b96703902fc8b32fd438a3b7cd7f55" },
    {120,  38, "02ceb6cbbcdbdf5ef7150682150f4ce2c6f4807b349827dcdbdd1f2efa885a2630" },
    {125,  77, "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e" },
    {130,  62, "03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852" },
    {135,  71, "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16" },
    {  0,   0, nullptr } // sentinel
};

// Look up known zone for a solved puzzle (-1 if unknown/unsolved)
static int GetKnownZone(int puzzle_num) {
    for (int i = 0; SOLVED_PUZZLES[i].pubkey_hex != nullptr; i++) {
        if (SOLVED_PUZZLES[i].puzzle_num == puzzle_num)
            return SOLVED_PUZZLES[i].zone_pct;
    }
    return -1; // unsolved — use round-robin selection
}

// ============================================================
// Smart zone selection:
// - Solved puzzles, first run: known zone (for testing/verification only)
// - Unsolved puzzles, or subsequent runs: lowest-DP zone (round robin)
// ============================================================
static int SelectNextZone(int puzzle_num, bool first_run) {
    // Forced zone (-zone-force N) overrides EVERYTHING else, on every call
    // for the whole run -- not just the first one. Used for controlled A/B
    // benchmarking where the zone must never vary between runs or between
    // successive attempts within one run, regardless of zone_registry.bin
    // state (leftover dps_collected/sessions from earlier runs can no
    // longer influence zone choice at all when this is set).
    if (g_zone_force_id >= 0 && g_zone_force_id < ZONE_COUNT) {
        if (first_run)
            printf("[ZoneMode] -zone-force %d%% active -- pinned for this entire run\n",
                   g_zone_force_id);
        return g_zone_force_id;
    }
    int known = GetKnownZone(puzzle_num);
    if (known >= 0 && first_run) {
        printf("[ZoneMode] SOLVED puzzle — starting at known zone %d%% (verification run)\n",
               known);
        return known;
    }
    return g_zone_registry.GetLowestDPZone();
}

// ============================================================
// Main zone mode entry point
// Called from main() when -zone-mode flag is set
// ============================================================
void RunZoneModeV2(EcPoint pubkey, int puzzle_num, int range_bits, EcInt* pk_res, bool* out_solved) {
    *out_solved = false;

    printf("\n");
    printf("================================================\n");
    printf("  ZONE MODE - Puzzle %d (%d-bit)\n", puzzle_num, range_bits);

    int known_zone = GetKnownZone(puzzle_num);
    if (known_zone >= 0)
        printf("  MODE: TEST (solved puzzle - zone %d%% known)\n", known_zone);
    else
        printf("  MODE: PRODUCTION (unsolved - round-robin zones)\n");
    printf("  Duration: %d min/zone\n", g_zone_duration_mins);
    printf("================================================\n\n");

    EnsureRegistry((u32)puzzle_num, (u32)range_bits, pubkey);
    g_zone_registry.PrintStatus();

    int    zone_range_bits = CalcZoneRangeBits(range_bits);
    double saved_gMax      = gMax;
    bool   first_run       = true;

    printf("[ZoneMode] Zone width: ~%d bits (1%% of %d-bit range)\n",
           zone_range_bits, range_bits);

    while (!gSolved) {

        int zone_id = SelectNextZone(puzzle_num, first_run);
        first_run   = false;
        ZoneEntry& zone = g_zone_registry.zones[zone_id];

        char s_hex[48] = {0}, e_hex[48] = {0};
        g_zone_registry.GetZoneHex(zone_id, s_hex, e_hex, sizeof(s_hex));

        printf("\n[ZoneMode] Zone %3d%% | DPs: %llu | Sessions: %u\n",
               zone_id,
               (unsigned long long)zone.dps_collected,
               zone.sessions);
        printf("[ZoneMode] 0x%s -> 0x%s\n", s_hex, e_hex);
        printf("[ZoneMode] Work: %s\n", zone.workfile);
        if (known_zone >= 0 && zone_id == known_zone)
            printf("[ZoneMode] *** KNOWN KEY ZONE — searching here ***\n");

        // Configure globals for this zone (gStart kept in sync purely for
        // display/workfile naming; the actual offset used to compute the
        // point handed to SolvePoint() is taken directly below so there is
        // no ambiguity about which value SolvePoint() is really using).
        SetStartFromZone(zone);
        g_work_filename = std::string(zone.workfile);
        gRange = (u32)zone_range_bits;

        EcPoint PntZone = PointForZoneStart(pubkey, gStart);

        // Persist this zone's DPs across visits instead of losing them
        // when SolvePoint() clears db on return. SolvePoint() already
        // knows how to LOAD gTamesFileName (existing, untouched logic);
        // g_save_tames_on_exit additionally makes it SAVE back to the same
        // file on every exit path (see RCKangaroo.cpp).
        std::string zone_dp_file = ZoneDPFilename(puzzle_num, zone_id);
        strncpy(gTamesFileName, zone_dp_file.c_str(), sizeof(gTamesFileName) - 1);
        gTamesFileName[sizeof(gTamesFileName) - 1] = 0;
        g_save_tames_on_exit = true;
        printf("[ZoneMode] DP checkpoint: %s\n", gTamesFileName);

        // Per-zone ops budget approximating -zone-duration minutes.
        // This is an estimate (assumes ~8 GK/s combined throughput), not a
        // hard wall-clock guarantee — actual zone runtime will scale with
        // real GPU speed.
        {
            double exp_ops  = 1.15 * pow(2.0, zone_range_bits / 2.0);
            double budget   = 8.0e9 * g_zone_duration_mins * 60.0;
            gMax = budget / exp_ops;
            if (gMax < 0.05) gMax = 0.05;
            if (gMax > 20.0) gMax = 20.0;
            printf("[ZoneMode] Ops limit: %.2fx expected (~%d min at 8 GK/s est.)\n",
                   gMax, g_zone_duration_mins);
        }

        time_t t0 = time(nullptr);

        EcInt found_key_local; // relative to PntZone, NOT the absolute private key
        bool solved = SolvePoint(PntZone, zone_range_bits, (int)gDP, &found_key_local);

        u64 elapsed = (u64)(time(nullptr) - t0);
        // g_last_solve_dp_count is set by SolvePoint() itself right before
        // it clears db on return -- sampling db.GetBlockCnt() here (after
        // the call) would always read 0, since SolvePoint() already wiped
        // it internally by this point. Because g_save_tames_on_exit is on,
        // db at that moment holds this zone's FULL accumulated history
        // (whatever was loaded from zone_dp_file + everything new this
        // visit), so this is the true running total, not just this visit's
        // delta -- UpdateZone() now treats it as such.
        u64 total_dps_now = g_last_solve_dp_count;

        g_zone_registry.UpdateZone(zone_id, total_dps_now, elapsed);

        // Real Ctrl+C / SIGTERM (distinct from gIsOpsLimit, which is what a
        // normal -zone-duration timeout sets) -- this zone's checkpoint is
        // already saved above via UpdateZone()/SolvePoint()'s own exit-path
        // save, so just stop the sweep instead of rotating to the next
        // zone. Without this check, g_shutdown_requested stays set (nothing
        // clears it) and every subsequent SolvePoint() call would see it
        // immediately and return near-instantly, busy-looping through all
        // remaining zones instead of actually exiting.
        if (g_shutdown_requested) {
            printf("[ZoneMode] Interrupted -- stopping (zone %d checkpoint saved).\n", zone_id);
            gMax = saved_gMax;
            g_save_tames_on_exit = false;
            gTamesFileName[0]    = 0;
            g_zone_registry.PrintStatus();
            g_zone_registry.SaveTextReport();
            return;
        }

        if (solved || gSolved) {
            // Recover the absolute private key: SolvePoint() only knows
            // about PntZone = pubkey - gStart*G, so it returns a value
            // relative to that offset. Undo the offset here.
            EcInt abs_key = found_key_local;
            abs_key.AddModP(gStart);

            g_zone_registry.RecordSolution(zone_id, (u64*)abs_key.data);
            if (known_zone >= 0)
                printf("[ZoneMode] VERIFICATION PASSED — zone mode's offset math and search are correct.\n");
            else
                printf("[ZoneMode] PUZZLE %d SOLVED — Zone %d!\n", puzzle_num, zone_id);

            *pk_res     = abs_key;
            *out_solved = true;
            gMax = saved_gMax;
            g_save_tames_on_exit = false;
            gTamesFileName[0]    = 0;
            g_zone_registry.PrintStatus();
            g_zone_registry.SaveTextReport();
            return;
        }

        // RAM management — clear if > 90% of 120GB budget
        {
            u64 cap = (u64)(120.0 * 1024.0 * 1024.0 * 1024.0 / 36.0);
            if (GetCurrentDPs() > cap * 90 / 100) {
                printf("[ZoneMode] RAM at 90%% cap — clearing DB\n");
                db.Clear();
            }
        }

        g_zone_registry.PrintStatus();
    }

    gMax = saved_gMax;
    g_save_tames_on_exit = false;
    gTamesFileName[0]    = 0;
    g_zone_registry.PrintStatus();
    g_zone_registry.SaveTextReport();
    printf("[ZoneMode] Session ended without a solve — zone_registry.bin saved, resume anytime with the same flags.\n");
}
