# fractal_jump_table.h — archived 2026-08-18

## Why this was pulled out of the active tree

The active jump-table generator in RCKangaroo.cpp (both the USE_LISSAJOUS=1
path and the classic USE_LISSAJOUS=0 fallback at line ~1081) already centers
J1 jump magnitude on `Range/2 + ~1` bits — the Teske-optimal mean jump size
(sqrt(range)) for minimizing expected kangaroo steps to collision. EcJumps2/
EcJumps3 separately provide large loop-escape jumps for the L1S2 detector.
See defs.h's comment above JMP_CNT:
  "Bimodal small/large jumps already comes from L1S2 switching between
   jmp1/jmp2 tables; JMP_CNT controls diversity within each table, not
   the small-vs-large ratio."

fractal_jump_table.h ignores puzzle size and hardcodes a fixed power-law
spread from 2^14 to 2^80 bits, weighted so >50% of jumps land in the
2^14-2^34 range. For Puzzle 140 (optimal mean ~2^70) that's roughly
2^36-2^56x smaller than optimal for most of the table — a severe
regression against the theory the existing generator already implements
correctly. Wiring it in as the primary J1 generator would very likely
increase real ops-to-solve, not reduce it.

Kept here in case the multi-scale idea is revisited with buckets rescaled
per-range (i.e. centered on sqrt(zone_width) instead of fixed absolute
bit ranges) and benchmarked against the current generator before ever
pointing it at an unsolved target.

Not deleted — just out of the active build path.
