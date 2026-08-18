# Archive note: `mandelbrot_jump.h` (Mandelbrot LUT jump-index selection)

**Status: archived, experimental, null result.** Removed from the active build.
Nothing in the live codebase includes this header anymore; the `-mandelbrot`
CLI flag, the `Kparams.mand_lut`/`use_mandelbrot` fields, the `SelectJumpIndex`
wrapper in `RCGpuCore.cu`, `SelectJumpIndexCPU` in `CpuKang.cpp`/
`CpuKang_JLP.cpp`, `SetMandelbrotLUT` in `GpuKang.h`, and the magnitude-sort
step in `RCKangaroo.cpp` have all been reverted. Every jump-selection call
site is back to its original `x & JMP_MASK`.

## What this was

An alternate jump-index selection scheme: instead of `x & (JMP_CNT-1)`
(uniform over the 512 pre-generated jump table entries), compute a Mandelbrot
escape-time value from bits of `x` and use a precomputed 512x512 LUT to bias
selection toward different "buckets" of jump-table indices, with the bucket
weighting varying per operational zone (a different Im-plane slice per zone).
The idea: distinguished-point search might benefit from escape-time-driven,
zone-dependent jump weighting instead of uniform selection.

## Why it's archived, not just left off by default

Because the honest answer, after actually testing it properly, is that it
doesn't do anything measurable. Specifically:

- **Final, clean A/B test** (puzzle 80, zone 82, `-zone-force 82` pinning the
  zone every run, `zone_registry.bin`/`.txt` and the zone's `.kangs` DP file
  deleted before *every single run* to eliminate all state confounds, n=10
  baseline vs n=10 Mandelbrot-with-magnitude-sorted-table): mean K 0.857 vs
  0.822, sd ~0.40 both groups, **t ~= 0.19**. Not distinguishable from zero
  effect at any reasonable threshold.
- An earlier batch (same setup, minus the registry reset) showed what looked
  like a real, large effect: median K dropped ~50%, t ~= 3.2 (nominally
  significant). That's what triggered writing this up as a "real finding" at
  one point. It didn't survive controlling for a confound: `zone_registry.bin`
  persists `dps_collected`/`sessions` per zone *across every process launch*
  and influences round-robin zone selection on any zone-mode retry. The
  baseline and Mandelbrot batches were run back-to-back sharing that file,
  unreset. Once it was wiped before every run and the zone was hard-pinned
  with `-zone-force`, the effect disappeared entirely. That's about as clean
  a "the earlier result was an artifact" signal as you get.
- A planned follow-up control (`MAND_FORCE_BUCKET` env var, added to force a
  fixed bucket with zero fractal computation, to check whether *any* skew
  toward one end of the table reproduces the effect) was never needed, since
  the confound-free test already showed nothing to explain.

## What WAS real, and worth keeping in mind if anyone revisits this

1. **The base jump table (`EcJumps1`) is magnitude-blind by index**, under
   the active (non-Lissajous) generator in `RCKangaroo.cpp`: each of the 512
   entries is an i.i.d. random draw, independent of its index. Empirically,
   correlation(index, log2(jump size)) ~= 0.07 (noise) on a real puzzle-80
   table, and every one of this file's `MAND_BUCKETS` "fine/local/.../long"
   groups had the same ~41.5-41.6 bit mean size, despite the bucket labels
   claiming a 14-80 bit spread. This means the *original* version of this
   LUT (before a fix) was reweighting selection over indices that carried no
   size information whatsoever -- it literally could not have worked as
   designed. A magnitude-sort step (`std::sort` on `EcJumps1` by `.dist`,
   gated behind `g_use_mandelbrot`, applied before the table is copied to any
   GPU) was added to fix this and make the bucket weighting mean something.
   That fix is what was actually tested in the final clean A/B run above --
   and it still showed nothing. So this isn't "the idea was never given a
   fair shot" -- it was, and it didn't pan out.
2. **A real multi-GPU bug was found and fixed during integration**: a single
   shared `MandelbrotLUT` instance's device pointer (`d_lut`, `cudaMalloc`'d)
   is only valid on whichever GPU was CUDA's "current device" at allocation
   time. Sharing one pointer across multiple `RCGpuKang` instances on
   different GPUs caused "illegal memory access" on every GPU but one. The
   fix (per-GPU `MandelbrotLUT` array, built inside the per-GPU prepare loop
   *after* that GPU's own `cudaSetDevice()`, plus a separate single instance
   for CPU workers since they only ever touch host memory) is documented in
   this file's own trailing comments, preserved below for reference. Anyone
   reviving this should keep that pattern -- don't reintroduce a single
   shared instance across GPUs.
3. **Herd bias and loop detection are both index-based, not magnitude-based**
   (`herd_bias` is added to `x` *before* index selection; loop detection
   compares `jmp_ind == jmp_next`, i.e. index equality) -- so neither
   interacts badly with reordering what a given index means. Verified against
   `RCGpuCore.cu` directly, not assumed.
4. **`EcJumps2` is never selected through the x-derived jump index at all**
   (it's read at fixed loop-counter offsets for L1S2 loop-detection
   bookkeeping) -- only `EcJumps1` (main stepping) and `EcJumps3` (rare
   loop-escape jumps, via `BuildR2JumpTable`'s own deliberately spread-out R2
   magnitude assignment) are ever touched by an x-derived selection index.

## What's staying in the live codebase, unrelated to this archival

- **`-zone-force N`** (`zone_registry.h`/`zone_mode.cpp`): pins zone-mode to
  a specific zone for the whole run, bypassing both the known-zone-first-run
  logic and round-robin. This is what made the clean A/B test above possible,
  and it's independently useful for *any* future search-strategy experiment
  in this codebase, not just this one. Keeping it.
- The zone-id-to-registry mapping fix and the multi-GPU pointer-safety
  pattern described above are worth remembering even though the feature
  they were built for is archived -- the same classes of bug (per-GPU device
  pointer sharing, unreset shared state between benchmark runs) can and will
  recur in any future GPU-side or zone-mode experiment here.

## If you want to revive this

The magnitude-sort fix, the zone-id wiring, and the per-GPU pointer pattern
are all preserved in this archived header's trailing comments and in git
history. Rebuild the same integration (12 call sites in `RCGpuCore.cu`, 3
each in `CpuKang.cpp`/`CpuKang_JLP.cpp`, `Kparams` fields in `defs.h`,
`SetMandelbrotLUT` in `GpuKang.h`, the CLI flag + globals + LUT build calls
in `RCKangaroo.cpp`), and re-run the clean A/B methodology above (registry +
DP file wipe before every run, `-zone-force` pinning, n>=10 per arm, a
significance check) before trusting any result. Don't skip the registry
reset -- that's the mistake that produced a false positive last time.
