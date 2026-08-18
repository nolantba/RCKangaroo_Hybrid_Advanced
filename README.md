# RCKangaroo Hybrid Advanced v3.2
**Bitcoin Puzzle Solver — Pollard Kangaroo ECDLP on secp256k1**

Based on RetiredCoder's kangaroo implementation. This fork targets Puzzle 135 and includes
a series of correctness fixes, stability improvements, and performance enhancements built
over extended development sessions.

---

## Hardware

```
GPUs:  3x NVIDIA GeForce RTX 3060 (SM 8.6, 12GB each)
CPU:   Dual Xeon E5-2696 v3 — 68 threads
RAM:   128 GB DDR4
HDD:   4 TB
CUDA:  Driver 13.0 / Runtime 11.5
OS:    Linux (Ubuntu-based)
```

**Sustained throughput:** ~7.97 GK/s

---

## Puzzle 135 Command

```bash
./rckangaroo \
  -range 135 \
  -start 4000000000000000000000000000000000 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -dp 22 \
  -gpu 012 \
  -cpu 68 \
  -workfile puzzle135_dp22.work \
  -autosave 300
```

**Why DP=22:** At 7.97 GK/s the expected DPs/sec is ~1,900. DP=22 gives ~12 minute
detection lag while keeping the table within the 128 GB RAM budget (~3.4B DPs max).
DP=42 would give a 25-year detection lag. DP=14 overflows RAM within hours.

---

## Enhancements

### 1. Per-GPU Independent Seeds
Each GPU now derives a unique seed from a shared base seed using a mixing constant:

```
gpu_seed = base_seed XOR (gpu_index * 0x9E3779B97F4A7C15)
```

This gives each GPU completely different jump tables and starting positions — three
genuinely independent trajectory families attacking the range simultaneously. Previously
all three GPUs shared the same jump tables and were doing overlapping work.

The base seed is auto-generated from the clock each run (best for normal use). To
reproduce an exact run, pass the printed hex value:

```bash
./rckangaroo ... -seed 0x1A3F8C2D9E4B7061
```

Seeds are shown at startup and in the GPU monitor every 10 seconds:

```
Base seed: 0x1A3F8C2D9E4B7061
GPU 0 seed: 0x1A3F8C2D9E4B7061
GPU 1 seed: 0x9E3779B96C752A45
GPU 2 seed: 0x3C6EF372EDABAE7A
```

**Work file resume behavior:** Fresh seed on every resume. This is intentional — new
trajectories add coverage on top of accumulated DPs rather than retracing old paths.

---

### 2. Solve Snapshot
When a key is found, a snapshot prints before the private key showing which seeds
were active and total DPs at the moment of collision:

```
*** KEY FOUND ***
DPs accumulated: 46302405
  GPU 0  seed: 0x00000000133F5650  kangaroos: 917504
  GPU 1  seed: 0x9E3779B96C752A45  kangaroos: 917504
  GPU 2  seed: 0x3C6EF372EDABAE7A  kangaroos: 917504

PRIVATE KEY: 00000000000000000000000000000000000000000000EA1A5C66DCC11B5AD180
```

---

### 3. Safe Ctrl+C Save
The original signal handler called `fwrite`/`printf` directly — undefined behavior
that caused "Failed to write DP records" whenever Ctrl+C arrived while the heap lock
was held (common in a CUDA program).

**Fix:** Signal handler only sets a `volatile sig_atomic_t` flag. The main loop detects
it, stops the DP processor thread first (eliminating the concurrent `push_back` race),
then calls `Save()` from the main thread with no contention.

Exit sequence on Ctrl+C:
```
Interrupted! Saving progress...
Work file saved: puzzle135_dp22.work (711249100800 ops, 69509861 DPs)
Work file saved successfully
Stopping work ...
```

---

### 4. RAM-Based Auto-Switch Cap
For low-DP long runs (DP=22 on puzzle 135) the theoretical DP table would require
57 trillion entries — far exceeding available RAM. The solver now caps the auto-switch
threshold at available RAM:

```
ram_cap_dps = (120 GB) / 36 bytes per record = ~3.4 billion DPs
```

At 93% of this cap the solver switches to pure-wild mode (tame DPs dropped, wilds
continue accumulating). The W-W buffer is capped at 5% of this threshold.

---

### 5. J3 Dead-Zone Fix
The loop-escape kernel selected jump table entries with:

```cpp
// BUG: 0x2FF mask clears bit 8, making entries 256-511 unreachable
u32 jmp_ind = x0[0] & (JMP_CNT - 1);  // JMP_CNT=768, mask=0x2FF

// FIX: modulo always covers the full 0-767 range
u32 jmp_ind = (u32)(x0[0] % JMP_CNT);
```

256 out of 768 jump table entries (33%) were silently never used.

---

### 6. Speed=0 Fix at High DP
At DP=22 the GPU generates a DP only once every ~4 million steps. For most kernel
launches `cnt=0` and `AddPointsToList` was never called, so `PntTotalOps` stayed
at 0 and speed displayed as 0 MKeys/s despite full GPU utilization.

**Fix:** `AddPointsToList` is now called unconditionally on every launch so
`PntTotalOps` increments correctly.

---

### 7. Loop Escape Visibility
Loop escape count is tracked per GPU and shown in the stats line:

```
MAIN: Speed: 7577 MKeys/s ... Loops: 847293 ...
```

Typical rate of 3,000-4,500 loop escapes/sec is stable and expected.

---

### 8. Default DP=14
Default `gDP` changed from 0 to 14. The DP warning only fires if the user sets a
value below the calculated optimum for the range.

---

### 9. fread Return Value Checks
All `fread()` calls in XorFilter.cpp now check return values and return false on
short reads, eliminating compiler warnings and catching file corruption early.

---

### 10. utils.cpp Duplicate Define Fix
A legacy `#ifndef` block defined `DB_FIND_LEN=5` and `DB_REC_LEN=28` (wrong values),
shadowing the canonical definitions (`DB_FIND_LEN=9`, `DB_REC_LEN=32`). Removed.

---

## Work File Resume

Work files accumulate DPs across sessions. On resume:
- All saved DPs reload into the hash table
- New random seed generates fresh trajectories for all 3 GPUs
- New DPs collide against all stored DPs including those from previous runs

Fresh trajectories on resume is intentional — each session adds new territory on top
of what was already accumulated rather than retracing old paths.

---

## Proven Results

| Puzzle | K-Factor | Speed      | DPs at Solve | HEX Key (suffix)           | Status    |
|--------|----------|------------|--------------|----------------------------|-----------|
| 85     | 0.505    | ~7.92 GK/s | 29,796,035   | `...11720C4F018D51B8CEBBA8` | ✅ Solved |
| 80     | 0.605    | ~7.95 GK/s | 23,321,074   | `...EA1A5C66DCC11B5AD180`   | ✅ Solved |
| 80     | 0.777    | ~7.95 GK/s | 31,549,672   | `...EA1A5C66DCC11B5AD180`   | ✅ Solved |
| 80     | 1.114    | —          | 46,302,405   | —                           | ✅ Solved |
| 80     | 0.543    | —          | —            | —                           | ✅ Solved |
| 75     | 0.782    | ~4.72 GK/s | —            | `...4C5CE114686A1336E07`    | ✅ Solved |

Puzzle 135 — in progress.

---

## Example Run — Puzzle 80

```
./rckangaroo -range 80 -start 80000000000000000000 \
  -pubkey 037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc \
  -dp 14 -gpu 012 -cpu 68
```

**Hardware:** 3× RTX 3060 (28 CU each)  
**Total speed:** ~7.95 GK/s  
**K-Factor at solve:** 0.605 ✓ (ahead of schedule — solved at 60% of expected ops)  
**DPs at solve:** 23,321,074 / 77,175,193 (30.2%)  
**Time to solve:** 1 min 28 s

```
GPU 0:  2.52 GK/s  |   66°C  |  169W  |  100% util  |  PCI 3
         seed: 0x9E3779B96B227B7B
GPU 1:  2.56 GK/s  |   62°C  |  170W  |  100% util  |  PCI 4
         seed: 0x3C6EF372EAFCFF44
GPU 2:  2.52 GK/s  |   72°C  |  168W  |   97% util  |  PCI 132
         seed: 0xDAA66D2C69B77351
CPU:  348.2 MK/s
Total:  7.95 GK/s  |  Avg Temp: 66°C  |  Power: 507W
K-Factor:  0.484  OK (ahead of schedule)
DPs:  23321074 / 77175193 (30.2%)  |  Rate:  335325/s
ETA:  0d 22h 47m  |  Ops: 2^39.16 / 2^40.20
```

```
┌─────────────────────────────────────────┐
│           SESSION SUMMARY               │
├─────────────────────────────────────────┤
│ Status   : SOLVED ✓                     │
│ Time     : 0d 00h 01m 28s              │
│ Ops      : 2^39.275                    │
│ Avg Speed: 7520 MK/s                   │
│ Peak Spd : 7521 MK/s                   │
│ Solve K  : 0.605                       │
│ Errors   : 0                           │
│ CSV log  : kfactor_log.csv             │
└─────────────────────────────────────────┘
```

```
╔════════════════════════════════════════════════════════════════════════╗
║                               KEY FOUND                                ║
╠════════════════════════════════════════════════════════════════════════╣
║  HEX: 00000000000000000000000000000000000000000000EA1A5C66DCC11B5AD180 ║
║  WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZiAPRB6KvN7FnKsw69PP7vW             ║
║  ADR: 14LSbY9ZkRK3DDGyPcWYsYFAQtiVUoDkwV                               ║
╠════════════════════════════════════════════════════════════════════════╣
║  SWP: bc1q6r28u9cupz94d5qvhxgt45yascge94yzxn4e6x                       ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Example Run — Puzzle 85

```
./rckangaroo -range 85 -start 1000000000000000000000 \
  -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a \
  -dp 16 -gpu 012 -cpu 68
```

**Hardware:** 3× RTX 3060 (28 CU each) + 68 CPU threads  
**Total speed:** ~7.92 GK/s (6,594 GPU + 348 CPU MK/s)  
**K-Factor at solve:** 0.505 ✓ (ahead of schedule)  
**DPs at solve:** 29,796,035 / 109,142,205 (27.3%)  
**Time to solve:** ~6 min

```
GPU 0:  2.51 GK/s  │  67°C  │  167W  │  100% util  │  PCI 3  │  seed: 0x9E3779B96CA5B956
GPU 1:  2.55 GK/s  │  64°C  │  168W  │  100% util  │  PCI 4  │  seed: 0x3C6EF372ED7B3D69
GPU 2:  2.51 GK/s  │  71°C  │  169W  │  100% util  │  PCI 132│  seed: 0xDAA66D2C6E30B17C
CPU:  348.2 MK/s
Total:  7.92 GK/s  │  Avg Temp: 67°C  │  Power: 504W
K-Factor:  0.437  ✓ (ahead of schedule)
```

```
╔════════════════════════════════════════════════════════════════════════╗
║                               KEY FOUND                                ║
╠════════════════════════════════════════════════════════════════════════╣
║  HEX: 00000000000000000000000000000000000000000011720C4F018D51B8CEBBA8 ║
║  WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZkCnRjpBspr8M5K8YnUtDNa             ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Build

```bash
make -j$(nproc)
```

Requires CUDA toolkit, g++ with C++17, and optionally `libnvidia-ml` for GPU monitoring.
