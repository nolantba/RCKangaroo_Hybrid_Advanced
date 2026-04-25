# RCKangaroo Hybrid Advanced

**Pollard's Kangaroo ECDLP solver for secp256k1 — GPU + CPU hybrid**

Based on [RetiredCoder's kangaroo](https://github.com/RetiredCoder/RCKangaroo). This fork targets Bitcoin Puzzle 135 and includes a series of correctness fixes, stability improvements, and GPU performance enhancements.

---

## Features

- **Multi-GPU parallel search** — each GPU runs independent seeds and jump tables, no overlapping trajectories
- **CPU+GPU hybrid** — Xeon/Threadripper CPU threads contribute alongside GPU kangaroos
- **PTX-optimized MulModP** — 8-mul ILP burst structure eliminates IMAD.WIDE stall chains on SM 8.6
- **Jacobian coordinates** — optional (`USE_JACOBIAN=1`) for faster point arithmetic
- **Smart restart** — auto-restarts with fresh trajectories when K-factor drifts above threshold
- **RAM-aware DP cap** — auto-switches to pure-wild mode before the DP table exhausts available RAM
- **Safe Ctrl+C save** — signal handler sets a flag only; work file written from main thread with no lock contention
- **Work file resume** — fresh seeds on resume intentionally adds new coverage on top of stored DPs

---

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA SM 8.0+ (RTX 30xx/40xx recommended) |
| CUDA | 11.5+ |
| Compiler | g++ 11, nvcc |
| RAM | 32 GB+ (128 GB recommended for DP=22 on Puzzle 135) |
| OS | Linux |

---

## Build

```bash
# Release build (SM 8.6, Jacobian on)
make SM=86 USE_JACOBIAN=1 PROFILE=release -j

# Debug build
make SM=86 USE_JACOBIAN=1 PROFILE=debug -j

# Clean
make clean
```

Other SM targets: `SM=80` (A100), `SM=89` (RTX 40xx), `SM=75` (RTX 20xx).

---

## Usage

```bash
./rckangaroo \
  -range 135 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -dp 22 \
  -gpu 012 \
  -cpu 68 \
  -workfile puzzle135_dp22.work \
  -autosave 300
```

### Key flags

| Flag | Description |
|------|-------------|
| `-range N` | Bit range of the target key |
| `-pubkey HEX` | Compressed public key |
| `-dp N` | Distinguished point threshold (higher = less RAM, longer detection lag) |
| `-gpu 012` | GPU indices to use |
| `-cpu N` | Number of CPU threads |
| `-workfile FILE` | Save/resume file for DPs |
| `-autosave N` | Auto-save interval in seconds |
| `-seed 0xHEX` | Fixed base seed (default: random each run) |

### Choosing DP

At ~7 GK/s on three RTX 3060s:

| DP | DPs/sec | Detection lag | RAM at 3B DPs |
|----|---------|---------------|---------------|
| 18 | ~30,000 | ~1.7 min | ~108 GB |
| 22 | ~1,900  | ~26 min  | ~108 GB |
| 24 | ~470    | ~106 min | ~108 GB |

DP=22 is the sweet spot for this hardware — fits within 128 GB RAM and keeps detection lag under 30 minutes.

---

## Puzzle files

Pre-formatted puzzle inputs are in the `puzzles/` directory:

```
puzzles/puzzle135.txt   ← primary target
puzzles/puzzle125.txt
puzzles/puzzle120.txt
...
```

---

## Proven results

| Puzzle | K-Factor | Speed      | Status     |
|--------|----------|------------|------------|
| 85     | 0.505    | ~7.92 GK/s | ✅ Solved  |
| 80     | 0.605    | ~7.95 GK/s | ✅ Solved  |

---

## Bug fixes over base

1. **J3 dead-zone** — bitmask `0x2FF` made jump entries 256–511 unreachable (33% of the table unused). Fixed with modulo.
2. **Speed=0 at high DP** — `PntTotalOps` only incremented when DPs were found; at DP=22 that's once per ~4M steps. Now incremented every kernel launch.
3. **Ctrl+C data corruption** — signal handler called `fwrite`/`printf` directly while heap lock was held. Fixed: handler sets a flag only, save happens from main thread.
4. **utils.cpp shadowed defines** — legacy `#ifndef` block defined `DB_FIND_LEN=5`/`DB_REC_LEN=28`, shadowing the correct values of 9/32. Removed.
5. **fread return unchecked** — XorFilter.cpp now validates all `fread()` return values.
6. **Peak speed display** — interval baseline was sliding on every display tick, preventing the 1-second window from ever accumulating. Fixed to only advance baseline on measurement.

---

## Project structure

```
├── RCKangaroo.cpp          Main loop, stats display, work file I/O
├── RCGpuCore_PTX.cu        GPU kernels (PTX path, SM 8.0–8.8)
├── RCGpuCore.cu            GPU kernels (generic path)
├── RCGpuUtils.h            PTX MulModP, field arithmetic macros
├── defs.h                  Block size, PNT_GROUP_CNT, compile-time config
├── GpuKang.cpp/h           GPU kangaroo management
├── CpuKang.cpp/h           CPU kangaroo threads
├── Ec.cpp/h                secp256k1 point arithmetic
├── WorkFile.cpp/h          DP save/resume
├── XorFilter.cpp/h         XOR filter for DP deduplication
├── GpuMonitor.cpp/h        Per-GPU power/temp/util display
├── Makefile
├── puzzles/                Puzzle input files
├── scripts/                Build and benchmark scripts
└── msvc/                   Visual Studio solution/project files
```

---

## License

See [LICENSE.TXT](LICENSE.TXT).

---

## Repository

```bash
git clone https://github.com/nolantba/RCKangaroo_Hybrid_Advanced.git
cd RCKangaroo_Hybrid_Advanced
make SM=86 USE_JACOBIAN=1 PROFILE=release -j
```

---

## Example output

### Live monitor
```
================================================================
  GPU Performance Monitor
================================================================
GPU 0:  1.32 GK/s  |   65°C  |  170W  |  100% util  |  PCI 3
         seed: 0x409AC75694CC7A37
GPU 1:  1.34 GK/s  |   64°C  |  170W  |  100% util  |  PCI 4
         seed: 0xE2C34D9D1512FE08
GPU 2:  1.32 GK/s  |   71°C  |  169W  |  100% util  |  PCI 132
         seed: 0x040BD3C39659721D
CPU:  348.2 MK/s
Total:  4.33 GK/s  |  Avg Temp: 66°C  |  Power: 509W
K-Factor:  0.266  OK (ahead of schedule)  |  P(solve):  23.4%
K trend:  ▁  0.30–0.31
DPs:  10267255 / 13642775 (75.3%)  |  Rate:  172440/s  |  Buffer:  0 / 134217728 (0.0%)
ETA:  0d 00h 00m:37s  |  Ops: 2^35.79 / 2^37.70
⚡ Cost:  509W gpu  +  621W sys  =  1130W total  |  $0.1412/hr  |  Session: 0.20 cents  |  Daily: $3.39  |  Annual: $1237
Algo:  SOTA+ ✗  |  Jacobian ✓  |  LISSA ✓  |  R2-4V ✓  |  J1 ~2^37.6  |  J2=2^65  |  J3=2^63
Seq:   phi+0.8783  psi+0.2910  H5-scr  H7-scr  |  salt=0x13AC0121
       V0[-------*-]  V1[--*------]  V2[Owen/5]  V3[Owen/7]
Dist:  0.0μ         1.0μ (μ)         2.0μ
  V0 [----------##########|#########----------] phi  [0.50–1.50μ]
  V1 [------##############|############-------] psi  [0.33–1.67μ]
  V2 [-----###############|##############-----] H(5) [0.25–1.75μ]
  V3 [--------############|###########--------] H(7) [0.40–1.60μ]
  Rnd [-------#--#-##-###-#|#-#####-#-##-#--#--] random (clustered near μ, gaps at edges)
================================================================
```

### Solve
```
┌──────────────────────────────────────────┐
│           SESSION SUMMARY               │
├──────────────────────────────────────────┤
│ Status    : SOLVED ✓                   │
│ Time      : 0d 00h 00m 10s             │
│ Ops       : 2^36.166                   │
│ Avg Speed : 7015 MK/s                  │
│ Peak Spd  : 7015 MK/s                  │
│ Solve K   : 0.397                      │
│ Errors    : 0                          │
│ CSV log   : kfactor_log.csv            │
└──────────────────────────────────────────┘
╔════════════════════════════════════════════════════════════════════════╗
║                                KEY FOUND                                 ║
╠════════════════════════════════════════════════════════════════════════╣
║  HEX: 0000000000000000000000000000000000000000000004C5CE114686A1336E07 ║
║  WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3yHuxB7NfERRuDTNy2kbr             ║
║  ADR: 1G4uP62J6ViZQEtDGU1N9TKCsXaDN4kPiX                               ║
╠════════════════════════════════════════════════════════════════════════╣
║  SWP: bc1q54rdcf9d830k8j3m7c7sf489q99mqm2fx2apwp                       ║
╚════════════════════════════════════════════════════════════════════════╝
  Verification: k*G == pubkey  OK
```
