// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <signal.h>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "GpuHerdManager.h"
#include "CpuKang.h"
#include "WorkFile.h"
#include "GpuMonitor.h"
#include "R2_JumpTable.h"
#include "lissajous_jump_generator.hpp"

// ANSI color codes
#define RC_RESET   "\033[0m"
#define RC_BOLD    "\033[1m"
#define RC_GREEN   "\033[32m"
#define RC_BGREEN  "\033[1;32m"
#define RC_YELLOW  "\033[33m"
#define RC_BYELLOW "\033[1;33m"
#define RC_CYAN    "\033[36m"
#define RC_BCYAN   "\033[1;36m"
#define RC_MAGENTA "\033[35m"
#define RC_RED     "\033[31m"
#define RC_BRED    "\033[1;31m"
#define RC_WHITE   "\033[37m"
#define RC_BWHITE "\033[1;37m"

EcJMP EcJumps1[JMP_CNT];                          // base J1 table (CPU workers)
EcJMP EcJumps1PerGpu[MAX_GPU_CNT][JMP_CNT];       // per-GPU diverse J1 tables
#if USE_R2_4V
EcJMP EcJumps1_4v[JMP_CNT * 4];                   // 4-variant R2: layout [jmp_ind*4+variant]
#endif

// Lissajous monitor state — set once at startup, read by GpuMonitor
double g_lissa_j1_bits = 0.0;   // log2 of mean J1 jump size (after shift)
int    g_lissa_range   = 0;      // Range value (for J2/J3 display)

// R2-4V scramble state — set by BuildR2JumpTable4V, read by GpuMonitor
uint32_t g_r2_salt = 0;          // per-run scramble salt (shown in monitor)
double   g_r2_off0 = 0.0;        // V0 phi shift in [0,1)
double   g_r2_off1 = 0.0;        // V1 psi shift in [0,1)

EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
RCCpuKang* CpuKangs[128]; // Support up to 128 CPU threads
int CpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;

// Per-GPU same-GPU collision counters (populated in CheckNewPoints)
std::atomic<uint64_t> g_gpu_local_colls[MAX_GPU_CNT] = {};
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
volatile u64 DroppedDPs = 0;
volatile u64 TotalDPsGenerated = 0;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
bool gDP_manual = false;   // true when user passed -dp explicitly
double g_sys_overhead_w = 621.0;  // non-GPU system watts (CPU+board+RAM)
double g_kwh_rate       = 0.125;  // electricity cost $/kWh
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
int gCpuThreads; // Number of CPU threads to use
char gTamesFileName[1024];
int gTameRatioPct = 33;  // SOTA+ tame ratio optimization
int gTameBitsOffset = 4; // SOTA+ tame bits offset
double gMax;
bool gGenMode; //tames generation mode
bool g_wild_only = false;    // Wild-only mode: preloaded tames used as static traps, no new TAME kangaroos
u64  g_loaded_tame_cnt = 0;  // Number of tames loaded from file (shown in display)

// Smart-restart settings
static int    g_restart_count     = 0;
static double g_restart_k_thresh  = 1.20;  // restart when K exceeds this (adaptive)
static double g_restart_dp_thresh = 0.40;  // only restart if >=40% DPs collected

// ── Multi-attempt statistics ─────────────────────────────────────────────────
#define MAX_ATTEMPTS 64
struct AttemptRecord {
    int    attempt;       // attempt number (1-based)
    double peak_k;        // worst K seen this attempt
    double best_k;        // best K seen this attempt
    u64    ops;           // total ops done
    u64    duration_s;    // wall seconds
};
static AttemptRecord g_attempts[MAX_ATTEMPTS];
static int           g_attempt_idx   = 0;   // next free slot
static u64           g_attempt_start_ms = 0; // start of current attempt
static double        g_attempt_peak_k   = 0.0;
static double        g_attempt_best_k   = 1e18;
// ─────────────────────────────────────────────────────────────────────────────

// ── K-factor sparkline (last 32 samples, one per 10s stats tick) ─────────────
#define SPARK_LEN 32
double g_spark[SPARK_LEN] = {};
int    g_spark_head = 0;  // ring-buffer write index
int    g_spark_cnt  = 0;  // samples filled so far
// ─────────────────────────────────────────────────────────────────────────────

bool gIsOpsLimit;

// Save/Resume work file support
RCWorkFile* g_work_file = nullptr;
AutoSaveManager* g_autosave = nullptr;
std::string g_work_filename;
uint64_t g_autosave_interval = 60;  // Default: 60 seconds

// SOTA++ Herds mode
bool g_use_herds = false;
bool g_force_resume = false;  // skip pubkey check on workfile load (-force-resume)
time_t g_start_time = 0;
bool g_resume_mode = false;

// ── Session tracking ────────────────────────────────────────────────────────
static double   g_best_k        = 1e18;   // best K-factor seen this session
static double   g_session_peak_speed = 0; // peak MK/s seen
static u64      g_session_start_ms   = 0; // GetTickCount64() at solve start
static u64      g_session_ops_base   = 0; // PntTotalOps at session start (for accurate speed on resume)
static FILE*    g_kcsv_fp        = nullptr; // kfactor_log.csv handle

// ── Session persistence (.state file alongside .work file) ───────────────────
// Stores: restart_count, best_k, adaptive thresh, attempt history
// Written on every restart and on clean exit; loaded on resume.
static std::string g_state_filename;   // e.g. "puzzle135_dp22.state"
static std::string g_kangs_filename;   // e.g. "puzzle135_dp22.kangs" — db serialization companion

static void SaveSessionState()
{
    if (g_state_filename.empty()) return;
    FILE* f = fopen(g_state_filename.c_str(), "w");
    if (!f) return;
    fprintf(f, "restart_count=%d\n",    g_restart_count);
    fprintf(f, "best_k=%.6f\n",         g_best_k < 1e17 ? g_best_k : 0.0);
    fprintf(f, "k_thresh=%.4f\n",       g_restart_k_thresh);
    fprintf(f, "attempt_count=%d\n",    g_attempt_idx);
    for (int i = 0; i < g_attempt_idx; i++)
        fprintf(f, "attempt %d pk=%.4f bk=%.4f ops=%llu dur=%llu\n",
                g_attempts[i].attempt,
                g_attempts[i].peak_k, g_attempts[i].best_k,
                (unsigned long long)g_attempts[i].ops,
                (unsigned long long)g_attempts[i].duration_s);
    fclose(f);
}

static void LoadSessionState()
{
    if (g_state_filename.empty()) return;
    FILE* f = fopen(g_state_filename.c_str(), "r");
    if (!f) return;
    char line[256];
    int  loaded_attempts = 0;
    while (fgets(line, sizeof(line), f)) {
        double dv; int iv; unsigned long long u1, u2;
        if (sscanf(line, "restart_count=%d",  &iv)  == 1) g_restart_count     = iv;
        if (sscanf(line, "best_k=%lf",         &dv)  == 1 && dv > 0) g_best_k = dv;
        if (sscanf(line, "k_thresh=%lf",        &dv)  == 1 && dv > 0) g_restart_k_thresh = dv;
        if (sscanf(line, "attempt_count=%d",    &iv)  == 1) loaded_attempts = iv;
        int ai; double pk, bk;
        if (sscanf(line, "attempt %d pk=%lf bk=%lf ops=%llu dur=%llu",
                   &ai, &pk, &bk, &u1, &u2) == 5 && g_attempt_idx < MAX_ATTEMPTS) {
            g_attempts[g_attempt_idx] = { ai, pk, bk, (u64)u1, (u64)u2 };
            g_attempt_idx++;
        }
    }
    fclose(f);
    if (g_restart_count > 0)
        printf(RC_BCYAN "Session state loaded: %d prior restarts, best K=%.4f, thresh=%.2f\r\n" RC_RESET,
               g_restart_count, g_best_k < 1e17 ? g_best_k : 0.0, g_restart_k_thresh);
}
// ─────────────────────────────────────────────────────────────────────────────

// Background DP processing thread
volatile bool g_dp_thread_stop = false;
void CheckNewPoints(); // forward declaration

// Auto-switch to pure-wild mode when table reaches 93% of estimated capacity
// When true: tame DPs are silently dropped, preserving table space for wilds
volatile bool g_capacity_reached = false;
volatile u64  g_capacity_threshold = 0;  // set once est_dps_cnt is known

// W-W buffer: dedicated secondary hash for WILD DPs only.
// Gives a second, independent collision chance for every wild DP —
// the main db may already have a tame there, but ww_db may have a different
// wild that produces a W-W resolution.  Capped at 5% of est_dps_cnt to
// keep RAM overhead small.  When full we stop inserting but keep checking.
TFastBase ww_db;
volatile u64  g_ww_cap = 0;     // max entries in ww_db (5% of est)
volatile bool g_ww_full = false;
volatile u64  g_ww_count = 0;   // FAST counter — avoids O(16M) GetBlockCnt() in hot path

// Shutdown flag — set ONLY by signal handler, checked in main loop.
// sig_atomic_t is the only type guaranteed safe to write from a signal handler.
volatile sig_atomic_t g_shutdown_requested = 0;

// ── Background .kangs checkpoint ─────────────────────────────────────────────
// Periodically saves the full DP database in a background thread.
// DB_LOCK is held for the entire SaveToFile duration; dp inserts queue up in
// the 33M pPntList buffer (7+ hours capacity — a 3-min save is no problem).
volatile bool g_kangs_bg_stop     = false;  // signal thread to exit
volatile bool g_kangs_bg_saving   = false;  // true while bg save in progress
static   u64  g_kangs_bg_interval_s = 1800; // default 30 min; -ksave N overrides
#ifdef _WIN32
static CRITICAL_SECTION g_db_cs;
#define DB_LOCK()    EnterCriticalSection(&g_db_cs)
#define DB_UNLOCK()  LeaveCriticalSection(&g_db_cs)
#else
static pthread_mutex_t g_db_mutex = PTHREAD_MUTEX_INITIALIZER;
#define DB_LOCK()    pthread_mutex_lock(&g_db_mutex)
#define DB_UNLOCK()  pthread_mutex_unlock(&g_db_mutex)
#endif
// ─────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
static unsigned __stdcall dp_proc_thread(void*)
#else
static void* dp_proc_thread(void*)
#endif
{
    // Spin-drain pPntList as fast as possible — no Sleep so we never fall behind
    while (!g_dp_thread_stop)
    {
        CheckNewPoints();
#ifdef _WIN32
        SwitchToThread();   // Yield to other threads without sleeping
#else
        sched_yield();
#endif
    }
    // Final drain after stop signal
    CheckNewPoints();
#ifdef _WIN32
    return 0;
#else
    return nullptr;
#endif
}

// ── Background kangs checkpoint thread ───────────────────────────────────────
#ifdef _WIN32
static unsigned __stdcall kangs_bg_save_thread(void*)
#else
static void* kangs_bg_save_thread(void*)
#endif
{
    u64 last_save_ms = GetTickCount64();
    while (!g_kangs_bg_stop) {
        // Wake every 100 ms to stay responsive to stop/shutdown signals
        for (int i = 0; i < 100 && !g_kangs_bg_stop && !g_shutdown_requested; i++)
            Sleep(100);
        if (g_kangs_bg_stop || g_shutdown_requested) break;
        if (g_kangs_filename.empty()) continue;
        u64 now = GetTickCount64();
        if ((now - last_save_ms) < g_kangs_bg_interval_s * 1000ULL) continue;

        // Time to checkpoint
        g_kangs_bg_saving = true;
        u64 cnt = db.GetBlockCnt();
        printf("\r\n[Checkpoint] Saving %llu DPs to %s ...\r\n",
               (unsigned long long)cnt, g_kangs_filename.c_str());

        DB_LOCK();   // excludes dp_proc_thread from writing to db during fwrite
        bool ok = db.SaveToFile((char*)g_kangs_filename.c_str());
        DB_UNLOCK();

        if (ok)
            printf("[Checkpoint] Saved %llu DPs  (next checkpoint in %.0f min)\r\n",
                   (unsigned long long)cnt, g_kangs_bg_interval_s / 60.0);
        else
            printf("[Checkpoint] ERROR: failed to save %s!\r\n", g_kangs_filename.c_str());

        last_save_ms = GetTickCount64();
        g_kangs_bg_saving = false;
    }
#ifdef _WIN32
    return 0;
#else
    return nullptr;
#endif
}
// ─────────────────────────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
u32 __stdcall cpu_kang_thr_proc(void* data)
{
	RCCpuKang* Kang = (RCCpuKang*)data;
	// Use original RC implementation (best performance for this codebase)
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
void* cpu_kang_thr_proc(void* data)
{
	RCCpuKang* Kang = (RCCpuKang*)data;
	// Use optimized version: larger batches (5K) while preserving cache locality
	Kang->Execute_Optimized();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	
	PntTotalOps += ops_cnt;
	TotalDPsGenerated += pnt_cnt;
	
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		DroppedDPs += pnt_cnt;
		
		static u64 last_warning = 0;
		if (DroppedDPs - last_warning >= 10000)
		{
			printf("\n⚠️  WARNING: DP BUFFER OVERFLOW!\n");
			printf("    Dropped: %llu DPs (%.1f%% loss)\n",
			       (unsigned long long)DroppedDPs,
			       100.0 * DroppedDPs / TotalDPsGenerated);
			printf("    FIX: Use -dp %d (current: %d)\n\n", gDP + 2, gDP);
			last_warning = DroppedDPs;
		}
		
		csAddPoints.Leave();
		return;
	}
	
	memcpy(pPntList + (u64)GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	// Swap buffer pointers instead of memcpy — holds lock for microseconds, not 33ms.
	// New DPs will go into the formerly-empty pPntList2 while we process old pPntList.
	int cnt = PntIndex;
	PntIndex = 0;
	u8* process_buf = pPntList;
	pPntList = pPntList2;   // GPU threads now write into the empty buffer
	pPntList2 = process_buf; // We process the full buffer below, outside the lock
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = process_buf + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		// p[40] carries: low nibble = kangaroo type (TAME/WILD1/WILD2),
		//                high nibble = source GPU ID (1-based; 0 = unknown/CPU/old file)
		nrec.type = gGenMode ? TAME : p[40];
		u8 ktype = nrec.type & 0x0F; // kangaroo type stripped of GPU ID

		// Auto-switch: once the table is 93% full, drop incoming tame DPs.
		// Wilds still collide with the already-stored tames, so T-W collisions
		// continue. We just stop consuming remaining table space on new tames.
		if (g_capacity_reached && ktype == TAME)
			continue;

		DB_LOCK();
		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
		DB_UNLOCK();

		// DPs are persisted via db.SaveToFile() on autosave — no per-DP AddDP needed

		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			u8 ptype = pref->type & 0x0F; // stored type stripped of GPU ID

			// Same-GPU collision tracking: both high-nibble GPU IDs known and equal
			{
				int gpu_a = (pref->type >> 4) & 0x0F; // 0 = old file / unknown
				int gpu_b = (nrec.type  >> 4) & 0x0F;
				if (gpu_a > 0 && gpu_b > 0 && gpu_a == gpu_b) {
					int gpu_idx = gpu_a - 1; // back to 0-based
					if (gpu_idx < MAX_GPU_CNT)
						g_gpu_local_colls[gpu_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}

			if (ptype == ktype)
			{
				if (ptype == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (ptype != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = ktype;
				WildType = ptype;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = ktype;
			}

			bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
			if (!res)
			{
				bool w12 = ((ptype == WILD1) && (ktype == WILD2)) || ((ptype == WILD2) && (ktype == WILD1));
				if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
					;// ToLog("W1 and W2 collides in mirror");
				else
				{
					printf("Collision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}
			gSolved = true;
			break;
		}
		if (gSolved) break;

		// ---------------------------------------------------------------
		// W-W buffer: secondary collision check for wild DPs only.
		// The main db above already handles T-W and same-type W-W.
		// The ww_db gives a second lookup specifically for cross-type
		// WILD1-WILD2 collisions whose x-prefix might map to a tame slot
		// in the main db (and thus miss the wild there).
		// ---------------------------------------------------------------
		if (!gGenMode && (ktype == WILD1 || ktype == WILD2))
		{
			// When ww_db is full: lookup-only (no insert) — still catch W-W collisions
			// against already-stored entries without growing memory further.
			DBRec* ww_pref;
			if (g_ww_full)
				ww_pref = (DBRec*)ww_db.FindDataBlock((u8*)&nrec);
			else
			{
				ww_pref = (DBRec*)ww_db.FindOrAddDataBlock((u8*)&nrec);
				if (ww_pref == NULL)
				{
					// Newly inserted — fast counter, no GetBlockCnt() ever
					g_ww_count++;
					if (g_ww_cap > 0 && g_ww_count >= g_ww_cap)
						g_ww_full = true;
				}
			}
			if (ww_pref != NULL)
			{
				// Restore the 3-byte prefix that TFastBase strips
				DBRec tmp_ww;
				memcpy(&tmp_ww, &nrec, 3);
				memcpy(((u8*)&tmp_ww) + 3, ww_pref, sizeof(DBRec) - 3);
				ww_pref = &tmp_ww;

				u8 wwtype = ww_pref->type & 0x0F;

				// Only resolve if the two wilds are different types (WILD1 vs WILD2)
				// and have different distances (skip same-kang duplicates)
				bool cross = ((wwtype == WILD1 && ktype == WILD2) ||
				              (wwtype == WILD2 && ktype == WILD1));
				if (cross && *(u64*)ww_pref->d != *(u64*)nrec.d)
				{
					// W-W resolution: neither side is TAME → ShiftRight(1) path
					EcInt ww_w, ww_t;
					memcpy(ww_w.data, ww_pref->d, sizeof(ww_pref->d));
					if (ww_pref->d[21] == 0xFF) memset(((u8*)ww_w.data) + 22, 0xFF, 18);
					memcpy(ww_t.data, nrec.d, sizeof(nrec.d));
					if (nrec.d[21] == 0xFF) memset(((u8*)ww_t.data) + 22, 0xFF, 18);

					bool res = Collision_SOTA(gPntToSolve, ww_t, ktype, ww_w, wwtype, false) ||
					           Collision_SOTA(gPntToSolve, ww_t, ktype, ww_w, wwtype, true);
					if (res)
					{
						printf("W-W buffer collision solved!\r\n");
						gSolved = true;
						break;
					}
				}
			}
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	// Timing (session duration used for time display; session_ms still drives K-factor delta)
	u64 session_ms = (g_session_start_ms > 0)
		? (GetTickCount64() - g_session_start_ms) : (GetTickCount64() - tm_start);
	if (session_ms == 0) session_ms = 1;
	u64 elapsed_ms = GetTickCount64() - tm_start; // total elapsed (for time display)
	if (elapsed_ms == 0) elapsed_ms = 1;

	// Use per-GPU instantaneous speeds from the monitor (same source as GPU monitor display).
	// This ensures the status line and GPU monitor always agree on speed.
	int cpu_speed = 0;
	int gpu_speed = 0;
	int total_speed = 0;
	if (g_gpu_monitor) {
		SystemStats mon_snap = g_gpu_monitor->GetSystemStats();
		cpu_speed = (int)mon_snap.cpu_speed_mkeys;
		for (int _si = 0; _si < GpuCnt; _si++)
			gpu_speed += (int)mon_snap.gpu_stats[_si].speed_mkeys;
		total_speed = gpu_speed + cpu_speed;
	}
	// Fallback: session-average if monitor unavailable or not yet populated
	if (total_speed == 0) {
		u64 session_new_ops = (PntTotalOps > g_session_ops_base) ? (PntTotalOps - g_session_ops_base) : PntTotalOps;
		total_speed = (int)(session_new_ops / (session_ms / 1000.0) / 1000000.0);
		gpu_speed   = total_speed - cpu_speed;
	}

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec_total = 0xFFFFFFFFFFFFFFFFull;
	if (total_speed)
		exp_sec_total = (u64)((exp_ops / 1000000) / total_speed); //in sec
	u64 exp_days = exp_sec_total / (3600 * 24);
	int exp_hours = (int)(exp_sec_total - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec_total - exp_days * (3600 * 24) - exp_hours * 3600) / 60;
	int exp_sec = (int)(exp_sec_total - exp_days * (3600 * 24) - exp_hours * 3600 - exp_min * 60);

	// Left time = since last restart; right time = total
	u64 sec_total = session_ms / 1000;
	u64 days = sec_total / (3600 * 24);
	int hours = (int)(sec_total - days * (3600 * 24)) / 3600;
	int min = (int)(sec_total - days * (3600 * 24) - hours * 3600) / 60;
	int sec = (int)(sec_total - days * (3600 * 24) - hours * 3600 - min * 60);
	u64 tot_sec = elapsed_ms / 1000;
	u64 tot_days = tot_sec / (3600 * 24);
	int tot_hours = (int)(tot_sec % 86400) / 3600;
	int tot_min = (int)(tot_sec % 3600) / 60;
	int tot_sec2 = (int)(tot_sec % 60);

	// Show total speed with GPU+CPU breakdown
	const char* mode_tag = gGenMode ? "GEN: " : (IsBench ? "BENCH: " :
	    (g_wild_only ? "MAIN[WILDONLY]: " : (g_capacity_reached ? "MAIN[WILD]: " : "MAIN: ")));

	// Show dropped DP warning inline if any DPs were lost to buffer overflow
	if (DroppedDPs > 0)
		printf("  *** DROPPED DPs: %llu (%.1f%%) — increase -dp value! ***\r\n",
			(unsigned long long)DroppedDPs,
			100.0 * DroppedDPs / (TotalDPsGenerated ? TotalDPsGenerated : 1));

	// Accumulate loop escapes across all GPUs for display
	u64 total_loop_escapes = 0;
	for (int i = 0; i < GpuCnt; i++)
		total_loop_escapes += GpuKangs[i]->GetTotalLoopEscapes();

	// Elapsed color: green while under projected, red once past it
	const char* elapsed_color = (exp_sec_total == 0xFFFFFFFFFFFFFFFFull || sec_total <= exp_sec_total)
	                            ? RC_BGREEN : RC_BRED;
	printf(RC_CYAN "%s" RC_RESET "Speed: " RC_BOLD "%d" RC_RESET " MKeys/s (%d GPU + %d CPU), Err: %d, DPs: " RC_YELLOW "%lluK/%lluK" RC_RESET ", WW: %lluK, Loops: %llu, Buf: %d/%d, Time: %s%llud:%02dh:%02dm:%02ds" RC_RESET "/" RC_BGREEN "%llud:%02dh:%02dm:%02ds" RC_RESET "\r\n",
		mode_tag,
		total_speed, gpu_speed, cpu_speed,
		gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000,
		g_ww_count/1000,
		total_loop_escapes,
		PntIndex, MAX_CNT_LIST,
		elapsed_color, days, hours, min, sec,
		exp_days, exp_hours, exp_min, exp_sec);

	// ── Gen mode: DP count progress ─────────────────────────────────────────
	if (gGenMode) {
		u64 dp_now = db.GetBlockCnt();
		double dp_rate = (sec_total > 0) ? (double)dp_now / sec_total : 0.0;
		if (gMax > 0.0) {
			u64 dp_target = (u64)gMax;
			double pct = (dp_target > 0) ? 100.0 * dp_now / dp_target : 0.0;
			u64 dp_remain = (dp_now < dp_target) ? (dp_target - dp_now) : 0;
			u64 eta_s = (dp_rate > 0 && dp_remain > 0) ? (u64)(dp_remain / dp_rate) : 0;
			printf(RC_BGREEN "GEN progress: %llu / %llu tames (%.1f%%)  |  Rate: %.0f/s  |  ETA: %llud %02dh %02dm\r\n" RC_RESET,
			       (unsigned long long)dp_now, (unsigned long long)dp_target, pct,
			       dp_rate,
			       eta_s / 86400, (int)((eta_s % 86400) / 3600), (int)((eta_s % 3600) / 60));
		} else {
			printf(RC_BGREEN "GEN progress: %llu tames collected  |  Rate: %.0f/s  |  (no target, Ctrl+C to stop)\r\n" RC_RESET,
			       (unsigned long long)dp_now, dp_rate);
		}
	}
	// ── K-factor tracking + live ETA ─────────────────────────────────────────
	if (!IsBench && !gGenMode && total_speed > 0 && exp_ops > 0) {
		double cur_k = (double)PntTotalOps / exp_ops;
		// update session bests (only after 10% progress to avoid noisy early readings)
		double dp_progress = (exp_ops > 0) ? (double)PntTotalOps / exp_ops : 0.0;
		if (cur_k > 0.01 && cur_k < g_best_k && dp_progress >= 0.10) g_best_k = cur_k;
		// Peak speed: max instantaneous 1-second throughput seen this session.
		// _prev_ops/_prev_ms only advance when a measurement fires (>= 1s window),
		// so sub-second display ticks accumulate into the window instead of sliding past it.
		{
			static u64 _prev_ops = 0;
			static u64 _prev_ms  = 0;
			static bool _initialized = false;
			// Seed baseline on first call, or after a smart restart resets ops
			if (!_initialized) {
				_prev_ops    = PntTotalOps;
				_prev_ms     = session_ms;
				_initialized = true;
			} else if (PntTotalOps >= _prev_ops && session_ms > _prev_ms) {
				u64 d_ops = PntTotalOps - _prev_ops;
				u64 d_ms  = session_ms  - _prev_ms;
				// Fire once per second; baseline only advances when a measurement is taken
				// so sub-second ticks keep accumulating into the same window
				if (d_ms >= 1000) {
					double inst = d_ops / (d_ms / 1000.0) / 1e6;
					if (inst > 1e9) inst = 1e9; // sanity cap
					if (inst > (double)g_session_peak_speed)
						g_session_peak_speed = (double)inst;
					// Only reset baseline after a successful measurement
					_prev_ops = PntTotalOps;
					_prev_ms  = session_ms;
				}
				// If d_ms < 1000: do NOT update _prev_ops/_prev_ms — let the window grow
			} else {
				// PntTotalOps reset (smart restart): reset baseline
				_prev_ops    = 0;
				_prev_ms     = 0;
				_initialized = false;
				return; // re-seed on next call
			}
		}

		// ETA based on remaining ops at current speed
		double remaining_ops = exp_ops - (double)PntTotalOps;
		if (remaining_ops < 0) remaining_ops = 0;
		u64 eta_s = (u64)(remaining_ops / ((double)total_speed * 1e6));
		u64 eta_d = eta_s / 86400;
		int eta_h = (int)((eta_s % 86400) / 3600);
		int eta_m = (int)((eta_s % 3600) / 60);

		// DP progress %
		double dp_pct = (est_dps_cnt > 0)
			? 100.0 * (double)db.GetBlockCnt() / (double)est_dps_cnt : 0.0;

		// K-factor color
		const char* kc = (cur_k < 1.15) ? RC_BGREEN : (cur_k < 1.3) ? RC_BYELLOW : RC_BRED;
		// Solve probability: P = 1 - exp(-K)  (exponential collision model)
		double solve_prob = (1.0 - exp(-cur_k)) * 100.0;
		int eta_sec = (int)(eta_s % 60);
		printf("       K: %s%.3f" RC_RESET "  │  ETA: " RC_CYAN "%llud:%02dh:%02dm:%02ds" RC_RESET
		       "  │  DP%%: " RC_YELLOW "%.1f%%" RC_RESET
		       "  │  P(solve): " RC_MAGENTA "%.1f%%" RC_RESET
		       "  │  Best K: " RC_BGREEN "%.3f" RC_RESET "\r\n",
		       kc, cur_k, eta_d, eta_h, eta_m, eta_sec, dp_pct, solve_prob,
		       g_best_k < 1e17 ? g_best_k : 0.0);

		// CSV log
		if (g_kcsv_fp) {
			fprintf(g_kcsv_fp, "%llu,%d,%.4f,%llu,%.2f,%llu,%.2f\n",
				(unsigned long long)sec_total,
				total_speed,
				cur_k,
				(unsigned long long)db.GetBlockCnt(),
				dp_pct,
				(unsigned long long)eta_s,
				solve_prob);
			fflush(g_kcsv_fp);
		}
	}
	// ─────────────────────────────────────────────────────────────────────────
}

static void PrintSessionSummary(bool solved, double final_k = -1.0);  // forward declaration
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 10) || (DP > 60))
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);

	// Initialize auto-switch threshold: 93% of expected DP count,
	// but cap at a RAM-safe limit so low-DP runs (e.g. DP=22 on puzzle 135)
	// don't try to fill a 50-trillion-entry table and crash.
	// Each DP record = DB_REC_LEN (32) bytes + ~4 bytes overhead.
	// We reserve 90% of a configurable RAM budget (default 120 GB) for the DB.
	u64 est_dps_cnt_init = (u64)(ops / dp_val);
	const double RAM_BUDGET_GB = 120.0; // leave ~8 GB headroom for OS + program
	u64 ram_cap_dps = (u64)((RAM_BUDGET_GB * 1024.0 * 1024.0 * 1024.0) / 36.0); // DB_REC_LEN(32) + 4 overhead
	u64 threshold_93pct = (u64)(0.93 * (double)est_dps_cnt_init);
	g_capacity_threshold = (threshold_93pct < ram_cap_dps) ? threshold_93pct : ram_cap_dps;
	g_capacity_reached = false;
	if (threshold_93pct > ram_cap_dps)
		printf("Auto-switch threshold: %llu DPs (RAM cap — 93%% of expected %llu exceeds 120 GB)\r\n",
			g_capacity_threshold, est_dps_cnt_init);
	else
		printf("Auto-switch threshold: %llu DPs (93%% of expected %llu)\r\n",
			g_capacity_threshold, est_dps_cnt_init);

	// W-W buffer: cap at 5% of threshold (not raw est_dps_cnt which may be huge)
	g_ww_cap = (u64)(0.05 * (double)g_capacity_threshold);
	if (g_ww_cap < 1000) g_ww_cap = 1000;  // floor for small ranges
	g_ww_full = false;
	g_ww_count = 0;
	ww_db.Clear();
	printf("W-W buffer capacity: %llu DPs (5%% of threshold)\r\n", g_ww_cap);

	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}

	u64 total_kangs = 0;
	if (GpuCnt > 0)
	{
		total_kangs = GpuKangs[0]->CalcKangCnt();
		for (int i = 1; i < GpuCnt; i++)
			total_kangs += GpuKangs[i]->CalcKangCnt();
	}
	total_kangs += CpuCnt * CPU_KANGS_PER_THREAD;
	printf("Total kangaroos: %llu (GPU: %llu, CPU: %d)\r\n", total_kangs, total_kangs - CpuCnt * CPU_KANGS_PER_THREAD, CpuCnt * CPU_KANGS_PER_THREAD);

	// ── Auto-DP selection ────────────────────────────────────────────────────
	// Optimal DP = Range/2 − log2(total_kangs) − 1
	// Targets ~2 DPs per kangaroo per expected path length, minimising both
	// RAM waste (too low) and GPU idle time between saves (too high).
	// Clamped to [14, 58] to stay within hardware and RAM limits.
	if (!gDP_manual) {
		int auto_dp = (int)floor(Range / 2.0 - log2((double)total_kangs) - 1.0);
		auto_dp = (auto_dp < 14) ? 14 : (auto_dp > 58) ? 58 : auto_dp;
		DP     = auto_dp;
		gDP    = (u32)auto_dp;
		dp_val = (double)(1ull << DP);
		// Recalculate RAM estimate with new DP
		ram  = (32.0 + 4.0 + 4.0) * ops / dp_val;
		ram += sizeof(TListRec) * 256.0 * 256.0 * 256.0;
		ram /= (1024.0 * 1024.0 * 1024.0);
		printf(RC_BGREEN "Auto-DP: %d" RC_RESET " (%.3f GB RAM est.) — pass -dp N to override\r\n", DP, ram);
	}
	// ── Recalculate est_dps_cnt_init and thresholds now that dp_val is final ─
	// (must happen AFTER auto-DP, otherwise the restart dp_pct check uses the
	//  wrong denominator and the smart restart never fires on small puzzles)
	est_dps_cnt_init = (u64)(ops / dp_val);
	{
		const double RAM_BUDGET_GB2 = 120.0;
		u64 ram_cap2   = (u64)((RAM_BUDGET_GB2 * 1024.0 * 1024.0 * 1024.0) / 36.0);
		u64 thresh_93  = (u64)(0.93 * (double)est_dps_cnt_init);
		g_capacity_threshold = (thresh_93 < ram_cap2) ? thresh_93 : ram_cap2;
	}
	g_ww_cap = (u64)(0.05 * (double)g_capacity_threshold);
	if (g_ww_cap < 1000) g_ww_cap = 1000;
	// ────────────────────────────────────────────────────────────────────────

	double path_single_kang = ops / total_kangs;
	double DPs_per_kang = path_single_kang / dp_val;

	// DP balance report
	printf("Estimated DPs per kangaroo: %.3f", DPs_per_kang);
	if (DPs_per_kang < 2.0)
		printf(" " RC_BYELLOW "⚠ very low — DP may be too high for this range/herd size" RC_RESET "\r\n");
	else if (DPs_per_kang < 5.0)
		printf(" " RC_YELLOW "✓ low-overhead (fast detection, higher RAM)" RC_RESET "\r\n");
	else if (DPs_per_kang < 15.0)
		printf(" " RC_BGREEN "✓ optimal" RC_RESET "\r\n");
	else if (DPs_per_kang < 30.0)
		printf(" " RC_YELLOW "✓ good balance" RC_RESET "\r\n");
	else
		printf(" " RC_BYELLOW "⚠ high — consider lowering DP (slower detection, less RAM)" RC_RESET "\r\n");

	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			g_loaded_tame_cnt = db.GetBlockCnt();
			printf("tames loaded: %llu records\r\n", (unsigned long long)g_loaded_tame_cnt);
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clearing\r\n");
				db.Clear();
				g_loaded_tame_cnt = 0;
			}
			else if (g_wild_only)
			{
				printf(RC_BGREEN "Wild-only mode: %llu tames loaded as static traps — running WILDs only\r\n" RC_RESET,
				       (unsigned long long)g_loaded_tame_cnt);
				g_capacity_reached = true;  // Discard all new TAME DPs from GPU/CPU
			}
		}
		else
			printf("tames loading failed\r\n");
	}

	SetRndSeed(0); //use same seed to make tames from file compatible
	// Restore ops counter from work file on resume so K-factor/ETA are accurate
	PntTotalOps = (g_resume_mode && TotalOps > 0) ? TotalOps : 0;
	PntIndex = 0;

//prepare jumps
	// Jump2/3: SOTA loop escape — kept large for cycle breaking (unchanged)
	// Jump1: Lissajous-distributed — Teske-optimal mean at 2^(Range/2)
	EcInt minjump, t;

	g_lissa_range = Range;  // store for monitor display

#if USE_LISSAJOUS
	// --- J1: Lissajous jump table (Teske-optimal distribution) ---
	// 512 entries shaped by quasi-random Lissajous curve (freq ratios 1:φ:e).
	// Irrational frequencies guarantee the curve fills its space without repeating,
	// giving equidistributed jump sizes with no accidental clusters.
	// Base range [2^33, 2^34] is safe for uint64_t; shifted up to 2^(Range/2).
	// Mean ≈ 2^(Range/2+0.5) — Teske theoretical optimum for kangaroo.
	{
		int jump1_exp = Range / 2;          // Teske optimal mean: sqrt(range)
		if (jump1_exp < 10) jump1_exp = 10;
		// base_bits: largest safe power-of-2 anchor that fits in uint64_t and
		// is <= jump1_exp so lissa_shift is always >= 0 for any puzzle size.
		// Puzzle 135: base=33, shift=34 → mean ~2^67.5 ✓
		// Puzzle 64:  base=32, shift=0  → mean ~2^32.5 ✓
		// Puzzle 40:  base=19, shift=1  → mean ~2^19.5 ✓
		int base_bits   = (jump1_exp < 33) ? jump1_exp : 33;
		int lissa_shift = jump1_exp - base_bits;  // always >= 0

		LissajousJumpGenerator::Config lissa_cfg;
		lissa_cfg.freq_x = 1.0;
		lissa_cfg.freq_y = 1.6180339887498948482; // φ — golden ratio
		lissa_cfg.freq_z = 2.7182818284590452354; // e — natural log base
		lissa_cfg.phase_x = 0.0;
		lissa_cfg.phase_y = M_PI / 2.0;
		lissa_cfg.phase_z = M_PI / 4.0;
		// Adaptive ranges: always centered at 2^(Range/2) regardless of puzzle size
		lissa_cfg.mean_range    = {1ULL << base_bits,       1ULL << (base_bits + 1)};
		lissa_cfg.std_dev_range = {1ULL << (base_bits - 2), 1ULL << (base_bits - 1)};
		lissa_cfg.skew_range    = {0, 200};
		lissa_cfg.table_size    = JMP_CNT;                    // exactly 512 entries

		LissajousJumpGenerator lissa(lissa_cfg);
		auto lissa_raw = lissa.generate_sampled_jumps((uint32_t)GetTickCount64());

		// Compute and store mean J1 size (in bits) for monitor display
		{
			double sum = 0.0;
			for (size_t k = 0; k < lissa_raw.size(); k++) sum += (double)lissa_raw[k];
			double mean_val = sum / (double)lissa_raw.size();
			g_lissa_j1_bits = log2(mean_val) + (double)lissa_shift;
		}

		for (int i = 0; i < JMP_CNT; i++)
		{
			uint64_t v = lissa_raw[i % lissa_raw.size()];
			if (v < 2) v = 2;                               // safety floor
			EcJumps1[i].dist.Set(v);                        // Set() zeroes upper words first
			if (lissa_shift > 0)
				EcJumps1[i].dist.ShiftLeft(lissa_shift);
			EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; // must be even
			EcJumps1[i].p = ec.MultiplyG_Lambda(EcJumps1[i].dist);
		}
	}
#else
	// Classic uniform-random J1
	{
		int jump1_exp = Range / 2 + 1;
		if (jump1_exp < 10) jump1_exp = 10;
		minjump.Set(1);
		minjump.ShiftLeft(jump1_exp);
		g_lissa_j1_bits = (double)(jump1_exp) + 0.58; // log2(1.5 * 2^exp) ≈ exp+0.58
		for (int i = 0; i < JMP_CNT; i++)
		{
			EcJumps1[i].dist = minjump;
			t.RndMax(minjump);
			EcJumps1[i].dist.Add(t);
			EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
			EcJumps1[i].p = ec.MultiplyG_Lambda(EcJumps1[i].dist);
		}
	}
#endif

	// ── Compile-time feature banner ──────────────────────────────────────────
	printf("Features: "
#if USE_SOTA_PLUS
		RC_BGREEN "SOTA+ \xe2\x9c\x93" RC_RESET
#else
		RC_BRED    "SOTA+ \xe2\x9c\x97" RC_RESET
#endif
		"  |  "
#if USE_JACOBIAN
		RC_BGREEN "Jacobian \xe2\x9c\x93" RC_RESET
#else
		RC_BRED    "Jacobian \xe2\x9c\x97" RC_RESET
#endif
		"  |  "
#if USE_LISSAJOUS
		RC_BGREEN "LISSA \xe2\x9c\x93" RC_RESET
#else
		RC_BRED    "LISSA \xe2\x9c\x97" RC_RESET
#endif
		"  |  "
#if USE_R2_4V
		RC_BGREEN "R2-4V \xe2\x9c\x93" RC_RESET
#else
		RC_BRED    "R2-4V \xe2\x9c\x97" RC_RESET
#endif
		"  |  JMP_CNT: " RC_CYAN "%d" RC_RESET
		"  |  STEP_CNT: " RC_CYAN "%d" RC_RESET "\r\n",
		JMP_CNT, STEP_CNT);
	// ─────────────────────────────────────────────────────────────────────────

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	BuildR2JumpTable(EcJumps2, JMP_CNT, minjump, ec, 1); // R2-distributed, table_id=1

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	BuildR2JumpTable(EcJumps3, JMP_CNT, minjump, ec, 2); // R2-distributed, table_id=2

#if USE_R2_4V
	// Build 4-variant R2 J1 table: mu = 2^(Range/2) = Teske-optimal mean
	{
		int jump1_exp = Range / 2;
		if (jump1_exp < 10) jump1_exp = 10;
		minjump.Set(1);
		minjump.ShiftLeft(jump1_exp); // mu
		BuildR2JumpTable4V(EcJumps1_4v, JMP_CNT, minjump, ec);
		printf("  R2-4V: built %d entries (4x%d: V0=phi,V1=psi,V2=H5,V3=H7), mu=2^%d\r\n",
		       JMP_CNT * 4, JMP_CNT, jump1_exp);
	}
#endif

	u64 base_seed = (u64)GetTickCount64();
	SetRndSeed(base_seed);
	printf("Base seed: 0x%016llX  (set -seed <hex> to reproduce)\r\n", (unsigned long long)base_seed);

#if USE_LISSAJOUS
	// Per-GPU sunflower-Lissajous tables: each GPU gets a unique angular slice
	// of the distribution via perGpuSalt — reduces correlated walks between GPUs.
	// Uses same config as base J1 table but sampled at different Vogel angles.
	{
		int jump1_exp  = Range / 2;
		if (jump1_exp < 10) jump1_exp = 10;
		int base_bits  = (jump1_exp < 33) ? jump1_exp : 33;
		int lissa_shift = jump1_exp - base_bits;

		LissajousJumpGenerator::Config lissa_cfg;
		lissa_cfg.freq_x        = 1.0;
		lissa_cfg.freq_y        = 1.6180339887498948482;
		lissa_cfg.freq_z        = 2.7182818284590452354;
		lissa_cfg.phase_x       = 0.0;
		lissa_cfg.phase_y       = M_PI / 2.0;
		lissa_cfg.phase_z       = M_PI / 4.0;
		lissa_cfg.mean_range    = {1ULL << base_bits,       1ULL << (base_bits + 1)};
		lissa_cfg.std_dev_range = {1ULL << (base_bits - 2), 1ULL << (base_bits - 1)};
		lissa_cfg.skew_range    = {0, 200};
		lissa_cfg.table_size    = JMP_CNT;
		LissajousJumpGenerator lissa(lissa_cfg);

		for (int gi = 0; gi < GpuCnt; gi++)
		{
			// Each GPU gets a unique seed → different random draws from the same
			// Lissajous distribution. Preserves correct mean/variance, no clamping.
			uint32_t gpu_seed = (uint32_t)((base_seed ^ ((uint64_t)(gi + 1) * 0x9E3779B97F4A7C15ULL)) & 0xFFFFFFFF);
			auto gpu_raw = lissa.generate_sampled_jumps(gpu_seed);
			for (int i = 0; i < JMP_CNT; i++)
			{
				uint64_t v = gpu_raw[i % gpu_raw.size()];
				if (v < 2) v = 2;
				EcJumps1PerGpu[gi][i].dist.Set(v);
				if (lissa_shift > 0)
					EcJumps1PerGpu[gi][i].dist.ShiftLeft(lissa_shift);
				EcJumps1PerGpu[gi][i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
				EcJumps1PerGpu[gi][i].p = ec.MultiplyG_Lambda(EcJumps1PerGpu[gi][i].dist);
			}
		}
		printf("  LISSA: %d unique GPU J1 tables (per-GPU seeded diversity)\r\n", GpuCnt);
	}
#else
	// Non-Lissajous: all GPUs share the same uniform J1 table
	for (int gi = 0; gi < GpuCnt; gi++)
		memcpy(EcJumps1PerGpu[gi], EcJumps1, sizeof(EcJMP) * JMP_CNT);
#endif

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
	Pnt_HalfRange = ec.MultiplyG_Lambda(Int_HalfRange);  // ~40% faster with GLV
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
	// Each GPU gets a unique seed so their kangaroos walk independent paths.
	const u64 SEED_MIX = 0x9E3779B97F4A7C15ULL; // golden-ratio mixing constant
	for (int i = 0; i < GpuCnt; i++)
	{
		u64 gpu_seed = base_seed ^ ((u64)(i + 1) * SEED_MIX);
		GpuKangs[i]->GpuSeed = gpu_seed;
		SetRndSeed(gpu_seed);
		printf("GPU %d seed: 0x%016llX\r\n", GpuKangs[i]->CudaIndex, (unsigned long long)gpu_seed);

		// Enable SOTA++ herds if requested and range is large enough
		if (g_use_herds && Range >= 100)
		{
			printf("[GPU %d] Enabling SOTA++ herds (range=%d bits)\r\n", GpuKangs[i]->CudaIndex, Range);
			GpuKangs[i]->SetUseHerds(true, Range);
		}
		else if (g_use_herds && Range < 100)
		{
			printf("[GPU %d] Herds disabled: range too small (%d < 100)\r\n", GpuKangs[i]->CudaIndex, Range);
		}

#if USE_R2_4V
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1_4v, EcJumps2, EcJumps3))
#else
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
#endif
		{
			GpuKangs[i]->Failed = true;
			printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
		}
	}

//prepare CPUs
	for (int i = 0; i < CpuCnt; i++)
	{
		CpuKangs[i] = new RCCpuKang();
		CpuKangs[i]->ThreadIndex = i;
		if (!CpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
		{
			CpuKangs[i]->Failed = true;
			printf("CPU worker %d Prepare failed\r\n", i);
		}
	}

	// Initialize GPU monitoring system
	if (GpuCnt > 0)
	{
		g_gpu_monitor = new GpuMonitor();
		if (!g_gpu_monitor->Initialize(GpuCnt))
		{
			printf("WARNING: GPU monitoring initialization failed\r\n");
			delete g_gpu_monitor;
			g_gpu_monitor = nullptr;
		}
	}

	// Initialise db mutex (Windows CS; pthread mutex is statically initialised)
#ifdef _WIN32
	InitializeCriticalSection(&g_db_cs);
#endif

	u64 tm0 = GetTickCount64();
	g_session_start_ms = tm0;
	g_session_ops_base = PntTotalOps;  // baseline for accurate speed on resume
	g_best_k = 1e18;
	g_session_peak_speed = 0;

	// Open K-factor CSV log
	g_kcsv_fp = fopen("kfactor_log.csv", "a");
	if (g_kcsv_fp) {
		// Write header if file is empty
		fseek(g_kcsv_fp, 0, SEEK_END);
		if (ftell(g_kcsv_fp) == 0)
			fprintf(g_kcsv_fp, "elapsed_s,speed_mks,k_factor,dp_count,dp_pct,eta_s,solve_prob_pct\n");
	}

	if (GpuCnt > 0)
		printf("GPUs started...\r\n");
	if (CpuCnt > 0)
		printf("CPU workers started (%d threads)...\r\n", CpuCnt);

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT + 128];
#else
	pthread_t thr_handles[MAX_GPU_CNT + 128];
#endif

	// Start background DP processing thread — keeps pPntList drained
	// regardless of main-loop Sleep() intervals or monitoring overhead
	g_dp_thread_stop = false;
	u32 ThreadID;
#ifdef _WIN32
	HANDLE dp_thread_handle = (HANDLE)_beginthreadex(NULL, 0, dp_proc_thread, NULL, 0, &ThreadID);
#else
	pthread_t dp_thread_handle;
	pthread_create(&dp_thread_handle, NULL, dp_proc_thread, NULL);
#endif
	printf("DP processor thread started.\r\n");

	// Start background kangs checkpoint thread
	g_kangs_bg_stop   = false;
	g_kangs_bg_saving = false;
#ifdef _WIN32
	HANDLE bg_save_handle = (HANDLE)_beginthreadex(NULL, 0, kangs_bg_save_thread, NULL, 0, &ThreadID);
#else
	pthread_t bg_save_handle;
	pthread_create(&bg_save_handle, NULL, kangs_bg_save_thread, NULL);
#endif
	printf("Background checkpoint thread started (every %.0f min).\r\n", g_kangs_bg_interval_s / 60.0);

	gSolved = false;
	ThrCnt = GpuCnt + CpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}
	for (int i = 0; i < CpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[GpuCnt + i] = (HANDLE)_beginthreadex(NULL, 0, cpu_kang_thr_proc, (void*)CpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[GpuCnt + i], NULL, cpu_kang_thr_proc, (void*)CpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	u64 tm_monitor = GetTickCount64();
	u64 tm_autosave = GetTickCount64();  // independent autosave timer (not tied to stats interval)
	u64 tm_gentame_save = GetTickCount64();  // gen mode periodic tame save (every 5 min)
	while (!gSolved)
	{
		// Check for Ctrl+C / SIGTERM — save is done here in the main thread,
		// NOT in the signal handler (fwrite/printf are not async-signal-safe).
		if (g_shutdown_requested)
		{
			printf("\r\n\r\nInterrupted! Saving progress...\r\n");

			// Stop background checkpoint thread first.
			// If a bg save is in progress, wait for it to finish (don't corrupt the file).
			g_kangs_bg_stop = true;
			if (g_kangs_bg_saving) {
				printf("Waiting for background checkpoint to complete...\r\n");
				while (g_kangs_bg_saving) Sleep(500);
				printf("Background checkpoint done.\r\n");
			}

			// Stop the DP processor thread FIRST so it can no longer call
			// AddDP→push_back while we fwrite the same vector below.
			g_dp_thread_stop = true;
#ifdef _WIN32
			WaitForSingleObject(dp_thread_handle, 5000);
#else
			pthread_join(dp_thread_handle, NULL);
#endif
			CheckNewPoints(); // drain any last DPs queued before thread stopped

			// Gen mode: save tames on Ctrl+C
			if (gGenMode && gTamesFileName[0])
			{
				printf("Saving tames: %llu records → %s\r\n",
				       (unsigned long long)db.GetBlockCnt(), gTamesFileName);
				db.Header[0] = gRange;
				if (db.SaveToFile(gTamesFileName))
					printf("Tames saved OK\r\n");
				else
					printf("ERROR: tames save failed!\r\n");
			}

			if (g_work_file && g_start_time > 0)
			{
				uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
				g_work_file->UpdateProgress(PntTotalOps, db.GetBlockCnt(), gTotalErrors, elapsed);
				if (g_work_file->Save())
					printf("Work file saved successfully\r\n");
				else
					printf("ERROR: Failed to save work file!\r\n");
				// Serialize full db to companion .kangs file
				if (!g_kangs_filename.empty())
				{
					if (db.SaveToFile((char*)g_kangs_filename.c_str()))
						printf("DB saved: %s (%llu DPs)\r\n", g_kangs_filename.c_str(), (unsigned long long)db.GetBlockCnt());
					else
						printf("ERROR: Failed to save .kangs file!\r\n");
				}
			}
			break;  // exit main loop — GPU threads stopped in cleanup below
		}

		// CheckNewPoints() is now handled by background dp_proc_thread
		Sleep(1);

		// 1-second monitoring tick: GPU thermal management + auto-switch check
		if (GetTickCount64() - tm_monitor > 1000)
		{
			// GPU monitoring and thermal management (only when monitor is active)
			if (g_gpu_monitor)
			{
				SystemStats sys_stats = g_gpu_monitor->GetSystemStats();
				sys_stats.start_time_ms = tm0;
				sys_stats.actual_ops = PntTotalOps;
				sys_stats.expected_ops = ops;
				sys_stats.dp_count = db.GetBlockCnt();
				sys_stats.dp_expected = (u64)(ops / dp_val);
				sys_stats.dp_buffer_used = PntIndex;
				sys_stats.dp_buffer_total = MAX_CNT_LIST;
				sys_stats.wild_only_active = g_wild_only;
				sys_stats.loaded_tame_cnt  = g_loaded_tame_cnt;

				// DP rate: (new DPs) / (actual elapsed seconds since last tick)
				// Must divide by real elapsed time — tick interval can exceed 1000ms
				// if the main loop was blocked (e.g., by a large .kangs save).
				static u64 s_prev_dp_count = 0;
				static u64 s_prev_dp_ms    = 0;
				u64 cur_dp   = sys_stats.dp_count;
				u64 now_dp_ms = GetTickCount64();
				// Initialise on first call so we don't show a spike from loaded DPs
				if (s_prev_dp_ms == 0) {
					s_prev_dp_count = cur_dp;
					s_prev_dp_ms    = now_dp_ms;
				}
				u64 dp_elapsed_ms = now_dp_ms - s_prev_dp_ms;
				if (dp_elapsed_ms < 1) dp_elapsed_ms = 1;
				sys_stats.dp_rate_per_sec = (double)(cur_dp > s_prev_dp_count
					? cur_dp - s_prev_dp_count : 0)
					/ (dp_elapsed_ms / 1000.0);
				s_prev_dp_count = cur_dp;
				s_prev_dp_ms    = now_dp_ms;

				// K-factor for monitor
				if (sys_stats.expected_ops > 0)
					sys_stats.current_k_factor = (double)PntTotalOps / sqrt(sys_stats.expected_ops);

				// Update per-GPU speeds
				for (int i = 0; i < GpuCnt; i++)
				{
					sys_stats.gpu_stats[i].speed_mkeys = GpuKangs[i]->GetStatsSpeed();
					sys_stats.gpu_stats[i].operations = 0; // GPU ops tracked globally via PntTotalOps
					sys_stats.gpu_stats[i].seed        = GpuKangs[i]->GpuSeed;
				}

				// Update CPU speed
				// CPU stats are in KKeys/s, convert to MKeys/s
				sys_stats.cpu_speed_mkeys = 0;
				for (int i = 0; i < CpuCnt; i++)
				{
					sys_stats.cpu_speed_mkeys += CpuKangs[i]->GetStatsSpeed() / 1000.0;
				}

				// ── Thermal throttle detection ──────────────────────────────────
				static double s_prev_gpu_speed[MAX_GPU_CNT] = {};
				for (int _i = 0; _i < GpuCnt; _i++) {
					double cur_spd = sys_stats.gpu_stats[_i].speed_mkeys;
					double prv_spd = s_prev_gpu_speed[_i];
					if (prv_spd > 100.0 && cur_spd > 0.0) {
						double drop_pct = (prv_spd - cur_spd) / prv_spd * 100.0;
						if (drop_pct > 15.0)
							printf(RC_BRED "⚠ GPU %d throttling: %.0f → %.0f MK/s (%.0f%% drop) — check temps!\r\n" RC_RESET,
							       _i, prv_spd, cur_spd, drop_pct);
					}
					s_prev_gpu_speed[_i] = cur_spd;
				}
				// ────────────────────────────────────────────────────────────

				// ── Herds stats (live per-GPU DP counts from GpuDpCount) ────────
				sys_stats.herds_active      = false;
				sys_stats.total_herd_dps    = 0;
				sys_stats.total_local_colls = 0;
				sys_stats.herds_per_gpu     = 0;
				sys_stats.kangs_per_herd    = 0;
				for (int i = 0; i < GpuCnt; i++) {
					memset(&sys_stats.herd_gpu[i], 0, sizeof(sys_stats.herd_gpu[i]));
					if (GpuKangs[i]->IsUsingHerds()) {
						sys_stats.herds_active = true;
						GpuHerdManager* hm = GpuKangs[i]->GetHerdManager();
						HerdGPUStats& hg = sys_stats.herd_gpu[i];
						// Use live atomic counter — updated every kernel launch
						hg.total_dps        = GpuKangs[i]->GpuDpCount.load(std::memory_order_relaxed);

						// Live same-GPU collision count (populated in CheckNewPoints)
						hg.local_collisions = g_gpu_local_colls[i].load(std::memory_order_relaxed);
						sys_stats.total_local_colls += hg.local_collisions;

						// Bias: tame DP fraction should be within ±15pp of gTameRatioPct
						{
							u64 tc = GpuKangs[i]->GpuTameDpCount.load(std::memory_order_relaxed);
							u64 wc = GpuKangs[i]->GpuWildDpCount.load(std::memory_order_relaxed);
							u64 tot = tc + wc;
							if (tot >= 100) {
								double frac = (double)tc / (double)tot;
								double exp_frac = (double)gTameRatioPct / 100.0;
								hg.bias_ok = (fabs(frac - exp_frac) < 0.15);
							} else {
								hg.bias_ok = true; // not enough samples yet
							}
						}

						if (hm) {
							hg.herds_per_gpu = hm->GetMemory()->config.herds_per_gpu;
							if (sys_stats.herds_per_gpu == 0) {
								sys_stats.herds_per_gpu  = hg.herds_per_gpu;
								sys_stats.kangs_per_herd = hm->GetMemory()->config.kangaroos_per_herd;
							}
						}
						sys_stats.total_herd_dps += hg.total_dps;
					}
				}
				// ────────────────────────────────────────────────────────────

				// CRITICAL: Write updated stats back to GPU monitor!
				g_gpu_monitor->SetSystemStats(sys_stats);
				g_gpu_monitor->UpdateAllGPUs();
				g_gpu_monitor->ApplyThermalLimits();
			}

			// Auto-switch to pure-wild mode at 93% of expected DP count.
			// Always checked at 1-second granularity regardless of GPU monitor state.
			// At DP=12 with 3 GPUs (~1.9M DPs/sec): ~1.9M overshoot max (was ~19M at 10s).
			if (!g_capacity_reached && g_capacity_threshold > 0)
			{
				u64 current_dps = db.GetBlockCnt();
				if (current_dps >= g_capacity_threshold)
				{
					g_capacity_reached = true;
					// Suggest the -dp value that would keep RAM under 8 GB
					// Each DP record ≈ 36 bytes; target 8 GB = ~238M records
					// dp_suggested = ceil(log2(exp_ops / 238e6))
					int dp_suggest = (int)ceil(log2(ops / 238e6));
					if (dp_suggest < (int)gDP + 2) dp_suggest = (int)gDP + 2;
					printf(RC_BYELLOW "\r\n*** AUTO-SWITCH: table at %llu/%llu DPs (93%%) — tame DPs dropped, pure-wild mode ***\r\n" RC_RESET,
						current_dps, g_capacity_threshold);
					printf(RC_BYELLOW "    Next run: use " RC_BCYAN "-dp %d" RC_BYELLOW
					       " to stay under 8 GB RAM  ***\r\n\r\n" RC_RESET, dp_suggest);
				}
			}

			tm_monitor = GetTickCount64();
		}

		// Statistics display (every 10 seconds)
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);

			// ── Sparkline: push current K into ring buffer ───────────────────
			if (ops > 0 && PntTotalOps > 0) {
				double _sk = (double)PntTotalOps / ops;
				g_spark[g_spark_head] = _sk;
				g_spark_head = (g_spark_head + 1) % SPARK_LEN;
				if (g_spark_cnt < SPARK_LEN) g_spark_cnt++;
				// Track attempt-level best/peak K
				if (_sk > g_attempt_peak_k) g_attempt_peak_k = _sk;
				if (_sk < g_attempt_best_k) g_attempt_best_k = _sk;
			}
			// ────────────────────────────────────────────────────────────────

			// Show detailed GPU stats every 10 seconds
			if (g_gpu_monitor)
			{
				g_gpu_monitor->PrintDetailedStats();
			}

			tm_stats = GetTickCount64();

			// Smart restart: if K is too high and we have enough DPs, reseed kangaroos
			// The GPU threads handle the actual restart (safe CUDA context)
			// All accumulated DPs stay in db as permanent collision traps
			if (!IsBench && !gGenMode && GpuCnt > 0 && ops > 0 && PntTotalOps > 0) {
				double cur_k  = (double)PntTotalOps / ops;
				double dp_pct = (est_dps_cnt_init > 0)
					? (double)db.GetBlockCnt() / (double)est_dps_cnt_init : 0.0;
				if (cur_k >= g_restart_k_thresh && dp_pct >= g_restart_dp_thresh) {
					u64 preserved = db.GetBlockCnt();
					const u64 MIX = 0x9E3779B97F4A7C15ULL;
					u64 new_base  = GetTickCount64()
						^ ((u64)(g_restart_count + 1) * 0xDEADBEEFCAFEBABEULL);
					for (int _i = 0; _i < GpuCnt; _i++) {
						GpuKangs[_i]->RestartSeed = new_base ^ ((u64)(_i+1) * MIX);
						GpuKangs[_i]->RestartPending.store(true, std::memory_order_release);
					}
					// Wait for all GPU threads to acknowledge
					bool _done = false;
					while (!_done) {
						_done = true;
						for (int _i = 0; _i < GpuCnt; _i++)
							if (GpuKangs[_i]->RestartPending.load(std::memory_order_acquire))
								{ _done = false; break; }
						Sleep(5);
					}
					// ── Record attempt stats ────────────────────────────────
					if (g_attempt_idx < MAX_ATTEMPTS) {
						u64 dur = (g_attempt_start_ms > 0)
						          ? (GetTickCount64() - g_attempt_start_ms) / 1000 : 0;
						g_attempts[g_attempt_idx++] = {
							g_restart_count + 1,
							g_attempt_peak_k,
							(g_attempt_best_k < 1e17 ? g_attempt_best_k : cur_k),
							PntTotalOps,
							dur
						};
					}

					// ── Adaptive K threshold ────────────────────────────────
					// After 2+ restarts, tighten threshold to 95% of median
					// peak-K seen so far (but never below 1.05 or above 1.30)
					if (g_attempt_idx >= 2) {
						double sum_pk = 0;
						for (int _a = 0; _a < g_attempt_idx; _a++)
							sum_pk += g_attempts[_a].peak_k;
						double median_pk = sum_pk / g_attempt_idx;
						double new_thresh = median_pk * 0.95;
						new_thresh = (new_thresh < 1.05) ? 1.05
						           : (new_thresh > 1.30) ? 1.30 : new_thresh;
						if (new_thresh < g_restart_k_thresh) {
							printf(RC_BCYAN "  Adaptive threshold: %.2f → %.2f\r\n" RC_RESET,
							       g_restart_k_thresh, new_thresh);
							g_restart_k_thresh = new_thresh;
						}
					}
					// ────────────────────────────────────────────────────────

					g_restart_count++;
					PntTotalOps = 0;
					g_session_start_ms = GetTickCount64(); // reset speed base
					g_session_ops_base = 0;                // reset ops baseline after restart
					g_attempt_start_ms = g_session_start_ms;
					g_attempt_peak_k   = 0.0;
					g_attempt_best_k   = 1e18;
					g_best_k = 1e18;
					g_session_peak_speed = 0;
					// Reset sparkline for new attempt
					memset(g_spark, 0, sizeof(g_spark));
					g_spark_head = 0; g_spark_cnt = 0;

					printf(RC_BYELLOW "\r\n*** Smart restart #%d"
					       "  (K=%.3f >= %.2f, %.1f%% DPs)  --  "
					       RC_BCYAN "%llu DPs kept as traps"
					       RC_BYELLOW "  ***\r\n\r\n" RC_RESET,
					       g_restart_count, cur_k, g_restart_k_thresh,
					       dp_pct*100.0, preserved);

					SaveSessionState();
				}
			}
		}

		// ── Independent autosave timer ───────────────────────────────────────
		// Saves ONLY the .work header (tiny, fast, non-blocking).
		// The .kangs file (2+ GB) is saved ONLY on Ctrl+C or solution found —
		// never during regular autosave, to prevent multi-minute main-loop freezes.
		if (g_autosave && g_start_time > 0 &&
		    (GetTickCount64() - tm_autosave) >= (g_autosave_interval * 1000ULL))
		{
			uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
			if (g_work_file) g_work_file->UpdateProgress(PntTotalOps, db.GetBlockCnt(), gTotalErrors, elapsed);
			g_autosave->ForceSave(PntTotalOps, db.GetBlockCnt(), gTotalErrors, elapsed);
			// NOTE: db.SaveToFile() intentionally NOT called here — it writes 2+ GB
			// and blocks the main loop. It runs on Ctrl+C and solution found only.
			tm_autosave = GetTickCount64();
		}
		// ─────────────────────────────────────────────────────────────────────

		// Gen mode: periodic tame save every 5 minutes
		if (gGenMode && gTamesFileName[0] &&
		    (GetTickCount64() - tm_gentame_save) >= 300000ULL)
		{
			u64 cnt = db.GetBlockCnt();
			printf("\r\n[GEN] Autosave: %llu tames → %s\r\n",
			       (unsigned long long)cnt, gTamesFileName);
			db.Header[0] = gRange;
			if (!db.SaveToFile(gTamesFileName))
				printf("[GEN] WARNING: autosave failed!\r\n");
			tm_gentame_save = GetTickCount64();
		}
		// ─────────────────────────────────────────────────────────────────────

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("Operations limit reached\r\n");
			break;
		}
	}

	// Stop background checkpoint thread
	g_kangs_bg_stop = true;
	while (g_kangs_bg_saving) Sleep(500); // wait if mid-save
#ifdef _WIN32
	if (bg_save_handle) { WaitForSingleObject(bg_save_handle, 15000); CloseHandle(bg_save_handle); }
#else
	pthread_join(bg_save_handle, NULL);
#endif

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	for (int i = 0; i < CpuCnt; i++)
		CpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(1);

	// Stop background DP processor and do final drain.
	// Guard against double-join: shutdown path may have already joined the thread.
	if (!g_dp_thread_stop)
	{
		g_dp_thread_stop = true;
#ifdef _WIN32
		WaitForSingleObject(dp_thread_handle, 5000);
		CloseHandle(dp_thread_handle);
#else
		pthread_join(dp_thread_handle, NULL);
#endif
		CheckNewPoints(); // Final sweep after everything stops
	}
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}
	for (int i = 0; i < CpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[GpuCnt + i]);
#else
		pthread_join(thr_handles[GpuCnt + i], NULL);
#endif
		delete CpuKangs[i];
		CpuKangs[i] = nullptr;
	}

	// Shutdown GPU monitoring
	if (g_gpu_monitor)
	{
		delete g_gpu_monitor;
		g_gpu_monitor = nullptr;
	}

	// User pressed Ctrl+C — save already done above, just exit cleanly.
	if (g_shutdown_requested)
	{
		PrintSessionSummary(false);
		db.Clear();
		ww_db.Clear();
		return false;
	}

	if (gIsOpsLimit)
	{
		if (gGenMode)
		{
			printf("saving tames...\r\n");
			db.Header[0] = gRange;
			if (db.SaveToFile(gTamesFileName))
				printf("tames saved\r\n");
			else
				printf("tames saving failed\r\n");
		}
		PrintSessionSummary(false);
		db.Clear();
		ww_db.Clear();
		return false;
	}

	double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf(RC_BGREEN "Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n" RC_RESET, K);
	PrintSessionSummary(true, K);
	db.Clear();
	ww_db.Clear();
	*pk_res = gPrivKey;
	return true;
}

// ── Session summary ──────────────────────────────────────────────────────────
static void PrintSessionSummary(bool solved, double final_k)
{
	if (g_session_start_ms == 0) return;
	u64 elapsed_ms = GetTickCount64() - g_session_start_ms;
	u64 elapsed_s  = elapsed_ms / 1000;
	u64 d = elapsed_s / 86400;
	int h = (int)((elapsed_s % 86400) / 3600);
	int m = (int)((elapsed_s % 3600) / 60);
	int s = (int)(elapsed_s % 60);

	double avg_speed = (elapsed_ms > 0)
		? (double)(PntTotalOps > g_session_ops_base ? PntTotalOps - g_session_ops_base : PntTotalOps)
		  / (elapsed_ms / 1000.0) / 1e6 : 0.0;

	double show_k = (solved && final_k > 0.0) ? final_k
	              : (g_best_k < 1e17)          ? g_best_k
	              :                              0.0;

	// Box inner width = 41 display chars
	// Format: | space label(10) colon space value padding space |
	//          1     1    10    1     1    ^^^               1   = 15 + value + pad
	// prefix = " LABEL    : " = 14 chars, leaving 41-14-1 = 26 for value
	const int INNER   = 41;
	const int PREFIX  = 14; // " Status   : "
	const int SUFFIX  = 1;  // trailing space before |
	const int VAL_MAX = INNER - PREFIX - SUFFIX; // 26

	// UTF-8 display width (multi-byte chars count as 1)
	auto dw = [](const char* s) -> int {
		int n = 0;
		for (int i = 0; s[i]; ) {
			unsigned char c = (unsigned char)s[i];
			if      (c < 0x80) i += 1;
			else if (c < 0xE0) i += 2;
			else if (c < 0xF0) i += 3;
			else               i += 4;
			n++;
		}
		return n;
	};

	// Print one row: | LABEL : <color>value<reset> padding |
	auto row = [&](const char* label, const char* color, const char* value) {
		int pad = VAL_MAX - dw(value);
		if (pad < 0) pad = 0;
		printf(RC_BWHITE "\xe2\x94\x82" RC_RESET " %-10s: %s%s" RC_RESET "%*s "
		       RC_BWHITE "\xe2\x94\x82\r\n" RC_RESET,
		       label, color, value, pad, "");
	};

	char buf[64];
	const char* kc = (show_k > 0.0 && show_k < 1.15) ? RC_BGREEN
	               : (show_k > 0.0 && show_k < 1.30) ? RC_BYELLOW : RC_WHITE;

	printf(RC_BWHITE "\r\n\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x90\r\n" RC_RESET);
	printf(RC_BWHITE "\xe2\x94\x82           SESSION SUMMARY               \xe2\x94\x82\r\n" RC_RESET);
	printf(RC_BWHITE "\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\xa4\r\n" RC_RESET);

	// Status
	row("Status", solved ? RC_BGREEN : RC_BYELLOW,
	    solved ? "SOLVED \xe2\x9c\x93" : "stopped");

	// Time
	snprintf(buf, sizeof(buf), "%llud %02dh %02dm %02ds", (unsigned long long)d, h, m, s);
	row("Time", RC_CYAN, buf);

	// Ops
	snprintf(buf, sizeof(buf), "2^%.3f", PntTotalOps > 1 ? log2((double)PntTotalOps) : 0.0);
	row("Ops", RC_WHITE, buf);

	// Avg speed
	snprintf(buf, sizeof(buf), "%.0f MK/s", avg_speed);
	row("Avg Speed", RC_CYAN, buf);

	// Peak speed — floor at avg_speed to catch any tail interval < 1s at solve time
	if (avg_speed > g_session_peak_speed) g_session_peak_speed = avg_speed;
	snprintf(buf, sizeof(buf), "%.0f MK/s", (double)g_session_peak_speed);
	row("Peak Spd", RC_CYAN, buf);

	// Solve K
	snprintf(buf, sizeof(buf), "%.3f", show_k);
	row("Solve K", kc, buf);

	// Errors
	snprintf(buf, sizeof(buf), "%u", gTotalErrors);
	row("Errors", RC_WHITE, buf);

	// CSV log
	if (g_kcsv_fp)
		row("CSV log", RC_WHITE, "kfactor_log.csv");

	printf(RC_BWHITE "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x98\r\n\r\n" RC_RESET);

	// Close CSV
	if (g_kcsv_fp) { fclose(g_kcsv_fp); g_kcsv_fp = nullptr; }
}
// ─────────────────────────────────────────────────────────────────────────────

// Signal handler for Ctrl+C and graceful shutdown.
// ONLY sets the flag — do NOT call printf/fopen/fwrite/malloc here;
// they are not async-signal-safe and will deadlock if the signal fires
// while the heap lock is held (very likely in a CUDA + STL program).
void SignalHandler(int signum)
{
	(void)signum;
	g_shutdown_requested = 1;
	// Do NOT set gSolved=true here — that would make the post-loop code
	// think the puzzle was solved and call SolvePoint on garbage data.
	// The main loop checks g_shutdown_requested on every tick and breaks cleanly.
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 12) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
			gDP_manual = true;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-wildonly") == 0)
		{
			g_wild_only = true;
		}
		else
		if (strcmp(argument, "-gentames") == 0)
		{
			// Explicit tame generation mode — output file must follow via -tames
			gGenMode = true;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		if (strcmp(argument, "-cpu") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 0) || (val > 128))
			{
				printf("error: invalid value for -cpu option (must be 0-128)\r\n");
				return false;
			}
			gCpuThreads = val;
		}
		else
		if (strcmp(argument, "-workfile") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -workfile option\r\n");
				return false;
			}
			g_work_filename = argv[ci];
			ci++;
			printf("Work file: %s\r\n", g_work_filename.c_str());
			// Derive .state and .kangs filenames from work file
			g_state_filename = g_work_filename;
			size_t dot = g_state_filename.rfind('.');
			if (dot != std::string::npos) g_state_filename = g_state_filename.substr(0, dot);
			g_state_filename += ".state";
			// .kangs: full db serialization (replaces dp_records vector approach)
			g_kangs_filename = g_work_filename;
			dot = g_kangs_filename.rfind('.');
			if (dot != std::string::npos) g_kangs_filename = g_kangs_filename.substr(0, dot);
			g_kangs_filename += ".kangs";
		}
		else
		if (strcmp(argument, "-autosave") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -autosave option\r\n");
				return false;
			}
			int val = atoi(argv[ci]);
			ci++;
			if (val < 0)
			{
				printf("error: invalid value for -autosave option (must be >= 0)\r\n");
				return false;
			}
			g_autosave_interval = val;
			printf("Auto-save interval: %llu seconds\r\n", (unsigned long long)g_autosave_interval);
		}
		else
		if (strcmp(argument, "-ksave") == 0)
		{
			if (ci >= argc) { printf("error: missed value after -ksave\r\n"); return false; }
			int val = atoi(argv[ci]); ci++;
			if (val < 60) { printf("error: -ksave interval must be >= 60 seconds\r\n"); return false; }
			g_kangs_bg_interval_s = (u64)val;
			printf("Kangs checkpoint interval: %.0f min\r\n", g_kangs_bg_interval_s / 60.0);
		}
		else
		if (strcmp(argument, "-watts") == 0)
		{
			if (ci >= argc) { printf("error: missed value after -watts\r\n"); return false; }
			double v = atof(argv[ci++]);
			if (v > 0.0) g_sys_overhead_w = v;
			else { printf("error: -watts must be > 0\r\n"); return false; }
		}
		else
		if (strcmp(argument, "-rate") == 0)
		{
			if (ci >= argc) { printf("error: missed value after -rate\r\n"); return false; }
			double v = atof(argv[ci++]);
			if (v > 0.0) g_kwh_rate = v;
			else { printf("error: -rate must be > 0\r\n"); return false; }
		}
		else
		if (strcmp(argument, "-herds") == 0)
		{
			g_use_herds = true;
			printf("SOTA++ herds mode enabled\r\n");
		}
		else
		if (strcmp(argument, "-force-resume") == 0)
		{
			g_force_resume = true;
			printf("Force-resume enabled — pubkey check will be skipped\r\n");
		}
		else
		if (strcmp(argument, "-status") == 0)
		{
			// Print progress summary from work file and exit — no GPU needed
			if (ci >= argc)
			{
				printf("error: -status requires a work file path\r\n");
				exit(1);
			}
			RCWorkFile sf;
			if (!sf.Load(argv[ci]))
			{
				printf("error: could not load work file: %s\r\n", argv[ci]);
				exit(1);
			}
			uint32_t range   = sf.GetRangeBits();
			uint32_t dp      = sf.GetDPBits();
			uint64_t ops     = sf.GetTotalOps();
			uint64_t dp_cnt  = sf.GetDPCount();
			uint64_t elapsed = sf.GetElapsedSeconds();

			double exp_ops    = 1.15 * pow(2.0, range / 2.0);
			double dp_val     = pow(2.0, dp);
			double exp_dps    = exp_ops / dp_val;
			double dp_pct     = (exp_dps > 0) ? 100.0 * dp_cnt / exp_dps : 0.0;
			double k          = (exp_ops > 0) ? ops / sqrt(exp_ops) : 0.0;
			uint64_t ed = elapsed / 86400;
			int eh = (int)((elapsed % 86400) / 3600);
			int em = (int)((elapsed % 3600) / 60);

			printf("\r\n");
			printf("Work file : %s\r\n", argv[ci]);
			printf("Range     : %u bits\r\n", range);
			printf("DP        : %u\r\n", dp);
			printf("Ops done  : 2^%.3f\r\n", ops > 1 ? log2((double)ops) : 0.0);
			printf("DPs       : %llu / %.0f (%.1f%%)\r\n",
				(unsigned long long)dp_cnt, exp_dps, dp_pct);
			printf("K-factor  : %.3f\r\n", k);
			printf("Elapsed   : %llud %02dh %02dm\r\n",
				(unsigned long long)ed, eh, em);
			printf("\r\n");
			exit(0);
		}
		else
		if (strcmp(argument, "-info") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -info option\r\n");
				return false;
			}
			// Handle -info mode: display work file information and exit
			RCWorkFile info_file;
			if (info_file.Load(argv[ci]))
			{
				info_file.PrintInfo();
				exit(0);
			}
			else
			{
				printf("Failed to load work file: %s\r\n", argv[ci]);
				exit(1);
			}
		}
		else
		if (strcmp(argument, "-merge") == 0)
		{
			// Handle -merge mode: merge multiple work files
			std::vector<std::string> input_files;
			std::string output_file;

			// Collect input files until we hit -output or end of args
			while (ci < argc && strcmp(argv[ci], "-output") != 0)
			{
				input_files.push_back(argv[ci]);
				ci++;
			}

			// Get output file after -output
			if (ci < argc && strcmp(argv[ci], "-output") == 0)
			{
				ci++;
				if (ci < argc)
				{
					output_file = argv[ci];
					ci++;
				}
			}

			if (input_files.size() < 2)
			{
				printf("error: -merge requires at least 2 input files\r\n");
				exit(1);
			}

			if (output_file.empty())
			{
				printf("error: -merge requires -output option\r\n");
				exit(1);
			}

			// Perform merge
			printf("Merging %zu work files...\r\n", input_files.size());
			if (RCWorkFile::Merge(input_files, output_file))
			{
				printf("Merge successful! Output: %s\r\n", output_file.c_str());
				exit(0);
			}
			else
			{
				printf("Merge failed!\r\n");
				exit(1);
			}
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	if (g_wild_only && !gTamesFileName[0])
	{
		printf("error: -wildonly requires -tames <file>\r\n");
		return false;
	}
	if (gGenMode && !gTamesFileName[0])
	{
		printf("error: -gentames requires -tames <output_file>\r\n");
		return false;
	}
	return true;
}

// ─── WIF conversion ──────────────────────────────────────────────────────────
// Self-contained: no OpenSSL / external deps.
// Produces compressed WIF (prefix 'K' or 'L') for mainnet.

static void sha256_block(const uint8_t* data, size_t len, uint8_t out[32])
{
    // RFC 6234 / FIPS 180-4 SHA-256
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
        0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
        0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
        0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
        0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
        0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    // Build padded message
    size_t total = ((len + 9 + 63) / 64) * 64;
    uint8_t* m = (uint8_t*)calloc(total, 1);
    memcpy(m, data, len);
    m[len] = 0x80;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int j = 0; j < 8; j++)
        m[total - 1 - j] = (uint8_t)(bitlen >> (j * 8));
    // Process blocks
    for (size_t off = 0; off < total; off += 64) {
        uint32_t w[64];
        for (int j = 0; j < 16; j++)
            w[j] = ((uint32_t)m[off+j*4]<<24)|((uint32_t)m[off+j*4+1]<<16)|
                   ((uint32_t)m[off+j*4+2]<<8)|(uint32_t)m[off+j*4+3];
        for (int j = 16; j < 64; j++) {
            uint32_t s0 = (w[j-15]>>7|(w[j-15]<<25))^(w[j-15]>>18|(w[j-15]<<14))^(w[j-15]>>3);
            uint32_t s1 = (w[j-2]>>17|(w[j-2]<<15))^(w[j-2]>>19|(w[j-2]<<13))^(w[j-2]>>10);
            w[j] = w[j-16] + s0 + w[j-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int j = 0; j < 64; j++) {
            uint32_t S1=(e>>6|(e<<26))^(e>>11|(e<<21))^(e>>25|(e<<7));
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K[j]+w[j];
            uint32_t S0=(a>>2|(a<<30))^(a>>13|(a<<19))^(a>>22|(a<<10));
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(m);
    for (int j = 0; j < 8; j++) {
        out[j*4+0]=(h[j]>>24)&0xff; out[j*4+1]=(h[j]>>16)&0xff;
        out[j*4+2]=(h[j]>>8)&0xff;  out[j*4+3]=h[j]&0xff;
    }
}

// Encode bytes to Base58Check WIF string (result written into `wif`, must be >=53 bytes)
static void PrivKeyToWIF(const char* hexkey, char* wif)
{
    static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Parse hex → 32 bytes
    uint8_t key[32] = {};
    for (int i = 0; i < 32; i++) {
        unsigned int v = 0;
        sscanf(hexkey + i * 2, "%02x", &v);
        key[i] = (uint8_t)v;
    }

    // Build payload: 0x80 | key[32] | 0x01 (compressed)
    uint8_t payload[34];
    payload[0] = 0x80;
    memcpy(payload + 1, key, 32);
    payload[33] = 0x01;

    // Double-SHA256 checksum
    uint8_t h1[32], h2[32];
    sha256_block(payload, 34, h1);
    sha256_block(h1, 32, h2);

    // Full 38-byte input for Base58
    uint8_t full[38];
    memcpy(full, payload, 34);
    memcpy(full + 34, h2, 4);

    // Base58 encode
    // Use big-integer division on the 38-byte number
    uint8_t tmp[38];
    memcpy(tmp, full, 38);
    char rev[60];
    int rlen = 0;
    bool nonzero = true;
    while (nonzero) {
        nonzero = false;
        uint32_t rem = 0;
        for (int i = 0; i < 38; i++) {
            uint32_t cur = rem * 256 + tmp[i];
            tmp[i] = (uint8_t)(cur / 58);
            rem = cur % 58;
            if (tmp[i]) nonzero = true;
        }
        rev[rlen++] = B58[rem];
    }
    // Leading '1's for zero bytes
    int lead = 0;
    for (int i = 0; i < 38 && full[i] == 0; i++) lead++;
    int wlen = 0;
    for (int i = 0; i < lead; i++) wif[wlen++] = '1';
    for (int i = rlen - 1; i >= 0; i--) wif[wlen++] = rev[i];
    wif[wlen] = '\0';
}

// ---- RIPEMD-160 (inline, no external deps) ----------------------------------
static void ripemd160_block(const uint8_t* data, size_t len, uint8_t out[20])
{
#define ROL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
    static const uint32_t KL[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    static const uint32_t KR[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
    static const int RL[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    static const int RR[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    static const int SL[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    static const int SR[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };
    // padding
    size_t total = ((len + 9 + 63) / 64) * 64;
    uint8_t* m = (uint8_t*)calloc(total, 1);
    memcpy(m, data, len);
    m[len] = 0x80;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int j = 0; j < 8; j++) m[total - 8 + j] = (uint8_t)(bitlen >> (j * 8));

    uint32_t h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;

    for (size_t off = 0; off < total; off += 64) {
        uint32_t X[16];
        for (int j = 0; j < 16; j++) {
            X[j] = (uint32_t)m[off+j*4] | ((uint32_t)m[off+j*4+1]<<8) |
                   ((uint32_t)m[off+j*4+2]<<16) | ((uint32_t)m[off+j*4+3]<<24);
        }
        uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
        uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;
        for (int j = 0; j < 80; j++) {
            int r = j / 16;
            uint32_t fl, fr;
            if (r==0){ fl=(bl^cl^dl);        fr=(br|(~dr))^cr; }
            else if(r==1){ fl=(bl&cl)|(~bl&dl); fr=(br&cr)|(dr&~cr); }
            else if(r==2){ fl=(bl|~cl)^dl;    fr=(br|~cr)^dr; }
            else if(r==3){ fl=(bl&dl)|(cl&~dl); fr=(br&dr)|(cr&~dr); }
            else        { fl=bl^(cl|~dl);     fr=br^cr^dr; }
            uint32_t tl = ROL32(al+fl+X[RL[j]]+KL[r], SL[j]) + el;
            al=el; el=dl; dl=ROL32(cl,10); cl=bl; bl=tl;
            uint32_t tr = ROL32(ar+fr+X[RR[j]]+KR[r], SR[j]) + er;
            ar=er; er=dr; dr=ROL32(cr,10); cr=br; br=tr;
        }
        uint32_t t = h1+cl+dr; h1=h2+dl+er; h2=h3+el+ar;
        h3=h4+al+br; h4=h0+bl+cr; h0=t;
    }
    free(m);
    uint32_t H[5]={h0,h1,h2,h3,h4};
    for (int j=0;j<5;j++){
        out[j*4+0]=(H[j])&0xff; out[j*4+1]=(H[j]>>8)&0xff;
        out[j*4+2]=(H[j]>>16)&0xff; out[j*4+3]=(H[j]>>24)&0xff;
    }
#undef ROL32
}


// ---- Native SegWit P2WPKH (bech32) address from EcPoint ----------------
// addr must be >= 48 bytes.  Result is "bc1q..." for mainnet.
static void PubKeyToSWP(const EcPoint& pub, char* addr)
{
    // 1. Compressed pubkey -> hash160
    uint8_t pk[33];
    pk[0] = ((pub.y.data[0] & 1) == 0) ? 0x02 : 0x03;
    for (int i = 0; i < 32; i++)
        pk[1+i] = (uint8_t)((pub.x.data[7-(i/4)] >> (8*(3-(i%4)))) & 0xFF);
    uint8_t sha[32], rmd[20];
    sha256_block(pk, 33, sha);
    ripemd160_block(sha, 32, rmd);

    // 2. Convert 20 bytes (160 bits) from 8-bit groups to 5-bit groups
    static const char CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    uint8_t data5[33];
    data5[0] = 0;       // witness version 0
    int outlen = 1;
    uint32_t acc = 0; int bits = 0;
    for (int i = 0; i < 20; i++) {
        acc = (acc << 8) | rmd[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            data5[outlen++] = (acc >> bits) & 0x1f;
        }
    }
    if (bits > 0) data5[outlen++] = (acc << (5 - bits)) & 0x1f;

    // 3. Bech32 checksum
    static const uint32_t GEN[5] = {0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    auto polymod = [&](const uint8_t* v, int vlen) -> uint32_t {
        uint32_t chk = 1;
        const char* h = "bc";
        for (int i = 0; h[i]; i++) {
            uint8_t c = (uint8_t)(h[i] >> 5);
            uint32_t b = (chk >> 25); chk = ((chk & 0x1ffffff) << 5) ^ c;
            for (int j=0;j<5;j++) if ((b>>j)&1) chk ^= GEN[j];
        }
        { uint32_t b = (chk >> 25); chk = ((chk & 0x1ffffff) << 5);
          for (int j=0;j<5;j++) if ((b>>j)&1) chk ^= GEN[j]; }
        for (int i = 0; h[i]; i++) {
            uint8_t c = (uint8_t)(h[i] & 0x1f);
            uint32_t b = (chk >> 25); chk = ((chk & 0x1ffffff) << 5) ^ c;
            for (int j=0;j<5;j++) if ((b>>j)&1) chk ^= GEN[j];
        }
        for (int i = 0; i < vlen; i++) {
            uint8_t c = v[i];
            uint32_t b = (chk >> 25); chk = ((chk & 0x1ffffff) << 5) ^ c;
            for (int j=0;j<5;j++) if ((b>>j)&1) chk ^= GEN[j];
        }
        return chk;
    };

    uint8_t enc[64]; int elen = outlen;
    memcpy(enc, data5, outlen);
    for (int i = 0; i < 6; i++) enc[elen++] = 0;
    uint32_t chk = polymod(enc, elen) ^ 1;
    uint8_t chkbytes[6];
    for (int i = 0; i < 6; i++) chkbytes[i] = (chk >> (5*(5-i))) & 0x1f;

    // 4. Encode
    int alen = 0;
    addr[alen++] = 'b'; addr[alen++] = 'c'; addr[alen++] = '1';
    for (int i = 0; i < outlen; i++) addr[alen++] = CHARSET[data5[i]];
    for (int i = 0; i < 6;      i++) addr[alen++] = CHARSET[chkbytes[i]];
    addr[alen] = 0;
}

// ---- P2PKH compressed address from EcPoint ----------------------------------
static void PubKeyToAddress(const EcPoint& pub, char* addr)
{
    uint8_t pk[33];
    pk[0] = ((pub.y.data[0] & 1) == 0) ? 0x02 : 0x03;
    for (int i = 0; i < 32; i++)
        pk[1+i] = (uint8_t)((pub.x.data[7-(i/4)] >> (8*(3-(i%4)))) & 0xFF);
    uint8_t sha[32], rmd[20];
    sha256_block(pk, 33, sha);
    ripemd160_block(sha, 32, rmd);
    uint8_t payload[21], chk[32], chk2[32];
    payload[0] = 0x00;
    memcpy(payload+1, rmd, 20);
    sha256_block(payload, 21, chk);
    sha256_block(chk, 32, chk2);
    uint8_t full[25];
    memcpy(full, payload, 21);
    memcpy(full+21, chk2, 4);
    static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t tmp[25]; memcpy(tmp, full, 25);
    char rev[40]; int rlen = 0;
    bool nonzero = true;
    while (nonzero) {
        nonzero = false;
        uint32_t rem = 0;
        for (int i = 0; i < 25; i++) {
            uint32_t cur = rem * 256 + tmp[i];
            tmp[i] = (uint8_t)(cur / 58);
            rem = cur % 58;
            if (tmp[i]) nonzero = true;
        }
        rev[rlen++] = B58[rem];
    }
    int lead = 0;
    while (lead < 25 && full[lead] == 0) { addr[lead] = '1'; lead++; }
    int alen = lead;
    for (int i = rlen-1; i >= 0; i--) addr[alen++] = rev[i];
    addr[alen] = 0;
}

// =============================================================================
int main(int argc, char* argv[])
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    printf("********************************************************************************\r\n");
    printf("*        RCKangaroo v3.2 Hybrid+SOTA+  (c) 2024 RetiredCoder + fmg75           *\r\n");
    printf("*        GPU+CPU Hybrid with SOTA+ Optimizations (+10-30%% performance)        *\r\n");
    printf("*        Nataanii's Optimized Fork - SOTA++ Herds Save Functions               *\r\n");
    printf("********************************************************************************\r\n\r\n");
    printf("This software is free and open-source: https://github.com/RetiredC\r\n");
    printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");
#ifdef _WIN32
    printf("Windows version\r\n");
#else
    printf("Linux version\r\n");
#endif
#ifdef DEBUG_MODE
    printf("DEBUG MODE\r\n\r\n");
#endif

    InitEc();
    gDP = 14;
    gDP_manual = false;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    gCpuThreads = 0;
    CpuCnt = 0;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
    if (!ParseCommandLine(argc, argv))
        return 0;

    InitGpus();
    CpuCnt = gCpuThreads;

    if (!GpuCnt && !CpuCnt)
    {
        printf("No workers configured!\r\n");
        return 0;
    }
    if (GpuCnt == 0 && CpuCnt > 0)
        printf("Running in CPU-only mode with %d threads\r\n", CpuCnt);

    pPntList  = (u8*)malloc((u64)MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc((u64)MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0; TotalSolved = 0; gTotalErrors = 0;
    IsBench = gPubKey.x.IsZero();

    // ── Work file + session state ────────────────────────────────────────────
    if (!g_work_filename.empty() && !IsBench && !gGenMode)
    {
        g_work_file  = new RCWorkFile(g_work_filename);
        g_start_time = time(NULL);
        if (WorkFileExists(g_work_filename))
        {
            printf("Found existing work file, resuming...\r\n");
            if (g_work_file->Load())
            {
                g_resume_mode = true;
                if (!g_work_file->IsCompatible(gRange, gDP, (const uint8_t*)gPubKey.x.data, (const uint8_t*)gPubKey.y.data, g_force_resume))
                {
                    printf("ERROR: Work file parameters don't match!\r\n");
                    delete g_work_file; g_work_file = nullptr; return 1;
                }
                TotalOps = g_work_file->GetTotalOps();
                printf("Resuming from: %llu operations\r\n", (unsigned long long)TotalOps);
                // Restore db from companion .kangs file (full serialization, no per-DP AddDP overhead)
                if (!g_kangs_filename.empty() && IsFileExist((char*)g_kangs_filename.c_str()))
                {
                    printf("Loading DB from %s...\r\n", g_kangs_filename.c_str());
                    if (db.LoadFromFile((char*)g_kangs_filename.c_str()))
                        printf("Resume complete! %llu DPs loaded\r\n", (unsigned long long)db.GetBlockCnt());
                    else
                        printf("WARNING: Failed to load .kangs file, starting with empty DB\r\n");
                }
                else
                {
                    printf("No .kangs file found — starting with empty DB (DPs from prior session lost)\r\n");
                }
                LoadSessionState();  // restore restart count, best-K, adaptive thresh
            }
            else
            {
                printf("Failed to load work file\r\n");
                delete g_work_file; g_work_file = nullptr; return 1;
            }
        }
        else
        {
            printf("Creating new work file...\r\n");
            if (!g_work_file->Create(gRange, gDP, (const uint8_t*)gPubKey.x.data, (const uint8_t*)gPubKey.y.data,
                                     (const uint64_t*)gStart.data, nullptr))
            {
                printf("Failed to create work file\r\n");
                delete g_work_file; g_work_file = nullptr; return 1;
            }
            printf("Work file created successfully\r\n");
        }
        if (g_autosave_interval > 0)
        {
            g_autosave = new AutoSaveManager(g_work_file, g_autosave_interval);
            printf(RC_BGREEN "Auto-save: every %llu seconds -> %s" RC_RESET "\r\n",
                   (unsigned long long)g_autosave_interval, g_work_filename.c_str());
        }
        signal(SIGINT,  SignalHandler);
        signal(SIGTERM, SignalHandler);
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Warn if -autosave was given but no -work file (saves would be silently lost)
    if (g_autosave_interval > 0 && g_work_filename.empty() && !IsBench && !gGenMode)
    {
        printf(RC_BYELLOW "WARNING: -autosave requires -work <file> to save into.\r\n"
               "         No work file specified — auto-save is DISABLED.\r\n"
               "         Example: -work puzzle135.wf -autosave 300\r\n" RC_RESET);
    }

    g_attempt_start_ms = GetTickCount64();

    if (!IsBench && !gGenMode)
    {
        printf("\r\nMAIN MODE\r\n\r\n");
        EcPoint PntToSolve, PntOfs;
        EcInt pk, pk_found;

        PntToSolve = gPubKey;
        if (!gStart.IsZero())
        {
            PntOfs = ec.MultiplyG_Lambda(gStart);
            PntOfs.y.NegModP();
            PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
        }
        char sx[100], sy[100];
        gPubKey.x.GetHexStr(sx); gPubKey.y.GetHexStr(sy);
        printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
        gStart.GetHexStr(sx);
        printf("Offset: %s\r\n", sx);

        if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
        {
            if (!gIsOpsLimit && !g_shutdown_requested)
                printf("FATAL ERROR: SolvePoint failed\r\n");
            goto label_end;
        }
        pk_found.AddModP(gStart);
        EcPoint tmp = ec.MultiplyG_Lambda(pk_found);
        if (!tmp.IsEqual(gPubKey))
        {
            printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
            goto label_end;
        }

        char s[100]; pk_found.GetHexStr(s);
        char wif[60] = {}, adr[40] = {}, swp[48] = {};
        PrivKeyToWIF(s, wif);
        EcPoint pub = ec.MultiplyG_Lambda(pk_found);
        PubKeyToAddress(pub, adr);
        PubKeyToSWP(pub, swp);

        // Verify k*G == original pubkey
        EcPoint chk = ec.MultiplyG_Lambda(pk_found);
        bool ok = chk.IsEqual(gPubKey);

        // KEY FOUND box (72 chars inner width)
        #define KBOX_W 72
        auto kpad = [](const char* v, int w) -> int { return w - (int)strlen(v); };
        // Top border
        printf("\r\n" RC_BWHITE "\xe2\x95\x94");
        for (int _i=0;_i<KBOX_W;_i++) printf("\xe2\x95\x90");
        printf("\xe2\x95\x97\r\n");
        // Title row
        { int _tp = (KBOX_W - 9) / 2;
          printf("\xe2\x95\x91");
          for (int _i=0;_i<_tp;_i++) printf(" ");
          printf(RC_BYELLOW " KEY FOUND " RC_BWHITE);
          for (int _i=0;_i<KBOX_W-_tp-9;_i++) printf(" ");
          printf("\xe2\x95\x91\r\n"); }
        // Separator
        printf("\xe2\x95\xa0");
        for (int _i=0;_i<KBOX_W;_i++) printf("\xe2\x95\x90");
        printf("\xe2\x95\xa3\r\n");
        // HEX row
        printf("\xe2\x95\x91  " RC_BWHITE "HEX: " RC_BGREEN "%s" RC_BWHITE, s);
        for (int _i=0;_i<kpad(s,KBOX_W-7);_i++) printf(" ");
        printf("\xe2\x95\x91\r\n");
        // WIF row
        printf("\xe2\x95\x91  " RC_BWHITE "WIF: " RC_BGREEN "%s" RC_BWHITE, wif);
        for (int _i=0;_i<kpad(wif,KBOX_W-7);_i++) printf(" ");
        printf("\xe2\x95\x91\r\n");
        // ADR row
        printf("\xe2\x95\x91  " RC_BWHITE "ADR: " RC_BGREEN "%s" RC_BWHITE, adr);
        for (int _i=0;_i<kpad(adr,KBOX_W-7);_i++) printf(" ");
        printf("\xe2\x95\x91\r\n");
        // Separator before SWP
        printf("\xe2\x95\xa0");
        for (int _i=0;_i<KBOX_W;_i++) printf("\xe2\x95\x90");
        printf("\xe2\x95\xa3\r\n");
        // SWP row
        printf("\xe2\x95\x91  " RC_BWHITE "SWP: " RC_BYELLOW "%s" RC_BWHITE, swp);
        for (int _i=0;_i<kpad(swp,KBOX_W-7);_i++) printf(" ");
        printf("\xe2\x95\x91\r\n");
        // Bottom border
        printf("\xe2\x95\x9a");
        for (int _i=0;_i<KBOX_W;_i++) printf("\xe2\x95\x90");
        printf("\xe2\x95\x9d" RC_RESET "\r\n");
        #undef KBOX_W
        printf("  Verification: k*G == pubkey  %s\r\n\r\n",
               ok ? RC_BGREEN "OK" RC_RESET : RC_BRED "FAIL" RC_RESET);


        if (g_work_file && g_start_time > 0)
        {
            uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
            g_work_file->UpdateProgress(PntTotalOps, db.GetBlockCnt(), gTotalErrors, elapsed);
            g_work_file->Save();
            if (!g_kangs_filename.empty()) db.SaveToFile((char*)g_kangs_filename.c_str());
        }
        SaveSessionState();

        FILE* fp = fopen("RESULTS.TXT", "a");
        if (fp)
        {
            fprintf(fp, "PRIVATE KEY: %s\nWIF: %s\nAddress: %s\nSegWit:  %s\n\n", s, wif, adr, swp);
            fclose(fp);
        }
        else
        {
            printf("WARNING: Cannot save to RESULTS.TXT!\r\n");
            while (1) Sleep(100);
        }

        // Save to key_backup_1.txt and key_backup_2.txt (redundant copies)
        const char* backup_files[2] = { "key_backup_1.txt", "key_backup_2.txt" };
        for (int _b = 0; _b < 2; _b++)
        {
            FILE* fb = fopen(backup_files[_b], "a");
            if (fb)
            {
                fprintf(fb, "PRIVATE KEY (HEX): %s\n", s);
                fprintf(fb, "PRIVATE KEY (WIF): %s\n", wif);
                fprintf(fb, "ADDRESS (P2PKH): %s\n", adr);
                fprintf(fb, "ADDRESS (SegWit): %s\n\n", swp);
                fclose(fb);
                printf("Saved to %s\r\n", backup_files[_b]);
            }
            else
            {
                printf("WARNING: Cannot save to %s!\r\n", backup_files[_b]);
            }
        }
    }
    else
    {
        if (gGenMode) printf("\r\nTAMES GENERATION MODE\r\n");
        else          printf("\r\nBENCHMARK MODE\r\n");
        while (1)
        {
            EcInt pk, pk_found; EcPoint PntToSolve;
            if (!gRange) gRange = 78;
            if (!gDP)    gDP    = 14;
            pk.RndBits(gRange);
            PntToSolve = ec.MultiplyG_Lambda(pk);
            if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
            {
                if (!gIsOpsLimit) printf("FATAL ERROR: SolvePoint failed\r\n");
                break;
            }
            if (!pk_found.IsEqual(pk)) { printf("FATAL ERROR: Found key is wrong!\r\n"); break; }
            TotalOps += PntTotalOps; TotalSolved++;
            u64 ops_per_pnt = TotalOps / TotalSolved;
            double K_bench = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
            printf("K: %.3f, ops: 2^%.3f, avg: 2^%.3f\r\n\r\n",
                   K_bench, log2((double)PntTotalOps), log2((doubl