// GPU Monitoring and Thermal Management Implementation

#include "GpuMonitor.h"
#include "utils.h"
#include <cstring>
#include <cmath>
#include <cstdint>

// ANSI color codes
#define GM_RESET    "\033[0m"
#define GM_BOLD     "\033[1m"
#define GM_WHITE    "\033[37m"
#define GM_BWHITE   "\033[1;37m"
#define GM_GREEN    "\033[32m"
#define GM_BGREEN   "\033[1;32m"
#define GM_YELLOW   "\033[33m"
#define GM_BYELLOW  "\033[1;33m"
#define GM_RED      "\033[31m"
#define GM_BRED     "\033[1;31m"
#define GM_CYAN     "\033[36m"
#define GM_BCYAN    "\033[1;36m"
#define GM_MAGENTA  "\033[35m"
#define GM_BMAGENTA "\033[1;35m"

GpuMonitor* g_gpu_monitor = nullptr;

// K-factor sparkline — ring buffer maintained in RCKangaroo.cpp
#define SPARK_LEN 32
extern double g_spark[SPARK_LEN];
extern int    g_spark_head;
extern int    g_spark_cnt;
extern double g_sys_overhead_w;
extern double g_kwh_rate;

// Lissajous monitor state — set once at startup in RCKangaroo.cpp
extern double g_lissa_j1_bits;
extern int    g_lissa_range;

// R2-4V scramble state — set by BuildR2JumpTable4V in RCKangaroo.cpp
extern uint32_t g_r2_salt;
extern double   g_r2_off0;
extern double   g_r2_off1;

GpuMonitor::GpuMonitor() {
    initialized = false;
    thermal_policy = THERMAL_BALANCED;
    performance_sample_idx = 0;
    memset(&sys_stats, 0, sizeof(sys_stats));
    memset(gpu_stats, 0, sizeof(gpu_stats));
    memset(gpu_performance_history, 0, sizeof(gpu_performance_history));
    memset(power_limit_default, 0, sizeof(power_limit_default));
    memset(power_limit_throttle, 0, sizeof(power_limit_throttle));
}

GpuMonitor::~GpuMonitor() {
    Shutdown();
}

bool GpuMonitor::Initialize(int gpu_count) {
    if (gpu_count <= 0 || gpu_count > MAX_GPU_CNT) {
        printf("GPU Monitor: Invalid GPU count: %d\n", gpu_count);
        return false;
    }

    sys_stats.gpu_count = gpu_count;

#ifdef USE_NVML
    // Initialize NVML — fall back to speed-only mode if NVML unavailable
    bool nvml_ok = InitNVML();
    if (!nvml_ok) {
        printf("GPU Monitor: NVML unavailable — speed/K display active, no temp/power data\n");
        for (int i = 0; i < gpu_count; i++) {
            gpu_stats[i].gpu_id = i;
            gpu_stats[i].pci_bus = i;
            power_limit_default[i] = 170;
            power_limit_throttle[i] = 145;
        }
        initialized = true;
        return true;
    }

    // Get NVML device handles
    for (int i = 0; i < gpu_count; i++) {
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &nvml_devices[i]);
        if (result != NVML_SUCCESS) {
            printf("GPU Monitor: Failed to get device handle for GPU %d: %s — continuing without it\n",
                   i, nvmlErrorString(result));
            nvml_devices[i] = nullptr;
            continue;
        }

        gpu_stats[i].gpu_id = i;

        // Get default power limit
        unsigned int power_limit;
        result = nvmlDeviceGetPowerManagementLimit(nvml_devices[i], &power_limit);
        if (result == NVML_SUCCESS) {
            power_limit_default[i] = power_limit / 1000; // mW to W
            power_limit_throttle[i] = (unsigned int)(power_limit_default[i] * 0.85); // 85% for throttling
        } else {
            power_limit_default[i] = 170; // RTX 3060 default
            power_limit_throttle[i] = 145;
        }

        // Get PCI bus ID
        nvmlPciInfo_t pci_info;
        result = nvmlDeviceGetPciInfo(nvml_devices[i], &pci_info);
        if (result == NVML_SUCCESS) {
            gpu_stats[i].pci_bus = pci_info.bus;
        }
    }

    UpdateThermalThresholds();
    initialized = true;

    printf("GPU Monitor: Initialized for %d GPUs\n", gpu_count);
    printf("Thermal Policy: ");
    switch (thermal_policy) {
        case THERMAL_AGGRESSIVE: printf("AGGRESSIVE (< 85°C)\n"); break;
        case THERMAL_BALANCED:   printf("BALANCED (< 80°C)\n"); break;
        case THERMAL_QUIET:      printf("QUIET (< 75°C)\n"); break;
    }

    return true;
#else
    printf("GPU Monitor: NVML support not compiled (USE_NVML not defined)\n");
    printf("GPU Monitor: Running with limited functionality\n");

    // Initialize basic stats without NVML
    for (int i = 0; i < gpu_count; i++) {
        gpu_stats[i].gpu_id = i;
        gpu_stats[i].pci_bus = i;
        power_limit_default[i] = 170;
        power_limit_throttle[i] = 145;
    }

    UpdateThermalThresholds();
    initialized = true;
    return true;
#endif
}

void GpuMonitor::Shutdown() {
    if (initialized) {
        RestorePowerLimits();
#ifdef USE_NVML
        nvmlShutdown();
#endif
        initialized = false;
    }
}

bool GpuMonitor::InitNVML() {
#ifdef USE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return false;
    }
    return true;
#else
    return false;
#endif
}

void GpuMonitor::UpdateThermalThresholds() {
    switch (thermal_policy) {
        case THERMAL_AGGRESSIVE:
            temp_warning = 80;
            temp_throttle = 83;
            temp_critical = 85;
            break;
        case THERMAL_BALANCED:
            temp_warning = 75;
            temp_throttle = 78;
            temp_critical = 80;
            break;
        case THERMAL_QUIET:
            temp_warning = 70;
            temp_throttle = 73;
            temp_critical = 75;
            break;
    }
}

bool GpuMonitor::UpdateStats(int gpu_id) {
    if (!initialized || gpu_id < 0 || gpu_id >= sys_stats.gpu_count) {
        return false;
    }

#ifdef USE_NVML
    GPUStats* stats = &gpu_stats[gpu_id];
    nvmlDevice_t device = nvml_devices[gpu_id];

    // If this GPU's NVML handle is null (failed at init), skip hardware queries
    if (!device)
        return true;

    // Temperature
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &stats->temp_c);

    // Power
    nvmlDeviceGetPowerUsage(device, &stats->power_mw);

    // Utilization
    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
        stats->util_pct = util.gpu;
    }

    // Memory
    nvmlMemory_t mem_info;
    if (nvmlDeviceGetMemoryInfo(device, &mem_info) == NVML_SUCCESS) {
        stats->mem_used_mb = (unsigned int)(mem_info.used / (1024 * 1024));
    }

    // Check for throttling
    stats->throttling = (stats->temp_c >= temp_throttle);

    return true;
#else
    // Without NVML, can't update hardware stats
    return false;
#endif
}

void GpuMonitor::UpdateAllGPUs() {
    sys_stats.total_gpu_speed = 0;
    sys_stats.avg_temp_c = 0;
    sys_stats.total_power_w = 0;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        UpdateStats(i);
        // CRITICAL: Copy speed and seed from sys_stats to gpu_stats for display!
        gpu_stats[i].speed_mkeys = sys_stats.gpu_stats[i].speed_mkeys;
        gpu_stats[i].seed        = sys_stats.gpu_stats[i].seed;
        sys_stats.total_gpu_speed += gpu_stats[i].speed_mkeys / 1000.0; // Convert to GKeys/s
        sys_stats.avg_temp_c += gpu_stats[i].temp_c;
        sys_stats.total_power_w += gpu_stats[i].power_mw / 1000;
    }

    if (sys_stats.gpu_count > 0) {
        sys_stats.avg_temp_c /= sys_stats.gpu_count;
    }

    // Update elapsed time and ETA
    sys_stats.elapsed_ms = GetTickCount64() - sys_stats.start_time_ms;

    // Calculate ETA based on current K-factor
    if (sys_stats.actual_ops > 0 && sys_stats.expected_ops > 0) {
        sys_stats.current_k_factor = sys_stats.actual_ops / sys_stats.expected_ops;
        double total_ops_needed = sys_stats.expected_ops * 1.15; // SOTA theoretical
        double ops_remaining = total_ops_needed - sys_stats.actual_ops;
        double current_speed = (sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0) * 1e9; // ops/sec
        if (current_speed > 0) {
            sys_stats.eta_ms = (u64)((ops_remaining / current_speed) * 1000.0);
        }
    }
}

GPUStats GpuMonitor::GetGPUStats(int gpu_id) {
    if (gpu_id >= 0 && gpu_id < sys_stats.gpu_count) {
        return gpu_stats[gpu_id];
    }
    GPUStats empty = {0};
    return empty;
}

SystemStats GpuMonitor::GetSystemStats() {
    return sys_stats;
}

void GpuMonitor::SetSystemStats(const SystemStats& stats) {
    sys_stats = stats;
}

void GpuMonitor::SetThermalPolicy(ThermalPolicy policy) {
    thermal_policy = policy;
    UpdateThermalThresholds();
    printf("GPU Monitor: Thermal policy changed to ");
    switch (policy) {
        case THERMAL_AGGRESSIVE: printf("AGGRESSIVE\n"); break;
        case THERMAL_BALANCED:   printf("BALANCED\n"); break;
        case THERMAL_QUIET:      printf("QUIET\n"); break;
    }
}

bool GpuMonitor::ApplyThermalLimits() {
#ifdef USE_NVML
    bool throttled = false;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        unsigned int temp = gpu_stats[i].temp_c;
        nvmlDevice_t device = nvml_devices[i];

        if (temp >= temp_critical) {
            // Critical: Reduce to 85% power
            unsigned int limit = power_limit_throttle[i] * 1000; // W to mW
            if (nvmlDeviceSetPowerManagementLimit(device, limit) == NVML_SUCCESS) {
                gpu_stats[i].throttling = true;
                throttled = true;
                if (temp >= temp_critical + 2) {
                    printf("⚠️  GPU %d CRITICAL TEMP: %u°C → Reduced to %uW\n",
                           i, temp, power_limit_throttle[i]);
                }
            }
        } else if (temp >= temp_throttle) {
            // Throttle: Reduce to 90% power
            unsigned int limit = (unsigned int)(power_limit_default[i] * 0.90) * 1000;
            if (nvmlDeviceSetPowerManagementLimit(device, limit) == NVML_SUCCESS) {
                gpu_stats[i].throttling = true;
                throttled = true;
            }
        } else if (temp < temp_warning && gpu_stats[i].throttling) {
            // Restore full power
            unsigned int limit = power_limit_default[i] * 1000;
            nvmlDeviceSetPowerManagementLimit(device, limit);
            gpu_stats[i].throttling = false;
        }
    }

    return throttled;
#else
    return false;
#endif
}

void GpuMonitor::RestorePowerLimits() {
#ifdef USE_NVML
    for (int i = 0; i < sys_stats.gpu_count; i++) {
        unsigned int limit = power_limit_default[i] * 1000;
        nvmlDeviceSetPowerManagementLimit(nvml_devices[i], limit);
        gpu_stats[i].throttling = false;
    }
#endif
}

void GpuMonitor::CalculateOptimalDistribution(int total_kangaroos, int* per_gpu_kangaroos) {
    // Calculate efficiency scores for each GPU
    double efficiency[MAX_GPU_CNT] = {0};
    double total_efficiency = 0;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        // Efficiency = (speed / power) * thermal_factor
        double thermal_factor = 1.0;
        if (gpu_stats[i].temp_c >= temp_throttle) {
            thermal_factor = 0.7; // Penalize hot GPUs
        } else if (gpu_stats[i].temp_c >= temp_warning) {
            thermal_factor = 0.85;
        }

        if (gpu_stats[i].power_mw > 0) {
            efficiency[i] = (gpu_stats[i].speed_mkeys / (gpu_stats[i].power_mw / 1000.0)) * thermal_factor;
        } else {
            efficiency[i] = 1.0; // Equal if no power data
        }
        total_efficiency += efficiency[i];
    }

    // Distribute kangaroos proportionally to efficiency
    if (total_efficiency > 0) {
        int distributed = 0;
        for (int i = 0; i < sys_stats.gpu_count; i++) {
            if (i == sys_stats.gpu_count - 1) {
                // Last GPU gets remainder to ensure total is exact
                per_gpu_kangaroos[i] = total_kangaroos - distributed;
            } else {
                per_gpu_kangaroos[i] = (int)(total_kangaroos * (efficiency[i] / total_efficiency));
                distributed += per_gpu_kangaroos[i];
            }
        }
    } else {
        // Fallback: Equal distribution
        int per_gpu = total_kangaroos / sys_stats.gpu_count;
        for (int i = 0; i < sys_stats.gpu_count; i++) {
            per_gpu_kangaroos[i] = per_gpu;
        }
    }
}

double GpuMonitor::GetGPUEfficiency(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= sys_stats.gpu_count) return 0;
    if (gpu_stats[gpu_id].power_mw == 0) return 0;
    return gpu_stats[gpu_id].speed_mkeys / (gpu_stats[gpu_id].power_mw / 1000.0);
}

void GpuMonitor::PrintDetailedStats() {
    // Inline lambdas for colour selection
    auto temp_color = [&](unsigned int t) -> const char* {
        if (t >= 90)                    return GM_BRED;
        if (t >= (unsigned)temp_warning) return GM_BYELLOW;
        return GM_GREEN;
    };
    auto kfac_color = [&](double k) -> const char* {
        if (k < 1.15) return GM_BGREEN;
        if (k < 1.3)  return GM_BYELLOW;
        return GM_BRED;
    };
    auto rate_color = [&](double r) -> const char* {
        if (r >= 50000) return GM_BGREEN;
        if (r >= 10000) return GM_BYELLOW;
        return GM_BRED;
    };

    printf("\n");
    printf(GM_BWHITE "================================================================\n" GM_RESET);
    printf(GM_BWHITE "  GPU Performance Monitor\n"                                        GM_RESET);
    printf(GM_BWHITE "================================================================\n" GM_RESET);

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        GPUStats* s = &gpu_stats[i];
        bool has_nvml = (nvml_devices[i] != nullptr);
        printf(GM_BWHITE "GPU %d:" GM_RESET, i);
        printf("  " GM_CYAN "%.2f GK/s" GM_RESET, s->speed_mkeys / 1000.0);
        if (has_nvml) {
            printf("  |  %s%3u\xc2\xb0""C" GM_RESET, temp_color(s->temp_c), s->temp_c);
            printf("  |  " GM_WHITE "%3uW" GM_RESET, s->power_mw / 1000);
            printf("  |  " GM_WHITE "%3u%% util" GM_RESET, s->util_pct);
            printf("  |  " GM_WHITE "PCI %d" GM_RESET, s->pci_bus);
            if (s->throttling)
                printf("  " GM_BRED "THROTTLING" GM_RESET);
            else if (s->temp_c >= (unsigned)temp_warning)
                printf("  " GM_BYELLOW "WARM" GM_RESET);
        } else {
            printf("  |  " GM_WHITE "temp/power: N/A (NVML unavailable)" GM_RESET);
        }
        printf("\n");
        printf("         " GM_MAGENTA "seed: 0x%016llX" GM_RESET "\n",
               (unsigned long long)s->seed);
    }

    printf("\n");
    printf(GM_BWHITE "CPU:  " GM_RESET GM_CYAN "%.1f MK/s" GM_RESET "\n",
           sys_stats.cpu_speed_mkeys);
    printf(GM_BWHITE "Total:" GM_RESET "  "
           GM_CYAN "%.2f GK/s" GM_RESET
           "  |  Avg Temp: %s%u\xc2\xb0""C" GM_RESET
           "  |  Power: " GM_WHITE "%uW" GM_RESET "\n",
           sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0,
           temp_color(sys_stats.avg_temp_c), sys_stats.avg_temp_c,
           sys_stats.total_power_w);

    // K-Factor  +  P(solve)
    double kf = sys_stats.current_k_factor;
    // Solve probability: P = 1 − exp(−K)  (exponential collision model)
    double solve_prob = (kf > 0.0) ? (1.0 - exp(-kf)) * 100.0 : 0.0;
    printf("\n" GM_BWHITE "K-Factor:" GM_RESET "  %s%.3f" GM_RESET "  ",
           kfac_color(kf), kf);
    if      (kf < 1.0)  printf(GM_BGREEN  "OK (ahead of schedule)" GM_RESET);
    else if (kf < 1.15) printf(GM_BGREEN  "OK (on track)"          GM_RESET);
    else if (kf < 1.3)  printf(GM_BYELLOW "SLOW (slightly slow)"   GM_RESET);
    else                printf(GM_BRED    "BAD (check for issues)"  GM_RESET);
    printf("  |  " GM_BWHITE "P(solve):" GM_RESET "  " GM_BMAGENTA "%.1f%%" GM_RESET "\n",
           solve_prob);

    // ── K-factor sparkline ───────────────────────────────────────────────────
    // Braille-style 8-level blocks: ▁▂▃▄▅▆▇█  (0.0 → 2.0 K range)
    if (g_spark_cnt > 0) {
        // Find min/max across filled samples for dynamic scaling
        double sk_min = 1e18, sk_max = 0.0;
        int    total  = (g_spark_cnt < SPARK_LEN) ? g_spark_cnt : SPARK_LEN;
        for (int i = 0; i < total; i++) {
            double v = g_spark[i];
            if (v > 0.0 && v < sk_min) sk_min = v;
            if (v > sk_max)            sk_max = v;
        }
        if (sk_max <= sk_min) sk_max = sk_min + 0.01;

        // Block chars: 8 levels
        static const char* bars[] = {
            "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
            "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"
        };

        printf(GM_BWHITE "K trend:" GM_RESET "  ");
        // Print oldest → newest (oldest is at head, since head is next write pos)
        for (int i = 0; i < total; i++) {
            int ridx = (g_spark_head - total + i + SPARK_LEN) % SPARK_LEN;
            double v = g_spark[ridx];
            if (v <= 0.0) { printf(" "); continue; }
            int bar = (int)((v - sk_min) / (sk_max - sk_min) * 7.0 + 0.5);
            bar = (bar < 0) ? 0 : (bar > 7) ? 7 : bar;
            // Color: green below 1.0, yellow 1.0-1.15, red above
            const char* sc = (v < 1.0) ? GM_BGREEN : (v < 1.15) ? GM_BYELLOW : GM_BRED;
            printf("%s%s" GM_RESET, sc, bars[bar]);
        }
        printf("  " GM_WHITE "%.2f–%.2f" GM_RESET "\n", sk_min, sk_max);
    }
    // ────────────────────────────────────────────────────────────────────────

    // DPs / Rate / Buffer
    double dp_pct  = (sys_stats.dp_expected > 0)
        ? 100.0 * (double)sys_stats.dp_count / (double)sys_stats.dp_expected : 0.0;
    double buf_pct = (sys_stats.dp_buffer_total > 0)
        ? 100.0 * sys_stats.dp_buffer_used / (double)sys_stats.dp_buffer_total : 0.0;

    printf(GM_BWHITE "DPs:" GM_RESET "  %llu / %llu "
           GM_YELLOW "(%.1f%%)" GM_RESET
           "  |  " GM_BWHITE "Rate:" GM_RESET "  %s%.0f/s" GM_RESET
           "  |  " GM_BWHITE "Buffer:" GM_RESET "  %u / %u (%.1f%%)\n",
           (unsigned long long)sys_stats.dp_count,
           (unsigned long long)sys_stats.dp_expected,
           dp_pct,
           rate_color(sys_stats.dp_rate_per_sec), sys_stats.dp_rate_per_sec,
           sys_stats.dp_buffer_used, sys_stats.dp_buffer_total, buf_pct);

    // ETA  (total_gpu_speed is in GK/s, cpu_speed_mkeys is in MK/s)
    if (sys_stats.total_gpu_speed > 0 && sys_stats.expected_ops > 1 && sys_stats.actual_ops > 1) {
        double total_spd_gks = sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0;
        double remaining     = sys_stats.expected_ops - sys_stats.actual_ops;
        if (remaining < 0) remaining = 0;
        u64 eta_s = (u64)(remaining / (total_spd_gks * 1e9));
        u64 eta_d = eta_s / 86400;
        int eta_h = (int)((eta_s % 86400) / 3600);
        int eta_m = (int)((eta_s % 3600) / 60);
        int eta_sc = (int)(eta_s % 60);
        printf(GM_BWHITE "ETA:" GM_RESET "  " GM_CYAN "%llud %02dh %02dm:%02ds" GM_RESET
               "  |  Ops: 2^%.2f / 2^%.2f\n",
               eta_d, eta_h, eta_m, eta_sc,
               log2(sys_stats.actual_ops),
               log2(sys_stats.expected_ops));
    }


    // cost tracker
    {
        double gpu_w     = (double)sys_stats.total_power_w;
        double total_w   = gpu_w + g_sys_overhead_w;
        double cost_hr   = total_w / 1000.0 * g_kwh_rate;
        double cost_day  = cost_hr * 24.0;
        double cost_yr   = cost_day * 365.0;
        double elap_hr   = (double)sys_stats.elapsed_ms / 3600000.0;
        double cost_sess = cost_hr * elap_hr;
        char sess_buf[32], hr_buf[32];
        if (cost_sess < 1.0)
            snprintf(sess_buf, sizeof(sess_buf), "%.2f cents", cost_sess * 100.0);
        else
            snprintf(sess_buf, sizeof(sess_buf), "$%.4f", cost_sess);
        snprintf(hr_buf, sizeof(hr_buf), "$%.4f", cost_hr);
        printf("\n" GM_BWHITE "\xe2\x9a\xa1 Cost:" GM_RESET
               "  " GM_WHITE "%.0fW" GM_RESET " gpu"
               "  +  " GM_WHITE "%.0fW" GM_RESET " sys"
               "  =  " GM_BYELLOW "%.0fW" GM_RESET " total"
               "  |  " GM_BYELLOW "%s/hr" GM_RESET
               "  |  Session: " GM_BGREEN "%s" GM_RESET
               "  |  Daily: " GM_CYAN "$%.2f" GM_RESET
               "  |  Annual: " GM_WHITE "$%.0f" GM_RESET "\n",
               gpu_w, g_sys_overhead_w, total_w,
               hr_buf, sess_buf, cost_day, cost_yr);
    }

    // ── Wild-only mode indicator ─────────────────────────────────────────────
    if (sys_stats.wild_only_active) {
        char tame_buf[32];
        if (sys_stats.loaded_tame_cnt >= 1000000)
            snprintf(tame_buf, sizeof(tame_buf), "%.2fM", sys_stats.loaded_tame_cnt / 1000000.0);
        else if (sys_stats.loaded_tame_cnt >= 1000)
            snprintf(tame_buf, sizeof(tame_buf), "%.1fK", sys_stats.loaded_tame_cnt / 1000.0);
        else
            snprintf(tame_buf, sizeof(tame_buf), "%llu", (unsigned long long)sys_stats.loaded_tame_cnt);
        printf(GM_BGREEN "Wild-only mode" GM_RESET
               "  |  Tames loaded: " GM_CYAN "%s" GM_RESET " (static traps)"
               "  |  Running: " GM_WHITE "100%% WILDs\n" GM_RESET,
               tame_buf);
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── Algorithm feature status + jump table monitor ───────────────────────
    {
        char j1_buf[32];
#if USE_LISSAJOUS
        snprintf(j1_buf, sizeof(j1_buf), "~2^%.1f", g_lissa_j1_bits);
#else
        snprintf(j1_buf, sizeof(j1_buf), "2^%.1f", g_lissa_j1_bits);
#endif
        printf(GM_BWHITE "Algo:  " GM_RESET
#if USE_SOTA_PLUS
               GM_BGREEN "SOTA+ \xe2\x9c\x93" GM_RESET
#else
               GM_BRED   "SOTA+ \xe2\x9c\x97" GM_RESET
#endif
               "  |  "
#if USE_JACOBIAN
               GM_BGREEN "Jacobian \xe2\x9c\x93" GM_RESET
#else
               GM_BRED   "Jacobian \xe2\x9c\x97" GM_RESET
#endif
               "  |  "
#if USE_LISSAJOUS
               GM_BGREEN "LISSA \xe2\x9c\x93" GM_RESET
#else
               GM_BRED   "LISSA \xe2\x9c\x97" GM_RESET
#endif
               "  |  "
#if USE_R2_4V
               GM_BGREEN "R2-4V \xe2\x9c\x93" GM_RESET
#else
               GM_BRED   "R2-4V \xe2\x9c\x97" GM_RESET
#endif
               "  |  J1 " GM_CYAN "%s" GM_RESET
               "  |  J2=2^%d  |  J3=2^%d\n",
               j1_buf,
               (g_lissa_range > 10) ? g_lissa_range - 10 : 0,
               (g_lissa_range > 12) ? g_lissa_range - 12 : 0);
    }  // end Algo block

#if USE_R2_4V
    // ── R2-4V scramble status line ───────────────────────────────────────────
    {
        // Color-code the offsets: green=well-spread (0.25-0.75), yellow=edges, red=near 0/1
        auto off_color = [](double v) -> const char* {
            if (v >= 0.25 && v <= 0.75) return GM_BGREEN;
            if (v >= 0.10 && v <= 0.90) return GM_BYELLOW;
            return GM_BRED;
        };
        // Salt freshness: non-zero = scrambled (green), zero = unscrambled fallback (red)
        const char* salt_col = (g_r2_salt != 0) ? GM_BGREEN : GM_BRED;

        printf(GM_BWHITE "Seq:   " GM_RESET
               "phi+" GM_CYAN "%.4f" GM_RESET
               "  psi+" GM_CYAN "%.4f" GM_RESET
               "  H5-scr  H7-scr"
               "  |  salt=%s0x%08X" GM_RESET "\n",
               g_r2_off0, g_r2_off1, salt_col, g_r2_salt);

        // Second mini-bar: visual spread indicator for V0 and V1
        // Shows [----*----] where * is the offset position across [0,1)
        char bar0[12], bar1[12];
        int pos0 = (int)(g_r2_off0 * 9.0);  // 0..8
        int pos1 = (int)(g_r2_off1 * 9.0);
        for (int b = 0; b < 9; b++) { bar0[b] = (b == pos0) ? '*' : '-'; bar1[b] = (b == pos1) ? '*' : '-'; }
        bar0[9] = bar1[9] = '\0';

        printf(GM_BWHITE "       " GM_RESET
               "V0[%s%s" GM_RESET "]  V1[%s%s" GM_RESET "]"
               "  V2[Owen/5]  V3[Owen/7]\n",
               off_color(g_r2_off0), bar0,
               off_color(g_r2_off1), bar1);

        // ── Jump distribution chart ───────────────────────────────────────
        // 40-char bar spanning [0.0μ, 2.0μ], 0.05μ per cell, μ=cell 20
        // Shows each variant's solid equidistributed coverage vs pseudo-random clustering
        printf(GM_BWHITE "Dist:  " GM_RESET
               GM_WHITE "0.0" GM_RESET
               "μ         "
               GM_WHITE "1.0" GM_RESET
               "μ" GM_WHITE " (μ)" GM_RESET
               "         "
               GM_WHITE "2.0μ" GM_RESET "\n");

        // variant definitions: lo/hi as cell indices (cell = x/0.05)
        struct VDef { const char* name; int lo; int hi; const char* col; const char* label; };
        static const VDef vd[4] = {
            { "V0", 10, 30, GM_BGREEN,   "phi  [0.50" "\xe2\x80\x93" "1.50\xce\xbc]" },
            { "V1",  6, 33, GM_BYELLOW,  "psi  [0.33" "\xe2\x80\x93" "1.67\xce\xbc]" },
            { "V2",  5, 35, GM_BCYAN,    "H(5) [0.25" "\xe2\x80\x93" "1.75\xce\xbc]" },
            { "V3",  8, 32, GM_BMAGENTA, "H(7) [0.40" "\xe2\x80\x93" "1.60\xce\xbc]" },
        };
        for (int vi = 0; vi < 4; vi++) {
            printf("  %s%s" GM_RESET " [", vd[vi].col, vd[vi].name);
            for (int i = 0; i < 40; i++) {
                if (i == 20)                              printf(GM_BWHITE "|" GM_RESET);
                else if (i >= vd[vi].lo && i < vd[vi].hi) printf("%s#" GM_RESET, vd[vi].col);
                else                                       printf(GM_WHITE "-" GM_RESET);
            }
            printf("] %s%s" GM_RESET "\n", vd[vi].col, vd[vi].label);
        }

        // Pseudo-random row: bell-curve density centred on μ, showing clustering + voids
        // Hardcoded density mask: 1=filled, 0=empty — mimics a gamma/normal dist with sigma~0.3
        static const int rnd[40] = {
            0,0,0,0,0, 0,0,1,0,0, 1,0,1,1,0, 1,1,1,0,1,
            1,1,0,1,1, 1,1,1,0,1, 0,1,1,0,1, 0,0,1,0,0
        };
        printf("  " GM_BRED "Rnd" GM_RESET " [");
        for (int i = 0; i < 40; i++) {
            if (i == 20)    printf(GM_BWHITE "|" GM_RESET);
            else if (rnd[i]) printf(GM_BRED "#" GM_RESET);
            else             printf(GM_WHITE "-" GM_RESET);
        }
        printf("] " GM_BRED "random" GM_RESET GM_WHITE " (clustered near \xce\xbc, gaps at edges)" GM_RESET "\n");
    }
#endif
    // ────────────────────────────────────────────────────────────────────────
    printf(GM_BWHITE "================================================================\n" GM_RESET);

    // ── Herds section (only when herds mode is active) ───────────────────────
    if (sys_stats.herds_active) {
        // Format helper: convert raw DP count to "NNN.NK" string
        auto fmt_k = [](u64 v, char* buf, int sz) {
            if (v >= 1000000)
                snprintf(buf, sz, "%.1fM", v / 1000000.0);
            else if (v >= 1000)
                snprintf(buf, sz, "%.1fK", v / 1000.0);
            else
                snprintf(buf, sz, "%llu", (unsigned long long)v);
        };

        char total_dp_buf[32];
        fmt_k(sys_stats.total_herd_dps, total_dp_buf, sizeof(total_dp_buf));

        printf(GM_BWHITE "Herds: " GM_RESET
               GM_BGREEN "ACTIVE" GM_RESET
               "  |  "
               GM_WHITE "%d GPU \xc3\x97 %d herds \xc3\x97 %d kangs" GM_RESET
               "  |  "
               GM_BWHITE "Herd DPs:" GM_RESET " " GM_CYAN "%s" GM_RESET
               "  |  "
               GM_BWHITE "Local Colls:" GM_RESET " " GM_WHITE "%llu" GM_RESET "\n",
               sys_stats.gpu_count,
               sys_stats.herds_per_gpu,
               sys_stats.kangs_per_herd,
               total_dp_buf,
               (unsigned long long)sys_stats.total_local_colls);

        for (int i = 0; i < sys_stats.gpu_count; i++) {
            const HerdGPUStats& hs = sys_stats.herd_gpu[i];
            char dp_buf[32];
            fmt_k(hs.total_dps, dp_buf, sizeof(dp_buf));
            printf("  " GM_BWHITE "GPU%d:" GM_RESET
                   "  DPs: " GM_CYAN "%s" GM_RESET "\n",
                   i, dp_buf);
        }
    }  // end herds_active block

    printf("\n");
    printf(GM_BWHITE "================================================================\n" GM_RESET);
    printf(GM_BWHITE "  GPU Performance Monitor\n"                                        GM_RESET);
    printf(GM_BWHITE "================================================================\n" GM_RESET);

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        GPUStats* s = &gpu_stats[i];
        bool has_nvml = (nvml_devices[i] != nullptr);
        printf(GM_BWHITE "GPU %d:" GM_RESET, i);
        printf("  " GM_CYAN "%.2f GK/s" GM_RESET, s->speed_mkeys / 1000.0);
        if (has_nvml) {
            printf("  |  %s%3u\xc2\xb0""C" GM_RESET, temp_color(s->temp_c), s->temp_c);
            printf("  |  " GM_WHITE "%3uW" GM_RESET, s->power_mw / 1000);
            printf("  |  " GM_WHITE "%3u%% util" GM_RESET, s->util_pct);
            printf("  |  " GM_WHITE "PCI %d" GM_RESET, s->pci_bus);
            if (s->throttling)
                printf("  " GM_BRED "THROTTLING" GM_RESET);
            else if (s->temp_c >= (unsigned)temp_warning)
                printf("  " GM_BYELLOW "WARM" GM_RESET);
        } else {
            printf("  |  " GM_WHITE "temp/power: N/A (NVML unavailable)" GM_RESET);
        }
        printf("\n");
        printf("         " GM_MAGENTA "seed: 0x%016llX" GM_RESET "\n",
               (unsigned long long)s->seed);
    }

    printf("\n");
    printf(GM_BWHITE "CPU:  " GM_RESET GM_CYAN "%.1f MK/s" GM_RESET "\n",
           sys_stats.cpu_speed_mkeys);
    printf(GM_BWHITE "Total:" GM_RESET "  "
           GM_CYAN "%.2f GK/s" GM_RESET
           "  |  Avg Temp: %s%u\xc2\xb0""C" GM_RESET
           "  |  Power: " GM_WHITE "%uW" GM_RESET "\n",
           sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0,
           temp_color(sys_stats.avg_temp_c), sys_stats.avg_temp_c,
           sys_stats.total_power_w);

    // K-Factor  +  P(solve)
    kf = sys_stats.current_k_factor;
    solve_prob = (kf > 0.0) ? (1.0 - exp(-kf)) * 100.0 : 0.0;
    printf("\n" GM_BWHITE "K-Factor:" GM_RESET "  %s%.3f" GM_RESET "  ",
           kfac_color(kf), kf);
    if      (kf < 1.0)  printf(GM_BGREEN  "OK (ahead of schedule)" GM_RESET);
    else if (kf < 1.15) printf(GM_BGREEN  "OK (on track)"          GM_RESET);
    else if (kf < 1.3)  printf(GM_BYELLOW "SLOW (slightly slow)"   GM_RESET);
    else                printf(GM_BRED    "BAD (check for issues)"  GM_RESET);
    printf("  |  " GM_BWHITE "P(solve):" GM_RESET "  " GM_BMAGENTA "%.1f%%" GM_RESET "\n",
           solve_prob);

    // ── K-factor sparkline ───────────────────────────────────────────────────
    if (g_spark_cnt > 0) {
        double sk_min = 1e18, sk_max = 0.0;
        int    total  = (g_spark_cnt < SPARK_LEN) ? g_spark_cnt : SPARK_LEN;
        for (int i = 0; i < total; i++) {
            double v = g_spark[i];
            if (v > 0.0 && v < sk_min) sk_min = v;
            if (v > sk_max)            sk_max = v;
        }
        if (sk_max <= sk_min) sk_max = sk_min + 0.01;
        static const char* bars[] = {
            "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
            "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"
        };
        printf(GM_BWHITE "K trend:" GM_RESET "  ");
        for (int i = 0; i < total; i++) {
            int ridx = (g_spark_head - total + i + SPARK_LEN) % SPARK_LEN;
            double v = g_spark[ridx];
            if (v <= 0.0) { printf(" "); continue; }
            int bar = (int)((v - sk_min) / (sk_max - sk_min) * 7.0 + 0.5);
            bar = (bar < 0) ? 0 : (bar > 7) ? 7 : bar;
            const char* sc = (v < 1.0) ? GM_BGREEN : (v < 1.15) ? GM_BYELLOW : GM_BRED;
            printf("%s%s" GM_RESET, sc, bars[bar]);
        }
        printf("  " GM_WHITE "%.2f\xe2\x80\x93%.2f" GM_RESET "\n", sk_min, sk_max);
    }
    // ────────────────────────────────────────────────────────────────────────

    // DPs / Rate / Buffer
    dp_pct  = (sys_stats.dp_expected > 0)
        ? 100.0 * (double)sys_stats.dp_count / (double)sys_stats.dp_expected : 0.0;
    buf_pct = (sys_stats.dp_buffer_total > 0)
        ? 100.0 * sys_stats.dp_buffer_used / (double)sys_stats.dp_buffer_total : 0.0;
    printf(GM_BWHITE "DPs:" GM_RESET "  %llu / %llu "
           GM_YELLOW "(%.1f%%)" GM_RESET
           "  |  " GM_BWHITE "Rate:" GM_RESET "  %s%.0f/s" GM_RESET
           "  |  " GM_BWHITE "Buffer:" GM_RESET "  %u / %u (%.1f%%)\n",
           (unsigned long long)sys_stats.dp_count,
           (unsigned long long)sys_stats.dp_expected,
           dp_pct,
           rate_color(sys_stats.dp_rate_per_sec), sys_stats.dp_rate_per_sec,
           sys_stats.dp_buffer_used, sys_stats.dp_buffer_total, buf_pct);

    // ETA
    if (sys_stats.total_gpu_speed > 0 && sys_stats.expected_ops > 1 && sys_stats.actual_ops > 1) {
        double total_spd_gks = sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0;
        double remaining     = sys_stats.expected_ops - sys_stats.actual_ops;
        if (remaining < 0) remaining = 0;
        u64 eta_s = (u64)(remaining / (total_spd_gks * 1e9));
        u64 eta_d = eta_s / 86400;
        int eta_h = (int)((eta_s % 86400) / 3600);
        int eta_m = (int)((eta_s % 3600) / 60);
        int eta_sc = (int)(eta_s % 60);
        printf(GM_BWHITE "ETA:" GM_RESET "  " GM_CYAN "%llud %02dh %02dm:%02ds" GM_RESET
               "  |  Ops: 2^%.2f / 2^%.2f\n",
               eta_d, eta_h, eta_m, eta_sc,
               log2(sys_stats.actual_ops),
               log2(sys_stats.expected_ops));
    }

    // Cost tracker
    {
        double gpu_w     = (double)sys_stats.total_power_w;
        double total_w   = gpu_w + g_sys_overhead_w;
        double cost_hr   = total_w / 1000.0 * g_kwh_rate;
        double cost_day  = cost_hr * 24.0;
        double cost_yr   = cost_day * 365.0;
        double elap_hr   = (double)sys_stats.elapsed_ms / 3600000.0;
        double cost_sess = cost_hr * elap_hr;
        char sess_buf[32], hr_buf[32];
        if (cost_sess < 1.0)
            snprintf(sess_buf, sizeof(sess_buf), "%.2f cents", cost_sess * 100.0);
        else
            snprintf(sess_buf, sizeof(sess_buf), "$%.4f", cost_sess);
        snprintf(hr_buf, sizeof(hr_buf), "$%.4f", cost_hr);
        printf("\n" GM_BWHITE "\xe2\x9a\xa1 Cost:" GM_RESET
               "  " GM_WHITE "%.0fW" GM_RESET " gpu"
               "  +  " GM_WHITE "%.0fW" GM_RESET " sys"
               "  =  " GM_BYELLOW "%.0fW" GM_RESET " total"
               "  |  " GM_BYELLOW "%s/hr" GM_RESET
               "  |  Session: " GM_BGREEN "%s" GM_RESET
               "  |  Daily: " GM_CYAN "$%.2f" GM_RESET
               "  |  Annual: " GM_WHITE "$%.0f" GM_RESET "\n",
               gpu_w, g_sys_overhead_w, total_w,
               hr_buf, sess_buf, cost_day, cost_yr);
    }

    printf(GM_BWHITE "================================================================\n" GM_RESET);
}

void GpuMonitor::PrintCompactStats() {
    printf("GPUs: ");
    for (int i = 0; i < sys_stats.gpu_count; i++) {
        printf("%d:%.1fGK/s,%u\xc2\xb0""C ", i, gpu_stats[i].speed_mkeys / 1000.0, gpu_stats[i].temp_c);
        if (gpu_stats[i].throttling) printf("\xe2\x9a\xa0\xef\xb8\x8f ");
    }
    printf("\xe2\x94\x82 Total: %.2f GK/s \xe2\x94\x82 K: %.3f",
           sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0,
           sys_stats.current_k_factor);
}

double GpuMonitor::CalculateMovingAverage(int gpu_id, int samples) {
    if (gpu_id < 0 || gpu_id >= sys_stats.gpu_count) return 0.0;
    if (samples > 60) samples = 60;
    double sum = 0.0; int count = 0;
    int start = performance_sample_idx - 1;
    for (int i = 0; i < samples; i++) {
        int sidx = (start - i + 60) % 60;
        double v = gpu_performance_history[gpu_id][sidx];
        if (v > 0.0) { sum += v; count++; }
    }
    return (count > 0) ? sum / count : 0.0;
}
