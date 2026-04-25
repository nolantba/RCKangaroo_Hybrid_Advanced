// IMPROVED LISSAJOUS JUMP GENERATOR FOR PUZZLE 135
// Key improvements: proper sampling, scaled ranges, variance injection
#ifndef LISSAJOUS_JUMP_GENERATOR_V2_HPP
#define LISSAJOUS_JUMP_GENERATOR_V2_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <random>

#ifndef GOLDEN_ANGLE
#define GOLDEN_ANGLE 2.39996322972865332 // 2π(1 - 1/φ)
#endif

class LissajousJumpGenerator {
public:
    struct Config {
        double freq_x{1.0};
        double freq_y{1.6180339887498948482}; // φ (Golden Ratio)
        double freq_z{2.5066282746310005024}; // sqrt(2*pi)
        
        double phase_x{0.0};
        double phase_y{M_PI / 2.0};
        double phase_z{M_PI / 4.0};
        
        struct Range {
            uint64_t min;
            uint64_t max;
        };
        
        // Default ranges — adaptive config in RCKangaroo.cpp overrides these at runtime
        Range mean_range{1ULL << 33, 1ULL << 34};           // base: scaled by lissa_shift
        Range std_dev_range{1ULL << 31, 1ULL << 32};        // ~25% variance
        Range skew_range{0, 1000};                          // Small skew for variety
        
        size_t table_size{1000000};
        
        // NEW: Distribution type
        enum class DistType {
            DETERMINISTIC,  // Just use mean (your current behavior)
            NORMAL,         // Gaussian distribution
            LOG_NORMAL,     // Log-normal (better for multiplicative processes)
            UNIFORM         // Uniform around mean
        };
        DistType dist_type{DistType::LOG_NORMAL};

        bool validate() const {
            return (table_size > 0) && (freq_x != 0.0) && (freq_y != 0.0) && (freq_z != 0.0) &&
                   (mean_range.min < mean_range.max) &&
                   (std_dev_range.min < std_dev_range.max) &&
                   (skew_range.min < skew_range.max);
        }

        std::string to_string() const {
            char buf[512];
            snprintf(buf, sizeof(buf),
                "Freq: [%.6f, %.6f, %.6f] Phases: [%.3f, %.3f, %.3f]\n"
                "Mean: [2^%.1f, 2^%.1f] StdDev: [2^%.1f, 2^%.1f] Skew: [%llu, %llu]\n"
                "Size: %zu (%.1f MB) Dist: %d",
                freq_x, freq_y, freq_z, phase_x, phase_y, phase_z,
                log2(mean_range.min), log2(mean_range.max),
                log2(std_dev_range.min), log2(std_dev_range.max),
                (unsigned long long)skew_range.min, (unsigned long long)skew_range.max, table_size,
                (table_size * sizeof(JumpParams)) / (1024.0 * 1024.0),
                static_cast<int>(dist_type));
            return buf;
        }
    };

    #pragma pack(push, 1)
    struct JumpParams {
        uint64_t mean;
        uint64_t std_dev;
        int64_t skew;
        
        // IMPROVED: Proper statistical sampling
        uint64_t sample(uint64_t seed = 0) const {
            // Use seed for deterministic but varied sampling
            std::mt19937_64 rng(seed ^ mean ^ std_dev);
            std::normal_distribution<double> dist(0.0, 1.0);
            
            double z = dist(rng);
            
            // Apply skew using cubic transformation
            if (skew != 0) {
                double skew_factor = skew / 1000.0;
                z = z + skew_factor * (z * z - 1.0);
            }
            
            // Convert to jump size with bounds checking
            double jump_d = static_cast<double>(mean) + z * static_cast<double>(std_dev);
            
            // Ensure positive and within reasonable bounds
            if (jump_d < 1.0) jump_d = 1.0;
            if (jump_d > static_cast<double>(mean) * 10.0) {
                jump_d = static_cast<double>(mean) * 10.0;
            }
            
            return static_cast<uint64_t>(jump_d);
        }
        
        // Deterministic variant (for GPU compatibility)
        uint64_t sample_deterministic() const {
            return mean;
        }
        
        // Fast approximation using index-based pseudo-randomness
        uint64_t sample_fast(uint64_t index) const {
            // XorShift-like mixing
            uint64_t x = index * 0x9E3779B97F4A7C15ULL;
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27;
            x *= 0x94D049BB133111EBULL;
            x ^= x >> 31;
            
            // Map to normal-like distribution using central limit theorem
            uint64_t sum = 0;
            for (int i = 0; i < 4; i++) {
                x = x * 1103515245ULL + 12345ULL;
                sum += (x >> 32);
            }
            uint64_t avg = sum / 4;
            
            // Scale to our distribution
            double ratio = static_cast<double>(avg) / static_cast<double>(UINT32_MAX);
            ratio = (ratio - 0.5) * 6.0; // Approximately 3 sigma range
            
            double jump_d = static_cast<double>(mean) + ratio * static_cast<double>(std_dev);
            if (jump_d < 1.0) jump_d = 1.0;
            
            return static_cast<uint64_t>(jump_d);
        }
    };
    #pragma pack(pop)

private:
    Config config_;
    std::vector<JumpParams> jump_table_;
    bool initialized_{false};

    double map_range(double value, double in_min, double in_max, double out_min, double out_max) const {
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }

    void generate_table() {
        jump_table_.clear();
        jump_table_.reserve(config_.table_size);
        
        for (size_t i = 0; i < config_.table_size; ++i) {
            double t = static_cast<double>(i);
            
            // Lissajous curve sampling
            double x = std::sin(config_.freq_x * t + config_.phase_x);
            double y = std::sin(config_.freq_y * t + config_.phase_y);
            double z = std::sin(config_.freq_z * t + config_.phase_z);

            JumpParams params;
            params.mean = static_cast<uint64_t>(map_range(x, -1.0, 1.0, 
                config_.mean_range.min, config_.mean_range.max));
            params.std_dev = static_cast<uint64_t>(map_range(y, -1.0, 1.0,
                config_.std_dev_range.min, config_.std_dev_range.max));
            params.skew = static_cast<int64_t>(map_range(z, -1.0, 1.0,
                config_.skew_range.min, config_.skew_range.max));

            jump_table_.push_back(params);
        }
    }

public:
    LissajousJumpGenerator(size_t size = 256) {
        Config config;
        config.table_size = 256;
        initialize(config);
    }

    explicit LissajousJumpGenerator(const Config& config) {
        initialize(config);
    }

    bool initialize(const Config& config) {
        if (!config.validate()) {
            std::cerr << "Invalid Lissajous configuration" << std::endl;
            return false;
        }
        
        config_ = config;
        generate_table();
        initialized_ = true;
        
        std::cout << "LissajousJumpGenerator initialized: " << config_.to_string() << std::endl;
        return true;
    }

    const JumpParams& get_jump_params(uint64_t index) const {
        return jump_table_[index % jump_table_.size()];
    }

    size_t get_table_size() const { return jump_table_.size(); }
    bool is_initialized() const { return initialized_; }
    const Config& config() const { return config_; }
    const JumpParams* data() const { return jump_table_.data(); }
    size_t memory_size() const { return sizeof(JumpParams) * jump_table_.size(); }

    static Config default_config() {
        return Config{};
    }

    // OPTIMIZED CONFIG FOR PUZZLE 135
    static Config optimized_config_135() {
        Config config;
        
        // Puzzle 135: range is 2^134 to 2^135-1
        // Optimal average jump ≈ 2^67.5 (sqrt of range)
        // Note: adaptive config in RCKangaroo.cpp uses safe base + ShiftLeft instead of
        // direct large literals (2^67 exceeds uint64_t range so use base 2^33, shift 34)
        config.mean_range    = {1ULL << 33, (1ULL << 33) + (1ULL << 32)};  // scaled at runtime
        config.std_dev_range = {1ULL << 31, 1ULL << 32};                    // ~25% variance
        config.skew_range = {0, 1000};                                 // Symmetric skew
        
        config.table_size = 256;  // Larger table for more variety
        config.dist_type = Config::DistType::LOG_NORMAL;
        
        // Adjust frequencies for better coverage
        config.freq_x = 1.0;
        config.freq_y = 1.6180339887498948482;  // φ
        config.freq_z = 2.7182818284590452354;  // e (instead of sqrt(2π))
        
        return config;
    }

    // Generic scaled config
    static Config optimized_config(size_t problem_bits) {
        Config config = default_config();
        
        if (problem_bits >= 130 && problem_bits <= 140) {
            return optimized_config_135();
        }
        
        // Generic scaling — cap shift at 63 to avoid uint64 overflow for large puzzles
        double target_jump_log2 = static_cast<double>(problem_bits) / 2.0;
        int safe_shift = static_cast<int>(target_jump_log2);
        if (safe_shift > 63) safe_shift = 63;
        uint64_t target_jump = 1ULL << safe_shift;
        
        config.mean_range = {target_jump / 2, target_jump * 2};
        config.std_dev_range = {target_jump / 8, target_jump / 2};
        config.skew_range = {0, 1000};
        
        config.table_size = std::min(static_cast<size_t>(10000000), 
                                     static_cast<size_t>(1000000 * (problem_bits / 64.0)));
        
        return config;
    }
    
    // Analysis helper
    void print_statistics() const {
        uint64_t sum_mean = 0;
        uint64_t sum_stddev = 0;
        uint64_t min_mean = UINT64_MAX;
        uint64_t max_mean = 0;
        
        for (const auto& params : jump_table_) {
            sum_mean += params.mean;
            sum_stddev += params.std_dev;
            if (params.mean < min_mean) min_mean = params.mean;
            if (params.mean > max_mean) max_mean = params.mean;
        }
        
        double avg_mean = static_cast<double>(sum_mean) / jump_table_.size();
        double avg_stddev = static_cast<double>(sum_stddev) / jump_table_.size();
        
        std::cout << "Table Statistics:" << std::endl;
        std::cout << "  Average mean: 2^" << log2(avg_mean) << std::endl;
        std::cout << "  Mean range: 2^" << log2(min_mean) << " to 2^" << log2(max_mean) << std::endl;
        std::cout << "  Average std_dev: 2^" << log2(avg_stddev) << std::endl;
        std::cout << "  Entries: " << jump_table_.size() << std::endl;
    }
    
    // Generate varied jump table for Kangaroo (pre-sample the distribution)
    std::vector<uint64_t> generate_sampled_jumps(uint32_t seed = 12345) const {
        std::vector<uint64_t> jumps;
        jumps.reserve(jump_table_.size());
        
        std::mt19937_64 rng(seed);
        
        for (size_t i = 0; i < jump_table_.size(); ++i) {
            // Each entry gets a varied sample based on its parameters
            uint64_t jump = jump_table_[i].sample_fast(seed ^ i ^ rng());
            jumps.push_back(jump);
        }
        
        return jumps;
    }
	
	// Sunflower (golden-angle) + Lissajous hybrid jump generator.
// Uses Vogel (sunflower) ordering to select indices from jump_table_,
// then samples each entry with your existing sample_fast().
std::vector<uint64_t> generate_sunflower_lissa_jumps(
        size_t n,
        uint64_t seed = 0x9e3779b97f4a7c15ULL,
        uint64_t perGpuSalt = 0) const
{
    if (!initialized_) return {};
    n = std::max<size_t>(1, n);

    std::vector<uint64_t> jumps;
    jumps.reserve(n);

    const double phase =
        std::fmod(double(perGpuSalt) * (GOLDEN_ANGLE/7.0), 2.0 * M_PI);

    std::mt19937_64 rng(seed ^ (perGpuSalt * 0x9E3779B185EBCA87ULL));
    std::uniform_real_distribution<double> J(-0.03, 0.03); // ±3% jitter

    const size_t T = jump_table_.size();
    for (size_t k = 0; k < n; ++k) {
        // Golden-angle point (theta currently unused but kept for future mixing)
        const double theta = phase + GOLDEN_ANGLE * double(k);
        (void)theta;

        // Area-uniform radius in [0,1]
        const double r = std::sqrt((double(k) + 0.5) / double(n));

        // Map radius -> index in [0, T-1]
        const double p = 1.0; // try 0.85..1.25 to bias toward small/large jumps
        const double scaled = std::clamp(std::pow(r, p) * double(T), 0.0, double(T - 1));
        const size_t idx = static_cast<size_t>(scaled);

        // Deterministic per-entry sampling with mixed index
        const uint64_t mix = (uint64_t)k ^ (perGpuSalt << 32) ^ seed;
        uint64_t jump = jump_table_[idx].sample_fast(mix);

        // Light jitter and clamp to your configured mean_range envelope
        if (jump > 0) {
            double jf = 1.0 + J(rng);
            uint64_t j2 = (uint64_t)std::llround(double(jump) * jf);
            const uint64_t lo = config_.mean_range.min;
            const uint64_t hi = std::max<uint64_t>(config_.mean_range.max, (uint64_t)(lo + 1ULL));
            if (j2 < lo) j2 = lo;
            if (j2 > hi) j2 = hi;
            jump = j2;
        }
        jumps.push_back(jump);
    }
    return jumps;
}

	
};

#endif // LISSAJOUS_JUMP_GENERATOR_V2_HPP