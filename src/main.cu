#include "collatz/collatz_kernel.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cuda_runtime.h>

// ── CUDA error checking ────────────────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " — " << cudaGetErrorString(err_) << "\n";           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

/// Check for errors from the last kernel launch.
#define CUDA_CHECK_LAST()                                                     \
    do {                                                                      \
        cudaError_t err_ = cudaGetLastError();                                \
        if (err_ != cudaSuccess) {                                            \
            std::cerr << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err_) << "\n";           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ── Config defaults ───────────────────────────────────────────────
static constexpr uint64_t kDefaultStart           = 1;
static constexpr uint64_t kDefaultEnd             = 0;        // 0 means "run forever"
static constexpr uint32_t kDefaultBatchSize       = 1u << 20;  // ~1 million
static constexpr uint32_t kDefaultMinChainLength   = 0;        // 0 means adaptive
static constexpr uint32_t kDefaultMaxSteps         = 1'000'000;
static constexpr const char* kDefaultOutput        = "collatz.csv";
static constexpr const char* kDefaultDivergent     = "collatz_divergent.csv";
static constexpr const char* kDefaultCheckpoint    = "collatz.ckpt";

// ── Graceful shutdown ──────────────────────────────────────────────
static std::atomic<bool> gShutdownRequested{false};

static void SignalHandler(int /*sig*/) {
    gShutdownRequested.store(true, std::memory_order_relaxed);
}

// ── CLI config ─────────────────────────────────────────────────────
struct Config {
    BigUint<>   start{kDefaultStart};
    BigUint<>   endVal{kDefaultEnd};           // 0 means "run forever"
    uint32_t    batchSize   = kDefaultBatchSize;
    uint32_t    minChain    = kDefaultMinChainLength;
    uint32_t    maxSteps    = kDefaultMaxSteps;
    std::string output     = kDefaultOutput;
    std::string divergent  = kDefaultDivergent;
    std::string checkpoint = kDefaultCheckpoint;
    bool        resume     = false;
    bool        fresh      = false;
    bool        oddOnly    = true;  // skip even numbers by default
    bool        help       = false;
};

// ── Device query ───────────────────────────────────────────────────
static void PrintDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "ERROR: no CUDA-capable devices found\n";
        std::exit(1);
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cerr << "GPU: " << props.name
              << " (SM " << props.major << "." << props.minor << ", "
              << props.multiProcessorCount << " SMs, "
              << (props.totalGlobalMem >> 20) << " MiB)\n";
}

// ── Checkpoint ─────────────────────────────────────────────────────
// Format: plain text, one field per line:
//   nextStart_limb0 nextStart_limb1 ... nextStart_limbN
//   batchSize longestChain
// Written atomically via tmp + rename.

static bool WriteCheckpoint(const std::string& Path,
                             const BigUint<>& NextStart,
                             uint32_t BatchSize,
                             uint32_t LongestChain) {
    std::string tmp = Path + ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        std::cerr << "WARNING: cannot write checkpoint " << tmp << "\n";
        return false;
    }
    for (int i = 0; i < COLLATZ_N_LIMBS; ++i) {
        if (i) out << ' ';
        out << NextStart.limbs[i];
    }
    out << '\n' << BatchSize << ' ' << LongestChain << '\n';
    out.close();
    if (std::rename(tmp.c_str(), Path.c_str()) != 0) {
        std::cerr << "WARNING: rename failed for checkpoint\n";
        return false;
    }
    return true;
}

static bool ReadCheckpoint(const std::string& Path,
                            BigUint<>& NextStart,
                            uint32_t& BatchSize,
                            uint32_t& LongestChain) {
    std::ifstream in(Path);
    if (!in) return false;
    for (int i = 0; i < COLLATZ_N_LIMBS; ++i) {
        if (!(in >> NextStart.limbs[i])) return false;
    }
    if (!(in >> BatchSize)) return false;
    // LongestChain is optional (backwards compat with old checkpoints)
    if (!(in >> LongestChain)) LongestChain = 0;
    return true;
}

// ── Argument parsing ───────────────────────────────────────────────

static void PrintUsage(const char* Prog) {
    std::cerr
        << "Usage: " << Prog << " [OPTIONS]\n"
        << "\n"
        << "  --start N           First number to test (default " << kDefaultStart << ")\n"
        << "                      Accepts decimal, 0x hex, or 2^N notation\n"
        << "  --end N             Last number to test (0 = infinite)\n"
        << "  --batch-size N      Numbers per GPU batch (default " << kDefaultBatchSize << ")\n"
        << "  --min-chain N       Minimum chain length to log (0 = adaptive, default)\n"
        << "  --max-steps N       Max steps before flagging as non-converging (default " << kDefaultMaxSteps << ")\n"
        << "  --output FILE       CSV output file (default " << kDefaultOutput << ")\n"
        << "  --divergent FILE    File for non-converging chains (default " << kDefaultDivergent << ")\n"
        << "  --checkpoint FILE   Checkpoint file (default " << kDefaultCheckpoint << ")\n"
        << "  --resume            Resume from checkpoint\n"
        << "  --fresh             Discard existing checkpoint and start fresh\n"
        << "  --no-odd-only       Test all numbers (default: skip even numbers)\n"
        << "  --help              Show this help\n"
        << "\n"
        << "Integers can exceed 2^64. Uses " << (COLLATZ_N_LIMBS * 64)
        << "-bit big integers (" << COLLATZ_N_LIMBS << " limbs).\n"
        << "Press Ctrl-C for a graceful stop (writes checkpoint).\n";
}

/// Parse a decimal string into a BigUint.
static BigUint<> ParseDecimal(const char* S) {
    BigUint<> result;
    for (const char* p = S; *p; ++p) {
        if (*p < '0' || *p > '9') {
            std::cerr << "ERROR: invalid digit '" << *p << "' in number\n";
            std::exit(1);
        }
        // result = result * 10 + digit
        BigUint<> r8 = result;
        r8.ShiftLeft1(); r8.ShiftLeft1(); r8.ShiftLeft1(); // *8
        BigUint<> r2 = result;
        r2.ShiftLeft1(); // *2
        result = r8;
        result.Add(r2);    // *10
        result.AddU64(static_cast<uint64_t>(*p - '0'));
    }
    return result;
}

/// Parse a hex string (without 0x prefix) into a BigUint.
static BigUint<> ParseHex(const char* S) {
    BigUint<> result;
    for (const char* p = S; *p; ++p) {
        // result <<= 4
        result.ShiftLeftN(4);
        uint64_t nib;
        if (*p >= '0' && *p <= '9')      nib = *p - '0';
        else if (*p >= 'a' && *p <= 'f') nib = *p - 'a' + 10;
        else if (*p >= 'A' && *p <= 'F') nib = *p - 'A' + 10;
        else {
            std::cerr << "ERROR: invalid hex digit '" << *p << "'\n";
            std::exit(1);
        }
        result.AddU64(nib);
    }
    return result;
}

/// Parse a number string. Accepts:
///   decimal:  "12345"
///   hex:      "0x1a2b" or "0X1A2B"
///   power:    "2^68"
static BigUint<> ParseBigUint(const char* S) {
    // 2^N notation
    const char* caret = std::strchr(S, '^');
    if (caret && caret > S && (caret[-1] == '2' || (caret == S + 1 && S[0] == '2'))) {
        // Verify everything before ^ is just "2"
        std::string baseStr(S, caret);
        if (baseStr != "2") {
            std::cerr << "ERROR: only 2^N power notation is supported\n";
            std::exit(1);
        }
        char* endp = nullptr;
        unsigned long exp = std::strtoul(caret + 1, &endp, 10);
        if (endp == caret + 1 || *endp != '\0') {
            std::cerr << "ERROR: invalid exponent in 2^N notation\n";
            std::exit(1);
        }
        if (exp >= static_cast<unsigned long>(COLLATZ_N_LIMBS) * 64) {
            std::cerr << "ERROR: 2^" << exp << " exceeds "
                      << COLLATZ_N_LIMBS * 64 << "-bit capacity\n";
            std::exit(1);
        }
        // O(1) construction via SetBit
        BigUint<> result;
        result.SetBit(static_cast<int>(exp));
        return result;
    }
    // 0x hex prefix
    if (S[0] == '0' && (S[1] == 'x' || S[1] == 'X'))
        return ParseHex(S + 2);
    // Plain decimal
    return ParseDecimal(S);
}

/// Parse a uint32_t from string with validation.
static uint32_t ParseUint32(const char* S, const char* FlagName) {
    char* endp = nullptr;
    unsigned long val = std::strtoul(S, &endp, 10);
    if (endp == S || *endp != '\0') {
        std::cerr << "ERROR: invalid number for " << FlagName << ": " << S << "\n";
        std::exit(1);
    }
    if (val > UINT32_MAX) {
        std::cerr << "ERROR: value too large for " << FlagName << ": " << S << "\n";
        std::exit(1);
    }
    return static_cast<uint32_t>(val);
}

static Config ParseArgs(int Argc, char** Argv) {
    Config cfg;
    for (int i = 1; i < Argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(Argv[i], name) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 >= Argc) {
                std::cerr << "ERROR: missing value for " << Argv[i] << "\n";
                std::exit(1);
            }
            return Argv[++i];
        };

        if (arg("--start"))          cfg.start      = ParseBigUint(next());
        else if (arg("--end"))       cfg.endVal    = ParseBigUint(next());
        else if (arg("--batch-size"))cfg.batchSize  = ParseUint32(next(), "--batch-size");
        else if (arg("--min-chain")) cfg.minChain   = ParseUint32(next(), "--min-chain");
        else if (arg("--max-steps")) cfg.maxSteps   = ParseUint32(next(), "--max-steps");
        else if (arg("--output"))    cfg.output      = next();
        else if (arg("--divergent")) cfg.divergent   = next();
        else if (arg("--checkpoint"))cfg.checkpoint  = next();
        else if (arg("--resume"))    cfg.resume      = true;
        else if (arg("--fresh"))     cfg.fresh       = true;
        else if (arg("--no-odd-only")) cfg.oddOnly   = false;
        else if (arg("--help"))      cfg.help        = true;
        else {
            std::cerr << "ERROR: unknown option " << Argv[i] << "\n";
            std::exit(1);
        }
    }

    // ── Validation ──
    if (cfg.resume && cfg.fresh) {
        std::cerr << "ERROR: --resume and --fresh are mutually exclusive\n";
        std::exit(1);
    }
    if (cfg.batchSize == 0) {
        std::cerr << "ERROR: --batch-size must be > 0\n";
        std::exit(1);
    }
    if (cfg.start.IsZero()) {
        std::cerr << "ERROR: --start must be >= 1\n";
        std::exit(1);
    }

    return cfg;
}

// ── CSV scanning ───────────────────────────────────────────────────

/// Scan existing CSV to find the longest chain length recorded.
/// Expects header "start_n,chain_length,max_value" and data rows.
static uint32_t ScanLongestChain(const std::string& Path) {
    std::ifstream in(Path);
    if (!in) return 0;

    std::string line;
    // Skip header
    if (!std::getline(in, line)) return 0;

    uint32_t longest = 0;
    while (std::getline(in, line)) {
        // Find the first comma (after start_n)
        auto pos1 = line.find(',');
        if (pos1 == std::string::npos) continue;
        // Find the second comma (after chain_length)
        auto pos2 = line.find(',', pos1 + 1);
        if (pos2 == std::string::npos) continue;
        // Extract chain_length between the two commas
        std::string chainStr = line.substr(pos1 + 1, pos2 - pos1 - 1);
        unsigned long val = std::strtoul(chainStr.c_str(), nullptr, 10);
        if (val > longest) longest = static_cast<uint32_t>(val);
    }
    return longest;
}

// ── Main loop ──────────────────────────────────────────────────────

/// Recompute a full Collatz chain on the CPU for a single starting number.
/// Used to recover maxValue/lastValue for the rare chains worth logging.
template <int N_LIMBS = COLLATZ_N_LIMBS>
static CollatzResult<N_LIMBS> RecomputeChain(const BigUint<N_LIMBS>& StartN,
                                              uint32_t MaxSteps) {
    CollatzResult<N_LIMBS> res;
    res.start = StartN;
    BigUint<N_LIMBS> n = StartN;
    res.maxValue = n;
    res.chainLength = 0;
    res.overflow = false;
    res.exceededLimit = false;

    while (!n.IsOne()) {
        if (MaxSteps > 0 && res.chainLength >= MaxSteps) {
            res.exceededLimit = true;
            break;
        }
        if (n.IsEven()) {
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            res.chainLength += tz;
        } else {
            if (n.TriplePlusOne()) {
                res.overflow = true;
                break;
            }
            if (n > res.maxValue) res.maxValue = n;
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            res.chainLength += 1 + tz;
        }
    }
    res.lastValue = n;
    return res;
}

int main(int argc, char** argv) {
    Config cfg = ParseArgs(argc, argv);
    if (cfg.help) {
        PrintUsage(argv[0]);
        return 0;
    }

    // Install signal handlers for graceful shutdown
    std::signal(SIGINT,  SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // Print GPU info
    PrintDeviceInfo();

    // Safety check: refuse to start if a checkpoint exists but --resume not given
    if (!cfg.resume) {
        std::ifstream ckptTest(cfg.checkpoint);
        if (ckptTest.good()) {
            ckptTest.close();
            if (cfg.fresh) {
                // User explicitly wants to start fresh — remove old checkpoint
                std::remove(cfg.checkpoint.c_str());
                std::cerr << "Discarded existing checkpoint (--fresh).\n";
            } else {
                std::cerr << "ERROR: checkpoint file '" << cfg.checkpoint
                          << "' exists from a previous run.\n"
                          << "  Use --resume  to continue from where you left off\n"
                          << "  Use --fresh   to discard the checkpoint and start over\n";
                return 1;
            }
        }
    }

    // Resume from checkpoint if requested
    uint32_t longestChain = 0;
    if (cfg.resume) {
        BigUint<> ckptStart;
        uint32_t ckptBatch;
        uint32_t ckptLongest = 0;
        if (ReadCheckpoint(cfg.checkpoint, ckptStart, ckptBatch, ckptLongest)) {
            cfg.start = ckptStart;
            longestChain = ckptLongest;
            std::cerr << "Resumed from checkpoint: start = "
                      << ckptStart.ToPowerString()
                      << ", longest = " << ckptLongest << "\n";
        } else {
            std::cerr << "WARNING: no checkpoint found, starting from --start\n";
        }
    }

    // If checkpoint didn't have longest chain (old format), scan CSV
    if (cfg.resume && longestChain == 0) {
        longestChain = ScanLongestChain(cfg.output);
        if (longestChain > 0) {
            std::cerr << "Longest chain in existing CSV: " << longestChain << "\n";
        }
    }

    // Check for existing non-converging chains
    uint64_t existingDivergent = 0;
    if (cfg.maxSteps > 0) {
        std::ifstream divCheck(cfg.divergent);
        if (divCheck.good()) {
            std::string line;
            if (std::getline(divCheck, line)) {  // skip header
                while (std::getline(divCheck, line)) ++existingDivergent;
            }
            if (existingDivergent > 0) {
                std::cerr << "WARNING: " << existingDivergent
                          << " non-converging chain(s) in " << cfg.divergent << "\n";
            }
        }
    }

    // Ensure start is odd when using odd-only mode
    if (cfg.oddOnly && cfg.start.IsEven() && !cfg.start.IsZero()) {
        cfg.start.AddU64(1);
        std::cerr << "Adjusted start to odd number: " << cfg.start.ToPowerString() << "\n";
    }

    // Open CSV for appending
    std::ofstream csv(cfg.output, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << cfg.output << " for writing\n";
        return 1;
    }

    // Write header if file is empty
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) {
        csv << "start_n,chain_length,max_value\n";
    }

    // Open divergent/non-converging file (only if maxSteps is active)
    std::ofstream divCsv;
    if (cfg.maxSteps > 0) {
        divCsv.open(cfg.divergent, std::ios::app);
        if (!divCsv) {
            std::cerr << "ERROR: cannot open " << cfg.divergent << " for writing\n";
            return 1;
        }
        divCsv.seekp(0, std::ios::end);
        if (divCsv.tellp() == 0) {
            divCsv << "start_n,steps_completed,current_value,max_value\n";
        }
    }

    // ── GPU pipeline ──
    const uint32_t batchSize = cfg.batchSize;
    const bool adaptive = (cfg.minChain == 0);

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // ── Adaptive mode buffers (GPU-filtered, minimal D→H) ──
    static constexpr uint32_t kMaxHitsPerBatch = 4096;

    uint32_t*    dHitCount[2] = {};
    AdaptiveHit* dHits[2]     = {};
    uint32_t     hHitCount[2] = {};
    AdaptiveHit* hHits[2]     = {};

    // ── Compact mode buffers (also used as bootstrap for adaptive when longestChain == 0) ──
    const size_t compactBytes = batchSize * sizeof(CompactResult);
    CompactResult* dResults[2] = {};
    CompactResult* hResults[2] = {};

    // Always allocate compact buffers (they're small: ~8 bytes × batchSize)
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dResults[b], compactBytes));
        CUDA_CHECK(cudaMallocHost(&hResults[b], compactBytes));
    }

    if (adaptive) {
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dHitCount[b], sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&dHits[b], kMaxHitsPerBatch * sizeof(AdaptiveHit)));
            CUDA_CHECK(cudaMallocHost(&hHits[b], kMaxHitsPerBatch * sizeof(AdaptiveHit)));
        }
    }

    BigUint<> current = cfg.start;
    bool hasEnd = !cfg.endVal.IsZero();

    uint64_t totalProcessed = 0;
    uint64_t totalDivergent = existingDivergent;
    auto wallStart = std::chrono::steady_clock::now();
    auto lastReport = wallStart;

    // In odd-only mode, each thread covers 2 numbers, so the stride per batch
    // is batchSize * 2 in the number space but we process batchSize results.
    const uint32_t stride = cfg.oddOnly ? 2 : 1;

    // Determine the actual count for this batch (may be less at the end)
    auto ComputeBatchCount = [&](const BigUint<>& BatchStart) -> uint32_t {
        if (!hasEnd) return batchSize;
        if (cfg.endVal < BatchStart) return 0;
        BigUint<> limit = BatchStart;
        limit.AddU64(static_cast<uint64_t>(batchSize) * stride);
        if (limit < cfg.endVal || limit == cfg.endVal) {
            return batchSize;
        }
        BigUint<> probe = BatchStart;
        for (uint32_t c = 0; c < batchSize; ++c) {
            if (cfg.endVal < probe) return c;
            probe.AddU64(stride);
        }
        return batchSize;
    };

    // Helper: launch kernel for the given buffer slot
    // In adaptive mode, fall back to compact kernel when longestChain == 0
    // (first batch has no threshold so adaptive would emit everything).
    auto UseAdaptive = [&]() { return adaptive && longestChain > 0; };

    auto LaunchBatch = [&](const BigUint<>& Start, uint32_t Count, int Buf) {
        if (UseAdaptive()) {
            CUDA_CHECK(cudaMemsetAsync(dHitCount[Buf], 0, sizeof(uint32_t), streams[Buf]));
            LaunchAdaptiveCollatzKernel<>(Start, Count, dHitCount[Buf], dHits[Buf],
                                         longestChain, kMaxHitsPerBatch,
                                         cfg.maxSteps, streams[Buf], cfg.oddOnly);
        } else {
            LaunchCompactCollatzKernel<>(Start, Count, dResults[Buf],
                                        cfg.maxSteps, streams[Buf], cfg.oddOnly);
        }
        CUDA_CHECK_LAST();
    };

    int buf = 0;
    bool bufUsedAdaptive[2] = {false, false};

    // Launch first batch
    uint32_t count = ComputeBatchCount(current);
    if (count == 0) {
        std::cerr << "Nothing to compute (start > end).\n";
        goto cleanup;
    }
    bufUsedAdaptive[buf] = UseAdaptive();
    LaunchBatch(current, count, buf);

    while (!gShutdownRequested.load(std::memory_order_relaxed)) {
        BigUint<> prevStart = current;
        uint32_t prevCount = count;
        int prevBuf = buf;

        current.AddU64(static_cast<uint64_t>(count) * stride);
        buf ^= 1;

        count = ComputeBatchCount(current);
        bool moreWork = (count > 0);

        if (moreWork) {
            bufUsedAdaptive[buf] = UseAdaptive();
            LaunchBatch(current, count, buf);
        }

        if (bufUsedAdaptive[prevBuf]) {
            // Transfer only the hit count (4 bytes)
            CUDA_CHECK(cudaMemcpyAsync(&hHitCount[prevBuf], dHitCount[prevBuf],
                            sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[prevBuf]));
            CUDA_CHECK(cudaStreamSynchronize(streams[prevBuf]));

            uint32_t nHits = hHitCount[prevBuf];
            if (nHits > kMaxHitsPerBatch) nHits = kMaxHitsPerBatch;

            if (nHits > 0) {
                CUDA_CHECK(cudaMemcpy(hHits[prevBuf], dHits[prevBuf],
                                nHits * sizeof(AdaptiveHit), cudaMemcpyDeviceToHost));

                for (uint32_t h = 0; h < nHits; ++h) {
                    const auto& hit = hHits[prevBuf][h];
                    BigUint<> startN = prevStart;
                    startN.AddU64(static_cast<uint64_t>(hit.index) * stride);

                    if (hit.flags & CompactResult::kOverflow) {
                        std::cerr << "\nOVERFLOW at n=" << startN.ToString()
                                  << " — increase limbs (currently "
                                  << COLLATZ_N_LIMBS * 64 << " bits)\n";
                        continue;
                    }
                    if (hit.flags & CompactResult::kExceededLimit) {
                        ++totalDivergent;
                        auto full = RecomputeChain(startN, cfg.maxSteps);
                        divCsv << full.start.ToString() << ','
                               << full.chainLength << ','
                               << full.lastValue.ToString() << ','
                               << full.maxValue.ToString() << '\n';
                        continue;
                    }
                    if (hit.chainLength > longestChain) {
                        longestChain = hit.chainLength;
                    }
                    auto full = RecomputeChain(startN, cfg.maxSteps);
                    csv << full.start.ToString() << ','
                        << full.chainLength << ','
                        << full.maxValue.ToString() << '\n';
                }
            }
        } else {
            // Compact mode: transfer and scan all results
            CUDA_CHECK(cudaMemcpyAsync(hResults[prevBuf], dResults[prevBuf],
                            prevCount * sizeof(CompactResult),
                            cudaMemcpyDeviceToHost, streams[prevBuf]));
            CUDA_CHECK(cudaStreamSynchronize(streams[prevBuf]));

            for (uint32_t i = 0; i < prevCount; ++i) {
                const auto& cr = hResults[prevBuf][i];
                if (cr.flags & CompactResult::kOverflow) {
                    BigUint<> startN = prevStart;
                    startN.AddU64(static_cast<uint64_t>(i) * stride);
                    std::cerr << "\nOVERFLOW at n=" << startN.ToString()
                              << " — increase limbs (currently "
                              << COLLATZ_N_LIMBS * 64 << " bits)\n";
                    continue;
                }
                if (cr.flags & CompactResult::kExceededLimit) {
                    ++totalDivergent;
                    BigUint<> startN = prevStart;
                    startN.AddU64(static_cast<uint64_t>(i) * stride);
                    auto full = RecomputeChain(startN, cfg.maxSteps);
                    divCsv << full.start.ToString() << ','
                           << full.chainLength << ','
                           << full.lastValue.ToString() << ','
                           << full.maxValue.ToString() << '\n';
                    continue;
                }
                bool isNewLongest = cr.chainLength > longestChain;
                if (isNewLongest) {
                    longestChain = cr.chainLength;
                }
                // In adaptive mode (compact bootstrap), only log new longest.
                // In fixed threshold mode, log everything >= minChain.
                if (adaptive ? isNewLongest : cr.chainLength >= cfg.minChain) {
                    BigUint<> startN = prevStart;
                    startN.AddU64(static_cast<uint64_t>(i) * stride);
                    auto full = RecomputeChain(startN, cfg.maxSteps);
                    csv << full.start.ToString() << ','
                        << full.chainLength << ','
                        << full.maxValue.ToString() << '\n';
                }
            }
        }
        csv.flush();
        if (divCsv.is_open()) divCsv.flush();

        totalProcessed += prevCount;

        auto now = std::chrono::steady_clock::now();
        double elapsedSinceReport =
            std::chrono::duration<double>(now - lastReport).count();
        if (elapsedSinceReport >= 5.0) {
            double totalElapsed =
                std::chrono::duration<double>(now - wallStart).count();
            double rate = totalProcessed / totalElapsed;
            // Format rate with SI suffix (K/M/B)
            auto FormatRate = [](double r) -> std::string {
                char buf[32];
                if (r >= 1e9)      std::snprintf(buf, sizeof(buf), "%.1fB", r / 1e9);
                else if (r >= 1e6) std::snprintf(buf, sizeof(buf), "%.1fM", r / 1e6);
                else if (r >= 1e3) std::snprintf(buf, sizeof(buf), "%.1fK", r / 1e3);
                else               std::snprintf(buf, sizeof(buf), "%.0f", r);
                return buf;
            };
            std::ostringstream status;
            status << "\r"
                   << "n/s:" << FormatRate(rate)
                   << (cfg.oddOnly ? " (odd)" : "")
                   << " longest: " << longestChain;
            status << " current: " << current.ToProgressString();
            if (totalDivergent > 0)
                status << " divergent: " << totalDivergent;
            std::string line = status.str();
            if (line.size() < 80) line.resize(80, ' ');
            std::cerr << line << std::flush;
            lastReport = now;
        }

        WriteCheckpoint(cfg.checkpoint, moreWork ? current : prevStart,
                         batchSize, longestChain);

        if (!moreWork) break;
    }

    if (gShutdownRequested.load(std::memory_order_relaxed)) {
        std::cerr << "\nShutting down gracefully...\n";
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        WriteCheckpoint(cfg.checkpoint, current, batchSize, longestChain);
        std::cerr << "Checkpoint written at " << current.ToPowerString() << "\n";
    }

cleanup:
    csv.close();
    if (divCsv.is_open()) divCsv.close();
    for (int b = 0; b < 2; ++b) {
        if (hResults[b]) CUDA_CHECK(cudaFreeHost(hResults[b]));
        if (dResults[b]) CUDA_CHECK(cudaFree(dResults[b]));
    }
    if (adaptive) {
        for (int b = 0; b < 2; ++b) {
            if (hHits[b])     CUDA_CHECK(cudaFreeHost(hHits[b]));
            if (dHitCount[b]) CUDA_CHECK(cudaFree(dHitCount[b]));
            if (dHits[b])     CUDA_CHECK(cudaFree(dHits[b]));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

    double totalElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStart).count();
    std::cerr << "\r" << std::string(80, ' ') << "\r";
    std::cerr << "Done. " << totalProcessed << " numbers processed in "
              << std::fixed << std::setprecision(2) << totalElapsed << "s\n";
    return 0;
}
