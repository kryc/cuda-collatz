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

    // ── Double-buffered GPU pipeline ──
    // Two CUDA streams so compute on batch N+1 overlaps with D→H copy of batch N.
    const uint32_t batchSize = cfg.batchSize;
    const size_t resultBytes = batchSize * sizeof(CollatzResult<>);

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    CollatzResult<>* dResults[2];
    CUDA_CHECK(cudaMalloc(&dResults[0], resultBytes));
    CUDA_CHECK(cudaMalloc(&dResults[1], resultBytes));

    CollatzResult<>* hResults[2];
    CUDA_CHECK(cudaMallocHost(&hResults[0], resultBytes));  // pinned for async copy
    CUDA_CHECK(cudaMallocHost(&hResults[1], resultBytes));

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

    int buf = 0;

    // Launch first batch
    uint32_t count = ComputeBatchCount(current);
    if (count == 0) {
        std::cerr << "Nothing to compute (start > end).\n";
        goto cleanup;
    }
    LaunchCollatzKernel<>(current, count, dResults[buf], cfg.maxSteps, streams[buf], cfg.oddOnly);
    CUDA_CHECK_LAST();

    while (!gShutdownRequested.load(std::memory_order_relaxed)) {
        BigUint<> prevStart = current;
        uint32_t prevCount = count;
        int prevBuf = buf;

        current.AddU64(static_cast<uint64_t>(count) * stride);
        buf ^= 1;

        count = ComputeBatchCount(current);
        bool moreWork = (count > 0);

        if (moreWork) {
            LaunchCollatzKernel<>(current, count, dResults[buf], cfg.maxSteps, streams[buf], cfg.oddOnly);
            CUDA_CHECK_LAST();
        }

        CUDA_CHECK(cudaMemcpyAsync(hResults[prevBuf], dResults[prevBuf],
                        prevCount * sizeof(CollatzResult<>),
                        cudaMemcpyDeviceToHost, streams[prevBuf]));
        CUDA_CHECK(cudaStreamSynchronize(streams[prevBuf]));

        for (uint32_t i = 0; i < prevCount; ++i) {
            const auto& r = hResults[prevBuf][i];
            if (r.overflow) {
                std::cerr << "\nOVERFLOW at n=" << r.start.ToString()
                          << " — increase limbs (currently "
                          << COLLATZ_N_LIMBS * 64 << " bits)\n";
                continue;
            }
            if (r.exceededLimit) {
                ++totalDivergent;
                divCsv << r.start.ToString() << ','
                        << r.chainLength << ','
                        << r.lastValue.ToString() << ','
                        << r.maxValue.ToString() << '\n';
                continue;
            }
            bool isNewLongest = r.chainLength > longestChain;
            if (isNewLongest) {
                longestChain = r.chainLength;
            }
            if (cfg.minChain == 0 ? isNewLongest : r.chainLength >= cfg.minChain) {
                csv << r.start.ToString() << ','
                    << r.chainLength << ','
                    << r.maxValue.ToString() << '\n';
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
    CUDA_CHECK(cudaFreeHost(hResults[0]));
    CUDA_CHECK(cudaFreeHost(hResults[1]));
    CUDA_CHECK(cudaFree(dResults[0]));
    CUDA_CHECK(cudaFree(dResults[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

    double totalElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStart).count();
    std::cerr << "\r" << std::string(80, ' ') << "\r";
    std::cerr << "Done. " << totalProcessed << " numbers processed in "
              << std::fixed << std::setprecision(2) << totalElapsed << "s\n";
    return 0;
}
