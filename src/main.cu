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

// ── Config defaults ───────────────────────────────────────────────
static constexpr uint64_t kDefaultStart           = 1;
static constexpr uint64_t kDefaultEnd             = 0;        // 0 means "run forever"
static constexpr uint32_t kDefaultBatchSize       = 1u << 20;  // ~1 million
static constexpr uint32_t kDefaultMinChainLength   = 1'000;
static constexpr uint32_t kDefaultMaxSteps         = 1'000'000;        // 0 = unlimited
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
    bool        help       = false;
};

// ── Checkpoint ─────────────────────────────────────────────────────
// Format: plain text, one field per line:
//   nextStart_limb0 nextStart_limb1 ... nextStart_limbN
//   batchSize
// Written atomically via tmp + rename.

static bool WriteCheckpoint(const std::string& Path,
                             const BigUint<>& NextStart,
                             uint32_t BatchSize) {
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
    out << '\n' << BatchSize << '\n';
    out.close();
    if (std::rename(tmp.c_str(), Path.c_str()) != 0) {
        std::cerr << "WARNING: rename failed for checkpoint\n";
        return false;
    }
    return true;
}

static bool ReadCheckpoint(const std::string& Path,
                            BigUint<>& NextStart,
                            uint32_t& BatchSize) {
    std::ifstream in(Path);
    if (!in) return false;
    for (int i = 0; i < COLLATZ_N_LIMBS; ++i) {
        if (!(in >> NextStart.limbs[i])) return false;
    }
    if (!(in >> BatchSize)) return false;
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
        << "  --min-chain N       Minimum chain length to log (default " << kDefaultMinChainLength << ")\n"
        << "  --max-steps N       Max steps before flagging as non-converging (default " << kDefaultMaxSteps << " = off)\n"
        << "  --output FILE       CSV output file (default " << kDefaultOutput << ")\n"
        << "  --divergent FILE    File for non-converging chains (default " << kDefaultDivergent << ")\n"
        << "  --checkpoint FILE   Checkpoint file (default " << kDefaultCheckpoint << ")\n"
        << "  --resume            Resume from checkpoint\n"
        << "  --fresh             Discard existing checkpoint and start fresh\n"
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
        result.ShiftLeft1(); result.ShiftLeft1();
        result.ShiftLeft1(); result.ShiftLeft1();
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
        unsigned long exp = std::strtoul(caret + 1, nullptr, 10);
        if (exp >= static_cast<unsigned long>(COLLATZ_N_LIMBS) * 64) {
            std::cerr << "ERROR: 2^" << exp << " exceeds "
                      << COLLATZ_N_LIMBS * 64 << "-bit capacity\n";
            std::exit(1);
        }
        BigUint<> result(1);
        for (unsigned long i = 0; i < exp; ++i)
            result.ShiftLeft1();
        return result;
    }
    // 0x hex prefix
    if (S[0] == '0' && (S[1] == 'x' || S[1] == 'X'))
        return ParseHex(S + 2);
    // Plain decimal
    return ParseDecimal(S);
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
        else if (arg("--batch-size"))cfg.batchSize  = static_cast<uint32_t>(std::atol(next()));
        else if (arg("--min-chain")) cfg.minChain   = static_cast<uint32_t>(std::atol(next()));
        else if (arg("--max-steps")) cfg.maxSteps   = static_cast<uint32_t>(std::atol(next()));
        else if (arg("--output"))    cfg.output      = next();
        else if (arg("--divergent")) cfg.divergent   = next();
        else if (arg("--checkpoint"))cfg.checkpoint  = next();
        else if (arg("--resume"))    cfg.resume      = true;
        else if (arg("--fresh"))     cfg.fresh       = true;
        else if (arg("--help"))      cfg.help        = true;
        else {
            std::cerr << "ERROR: unknown option " << Argv[i] << "\n";
            std::exit(1);
        }
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
    if (cfg.resume) {
        BigUint<> ckptStart;
        uint32_t ckptBatch;
        if (ReadCheckpoint(cfg.checkpoint, ckptStart, ckptBatch)) {
            cfg.start = ckptStart;
            std::cerr << "Resumed from checkpoint: start = "
                      << ckptStart.ToString() << "\n";
        } else {
            std::cerr << "WARNING: no checkpoint found, starting from --start\n";
        }
    }

    // Scan existing CSV for longest chain if resuming
    uint32_t longestChain = 0;
    if (cfg.resume) {
        longestChain = ScanLongestChain(cfg.output);
        if (longestChain > 0) {
            std::cerr << "Longest chain in existing CSV: " << longestChain << "\n";
        }
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
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    CollatzResult<>* dResults[2];
    cudaMalloc(&dResults[0], resultBytes);
    cudaMalloc(&dResults[1], resultBytes);

    CollatzResult<>* hResults[2];
    cudaMallocHost(&hResults[0], resultBytes);  // pinned memory for async copy
    cudaMallocHost(&hResults[1], resultBytes);

    BigUint<> current = cfg.start;
    bool hasEnd = !cfg.endVal.IsZero();

    uint64_t totalProcessed = 0;
    auto wallStart = std::chrono::steady_clock::now();
    auto lastReport = wallStart;

    // Determine the actual count for this batch (may be less at the end)
    auto ComputeBatchCount = [&](const BigUint<>& BatchStart) -> uint32_t {
        if (!hasEnd) return batchSize;
        // How many numbers from BatchStart to endVal (inclusive)?
        if (cfg.endVal < BatchStart) return 0;
        // If the whole batch fits before endVal, return full batchSize.
        // Otherwise, linear-scan the tail (small, only at the very end).
        BigUint<> limit = BatchStart;
        limit.AddU64(batchSize);
        if (limit < cfg.endVal || limit == cfg.endVal) {
            return batchSize;
        }
        // Linear scan for remaining count (small, only when near end)
        BigUint<> probe = BatchStart;
        for (uint32_t c = 0; c < batchSize; ++c) {
            if (cfg.endVal < probe) return c;
            probe.AddU64(1);
        }
        return batchSize;
    };

    int buf = 0;          // current buffer index (0 or 1)

    // Launch first batch
    uint32_t count = ComputeBatchCount(current);
    if (count == 0) {
        std::cerr << "Nothing to compute (start > end).\n";
        goto cleanup;
    }
    LaunchCollatzKernel<>(current, count, dResults[buf], cfg.maxSteps, streams[buf]);

    while (!gShutdownRequested.load(std::memory_order_relaxed)) {
        // Advance to next batch
        BigUint<> prevStart = current;
        uint32_t prevCount = count;
        int prevBuf = buf;

        current.AddU64(count);
        buf ^= 1;

        count = ComputeBatchCount(current);
        bool moreWork = (count > 0);

        // Launch next batch (overlaps with D→H of previous)
        if (moreWork) {
            LaunchCollatzKernel<>(current, count, dResults[buf], cfg.maxSteps, streams[buf]);
        }

        // Copy previous batch results D→H asynchronously, then sync
        cudaMemcpyAsync(hResults[prevBuf], dResults[prevBuf],
                        prevCount * sizeof(CollatzResult<>),
                        cudaMemcpyDeviceToHost, streams[prevBuf]);
        cudaStreamSynchronize(streams[prevBuf]);

        // Process results
        for (uint32_t i = 0; i < prevCount; ++i) {
            const auto& r = hResults[prevBuf][i];
            if (r.overflow) {
                std::cerr << "\nOVERFLOW at n=" << r.start.ToHexString()
                          << " — increase limbs (currently "
                          << COLLATZ_N_LIMBS * 64 << " bits)\n";
                continue;
            }
            if (r.exceededLimit) {
                // Chain did not converge within maxSteps
                divCsv << r.start.ToHexString() << ','
                        << r.chainLength << ','
                        << r.lastValue.ToHexString() << ','
                        << r.maxValue.ToHexString() << '\n';
                continue;
            }
            if (r.chainLength > longestChain) {
                longestChain = r.chainLength;
            }
            if (r.chainLength >= cfg.minChain) {
                csv << r.start.ToHexString() << ','
                    << r.chainLength << ','
                    << r.maxValue.ToHexString() << '\n';
            }
        }
        csv.flush();
        if (divCsv.is_open()) divCsv.flush();

        totalProcessed += prevCount;

        // Progress report every ~5 seconds
        auto now = std::chrono::steady_clock::now();
        double elapsedSinceReport =
            std::chrono::duration<double>(now - lastReport).count();
        if (elapsedSinceReport >= 5.0) {
            double totalElapsed =
                std::chrono::duration<double>(now - wallStart).count();
            double rate = totalProcessed / totalElapsed;
            std::ostringstream status;
            status << "\r" << totalProcessed << " numbers in "
                   << std::fixed << std::setprecision(1) << totalElapsed
                   << "s  " << std::setprecision(0) << rate << " n/s  "
                   << "longest: " << longestChain << "  "
                   << "current: " << current.ToString();
            // Pad with spaces to clear any previous longer line
            std::string line = status.str();
            if (line.size() < 80) line.resize(80, ' ');
            std::cerr << line << std::flush;
            lastReport = now;
        }

        // Checkpoint
        WriteCheckpoint(cfg.checkpoint, moreWork ? current : prevStart,
                         batchSize);

        if (!moreWork) break;
    }

    // If we were interrupted, sync any in-flight work and checkpoint
    if (gShutdownRequested.load(std::memory_order_relaxed)) {
        std::cerr << "\nShutting down gracefully...\n";
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        WriteCheckpoint(cfg.checkpoint, current, batchSize);
        std::cerr << "Checkpoint written at " << current.ToString() << "\n";
    }

cleanup:
    csv.close();
    if (divCsv.is_open()) divCsv.close();
    cudaFreeHost(hResults[0]);
    cudaFreeHost(hResults[1]);
    cudaFree(dResults[0]);
    cudaFree(dResults[1]);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    double totalElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallStart).count();
    std::cerr << "\r" << std::string(80, ' ') << "\r";
    std::cerr << "Done. " << totalProcessed << " numbers processed in "
              << std::fixed << std::setprecision(2) << totalElapsed << "s\n";
    return 0;
}
