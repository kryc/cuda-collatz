#pragma once

#include "collatz/big_uint.cuh"
#include <cstdint>

/// Result for a single starting number's Collatz chain.
template <int N_LIMBS = COLLATZ_N_LIMBS>
struct CollatzResult {
    BigUint<N_LIMBS> start;      ///< Starting number
    BigUint<N_LIMBS> maxValue;  ///< Maximum value reached in the chain
    BigUint<N_LIMBS> lastValue; ///< Value of n when computation stopped (1 if converged)
    uint32_t chainLength;       ///< Number of steps to reach 1
    bool overflow;               ///< True if BigUint overflowed during computation
    bool exceededLimit;         ///< True if chainLength hit MaxSteps without reaching 1
};

/// Launch the Collatz kernel for a batch of consecutive starting numbers.
/// @param Start       The first number in the batch.
/// @param Count       How many consecutive numbers to process.
/// @param DResults    Device pointer to an array of at least `Count` CollatzResult.
/// @param MaxSteps    Maximum steps before giving up (0 = unlimited).
/// @param Stream      CUDA stream (0 for default).
/// @param OddOnly     If true, process only odd numbers: Start + 2*idx.
template <int N_LIMBS = COLLATZ_N_LIMBS>
void LaunchCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* DResults,
    uint32_t MaxSteps = 0,
    cudaStream_t Stream = 0,
    bool OddOnly = false);

/// Compact result for the fast kernel path — only chain length + status flags.
/// At 8 bytes vs ~80 bytes per full result, this reduces D→H bandwidth ~10×
/// and eliminates the per-step maxValue comparison from the hot loop.
struct CompactResult {
    uint32_t chainLength;
    uint8_t  flags;  // bitmask: kOverflow | kExceededLimit

    static constexpr uint8_t kOverflow      = 1;
    static constexpr uint8_t kExceededLimit  = 2;
};

/// Launch the compact Collatz kernel — outputs only chain length + flags.
template <int N_LIMBS = COLLATZ_N_LIMBS>
void LaunchCompactCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    CompactResult* DResults,
    uint32_t MaxSteps = 0,
    cudaStream_t Stream = 0,
    bool OddOnly = false);

/// A single hit from the adaptive kernel — only written for noteworthy chains.
struct AdaptiveHit {
    uint32_t index;        ///< Thread index within the batch
    uint32_t chainLength;
    uint8_t  flags;        ///< CompactResult::kOverflow | kExceededLimit
};

/// Launch the adaptive Collatz kernel — filters on GPU, outputs only hits.
/// Only results with chainLength > MinChain or non-zero flags are written.
/// @param DHitCount   Device pointer to a uint32_t counter (zeroed via cudaMemsetAsync before launch).
/// @param DHits       Device array for output hits.
/// @param MinChain    Only chains strictly longer than this are emitted.
/// @param MaxHits     Maximum entries to write (prevents buffer overflow).
template <int N_LIMBS = COLLATZ_N_LIMBS>
void LaunchAdaptiveCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    uint32_t* DHitCount,
    AdaptiveHit* DHits,
    uint32_t MinChain,
    uint32_t MaxHits,
    uint32_t MaxSteps = 0,
    cudaStream_t Stream = 0,
    bool OddOnly = false);
