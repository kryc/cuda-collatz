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
template <int N_LIMBS = COLLATZ_N_LIMBS>
void LaunchCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* DResults,
    uint32_t MaxSteps = 0,
    cudaStream_t Stream = 0);
