#include "collatz/collatz_kernel.cuh"

static constexpr int kBlockSize = 256;
static constexpr int kMinBlocksPerSM = 2;

template <int N_LIMBS, bool OddOnly>
__global__ void __launch_bounds__(kBlockSize, kMinBlocksPerSM) CollatzKernel(
    BigUint<N_LIMBS> Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* __restrict__ Results,
    uint32_t MaxSteps)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Count) return;

    BigUint<N_LIMBS> n = Start;
    if constexpr (OddOnly) {
        // Start + 2*idx: process only odd numbers
        n.AddU64(static_cast<uint64_t>(idx) * 2);
    } else {
        n.AddU64(static_cast<uint64_t>(idx));
    }

    CollatzResult<N_LIMBS> res;
    res.start = n;
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
            // Even: batch-shift all trailing zeros at once
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            res.chainLength += tz;
        } else {
            // Odd: compute 3n+1, check for overflow
            bool ov = n.TriplePlusOne();  // n = 3n + 1
            if (ov) {
                res.overflow = true;
                break;
            }
            // The intermediate 3n+1 value may be the chain maximum
            if (n > res.maxValue) {
                res.maxValue = n;
            }
            // 3n+1 is always even; batch-shift trailing zeros
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            res.chainLength += 1 + tz;  // 1 for 3n+1, tz for the divisions
        }
    }

    res.lastValue = n;
    Results[idx] = res;
}

template <int N_LIMBS>
void LaunchCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* DResults,
    uint32_t MaxSteps,
    cudaStream_t Stream,
    bool OddOnly)
{
    if (Count == 0) return;
    const int gridSize = (Count + kBlockSize - 1) / kBlockSize;
    if (OddOnly) {
        CollatzKernel<N_LIMBS, true><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DResults, MaxSteps);
    } else {
        CollatzKernel<N_LIMBS, false><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DResults, MaxSteps);
    }
}

// Explicit instantiations for common limb counts
template void LaunchCollatzKernel<1>(const BigUint<1>&, uint32_t, CollatzResult<1>*, uint32_t, cudaStream_t, bool);
template void LaunchCollatzKernel<2>(const BigUint<2>&, uint32_t, CollatzResult<2>*, uint32_t, cudaStream_t, bool);
template void LaunchCollatzKernel<4>(const BigUint<4>&, uint32_t, CollatzResult<4>*, uint32_t, cudaStream_t, bool);

// Always instantiate for the configured limb count (may duplicate one of the above — that's fine)
template void LaunchCollatzKernel<COLLATZ_N_LIMBS>(const BigUint<COLLATZ_N_LIMBS>&, uint32_t, CollatzResult<COLLATZ_N_LIMBS>*, uint32_t, cudaStream_t, bool);

// ── Compact kernel: only chain length + flags ──────────────────────
// No start/maxValue/lastValue tracking; eliminates the per-step
// maxValue comparison from the hot loop.

template <int N_LIMBS, bool OddOnly>
__global__ void __launch_bounds__(kBlockSize, kMinBlocksPerSM) CompactCollatzKernel(
    BigUint<N_LIMBS> Start,
    uint32_t Count,
    CompactResult* __restrict__ Results,
    uint32_t MaxSteps)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Count) return;

    BigUint<N_LIMBS> n = Start;
    if constexpr (OddOnly) {
        n.AddU64(static_cast<uint64_t>(idx) * 2);
    } else {
        n.AddU64(static_cast<uint64_t>(idx));
    }

    uint32_t steps = 0;
    uint8_t  flags = 0;

    // Strip initial even factors (only possible when !OddOnly)
    if constexpr (!OddOnly) {
        if (n.IsEven() && !n.IsZero()) {
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            steps += tz;
        }
    }

    // n is now odd — fused loop: 3n+1 and shift in one call, no branch
    while (!n.IsOne()) {
        if (MaxSteps > 0 && steps >= MaxSteps) {
            flags |= CompactResult::kExceededLimit;
            break;
        }
        bool overflow;
        int tz = n.TriplePlusOneAndShift(overflow);
        if (overflow) {
            flags |= CompactResult::kOverflow;
            break;
        }
        steps += 1 + tz;
    }

    CompactResult cr;
    cr.chainLength = steps;
    cr.flags = flags;
    Results[idx] = cr;
}

template <int N_LIMBS>
void LaunchCompactCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    CompactResult* DResults,
    uint32_t MaxSteps,
    cudaStream_t Stream,
    bool OddOnly)
{
    if (Count == 0) return;
    const int gridSize = (Count + kBlockSize - 1) / kBlockSize;
    if (OddOnly) {
        CompactCollatzKernel<N_LIMBS, true><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DResults, MaxSteps);
    } else {
        CompactCollatzKernel<N_LIMBS, false><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DResults, MaxSteps);
    }
}

// Compact kernel instantiations
template void LaunchCompactCollatzKernel<1>(const BigUint<1>&, uint32_t, CompactResult*, uint32_t, cudaStream_t, bool);
template void LaunchCompactCollatzKernel<2>(const BigUint<2>&, uint32_t, CompactResult*, uint32_t, cudaStream_t, bool);
template void LaunchCompactCollatzKernel<4>(const BigUint<4>&, uint32_t, CompactResult*, uint32_t, cudaStream_t, bool);
template void LaunchCompactCollatzKernel<COLLATZ_N_LIMBS>(const BigUint<COLLATZ_N_LIMBS>&, uint32_t, CompactResult*, uint32_t, cudaStream_t, bool);

// ── Adaptive kernel: GPU-side filtering, outputs only hits ─────────

template <int N_LIMBS, bool OddOnly>
__global__ void __launch_bounds__(kBlockSize, kMinBlocksPerSM) AdaptiveCollatzKernel(
    BigUint<N_LIMBS> Start,
    uint32_t Count,
    uint32_t* __restrict__ HitCount,
    AdaptiveHit* __restrict__ Hits,
    uint32_t MinChain,
    uint32_t MaxHits,
    uint32_t MaxSteps)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Count) return;

    BigUint<N_LIMBS> n = Start;
    if constexpr (OddOnly) {
        n.AddU64(static_cast<uint64_t>(idx) * 2);
    } else {
        n.AddU64(static_cast<uint64_t>(idx));
    }

    uint32_t steps = 0;
    uint8_t  flags = 0;

    // Strip initial even factors (only possible when !OddOnly)
    if constexpr (!OddOnly) {
        if (n.IsEven() && !n.IsZero()) {
            int tz = n.CountTrailingZeros();
            n.ShiftRightN(tz);
            steps += tz;
        }
    }

    // n is now odd — fused loop: 3n+1 and shift in one call, no branch
    while (!n.IsOne()) {
        if (MaxSteps > 0 && steps >= MaxSteps) {
            flags |= CompactResult::kExceededLimit;
            break;
        }
        bool overflow;
        int tz = n.TriplePlusOneAndShift(overflow);
        if (overflow) {
            flags |= CompactResult::kOverflow;
            break;
        }
        steps += 1 + tz;
    }

    if (flags != 0 || steps > MinChain) {
        uint32_t slot = atomicAdd(HitCount, 1u);
        if (slot < MaxHits) {
            AdaptiveHit hit;
            hit.index = idx;
            hit.chainLength = steps;
            hit.flags = flags;
            Hits[slot] = hit;
        }
    }
}

template <int N_LIMBS>
void LaunchAdaptiveCollatzKernel(
    const BigUint<N_LIMBS>& Start,
    uint32_t Count,
    uint32_t* DHitCount,
    AdaptiveHit* DHits,
    uint32_t MinChain,
    uint32_t MaxHits,
    uint32_t MaxSteps,
    cudaStream_t Stream,
    bool OddOnly)
{
    if (Count == 0) return;
    const int gridSize = (Count + kBlockSize - 1) / kBlockSize;
    if (OddOnly) {
        AdaptiveCollatzKernel<N_LIMBS, true><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DHitCount, DHits, MinChain, MaxHits, MaxSteps);
    } else {
        AdaptiveCollatzKernel<N_LIMBS, false><<<gridSize, kBlockSize, 0, Stream>>>(
            Start, Count, DHitCount, DHits, MinChain, MaxHits, MaxSteps);
    }
}

// Adaptive kernel instantiations
template void LaunchAdaptiveCollatzKernel<1>(const BigUint<1>&, uint32_t, uint32_t*, AdaptiveHit*, uint32_t, uint32_t, uint32_t, cudaStream_t, bool);
template void LaunchAdaptiveCollatzKernel<2>(const BigUint<2>&, uint32_t, uint32_t*, AdaptiveHit*, uint32_t, uint32_t, uint32_t, cudaStream_t, bool);
template void LaunchAdaptiveCollatzKernel<4>(const BigUint<4>&, uint32_t, uint32_t*, AdaptiveHit*, uint32_t, uint32_t, uint32_t, cudaStream_t, bool);
template void LaunchAdaptiveCollatzKernel<COLLATZ_N_LIMBS>(const BigUint<COLLATZ_N_LIMBS>&, uint32_t, uint32_t*, AdaptiveHit*, uint32_t, uint32_t, uint32_t, cudaStream_t, bool);
