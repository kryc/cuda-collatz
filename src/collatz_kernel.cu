#include "collatz/collatz_kernel.cuh"

static constexpr int kBlockSize = 256;

template <int N_LIMBS>
__global__ void __launch_bounds__(kBlockSize) CollatzKernel(
    BigUint<N_LIMBS> Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* __restrict__ Results,
    uint32_t MaxSteps)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Count) return;

    BigUint<N_LIMBS> n = Start;
    n.AddU64(static_cast<uint64_t>(idx));

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
    cudaStream_t Stream)
{
    if (Count == 0) return;
    const int gridSize = (Count + kBlockSize - 1) / kBlockSize;
    CollatzKernel<N_LIMBS><<<gridSize, kBlockSize, 0, Stream>>>(
        Start, Count, DResults, MaxSteps);
}

// Explicit instantiations for common limb counts
template void LaunchCollatzKernel<1>(const BigUint<1>&, uint32_t, CollatzResult<1>*, uint32_t, cudaStream_t);
template void LaunchCollatzKernel<2>(const BigUint<2>&, uint32_t, CollatzResult<2>*, uint32_t, cudaStream_t);
template void LaunchCollatzKernel<4>(const BigUint<4>&, uint32_t, CollatzResult<4>*, uint32_t, cudaStream_t);
