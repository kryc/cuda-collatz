#include "collatz/collatz_kernel.cuh"

template <int N_LIMBS>
__global__ void CollatzKernel(
    BigUint<N_LIMBS> Start,
    uint32_t Count,
    CollatzResult<N_LIMBS>* Results,
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
            // Even: n = n / 2, one step
            n.ShiftRight1();
            res.chainLength += 1;
        } else {
            // Odd: compute 3n+1, check for overflow, capture max BEFORE shift
            bool ov = n.TriplePlusOne();  // n = 3n + 1
            if (ov) {
                res.overflow = true;
                break;
            }
            // The intermediate 3n+1 value may be the chain maximum
            if (n > res.maxValue) {
                res.maxValue = n;
            }
            // Now divide by 2 (3n+1 is always even), counts as 2 steps
            n.ShiftRight1();
            res.chainLength += 2;
        }
        // Update max after the step (for even steps, or after the >>1 of odd)
        if (n > res.maxValue) {
            res.maxValue = n;
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
    const int blockSize = 256;
    const int gridSize = (Count + blockSize - 1) / blockSize;
    CollatzKernel<N_LIMBS><<<gridSize, blockSize, 0, Stream>>>(
        Start, Count, DResults, MaxSteps);
}

// Explicit instantiations for common limb counts
template void LaunchCollatzKernel<1>(const BigUint<1>&, uint32_t, CollatzResult<1>*, uint32_t, cudaStream_t);
template void LaunchCollatzKernel<2>(const BigUint<2>&, uint32_t, CollatzResult<2>*, uint32_t, cudaStream_t);
template void LaunchCollatzKernel<4>(const BigUint<4>&, uint32_t, CollatzResult<4>*, uint32_t, cudaStream_t);
