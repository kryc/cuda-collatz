#include <gtest/gtest.h>
#include "collatz/collatz_kernel.cuh"
#include <vector>
#include <cuda_runtime.h>

// ── CUDA error checking helper ──
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

/// Helper: run the Collatz kernel for a single starting number, return result.
template <int N_LIMBS>
CollatzResult<N_LIMBS> RunSingle(uint64_t StartVal) {
    BigUint<N_LIMBS> start(StartVal);
    CollatzResult<N_LIMBS>* dResults = nullptr;
    EXPECT_EQ(cudaMalloc(&dResults, sizeof(CollatzResult<N_LIMBS>)), cudaSuccess);

    LaunchCollatzKernel<N_LIMBS>(start, 1, dResults);

    cudaError_t syncErr = cudaDeviceSynchronize();
    EXPECT_EQ(syncErr, cudaSuccess) << cudaGetErrorString(syncErr);

    CollatzResult<N_LIMBS> result;
    EXPECT_EQ(cudaMemcpy(&result, dResults, sizeof(CollatzResult<N_LIMBS>),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dResults);
    return result;
}

/// Helper: run the Collatz kernel for a batch of consecutive starting numbers.
template <int N_LIMBS>
std::vector<CollatzResult<N_LIMBS>> RunBatch(uint64_t StartVal, uint32_t Count) {
    BigUint<N_LIMBS> start(StartVal);
    CollatzResult<N_LIMBS>* dResults = nullptr;
    size_t bytes = Count * sizeof(CollatzResult<N_LIMBS>);
    EXPECT_EQ(cudaMalloc(&dResults, bytes), cudaSuccess);

    LaunchCollatzKernel<N_LIMBS>(start, Count, dResults);

    cudaError_t syncErr = cudaDeviceSynchronize();
    EXPECT_EQ(syncErr, cudaSuccess) << cudaGetErrorString(syncErr);

    std::vector<CollatzResult<N_LIMBS>> results(Count);
    EXPECT_EQ(cudaMemcpy(results.data(), dResults, bytes,
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dResults);
    return results;
}

// ═══════════════════════════════════════════════════════════════════
// Trivial / small cases
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, N1_ZeroSteps) {
    auto r = RunSingle<2>(1);
    EXPECT_EQ(r.chainLength, 0u);
    EXPECT_EQ(r.start, BigUint<2>(1));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N2_OneStep) {
    // 2 → 1 (one step: divide by 2)
    auto r = RunSingle<2>(2);
    EXPECT_EQ(r.chainLength, 1u);
    EXPECT_EQ(r.maxValue, BigUint<2>(2));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N4_TwoSteps) {
    // 4 → 2 → 1
    auto r = RunSingle<2>(4);
    EXPECT_EQ(r.chainLength, 2u);
    EXPECT_EQ(r.maxValue, BigUint<2>(4));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N8_ThreeSteps) {
    // 8 → 4 → 2 → 1
    auto r = RunSingle<2>(8);
    EXPECT_EQ(r.chainLength, 3u);
    EXPECT_EQ(r.maxValue, BigUint<2>(8));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N16_FourSteps) {
    auto r = RunSingle<2>(16);
    EXPECT_EQ(r.chainLength, 4u);
    EXPECT_EQ(r.maxValue, BigUint<2>(16));
    EXPECT_FALSE(r.overflow);
}

// ═══════════════════════════════════════════════════════════════════
// Famous sequences
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, N27_111Steps) {
    // n=27 is famous: 111 steps, max value 9232
    auto r = RunSingle<2>(27);
    EXPECT_EQ(r.chainLength, 111u);
    EXPECT_EQ(r.maxValue, BigUint<2>(9232));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N97_118Steps) {
    auto r = RunSingle<2>(97);
    EXPECT_EQ(r.chainLength, 118u);
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N871_178Steps) {
    auto r = RunSingle<2>(871);
    EXPECT_EQ(r.chainLength, 178u);
    EXPECT_FALSE(r.overflow);
}

// ═══════════════════════════════════════════════════════════════════
// Max value tracking — verify intermediate 3n+1 is captured
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, N3_MaxIs10) {
    // 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1
    // The max should be 16 (not 5, which would happen if we missed intermediates)
    auto r = RunSingle<2>(3);
    EXPECT_EQ(r.chainLength, 7u);
    EXPECT_EQ(r.maxValue, BigUint<2>(16));
    EXPECT_FALSE(r.overflow);
}

TEST(CollatzKernel, N7_MaxIs52) {
    // 7→22→11→34→17→52→26→13→40→20→10→5→16→8→4→2→1
    // Max = 52
    auto r = RunSingle<2>(7);
    EXPECT_EQ(r.chainLength, 16u);
    EXPECT_EQ(r.maxValue, BigUint<2>(52));
    EXPECT_FALSE(r.overflow);
}

// ═══════════════════════════════════════════════════════════════════
// Batch correctness — n=1..50 known chain lengths
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, Batch_1to50) {
    // Known chain lengths for n = 1 through 50
    const uint32_t expected[] = {
        0, 1, 7, 2, 5, 8, 16, 3, 19, 6,
        14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        7, 15, 15, 10, 23, 10, 111, 18, 18, 18,
        106, 5, 26, 13, 13, 21, 21, 21, 34, 8,
        109, 8, 29, 16, 16, 16, 104, 11, 24, 24
    };

    auto results = RunBatch<2>(1, 50);
    ASSERT_EQ(results.size(), 50u);

    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(results[i].chainLength, expected[i])
            << "Mismatch at n=" << (i + 1);
        EXPECT_FALSE(results[i].overflow)
            << "Unexpected overflow at n=" << (i + 1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Overflow detection — use 1-limb (64-bit) BigUint to force overflow
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, Overflow_SingleLimb) {
    // n = 2^63 + 1 (odd) with only 64 bits → 3n+1 overflows immediately
    // The kernel should flag overflow.
    BigUint<1> start((1ULL << 63) | 1ULL);
    CollatzResult<1>* dResults = nullptr;
    CUDA_CHECK(cudaMalloc(&dResults, sizeof(CollatzResult<1>)));

    LaunchCollatzKernel<1>(start, 1, dResults);
    CUDA_CHECK(cudaDeviceSynchronize());

    CollatzResult<1> result;
    CUDA_CHECK(cudaMemcpy(&result, dResults, sizeof(CollatzResult<1>),
                           cudaMemcpyDeviceToHost));
    cudaFree(dResults);

    EXPECT_TRUE(result.overflow);
}

// ═══════════════════════════════════════════════════════════════════
// Larger starting values (verify no crash / no overflow with 2-limb)
// ═══════════════════════════════════════════════════════════════════

TEST(CollatzKernel, LargeStart_NoOverflow) {
    // Start near 2^64 with 128-bit integers — should complete fine
    auto r = RunSingle<2>(UINT64_MAX - 1);
    EXPECT_GT(r.chainLength, 0u);
    EXPECT_FALSE(r.overflow);
}

// ═══════════════════════════════════════════════════════════════════
// Max-steps limit (exceededLimit flag)
// ═══════════════════════════════════════════════════════════════════

/// Helper: run single with MaxSteps
template <int N_LIMBS>
CollatzResult<N_LIMBS> RunSingleLimited(uint64_t StartVal, uint32_t MaxSteps) {
    BigUint<N_LIMBS> start(StartVal);
    CollatzResult<N_LIMBS>* dResults = nullptr;
    EXPECT_EQ(cudaMalloc(&dResults, sizeof(CollatzResult<N_LIMBS>)), cudaSuccess);

    LaunchCollatzKernel<N_LIMBS>(start, 1, dResults, MaxSteps);

    cudaError_t syncErr = cudaDeviceSynchronize();
    EXPECT_EQ(syncErr, cudaSuccess) << cudaGetErrorString(syncErr);

    CollatzResult<N_LIMBS> result;
    EXPECT_EQ(cudaMemcpy(&result, dResults, sizeof(CollatzResult<N_LIMBS>),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dResults);
    return result;
}

TEST(CollatzKernel, MaxSteps_Unlimited_Converges) {
    // MaxSteps=0 means unlimited — n=27 should converge normally (111 steps)
    auto r = RunSingleLimited<2>(27, 0);
    EXPECT_EQ(r.chainLength, 111u);
    EXPECT_FALSE(r.overflow);
    EXPECT_FALSE(r.exceededLimit);
    EXPECT_TRUE(r.lastValue.IsOne());
}

TEST(CollatzKernel, MaxSteps_Sufficient_Converges) {
    // n=27 takes 111 steps — giving it 200 should be enough
    auto r = RunSingleLimited<2>(27, 200);
    EXPECT_EQ(r.chainLength, 111u);
    EXPECT_FALSE(r.exceededLimit);
    EXPECT_TRUE(r.lastValue.IsOne());
}

TEST(CollatzKernel, MaxSteps_TooFew_ExceedsLimit) {
    // n=27 takes 111 steps — cap at 10 steps
    // Note: odd branch adds 2 steps atomically, so chainLength may slightly
    // exceed MaxSteps (by 1) before the next check.
    auto r = RunSingleLimited<2>(27, 10);
    EXPECT_TRUE(r.exceededLimit);
    EXPECT_FALSE(r.overflow);
    EXPECT_GE(r.chainLength, 10u);
    EXPECT_LE(r.chainLength, 11u);  // odd branch can overshoot by 1
    EXPECT_FALSE(r.lastValue.IsOne());  // chain didn't finish
}

TEST(CollatzKernel, MaxSteps_ExactBoundary) {
    // n=27 takes exactly 111 steps — cap at exactly 111
    auto r = RunSingleLimited<2>(27, 111);
    EXPECT_FALSE(r.exceededLimit);
    EXPECT_EQ(r.chainLength, 111u);
    EXPECT_TRUE(r.lastValue.IsOne());
}

TEST(CollatzKernel, MaxSteps_OneStep_AllButTrivial) {
    // n=2 takes 1 step — cap at 1 should still converge
    auto r = RunSingleLimited<2>(2, 1);
    EXPECT_FALSE(r.exceededLimit);
    EXPECT_EQ(r.chainLength, 1u);
    EXPECT_TRUE(r.lastValue.IsOne());
}

TEST(CollatzKernel, MaxSteps_LastValue_Preserved) {
    // n=3 → 10 → 5 → 16 → 8 → 4 → 2 → 1 (7 steps, but shortcut makes it fewer)
    // With MaxSteps=1: n=3 is odd → 3*3+1=10, 10/2=5, +2 steps → chainLength=2
    // chainLength(2) >= MaxSteps(1) won't trigger because we check BEFORE the step
    // Actually the check happens at top of loop: chainLength=0 < 1, proceed; 
    // after step chainLength=2 (odd branch); next iteration 2 >= 1 → exceeded
    auto r = RunSingleLimited<2>(3, 1);
    EXPECT_TRUE(r.exceededLimit);
    // lastValue should not be 1 since we stopped early
    EXPECT_FALSE(r.lastValue.IsOne());
}
