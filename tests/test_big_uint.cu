#include <gtest/gtest.h>
#include "collatz/big_uint.cuh"

// ═══════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintConstruction, ZeroInit) {
    BigUint<2> x;
    EXPECT_EQ(x.limbs[0], 0u);
    EXPECT_EQ(x.limbs[1], 0u);
}

TEST(BigUintConstruction, FromUint64_Zero) {
    BigUint<2> x(0);
    EXPECT_EQ(x.limbs[0], 0u);
    EXPECT_EQ(x.limbs[1], 0u);
}

TEST(BigUintConstruction, FromUint64_One) {
    BigUint<2> x(1);
    EXPECT_EQ(x.limbs[0], 1u);
    EXPECT_EQ(x.limbs[1], 0u);
}

TEST(BigUintConstruction, FromUint64_Max) {
    BigUint<2> x(UINT64_MAX);
    EXPECT_EQ(x.limbs[0], UINT64_MAX);
    EXPECT_EQ(x.limbs[1], 0u);
}

TEST(BigUintConstruction, FourLimbs) {
    BigUint<4> x(42);
    EXPECT_EQ(x.limbs[0], 42u);
    EXPECT_EQ(x.limbs[1], 0u);
    EXPECT_EQ(x.limbs[2], 0u);
    EXPECT_EQ(x.limbs[3], 0u);
}

// ═══════════════════════════════════════════════════════════════════
// Queries: IsOne, IsZero, IsEven, IsOdd
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintQueries, IsOne_True) {
    EXPECT_TRUE(BigUint<2>(1).IsOne());
}

TEST(BigUintQueries, IsOne_Zero) {
    EXPECT_FALSE(BigUint<2>(0).IsOne());
}

TEST(BigUintQueries, IsOne_Two) {
    EXPECT_FALSE(BigUint<2>(2).IsOne());
}

TEST(BigUintQueries, IsOne_HighLimbSet) {
    BigUint<2> x(1);
    x.limbs[1] = 1;  // high limb set, so not == 1
    EXPECT_FALSE(x.IsOne());
}

TEST(BigUintQueries, IsZero_True) {
    EXPECT_TRUE(BigUint<2>(0).IsZero());
}

TEST(BigUintQueries, IsZero_False) {
    EXPECT_FALSE(BigUint<2>(1).IsZero());
}

TEST(BigUintQueries, IsZero_HighLimbOnly) {
    BigUint<2> x(0);
    x.limbs[1] = 1;
    EXPECT_FALSE(x.IsZero());
}

TEST(BigUintQueries, IsEven_Zero) {
    EXPECT_TRUE(BigUint<2>(0).IsEven());
}

TEST(BigUintQueries, IsEven_Two) {
    EXPECT_TRUE(BigUint<2>(2).IsEven());
}

TEST(BigUintQueries, IsEven_LargeEven) {
    EXPECT_TRUE(BigUint<2>(0xFFFFFFFFFFFFFFFEULL).IsEven());
}

TEST(BigUintQueries, IsOdd_One) {
    EXPECT_TRUE(BigUint<2>(1).IsOdd());
}

TEST(BigUintQueries, IsOdd_Three) {
    EXPECT_TRUE(BigUint<2>(3).IsOdd());
}

TEST(BigUintQueries, IsOdd_Even) {
    EXPECT_FALSE(BigUint<2>(2).IsOdd());
}

// ═══════════════════════════════════════════════════════════════════
// CountTrailingZeros
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintQueries, CTZ_One) {
    EXPECT_EQ(BigUint<2>(1).CountTrailingZeros(), 0);
}

TEST(BigUintQueries, CTZ_Two) {
    EXPECT_EQ(BigUint<2>(2).CountTrailingZeros(), 1);
}

TEST(BigUintQueries, CTZ_PowerOf2) {
    EXPECT_EQ(BigUint<2>(1ULL << 20).CountTrailingZeros(), 20);
}

TEST(BigUintQueries, CTZ_OddNumber) {
    EXPECT_EQ(BigUint<2>(0xFF01).CountTrailingZeros(), 0);
}

TEST(BigUintQueries, CTZ_HighLimbOnly) {
    BigUint<2> x(0);
    x.limbs[1] = 8;  // bit 67 set (64 + 3)
    EXPECT_EQ(x.CountTrailingZeros(), 67);
}

TEST(BigUintQueries, CTZ_BothLimbs) {
    BigUint<2> x(4);  // bit 2 set
    x.limbs[1] = 1;
    EXPECT_EQ(x.CountTrailingZeros(), 2);  // lowest set bit wins
}

TEST(BigUintQueries, CTZ_Zero) {
    EXPECT_EQ(BigUint<2>(0).CountTrailingZeros(), 128);  // N_LIMBS * 64
}

// ═══════════════════════════════════════════════════════════════════
// Comparison
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintComparison, Equal_Same) {
    EXPECT_EQ(BigUint<2>(42), BigUint<2>(42));
}

TEST(BigUintComparison, Equal_Zero) {
    EXPECT_EQ(BigUint<2>(0), BigUint<2>(0));
}

TEST(BigUintComparison, NotEqual) {
    EXPECT_NE(BigUint<2>(1), BigUint<2>(2));
}

TEST(BigUintComparison, NotEqual_HighLimb) {
    BigUint<2> a(0), b(0);
    a.limbs[1] = 1;
    EXPECT_NE(a, b);
}

TEST(BigUintComparison, LessThan_LowLimb) {
    EXPECT_LT(BigUint<2>(1), BigUint<2>(2));
}

TEST(BigUintComparison, LessThan_HighLimb) {
    BigUint<2> a(UINT64_MAX), b(0);
    b.limbs[1] = 1;  // b = 2^64, a = 2^64-1
    EXPECT_LT(a, b);
}

TEST(BigUintComparison, GreaterThan) {
    EXPECT_GT(BigUint<2>(10), BigUint<2>(5));
}

TEST(BigUintComparison, GreaterThan_HighLimb) {
    BigUint<2> a(0), b(UINT64_MAX);
    a.limbs[1] = 1;  // a = 2^64, b = 2^64-1
    EXPECT_GT(a, b);
}

TEST(BigUintComparison, LessEqual) {
    EXPECT_LE(BigUint<2>(5), BigUint<2>(5));
    EXPECT_LE(BigUint<2>(4), BigUint<2>(5));
}

TEST(BigUintComparison, GreaterEqual) {
    EXPECT_GE(BigUint<2>(5), BigUint<2>(5));
    EXPECT_GE(BigUint<2>(6), BigUint<2>(5));
}

TEST(BigUintComparison, NotLessThan_Equal) {
    EXPECT_FALSE(BigUint<2>(5) < BigUint<2>(5));
}

// ═══════════════════════════════════════════════════════════════════
// Addition
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintAdd, ZeroPlusZero) {
    BigUint<2> a(0);
    a.Add(BigUint<2>(0));
    EXPECT_EQ(a, BigUint<2>(0));
}

TEST(BigUintAdd, OnePlusOne) {
    BigUint<2> a(1);
    a.Add(BigUint<2>(1));
    EXPECT_EQ(a, BigUint<2>(2));
}

TEST(BigUintAdd, CarryWithinLimb) {
    BigUint<2> a(UINT64_MAX - 1);
    uint64_t carry = a.Add(BigUint<2>(1));
    EXPECT_EQ(a, BigUint<2>(UINT64_MAX));
    EXPECT_EQ(carry, 0u);
}

TEST(BigUintAdd, CarryAcrossLimbs) {
    BigUint<2> a(UINT64_MAX);
    uint64_t carry = a.Add(BigUint<2>(1));
    // Should be 2^64
    EXPECT_EQ(a.limbs[0], 0u);
    EXPECT_EQ(a.limbs[1], 1u);
    EXPECT_EQ(carry, 0u);
}

TEST(BigUintAdd, CarryOut) {
    // All limbs max + 1 → overflow
    BigUint<2> a(UINT64_MAX);
    a.limbs[1] = UINT64_MAX;
    uint64_t carry = a.Add(BigUint<2>(1));
    EXPECT_EQ(carry, 1u);
}

TEST(BigUintAdd, AddU64) {
    BigUint<2> a(100);
    a.AddU64(200);
    EXPECT_EQ(a, BigUint<2>(300));
}

TEST(BigUintAdd, AddU64_CarryAcross) {
    BigUint<2> a(UINT64_MAX);
    a.AddU64(2);
    EXPECT_EQ(a.limbs[0], 1u);
    EXPECT_EQ(a.limbs[1], 1u);
}

// ═══════════════════════════════════════════════════════════════════
// Shift
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintShift, RightShift_Two) {
    BigUint<2> a(2);
    a.ShiftRight1();
    EXPECT_EQ(a, BigUint<2>(1));
}

TEST(BigUintShift, RightShift_One) {
    BigUint<2> a(1);
    a.ShiftRight1();
    EXPECT_EQ(a, BigUint<2>(0));
}

TEST(BigUintShift, RightShift_Odd) {
    BigUint<2> a(7);  // 111 → 011 = 3
    a.ShiftRight1();
    EXPECT_EQ(a, BigUint<2>(3));
}

TEST(BigUintShift, RightShift_CrossLimb) {
    // Value: 2^64 (limbs[1]=1, limbs[0]=0)
    // After >>1: 2^63 (limbs[1]=0, limbs[0]=0x8000000000000000)
    BigUint<2> a(0);
    a.limbs[1] = 1;
    a.ShiftRight1();
    EXPECT_EQ(a.limbs[0], 0x8000000000000000ULL);
    EXPECT_EQ(a.limbs[1], 0u);
}

TEST(BigUintShift, RightShift_LargeValue) {
    // 0xFFFFFFFFFFFFFFFF'FFFFFFFFFFFFFFFE >> 1
    //   = 0x7FFFFFFFFFFFFFFF'FFFFFFFFFFFFFFFF
    BigUint<2> a(0xFFFFFFFFFFFFFFFEULL);
    a.limbs[1] = UINT64_MAX;
    a.ShiftRight1();
    EXPECT_EQ(a.limbs[0], UINT64_MAX);
    EXPECT_EQ(a.limbs[1], 0x7FFFFFFFFFFFFFFFULL);
}

TEST(BigUintShift, LeftShift_One) {
    BigUint<2> a(1);
    uint64_t out = a.ShiftLeft1();
    EXPECT_EQ(a, BigUint<2>(2));
    EXPECT_EQ(out, 0u);
}

TEST(BigUintShift, LeftShift_CrossLimb) {
    BigUint<2> a(0x8000000000000000ULL);
    uint64_t out = a.ShiftLeft1();
    EXPECT_EQ(a.limbs[0], 0u);
    EXPECT_EQ(a.limbs[1], 1u);
    EXPECT_EQ(out, 0u);
}

TEST(BigUintShift, LeftShift_Overflow) {
    BigUint<2> a(0);
    a.limbs[1] = 0x8000000000000000ULL;
    uint64_t out = a.ShiftLeft1();
    EXPECT_EQ(out, 1u);  // bit shifted out of top limb
}

// ── ShiftRightN (multi-bit right shift) ──

TEST(BigUintShift, RightShiftN_Zero) {
    BigUint<2> a(42);
    a.ShiftRightN(0);
    EXPECT_EQ(a, BigUint<2>(42));
}

TEST(BigUintShift, RightShiftN_ByOne) {
    BigUint<2> a(2);
    a.ShiftRightN(1);
    EXPECT_EQ(a, BigUint<2>(1));
}

TEST(BigUintShift, RightShiftN_ByFour) {
    BigUint<2> a(0x30);  // 48
    a.ShiftRightN(4);
    EXPECT_EQ(a, BigUint<2>(3));
}

TEST(BigUintShift, RightShiftN_CrossLimb) {
    BigUint<2> a(0);
    a.limbs[1] = 1;  // 2^64
    a.ShiftRightN(1);
    EXPECT_EQ(a.limbs[0], 0x8000000000000000ULL);
    EXPECT_EQ(a.limbs[1], 0u);
}

TEST(BigUintShift, RightShiftN_FullLimb) {
    BigUint<2> a(0);
    a.limbs[1] = 0xFF;
    a.ShiftRightN(64);
    EXPECT_EQ(a.limbs[0], 0xFFu);
    EXPECT_EQ(a.limbs[1], 0u);
}

TEST(BigUintShift, RightShiftN_LimbPlusBits) {
    BigUint<2> a(0);
    a.limbs[1] = 0x80;  // bit 71
    a.ShiftRightN(68);   // shift right by 64+4
    EXPECT_EQ(a.limbs[0], 8u);  // 0x80 >> 4 = 8
    EXPECT_EQ(a.limbs[1], 0u);
}

TEST(BigUintShift, RightShiftN_AllBits) {
    BigUint<2> a(UINT64_MAX);
    a.limbs[1] = UINT64_MAX;
    a.ShiftRightN(128);
    EXPECT_EQ(a.limbs[0], 0u);
    EXPECT_EQ(a.limbs[1], 0u);
}

// ═══════════════════════════════════════════════════════════════════
// TriplePlusOne (3n + 1)
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintTriplePlusOne, One) {
    BigUint<2> a(1);
    bool overflow = a.TriplePlusOne();
    EXPECT_EQ(a, BigUint<2>(4));  // 3*1 + 1 = 4
    EXPECT_FALSE(overflow);
}

TEST(BigUintTriplePlusOne, Three) {
    BigUint<2> a(3);
    bool overflow = a.TriplePlusOne();
    EXPECT_EQ(a, BigUint<2>(10));  // 3*3 + 1 = 10
    EXPECT_FALSE(overflow);
}

TEST(BigUintTriplePlusOne, TwentySeven) {
    BigUint<2> a(27);
    bool overflow = a.TriplePlusOne();
    EXPECT_EQ(a, BigUint<2>(82));  // 3*27 + 1 = 82
    EXPECT_FALSE(overflow);
}

TEST(BigUintTriplePlusOne, LargeNoOverflow) {
    // n = 2^62, 3n+1 = 3*2^62 + 1 — fits in 64 bits
    BigUint<2> a(1ULL << 62);
    bool overflow = a.TriplePlusOne();
    EXPECT_EQ(a.limbs[0], 3ULL * (1ULL << 62) + 1);
    EXPECT_FALSE(overflow);
}

TEST(BigUintTriplePlusOne, CrossLimbCarry) {
    // n = 2^63, 3n+1 = 3*2^63 + 1 = 2^64 + 2^63 + 1
    // limbs[0] should be 2^63 + 1, limbs[1] should be 1
    BigUint<2> a(1ULL << 63);
    bool overflow = a.TriplePlusOne();
    EXPECT_EQ(a.limbs[0], (1ULL << 63) + 1);
    EXPECT_EQ(a.limbs[1], 1u);
    EXPECT_FALSE(overflow);
}

TEST(BigUintTriplePlusOne, OverflowSingleLimb) {
    // Use 1-limb BigUint (64 bits). n close to max → 3n+1 overflows.
    // n = 2^63, 3n+1 = 3*2^63 + 1 which doesn't fit in 64 bits.
    BigUint<1> a(1ULL << 63);
    bool overflow = a.TriplePlusOne();
    EXPECT_TRUE(overflow);
}

TEST(BigUintTriplePlusOne, OverflowTwoLimbs) {
    // n where all limbs are large enough that 3n+1 overflows 128 bits.
    // n = 2^127 (limbs[1] = 2^63). 3n+1 = 3*2^127 + 1 > 2^128.
    BigUint<2> a(0);
    a.limbs[1] = (1ULL << 63);
    bool overflow = a.TriplePlusOne();
    EXPECT_TRUE(overflow);
}

// ═══════════════════════════════════════════════════════════════════
// ToString
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintToString, Zero) {
    EXPECT_EQ(BigUint<2>(0).ToString(), "0");
}

TEST(BigUintToString, One) {
    EXPECT_EQ(BigUint<2>(1).ToString(), "1");
}

TEST(BigUintToString, SmallNumber) {
    EXPECT_EQ(BigUint<2>(12345).ToString(), "12345");
}

TEST(BigUintToString, MaxUint64) {
    EXPECT_EQ(BigUint<2>(UINT64_MAX).ToString(), "18446744073709551615");
}

TEST(BigUintToString, PowerOfTwo64) {
    // 2^64 = 18446744073709551616
    BigUint<2> a(0);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToString(), "18446744073709551616");
}

TEST(BigUintToString, LargeValue) {
    // 2^64 + 1 = 18446744073709551617
    BigUint<2> a(1);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToString(), "18446744073709551617");
}

TEST(BigUintToString, Max128Bit) {
    // 2^128 - 1 = 340282366920938463463374607431768211455
    BigUint<2> a(UINT64_MAX);
    a.limbs[1] = UINT64_MAX;
    EXPECT_EQ(a.ToString(), "340282366920938463463374607431768211455");
}

// ═══════════════════════════════════════════════════════════════════
// ToHexString
// ═══════════════════════════════════════════════════════════════════

TEST(BigUintToHexString, Zero) {
    EXPECT_EQ(BigUint<2>(0).ToHexString(), "0x0");
}

TEST(BigUintToHexString, One) {
    EXPECT_EQ(BigUint<2>(1).ToHexString(), "0x1");
}

TEST(BigUintToHexString, SmallNumber) {
    EXPECT_EQ(BigUint<2>(255).ToHexString(), "0xff");
}

TEST(BigUintToHexString, Hex0x3039) {
    EXPECT_EQ(BigUint<2>(12345).ToHexString(), "0x3039");
}

TEST(BigUintToHexString, MaxUint64) {
    EXPECT_EQ(BigUint<2>(UINT64_MAX).ToHexString(), "0xffffffffffffffff");
}

TEST(BigUintToHexString, PowerOfTwo64) {
    // 2^64
    BigUint<2> a(0);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToHexString(), "0x10000000000000000");
}

TEST(BigUintToHexString, LargeValue) {
    // 2^64 + 1
    BigUint<2> a(1);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToHexString(), "0x10000000000000001");
}

TEST(BigUintToHexString, Max128Bit) {
    // 2^128 - 1
    BigUint<2> a(UINT64_MAX);
    a.limbs[1] = UINT64_MAX;
    EXPECT_EQ(a.ToHexString(), "0xffffffffffffffffffffffffffffffff");
}

// ── ToPowerString ──────────────────────────────────────────────────

TEST(BigUintToPowerString, Zero) {
    EXPECT_EQ(BigUint<2>(0).ToPowerString(), "0");
}

TEST(BigUintToPowerString, One) {
    EXPECT_EQ(BigUint<2>(1).ToPowerString(), "1");
}

TEST(BigUintToPowerString, Two) {
    EXPECT_EQ(BigUint<2>(2).ToPowerString(), "2^1");
}

TEST(BigUintToPowerString, Three) {
    EXPECT_EQ(BigUint<2>(3).ToPowerString(), "2^1 + 1");
}

TEST(BigUintToPowerString, PowerOfTwo) {
    // 1024 = 2^10
    EXPECT_EQ(BigUint<2>(1024).ToPowerString(), "2^10");
}

TEST(BigUintToPowerString, PowerOfTwoPlusOffset) {
    // 1024 + 37 = 1061
    EXPECT_EQ(BigUint<2>(1061).ToPowerString(), "2^10 + 37");
}

TEST(BigUintToPowerString, LargeNumber) {
    // 1048576 = 2^20
    EXPECT_EQ(BigUint<2>(1048576).ToPowerString(), "2^20");
}

TEST(BigUintToPowerString, LargeWithOffset) {
    // 2^20 + 500000
    EXPECT_EQ(BigUint<2>(1048576 + 500000).ToPowerString(), "2^20 + 500000");
}

TEST(BigUintToPowerString, PowerOfTwo64) {
    // 2^64
    BigUint<2> a(0);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToPowerString(), "2^64");
}

TEST(BigUintToPowerString, PowerOfTwo64PlusSmall) {
    // 2^64 + 42
    BigUint<2> a(42);
    a.limbs[1] = 1;
    EXPECT_EQ(a.ToPowerString(), "2^64 + 42");
}

TEST(BigUintToPowerString, HighBitWithLargeRemainder) {
    // 2^65 + 2^64 = 3 * 2^64  →  bit 65 set, remainder = 2^64
    // remainder doesn't fit in u64, so shown as hex
    BigUint<2> a(0);
    a.limbs[1] = 3;  // bits 64 and 65 set
    EXPECT_EQ(a.ToPowerString(), "2^65 + 0x10000000000000000");
}
