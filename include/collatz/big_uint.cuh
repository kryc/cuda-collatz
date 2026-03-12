#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

#ifdef __CUDA_ARCH__
#define BU_HOST_DEVICE __host__ __device__
#else
#define BU_HOST_DEVICE
#endif

// Fallback: always mark as host+device when compiled by nvcc
#ifdef __CUDACC__
#undef BU_HOST_DEVICE
#define BU_HOST_DEVICE __host__ __device__
#endif

/// Default number of 64-bit limbs. 2 limbs = 128 bits.
#ifndef COLLATZ_N_LIMBS
#define COLLATZ_N_LIMBS 2
#endif

/// A fixed-width unsigned integer stored as N 64-bit limbs (little-endian).
/// limbs[0] is the least-significant word.
template <int N_LIMBS = COLLATZ_N_LIMBS>
struct BigUint {
    uint64_t limbs[N_LIMBS];

    // ── Construction ──

    /// Default: zero-initialised.
    BU_HOST_DEVICE BigUint() {
        for (int i = 0; i < N_LIMBS; ++i) limbs[i] = 0;
    }

    /// Construct from a single uint64_t value.
    BU_HOST_DEVICE explicit BigUint(uint64_t V) {
        limbs[0] = V;
        for (int i = 1; i < N_LIMBS; ++i) limbs[i] = 0;
    }

    // ── Queries ──

    /// True if value == 1.
    BU_HOST_DEVICE bool IsOne() const {
        if (limbs[0] != 1) return false;
        for (int i = 1; i < N_LIMBS; ++i)
            if (limbs[i] != 0) return false;
        return true;
    }

    /// True if value == 0.
    BU_HOST_DEVICE bool IsZero() const {
        for (int i = 0; i < N_LIMBS; ++i)
            if (limbs[i] != 0) return false;
        return true;
    }

    /// True if even (bit 0 clear).
    BU_HOST_DEVICE bool IsEven() const {
        return (limbs[0] & 1) == 0;
    }

    /// True if odd (bit 0 set).
    BU_HOST_DEVICE bool IsOdd() const {
        return (limbs[0] & 1) == 1;
    }

    // ── Comparison ──

    BU_HOST_DEVICE bool operator==(const BigUint& O) const {
        for (int i = 0; i < N_LIMBS; ++i)
            if (limbs[i] != O.limbs[i]) return false;
        return true;
    }

    BU_HOST_DEVICE bool operator!=(const BigUint& O) const {
        return !(*this == O);
    }

    /// True if *this < other.  Compare from most-significant limb down.
    BU_HOST_DEVICE bool operator<(const BigUint& O) const {
        for (int i = N_LIMBS - 1; i >= 0; --i) {
            if (limbs[i] < O.limbs[i]) return true;
            if (limbs[i] > O.limbs[i]) return false;
        }
        return false;  // equal
    }

    BU_HOST_DEVICE bool operator>(const BigUint& O) const {
        return O < *this;
    }

    BU_HOST_DEVICE bool operator<=(const BigUint& O) const {
        return !(O < *this);
    }

    BU_HOST_DEVICE bool operator>=(const BigUint& O) const {
        return !(*this < O);
    }

    // ── Arithmetic ──

    /// Add another BigUint. Returns carry out (0 or 1).
    BU_HOST_DEVICE uint64_t Add(const BigUint& O) {
        uint64_t carry = 0;
        for (int i = 0; i < N_LIMBS; ++i) {
            uint64_t a = limbs[i];
            uint64_t b = O.limbs[i];
            uint64_t sum = a + b;
            uint64_t c1 = (sum < a) ? 1ULL : 0ULL;
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
            limbs[i] = sum2;
            carry = c1 + c2;
        }
        return carry;
    }

    /// Add a small uint64_t value. Returns carry out (0 or 1).
    BU_HOST_DEVICE uint64_t AddU64(uint64_t V) {
        uint64_t carry = V;
        for (int i = 0; i < N_LIMBS && carry; ++i) {
            uint64_t sum = limbs[i] + carry;
            carry = (sum < limbs[i]) ? 1ULL : 0ULL;
            limbs[i] = sum;
        }
        return carry;
    }

    /// Right-shift by 1 bit (divide by 2).
    BU_HOST_DEVICE void ShiftRight1() {
        for (int i = 0; i < N_LIMBS - 1; ++i) {
            limbs[i] = (limbs[i] >> 1) | (limbs[i + 1] << 63);
        }
        limbs[N_LIMBS - 1] >>= 1;
    }

    /// Left-shift by 1 bit (multiply by 2). Returns the bit shifted out.
    BU_HOST_DEVICE uint64_t ShiftLeft1() {
        uint64_t carry = 0;
        for (int i = 0; i < N_LIMBS; ++i) {
            uint64_t newCarry = limbs[i] >> 63;
            limbs[i] = (limbs[i] << 1) | carry;
            carry = newCarry;
        }
        return carry;
    }

    /// Compute 3*n + 1 in-place. Returns true if overflow occurred
    /// (result doesn't fit in N_LIMBS limbs).
    BU_HOST_DEVICE bool TriplePlusOne() {
        // Compute 3n+1 = (n << 1) + n + 1
        BigUint orig = *this;
        uint64_t shiftOut = ShiftLeft1();          // *this = n << 1
        uint64_t addCarry = Add(orig);                // *this = (n << 1) + n = 3n
        uint64_t plus1Carry = AddU64(1);             // *this = 3n + 1
        return (shiftOut | addCarry | plus1Carry) != 0;
    }

    // ── Conversion ──

    /// Convert to uint64_t. Only valid if value fits; no check performed.
    BU_HOST_DEVICE uint64_t ToUint64() const {
        return limbs[0];
    }

    /// Convert to decimal string (host-only).
    std::string ToString() const {
        if (IsZero()) return "0";

        // Work on a copy; repeatedly divide by 10.
        BigUint tmp = *this;
        std::string result;

        while (!tmp.IsZero()) {
            // Divide tmp by 10, collect remainder.
            uint64_t remainder = 0;
            for (int i = N_LIMBS - 1; i >= 0; --i) {
                // Combine remainder from previous limb with current limb.
                // We need to compute (remainder * 2^64 + limbs[i]) / 10.
                // Use __uint128_t on host for this.
                __uint128_t val = ((__uint128_t)remainder << 64) | tmp.limbs[i];
                tmp.limbs[i] = (uint64_t)(val / 10);
                remainder = (uint64_t)(val % 10);
            }
            result.push_back('0' + (char)remainder);
        }

        // Reverse to get most-significant digit first.
        std::reverse(result.begin(), result.end());
        return result;
    }

    /// Convert to hexadecimal string with "0x" prefix (host-only).
    std::string ToHexString() const {
        if (IsZero()) return "0x0";

        // Find the most-significant non-zero limb.
        int msLimb = N_LIMBS - 1;
        while (msLimb > 0 && limbs[msLimb] == 0) --msLimb;

        static const char kHexDigits[] = "0123456789abcdef";
        std::string result = "0x";

        // First limb: no leading zeros.
        uint64_t v = limbs[msLimb];
        bool started = false;
        for (int b = 60; b >= 0; b -= 4) {
            int nib = (v >> b) & 0xF;
            if (nib || started) {
                result.push_back(kHexDigits[nib]);
                started = true;
            }
        }

        // Remaining limbs: full 16-digit hex, zero-padded.
        for (int i = msLimb - 1; i >= 0; --i) {
            v = limbs[i];
            for (int b = 60; b >= 0; b -= 4) {
                result.push_back(kHexDigits[(v >> b) & 0xF]);
            }
        }
        return result;
    }
};
