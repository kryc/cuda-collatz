#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <ostream>

#ifdef __CUDACC__
#define BU_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define BU_HOST_DEVICE inline
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
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) limbs[i] = 0;
    }

    /// Construct from a single uint64_t value.
    BU_HOST_DEVICE explicit BigUint(uint64_t V) {
        limbs[0] = V;
        #pragma unroll
        for (int i = 1; i < N_LIMBS; ++i) limbs[i] = 0;
    }

    // ── Queries ──

    /// True if value == 1.
    BU_HOST_DEVICE bool IsOne() const {
        if (limbs[0] != 1) return false;
        #pragma unroll
        for (int i = 1; i < N_LIMBS; ++i)
            if (limbs[i] != 0) return false;
        return true;
    }

    /// True if value == 0.
    BU_HOST_DEVICE bool IsZero() const {
        #pragma unroll
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

    /// Number of bits needed to represent this value (floor(log2(n)) + 1).
    /// Returns 0 when the value is zero.
    BU_HOST_DEVICE int BitLength() const {
        for (int i = N_LIMBS - 1; i >= 0; --i) {
            if (limbs[i] != 0) {
#ifdef __CUDA_ARCH__
                return i * 64 + 64 - __clzll(static_cast<long long>(limbs[i]));
#else
                return i * 64 + 64 - __builtin_clzll(limbs[i]);
#endif
            }
        }
        return 0;
    }

    /// Count the number of trailing zero bits.
    BU_HOST_DEVICE int CountTrailingZeros() const {
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            if (limbs[i] != 0) {
#ifdef __CUDA_ARCH__
                return i * 64 + __ffsll(static_cast<long long>(limbs[i])) - 1;
#else
                return i * 64 + __builtin_ctzll(limbs[i]);
#endif
            }
        }
        return N_LIMBS * 64;
    }

    // ── Comparison ──

    BU_HOST_DEVICE bool operator==(const BigUint& O) const {
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i)
            if (limbs[i] != O.limbs[i]) return false;
        return true;
    }

    BU_HOST_DEVICE bool operator!=(const BigUint& O) const {
        return !(*this == O);
    }

    /// True if *this < other.  Compare from most-significant limb down.
    BU_HOST_DEVICE bool operator<(const BigUint& O) const {
        #pragma unroll
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
        #pragma unroll
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
        #pragma unroll
        for (int i = 0; i < N_LIMBS && carry; ++i) {
            uint64_t sum = limbs[i] + carry;
            carry = (sum < limbs[i]) ? 1ULL : 0ULL;
            limbs[i] = sum;
        }
        return carry;
    }

    /// Right-shift by 1 bit (divide by 2).
    BU_HOST_DEVICE void ShiftRight1() {
        #pragma unroll
        for (int i = 0; i < N_LIMBS - 1; ++i) {
            limbs[i] = (limbs[i] >> 1) | (limbs[i + 1] << 63);
        }
        limbs[N_LIMBS - 1] >>= 1;
    }

    /// Right-shift by N bits (divide by 2^N).
    BU_HOST_DEVICE void ShiftRightN(int N) {
        int limbShift = N / 64;
        int bitShift  = N % 64;
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            int src = i + limbShift;
            if (src < N_LIMBS) {
                limbs[i] = limbs[src] >> bitShift;
                if (bitShift > 0 && src + 1 < N_LIMBS) {
                    limbs[i] |= limbs[src + 1] << (64 - bitShift);
                }
            } else {
                limbs[i] = 0;
            }
        }
    }

    /// Left-shift by 1 bit (multiply by 2). Returns the bit shifted out.
    BU_HOST_DEVICE uint64_t ShiftLeft1() {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            uint64_t newCarry = limbs[i] >> 63;
            limbs[i] = (limbs[i] << 1) | carry;
            carry = newCarry;
        }
        return carry;
    }

    /// Left-shift by N bits (multiply by 2^N). Bits shifted out are lost.
    BU_HOST_DEVICE void ShiftLeftN(int N) {
        int limbShift = N / 64;
        int bitShift  = N % 64;
        #pragma unroll
        for (int i = N_LIMBS - 1; i >= 0; --i) {
            int src = i - limbShift;
            if (src >= 0) {
                limbs[i] = limbs[src] << bitShift;
                if (bitShift > 0 && src - 1 >= 0) {
                    limbs[i] |= limbs[src - 1] >> (64 - bitShift);
                }
            } else {
                limbs[i] = 0;
            }
        }
    }

    /// Set a single bit at position Pos (0 = LSB).
    BU_HOST_DEVICE void SetBit(int Pos) {
        int limb = Pos / 64;
        int bit  = Pos % 64;
        if (limb < N_LIMBS) {
            limbs[limb] |= (1ULL << bit);
        }
    }

    /// Compute 3*n + 1 in-place. Returns true if overflow occurred
    /// (result doesn't fit in N_LIMBS limbs).
    BU_HOST_DEVICE bool TriplePlusOne() {
        uint64_t carry = 1;  // the +1
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            uint64_t x = limbs[i];
#ifdef __CUDA_ARCH__
            uint64_t lo = x * 3ULL;
            uint64_t hi = __umul64hi(x, 3ULL);
            uint64_t sum = lo + carry;
            hi += (sum < lo) ? 1ULL : 0ULL;
            limbs[i] = sum;
            carry = hi;
#else
            __uint128_t wide = (__uint128_t)x * 3 + carry;
            limbs[i] = static_cast<uint64_t>(wide);
            carry = static_cast<uint64_t>(wide >> 64);
#endif
        }
        return carry != 0;
    }

    /// Fused 3n+1 then right-shift all trailing zeros.
    /// Assumes n is odd. After this call the value is odd (or 1).
    /// Returns the number of trailing zeros shifted out.
    /// Sets `overflow` to true if 3n+1 exceeds N_LIMBS capacity.
    BU_HOST_DEVICE int TriplePlusOneAndShift(bool& overflow) {
        // Step 1: compute 3n+1
        uint64_t carry = 1;  // the +1
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            uint64_t x = limbs[i];
#ifdef __CUDA_ARCH__
            uint64_t lo = x * 3ULL;
            uint64_t hi = __umul64hi(x, 3ULL);
            uint64_t sum = lo + carry;
            hi += (sum < lo) ? 1ULL : 0ULL;
            limbs[i] = sum;
            carry = hi;
#else
            __uint128_t wide = (__uint128_t)x * 3 + carry;
            limbs[i] = static_cast<uint64_t>(wide);
            carry = static_cast<uint64_t>(wide >> 64);
#endif
        }
        if (carry != 0) {
            overflow = true;
            return 0;
        }
        overflow = false;

        // Step 2: count trailing zeros
        int tz = 0;
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) {
            if (limbs[i] != 0) {
#ifdef __CUDA_ARCH__
                tz = i * 64 + __ffsll(static_cast<long long>(limbs[i])) - 1;
#else
                tz = i * 64 + __builtin_ctzll(limbs[i]);
#endif
                break;
            }
        }

        // Step 3: shift right by tz
        if (tz > 0) {
            int limbShift = tz / 64;
            int bitShift  = tz % 64;
            #pragma unroll
            for (int i = 0; i < N_LIMBS; ++i) {
                int src = i + limbShift;
                if (src < N_LIMBS) {
                    limbs[i] = limbs[src] >> bitShift;
                    if (bitShift > 0 && src + 1 < N_LIMBS) {
                        limbs[i] |= limbs[src + 1] << (64 - bitShift);
                    }
                } else {
                    limbs[i] = 0;
                }
            }
        }

        return tz;
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

    /// Format as "2^x + y" for human-readable magnitude display (host-only).
    std::string ToPowerString() const {
        if (IsZero()) return "0";
        if (IsOne())  return "1";

        // Find the most-significant non-zero limb.
        int msLimb = N_LIMBS - 1;
        while (msLimb > 0 && limbs[msLimb] == 0) --msLimb;

        // Highest set bit position.
        int highBit = msLimb * 64 + (63 - __builtin_clzll(limbs[msLimb]));

        // Compute remainder = value - 2^highBit (clear the top bit).
        BigUint remainder = *this;
        remainder.limbs[msLimb] &= ~(1ULL << (highBit % 64));

        if (remainder.IsZero()) {
            return "2^" + std::to_string(highBit);
        }

        // Check if remainder fits in a single limb.
        bool fitsInU64 = true;
        for (int i = 1; i < N_LIMBS; ++i) {
            if (remainder.limbs[i] != 0) { fitsInU64 = false; break; }
        }

        std::string base = "2^" + std::to_string(highBit) + " + ";
        if (fitsInU64) {
            return base + std::to_string(remainder.limbs[0]);
        }
        return base + remainder.ToHexString();
    }

    /// Format as "2^x (P%)" showing percentage progress through the current
    /// power-of-2 range [2^x, 2^(x+1)) (host-only).
    std::string ToProgressString() const {
        if (IsZero()) return "0";
        if (IsOne())  return "1";

        int msLimb = N_LIMBS - 1;
        while (msLimb > 0 && limbs[msLimb] == 0) --msLimb;
        int highBit = msLimb * 64 + (63 - __builtin_clzll(limbs[msLimb]));

        // remainder = value - 2^highBit
        BigUint remainder = *this;
        remainder.limbs[msLimb] &= ~(1ULL << (highBit % 64));

        if (remainder.IsZero()) {
            return "2^" + std::to_string(highBit) + " (0%)";
        }

        // Percentage = remainder / 2^highBit * 100
        // Shift remainder so top bits give us precision, then divide.
        // We want ~3 digits of precision, so shift up 10 bits (×1024),
        // then take the top limb-and-a-bit after aligning.
        // Simpler: use the top 53 bits of remainder and highBit.
        double pct;
        if (highBit <= 53) {
            // Everything fits in a double
            double rem = 0;
            for (int i = msLimb; i >= 0; --i)
                rem = rem * 18446744073709551616.0 + static_cast<double>(remainder.limbs[i]);
            double denom = static_cast<double>(1ULL << highBit);
            pct = rem / denom * 100.0;
        } else {
            // Use top bits: shift remainder right by (highBit - 53)
            BigUint shifted = remainder;
            int shiftAmt = highBit - 53;
            shifted.ShiftRightN(shiftAmt);
            double rem = static_cast<double>(shifted.limbs[0]);
            double denom = static_cast<double>(1ULL << 53);
            pct = rem / denom * 100.0;
        }

        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f%%", pct);
        return "2^" + std::to_string(highBit) + " (" + buf + ")";
    }

    /// Stream insertion operator (host-only, uses ToHexString).
    friend std::ostream& operator<<(std::ostream& Os, const BigUint& V) {
        return Os << V.ToHexString();
    }
};
