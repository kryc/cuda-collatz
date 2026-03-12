# Collatz Chain Finder

A CUDA-accelerated tool for computing [Collatz (3x+1)](https://en.wikipedia.org/wiki/Collatz_conjecture) chain lengths and peak values, with big integer support for numbers beyond 2⁶⁴.

## Building

Requires CUDA Toolkit ≥ 12.4, CMake ≥ 3.18, and a C++20 compiler. Use **clang/clang++** as the host compiler — GCC 15 causes linker issues with GTest.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j$(nproc)
```

### Build options

| CMake variable | Default | Description |
|---|---|---|
| `CUDA_ARCH` | `90` | CUDA compute capability (e.g. `80` for A100, `90` for H100/RTX 5080) |
| `COLLATZ_N_LIMBS` | `2` | Number of 64-bit limbs for big integers (2 = 128-bit, 4 = 256-bit) |

```bash
# Example: build for Ada Lovelace (SM 89) with 256-bit integers
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCUDA_ARCH=89 -DCOLLATZ_N_LIMBS=4
```

## Usage

```bash
./build/collatz [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--start N` | 1 | First number to test (decimal, hex `0x…`, or `2^N`) |
| `--end N` | ∞ | Last number to test (0 = run forever) |
| `--batch-size N` | 1048576 | Numbers per GPU batch |
| `--min-chain N` | 1000 | Minimum chain length to log |
| `--max-steps N` | 1000000 | Max steps before flagging chain as non-converging (0 = unlimited) |
| `--no-odd-only` | — | Test all numbers instead of only odd numbers |
| `--output FILE` | collatz.csv | CSV output file |
| `--divergent FILE` | collatz_divergent.csv | File for non-converging chains |
| `--checkpoint FILE` | collatz.ckpt | Checkpoint file |
| `--resume` | — | Resume from last checkpoint |
| `--fresh` | — | Discard existing checkpoint and start fresh |
| `--help` | — | Show usage help |

By default, only odd numbers are tested — every even number trivially halves down to an odd number with a shorter chain, so skipping them doubles throughput with no loss of interesting results. Pass `--no-odd-only` if you need to test every number.

### Examples

Search the first billion numbers for long chains:
```bash
./build/collatz --start 1 --end 1000000000
```

Log everything from 1 to 1000:
```bash
./build/collatz --start 1 --end 1000 --min-chain 0
```

Resume an interrupted run:
```bash
./build/collatz --resume
```

Discard a previous checkpoint and start over:
```bash
./build/collatz --fresh --start 1
```

Press **Ctrl-C** to stop gracefully — a checkpoint is written so you can `--resume` later.

### Checkpoint safety

If a checkpoint file exists from a previous run, the program will **refuse to start** unless you explicitly pass `--resume` or `--fresh`. This prevents accidentally losing progress.

## Output

Results are appended to a CSV file with values in hexadecimal:

```
start_n,chain_length,max_value
cc7c7,524,b14e4100
367,178,2E9F0
```

The periodic status line on stderr also shows the current longest chain length found so far.

### Non-converging chains

When `--max-steps` is set, chains that don't reach 1 within the limit are written to a separate file (default `collatz_divergent.csv`):

```
start_n,steps_completed,current_value,max_value
1b,50,236,6f4
```

Columns: starting number (hex), how many steps were completed, value of *n* when the limit was hit (hex), and the peak value seen during the chain (hex).

## Big Integers

Numbers are represented as fixed-width unsigned integers using 64-bit limbs. The default is 128 bits (2 limbs), supporting values up to ~3.4×10³⁸. Recompile with a different limb count for wider arithmetic:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCOLLATZ_N_LIMBS=4  # 256 bits
```

If a number overflows during computation, it is flagged on stderr and skipped.

## Tests

```bash
cmake --build build -j$(nproc)
./build/collatz_tests
```

147 tests covering:
- **BigUint** — construction, queries, comparison, add, shift, shift-N, set-bit, triple-plus-one, toString, toHexString, toPowerString, stream operator, copy semantics, `BigUint<4>` multi-limb operations
- **Collatz kernel** — chain lengths, max values, batching, overflow detection, max-steps limiting, starts beyond 2⁶⁴, multi-block dispatch, odd-only mode

## Startup

On launch the program prints GPU device info (name, compute capability, SM count, clock, memory) and validates all CLI arguments before starting computation. All CUDA API calls are checked — any error prints the failing call with file/line and exits immediately.
