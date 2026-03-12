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
| `--output FILE` | collatz.csv | CSV output file |
| `--divergent FILE` | collatz_divergent.csv | File for non-converging chains |
| `--checkpoint FILE` | collatz.ckpt | Checkpoint file |
| `--resume` | — | Resume from last checkpoint |
| `--fresh` | — | Discard existing checkpoint and start fresh |
| `--help` | — | Show usage help |

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
cmake -B build -DCMAKE_CXX_FLAGS="-DCOLLATZ_N_LIMBS=4"  # 256 bits
```

If a number overflows during computation, it is flagged on stderr and skipped.

## Tests

```bash
cmake --build build -j$(nproc)
./build/collatz_tests
```

99 tests covering big integer arithmetic (construction, queries, comparison, add, shift, triple-plus-one, toString, toHexString) and Collatz kernel correctness (chain lengths, max values, batching, overflow detection, max-steps limiting).
