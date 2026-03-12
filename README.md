# Collatz Chain Finder

A CUDA-accelerated tool for computing [Collatz (3x+1)](https://en.wikipedia.org/wiki/Collatz_conjecture) chain lengths and peak values, with big integer support for numbers beyond 2⁶⁴.

## Building

Requires CUDA Toolkit ≥ 12.4, CMake ≥ 3.18, and a C++17 compiler.

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=clang++
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
| `--max-steps N` | 0 (off) | Max steps before flagging chain as non-converging |
| `--output FILE` | collatz.csv | CSV output file |
| `--divergent FILE` | collatz_divergent.csv | File for non-converging chains |
| `--checkpoint FILE` | collatz.ckpt | Checkpoint file |
| `--resume` | — | Resume from last checkpoint |

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

Press **Ctrl-C** to stop gracefully — a checkpoint is written so you can `--resume` later.

## Output

Results are appended to a CSV file:

```
start_n,chain_length,max_value
837799,524,2974984576
871,178,190996
```

### Non-converging chains

When `--max-steps` is set, chains that don't reach 1 within the limit are written to a separate file (default `collatz_divergent.csv`):

```
start_n,steps_completed,current_value,max_value
27,50,566,1780
```

Columns: starting number, how many steps were completed, value of *n* when the limit was hit, and the peak value seen during the chain.

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

77 tests covering big integer arithmetic, Collatz kernel correctness, and max-steps limiting.
