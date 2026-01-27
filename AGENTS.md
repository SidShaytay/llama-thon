# AGENTS.md

This file provides comprehensive guidance to AI agents (Claude Code, etc.) when working with this repository.

## Project Overview

LLM benchmarking project for testing model inference performance using `llama-bench` across different GPU backends (ROCm versions, Vulkan) on AMD hardware (targeting Strix Halo). Results are stored as CSV and log files.

## Repository Structure

```
llama-thon/
├── llama-thon.sh               # Main benchmarking script
├── update_llama_cpp.py         # Helper to fetch/build llama.cpp variants
├── run-rocm-smoke-test.sh      # Quick ROCm sanity check
├── results/                    # Benchmark output (per-model subdirs)
│   ├── <model-name>/
│   │   ├── results.csv         # Aggregated results for this model
│   │   └── *.log               # Individual run logs
│   └── rerun_*.csv             # Auto-generated rerun files (if failures)
├── MODEL_RECOMMENDATIONS.md    # Analysis and model picks
├── AGENTS.md                   # This file (agent instructions)
├── CLAUDE.md                   # Points to AGENTS.md
└── README.md                   # Public documentation
```

## Running Benchmarks

Full run (Cartesian product of models x runners x bench_arg_sets):
```bash
./llama-thon.sh
```

Rerun specific failures from CSV:
```bash
./llama-thon.sh results/rerun_20260125_143022.csv
```

Results saved to `results/<model-name>/`:
- `results.csv` - Parsed benchmark data (written incrementally, survives interruption)
- `YYYYMMDD_HHMMSS_backend-quant-args.log` - Raw llama-bench output per run
- `rerun_YYYYMMDD_HHMMSS.csv` - Auto-generated rerun file (only if failures occur)

## Script Structure

**llama-thon.sh** is organized into sections:

1. **CONFIGURATION** (top) - Arrays to modify:
   - `models[]` - GGUF model paths to benchmark
   - `runners[]` - llama-bench invocations (toolbox or local)
   - `bench_arg_sets[]` - Different benchmark parameter sets

2. **HELPER FUNCTIONS** - `run_in_dir`, `sanitize_for_filename`, GTT memory monitoring

3. **SYSTEM INFO FUNCTIONS** - OS, kernel, firmware detection

4. **LOG PARSING FUNCTIONS** - Extract build version, pp/tg values from llama-bench output

5. **INFERENCE FUNCTIONS** - Derive backend, build type, model name, quantization from paths

6. **CSV FUNCTIONS** - Write/append with immediate `sync` for durability

7. **BENCHMARK EXECUTION** - Core function `run_single_benchmark` executes one (model, runner, bench_args) tuple:
   - Monitors GTT memory usage during benchmark
   - Captures wall clock duration
   - Writes results to CSV immediately (crash-safe)
   - Logs failures to rerun CSV automatically

8. **MAIN EXECUTION** - Two modes:
   - Full run: Triple loop (models x runners x bench_arg_sets)
   - Rerun mode: Read tuples from CSV file and execute each
   - Displays failure count and rerun command at end

## Key Concepts

**Runners**: llama-bench via toolbox containers (`toolbox run -c <container> -- llama-bench`) or direct paths. Examples:
```bash
"toolbox run -c llama-rocm-6.4.4 -- llama-bench"
"toolbox run -c llama-cpp-rocm-7.250122 -- /usr/bin/llama-bench"
"/home/sid/Projects/ai/llama.cpp-b7813-vulkan-prebuilt/llama-bench"
```

**Models**: GGUF files; name and quantization inferred from HuggingFace-style paths. Examples:
```
/mnt/data/projects/ai/models/hub/models--unsloth--GLM-4.7-Flash-GGUF/.../GLM-4.7-Flash-UD-Q4_K_XL.gguf
```
Model name: "GLM-4.7-Flash (Unsloth)", Quantization: "Q4_K_XL"

**Backends**: ROCm (6.4.4, 7.1.1, 7.2, nightlies) and Vulkan (radv, amdvlk)

**bench_arg_sets**: Each string is a complete set of llama-bench args:
```bash
"-fa 1 -ngl 99 -mmp 0"                                    # Standard (pp512, tg128)
"-fa 1 -ngl 99 -mmp 0 -p 2048 -n 512 -d 32768 -ub 2048"   # Long context
```

## CSV Output Columns

`OS,Kernel,linux-firmware,llama-cpp,built,backend,model,quantization,pp_size,pp,tg_size,tg,gtt_peak_mb,duration_sec,params,notes`

| Column | Description | Example |
|--------|-------------|---------|
| OS | Operating system | Fedora Linux 43 |
| Kernel | Kernel version | 6.18.3-200.fc43.x86_64 |
| linux-firmware | Firmware package version | 20260110-1.fc43.noarch |
| llama-cpp | llama.cpp build number | b7823 |
| built | Build type | toolbox, prebuilt, local |
| backend | GPU backend | rocm-7.2, vulkan-radv |
| model | Model name (inferred) | GLM-4.7-Flash (Unsloth) |
| quantization | Quantization level | Q4_K_M, Q6_K, IQ3_XXS |
| pp_size | Prompt processing batch size | 512, 2048 |
| pp | Prompt processing tokens/sec | 683.42 |
| tg_size | Token generation batch size | 128, 512 |
| tg | Token generation tokens/sec | 45.21 |
| gtt_peak_mb | Peak GTT memory increase (MB) | 12800 |
| duration_sec | Wall clock time (seconds) | 45.2 |
| params | llama-bench arguments used | -fa 1 -ngl 99 -mmp 0 |
| notes | Failures, warnings, log path | |

## Rerun CSV Format

Auto-generated when failures occur, or create manually:
```csv
model,runner,bench_args
/path/to/model.gguf,toolbox run -c llama-rocm-7.1.1 -- llama-bench,-fa 1 -ngl 99 -mmp 0
```

---

## update_llama_cpp.py

Helper script that fetches and builds llama.cpp variants for different acceleration backends.

### Variants

1. **Prebuilt Vulkan**
   - Source: https://github.com/ggml-org/llama.cpp/releases/latest
   - Asset: Linux Vulkan x64 archive
   - Install dir: `~/Apps/llama.cpp-<tag>-vulkan-prebuilt`

2. **Prebuilt ROCm**
   - Source: https://github.com/lemonade-sdk/llamacpp-rocm/releases/latest
   - Asset: Linux ROCm `gfx1151` x64 archive
   - Install dir: `~/Apps/llama.cpp-<tag>-rocm-prebuilt`
   - Version tag resolved from release notes' llama.cpp commit hash

3. **Build from local source (Vulkan)**
   - Source path: `/home/sid/Projects/ai/llama.cpp`
   - Install dir: `~/Apps/llama.cpp-<count>-vulkan`
   - Requires Vulkan SDK headers, loader, and `glslc`

4. **Build via toolbox Dockerfile (Vulkan)**
   - Dockerfile: Local or fallback from GitHub
   - Container tag: `llama.cpp-<count>-vulkan`

5. **Build via toolbox Dockerfile (ROCm)**
   - Dockerfile: Local or fallback from GitHub
   - Container tag: `llama.cpp-<count>-rocm-7nightlies`

6. **Update toolboxes**
   - Repo: `/home/sid/Projects/ai/amd-strix-halo-toolboxes`
   - Runs `git pull` then `./refresh-toolboxes all`

### Usage

```bash
# Prebuilt (latest release assets)
./update_llama_cpp.py vulkan-prebuilt
./update_llama_cpp.py rocm-prebuilt

# Build from local source
./update_llama_cpp.py vulkan-build

# Build via toolbox Dockerfiles
./update_llama_cpp.py vulkan-toolbox
./update_llama_cpp.py rocm-toolbox

# Update toolboxes
./update_llama_cpp.py update-toolboxes

# Run all steps
./update_llama_cpp.py all

# Overwrite existing install dirs
./update_llama_cpp.py --force vulkan-prebuilt
```

### Configuration

- Install base: `~/Apps`
- Container runtime: `podman` (fallback: `docker`), override with `LLAMA_CPP_CONTAINER_RUNTIME`
- Toolbox builds use local context when available; remote Dockerfiles are fallback

---

## Design Goals

1. Keep it simple, easy to read yet modular
2. Crash-safe: results written incrementally with `sync`
3. Reproducible: all system info captured in CSV
4. Resumable: automatic rerun CSV generation on failures
