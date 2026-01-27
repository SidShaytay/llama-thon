#!/usr/bin/env bash
set -uo pipefail

toolbox run -c llama-rocm-6.4.4 -- llama-bench -fa 1 -ngl 99 -mmp 0 -m /mnt/data/projects/ai/models/hub/models--unsloth--gemma-3-4b-it-GGUF/snapshots/5a3566e716d80f709ed7b79817eaf7733d2a1fce/gemma-3-4b-it-UD-Q4_K_XL.gguf
#toolbox run -c llama-rocm-7.1.1 -- llama-bench -fa 1 -ngl 99 -mmp 0 -m /mnt/data/projects/ai/models/hub/models--unsloth--gemma-3-4b-it-GGUF/snapshots/5a3566e716d80f709ed7b79817eaf7733d2a1fce/gemma-3-4b-it-UD-Q4_K_XL.gguf
