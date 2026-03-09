#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# CONFIGURATION
#
# Intentionally kept as absolute paths and commentable lines for 
# visual directness
# ============================================================================

# Models to benchmark
models=(
	# Small
#	"/mnt/data/projects/ai/models/hub/models--unsloth--gemma-3-4b-it-GGUF/snapshots/5a3566e716d80f709ed7b79817eaf7733d2a1fce/gemma-3-4b-it-UD-Q4_K_XL.gguf"
#	"/mnt/data/projects/ai/models/hub/models--unsloth--gemma-3-4b-it-GGUF/snapshots/5a3566e716d80f709ed7b79817eaf7733d2a1fce/gemma-3-4b-it-UD-Q6_K_XL.gguf"
#	"/mnt/data/projects/ai/models/hub/models--unsloth--gemma-3-4b-it-GGUF/snapshots/5a3566e716d80f709ed7b79817eaf7733d2a1fce/gemma-3-4b-it-UD-Q8_K_XL.gguf"

	# Medium
#	"/mnt/data/projects/ai/models/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/66205ea11ef638d83850389bd921e6c199f7504e/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
#    "/mnt/data/projects/ai/models/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/66205ea11ef638d83850389bd921e6c199f7504e/GLM-4.7-Flash-Q6_K.gguf"
#    "/mnt/data/projects/ai/models/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/66205ea11ef638d83850389bd921e6c199f7504e/GLM-4.7-Flash-Q8_0.gguf"
	/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf
#	/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3.5-27B-GGUF/snapshots/3221f178a6b842d04f1fb42f1c413534adcc0a6a/Qwen3.5-27B-UD-Q5_K_XL.gguf
#	/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-UD-Q5_K_XL.gguf

	# X-Large
	#"/mnt/data/projects/ai/models/hub/models--unsloth--MiniMax-M2.5-GGUF/snapshots/7c50dca0e5592483ad308ecffc876aecac725660/UD-IQ3_XXS/MiniMax-M2.5-UD-IQ3_XXS-00001-of-00003.gguf"
#	"/mnt/data/projects/ai/models/hub/models--unsloth--MiniMax-M2.5-GGUF/snapshots/7c50dca0e5592483ad308ecffc876aecac725660/UD-IQ2_XXS/MiniMax-M2.5-UD-IQ2_XXS-00001-of-00003.gguf"
#	"/mnt/data/projects/ai/models/hub/models--bartowski--stepfun-ai_Step-3.5-Flash-GGUF/snapshots/a2943ef56a0a240bdfb9a2feee97a775f612476d/stepfun-ai_Step-3.5-Flash-IQ3_XXS/stepfun-ai_Step-3.5-Flash-IQ3_XXS-00001-of-00002.gguf"

	# Misc
	#"/mnt/data/projects/ai/models/hub/models--ggml-org--gemma-3-4b-it-GGUF/snapshots/d0976223747697cb51e056d85c532013931fe52e/gemma-3-4b-it-Q8_0.gguf"
    #"/mnt/data/projects/ai/models/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots/d932fcea62f83e088d8f076a2cd2d7eb02dfa682/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    #"/mnt/data/projects/ai/models/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/e1dc459feff949ff451ce107337a2026daa80df8/gpt-oss-20b-mxfp4.gguf"
    #"/mnt/data/projects/ai/models/hub/models--unsloth--GLM-4.5-Air-GGUF/snapshots/506d64aa8c5cfe9dbbf00bc7a15739438f83204d/UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
    #"/mnt/data/projects/ai/models/hub/models--mradermacher--MiniMax-M2.1-i1-GGUF/snapshots/4620d187ee0fbbdb7cb1f423511a5b5b3920bb6b/MiniMax-M2.1.i1-IQ3_M.gguf"
    #"/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/a6adef130ffb23ddaf1a62fec9dced968c9bc482/Qwen3-8B-Q4_1.gguf"
    #"/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/7ce945e58ed3f09f9cf9c33a2122d86ac979b457/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    #"/mnt/data/projects/ai/models/hub/models--unsloth--Qwen3-Next-80B-A3B-Instruct-GGUF/snapshots/b43eaa6f3cd0111a82b344b3030c1859d551d636/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf"
)

# Runners (llama-bench invocations)
runners=(
	# Strix Halo Toolboxes
	"toolbox run -c llama-rocm-6.4.4 -- llama-bench"
	# fails on current kernel+rocm config
	#"toolbox run -c llama-rocm-7.1.1 -- llama-bench"
#	"toolbox run -c llama-rocm-7.2 -- llama-bench"
#	"toolbox run -c llama-rocm7-nightlies -- llama-bench"
#	"toolbox run -c llama-vulkan-radv -- llama-bench"

	# rocm 7 nightly - pre-built vs local builds
#	"toolbox run -c llama.cpp-b7840-rocm-7nightlies -- llama-bench"
	#"/home/sid/Apps/llama.cpp-b7837-rocm-prebuilt/llama-bench"

	# vulkan - pre-built vs local builds
#	"/home/sid/Apps/llama.cpp-b7839-vulkan-prebuilt/llama-bench"
#	"run_in_dir /home/sid/Apps/llama.cpp-b7840-vulkan ./llama-bench"
)

# Benchmark argument sets (each string is a complete set of args)
bench_arg_sets=(
#	"-fa 1 -ngl 99 -mmp 0 -p 2048 -n 512 -d 32768 -ub 2048"
	"-fa 1 -ngl 99 -mmp 0"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Run a command in a specific directory (for shared library access)
run_in_dir() {
	local dir="$1"
	shift || return 2
	if [[ ! -d "${dir}" ]]; then
		echo "run_in_dir: directory not found: ${dir}"
		echo "run_in_dir: directory not found: ${dir}" >&2
		return 2
	fi
	if [[ -z "${1:-}" ]]; then
		echo "run_in_dir: missing command"
		echo "run_in_dir: missing command" >&2
		return 2
	fi
	if [[ -x "${dir}/$1" ]]; then
		( cd "${dir}" && "$@" )
		return $?
	fi
	if [[ "$1" = /* && -x "$1" ]]; then
		( cd "${dir}" && "$@" )
		return $?
	fi
	echo "run_in_dir: executable not found: ${dir}/$1"
	echo "run_in_dir: executable not found: ${dir}/$1" >&2
	return 2
}

# Sanitize string for use in filenames
sanitize_for_filename() {
	local raw="$1"
	local sanitized="${raw//[^A-Za-z0-9.]/.}"
	printf '%s' "${sanitized}" | tr -s '.' | sed -E 's/^\.+|\.+$//g'
}

# Monitor GTT memory usage in background and return peak delta from baseline
# Returns peak GTT usage increase (process contribution) in MB
monitor_gtt_memory() {
	local pid_file="$1"
	local baseline="$2"
	local gtt_path="/sys/class/drm/card1/device/mem_info_gtt_used"
	local peak_delta=0
	local current delta

	[[ -f "${gtt_path}" ]] || { echo "0"; return; }

	while [[ -f "${pid_file}" ]]; do
		current=$(cat "${gtt_path}" 2>/dev/null || echo "0")
		delta=$((current - baseline))
		((delta > peak_delta)) && peak_delta=$delta
		sleep 0.1
	done

	# Convert bytes to MB
	echo $((peak_delta / 1024 / 1024))
}

# Get current GTT usage (baseline)
get_gtt_baseline() {
	local gtt_path="/sys/class/drm/card1/device/mem_info_gtt_used"
	[[ -f "${gtt_path}" ]] || { echo "0"; return; }
	cat "${gtt_path}" 2>/dev/null || echo "0"
}

# ============================================================================
# SYSTEM INFO FUNCTIONS
# ============================================================================

get_os_info() {
	if [[ -f /etc/os-release ]]; then
		# shellcheck source=/dev/null
		source /etc/os-release
		echo "${NAME} ${VERSION_ID}"
	else
		echo "unknown"
	fi
}

get_kernel_version() {
	uname -r
}

get_linux_firmware_version() {
	if command -v rpm &>/dev/null; then
		rpm -q linux-firmware 2>/dev/null | sed 's/linux-firmware-//' || echo "unknown"
	elif command -v dpkg &>/dev/null; then
		dpkg -l linux-firmware 2>/dev/null | awk '/linux-firmware/{print $3}' || echo "unknown"
	else
		echo "unknown"
	fi
}

# ============================================================================
# LOG PARSING FUNCTIONS
# ============================================================================

# Extract llama.cpp build version: "build: a14b960bc (7816)" -> "b7816"
extract_build_version() {
	local log="$1"
	local build_num
	build_num=$(echo "${log}" | grep -oP 'build: [a-f0-9]+ \(\K[0-9]+' | head -1)
	if [[ -n "${build_num}" ]]; then
		echo "b${build_num}"
	else
		echo "unknown"
	fi
}

# Extract performance value from llama-bench output
# Usage: extract_perf_value "$log" "pp512"
# Handles optional defrag suffix: "pp2048 @ d32768"
extract_perf_value() {
	local log="$1"
	local test_type="$2"
	local value
	# Match test type with optional " @ dNNNNN" suffix (defrag context size)
	value=$(echo "${log}" | grep -E "\|\s*${test_type}(\s+@\s+d[0-9]+)?\s*\|" | grep -oP '\|\s*\K[0-9]+\.[0-9]+(?=\s*±)' | tail -1)
	echo "${value:-UNKNOWN}"
}

# Detect test types present in output (e.g., "pp2048 tg128")
detect_test_types() {
	local log="$1"
	local pp_type tg_type
	pp_type=$(echo "${log}" | grep -oP '\|\s*\Kpp[0-9]+' | head -1)
	tg_type=$(echo "${log}" | grep -oP '\|\s*\Ktg[0-9]+' | head -1)
	echo "${pp_type:-pp512} ${tg_type:-tg128}"
}

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

# Infer backend from runner string
infer_backend() {
	local runner="$1"
	if [[ "${runner}" =~ llama-rocm-([0-9.]+) ]]; then
		echo "rocm-${BASH_REMATCH[1]}"
	elif [[ "${runner}" =~ rocm7-nightlies ]]; then
		echo "rocm7-nightlies"
	elif [[ "${runner}" =~ rocm-7\.([0-9]+) ]]; then
		echo "rocm-7.${BASH_REMATCH[1]}"
	elif [[ "${runner}" =~ rocm.*([0-9]{6}) ]]; then
		echo "rocm7-nightly-${BASH_REMATCH[1]}"
	elif [[ "${runner}" =~ vulkan-radv ]]; then
		echo "vulkan-radv"
	elif [[ "${runner}" =~ vulkan-amdvlk ]]; then
		echo "vulkan-amdvlk"
	elif [[ "${runner}" =~ vulkan ]]; then
		echo "vulkan"
	elif [[ "${runner}" =~ rocm ]]; then
		echo "rocm"
	else
		echo "unknown"
	fi
}

# Infer build type: "toolbox", "prebuilt", or "local"
infer_build_type() {
	local runner="$1"
	if [[ "${runner}" =~ ^toolbox ]]; then
		echo "toolbox"
	elif [[ "${runner}" =~ prebuilt ]]; then
		echo "prebuilt"
	else
		echo "local"
	fi
}

# Extract model name from HuggingFace-style path
extract_model_name() {
	local path="$1"
	local provider model_name
	if [[ "${path}" =~ models--([^/]+)--([^/]+) ]]; then
		provider="${BASH_REMATCH[1]}"
		model_name="${BASH_REMATCH[2]}"
		model_name="${model_name%-GGUF}"
		provider="$(echo "${provider}" | sed 's/\b\(.\)/\u\1/g')"
		echo "${model_name} (${provider})"
	else
		basename "${path}" .gguf
	fi
}

# Extract quantization from filename (Q4_K_M, IQ3_XXS, mxfp4, etc.)
extract_quantization() {
	local path="$1"
	local filename quant
	filename=$(basename "${path}")
	if [[ "${filename}" =~ ((UD-)?I?Q[0-9]+_[A-Z0-9_]+) ]]; then
		quant="${BASH_REMATCH[1]}"
		quant=$(echo "${quant}" | sed 's/-[0-9]*-of-[0-9]*//')
		echo "${quant}"
	elif [[ "${filename}" =~ (mxfp[0-9]+) ]]; then
		echo "${BASH_REMATCH[1]}"
	else
		echo "unknown"
	fi
}

# ============================================================================
# CSV FUNCTIONS
# ============================================================================

# Escape value for CSV (handle commas and quotes)
csv_escape() {
	local val="$1"
	if [[ "${val}" =~ [,\"] ]]; then
		val="${val//\"/\"\"}"
		echo "\"${val}\""
	else
		echo "${val}"
	fi
}

# Write CSV header
write_csv_header() {
	local csv_file="$1"
	echo "OS,Kernel,linux-firmware,start-time,llama-cpp,binary-in,backend,model,quantization,pp_size,pp,tg_size,tg,gtt_peak_mb,duration_sec,params,notes" > "${csv_file}"
	sync "${csv_file}"
}

# Append row to CSV and flush to disk
append_csv_row() {
	local csv_file="$1"
	local os="$2"
	local kernel="$3"
	local firmware="$4"
	local start_time="$5"
	local llama_cpp="$6"
	local built="$7"
	local backend="$8"
	local model="$9"
	local quant="${10}"
	local pp_size="${11}"
	local pp="${12}"
	local tg_size="${13}"
	local tg="${14}"
	local gtt_peak="${15}"
	local duration="${16}"
	local params="${17}"
	local notes="${18}"

	printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
		"$(csv_escape "${os}")" \
		"$(csv_escape "${kernel}")" \
		"$(csv_escape "${firmware}")" \
		"$(csv_escape "${start_time}")" \
		"$(csv_escape "${llama_cpp}")" \
		"$(csv_escape "${built}")" \
		"$(csv_escape "${backend}")" \
		"$(csv_escape "${model}")" \
		"$(csv_escape "${quant}")" \
		"$(csv_escape "${pp_size}")" \
		"$(csv_escape "${pp}")" \
		"$(csv_escape "${tg_size}")" \
		"$(csv_escape "${tg}")" \
		"$(csv_escape "${gtt_peak}")" \
		"$(csv_escape "${duration}")" \
		"$(csv_escape "${params}")" \
		"$(csv_escape "${notes}")" >> "${csv_file}"

	# Flush to disk immediately
	sync "${csv_file}"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Global variables for rerun tracking
RERUN_CSV=""
FAILED_COUNT=0

# Append failed benchmark to rerun CSV
append_to_rerun_csv() {
	local model="$1"
	local runner="$2"
	local bench_args="$3"

	if [[ -z "${RERUN_CSV}" ]]; then
		RERUN_CSV="${results_dir}/rerun_$(date +%Y%m%d_%H%M%S).csv"
		echo "model,runner,bench_args" > "${RERUN_CSV}"
		sync "${RERUN_CSV}"
	fi

	printf '%s,%s,%s\n' \
		"$(csv_escape "${model}")" \
		"$(csv_escape "${runner}")" \
		"$(csv_escape "${bench_args}")" >> "${RERUN_CSV}"
	sync "${RERUN_CSV}"
	((FAILED_COUNT++))
}

# Run a single benchmark (model, runner, bench_args)
# Uses global: sys_os, sys_kernel, sys_firmware, results_dir
run_single_benchmark() {
	local model="$1"
	local runner="$2"
	local bench_args="$3"

	local model_dir csv_file
	local bench_args_array runner_cmd
	local log_file bench_output
	local llama_cpp_ver built_type backend model_name model_slug quant
	local pp_type tg_type pp_size tg_size pp_val tg_val gtt_peak notes
	local gtt_monitor_pid gtt_pid_file gtt_baseline gtt_result_file
	local log_timestamp args_slug
	local failed=0
	local start_time end_time duration
	local start_time_str

	# Extract model name and create model-specific directory
	model_name="$(extract_model_name "${model}")"
	model_slug="$(sanitize_for_filename "${model_name}")"
	quant="$(extract_quantization "${model}")"
	model_dir="${results_dir}/${model_slug}"
	mkdir -p "${model_dir}/logs"

	# CSV file at model level (aggregates all runs)
	csv_file="${model_dir}/${model_slug}-speedtests.csv"

	# Initialize CSV with header if it doesn't exist
	if [[ ! -f "${csv_file}" ]]; then
		write_csv_header "${csv_file}"
	fi

	echo "Model: ${model}"
	echo "  Runner: ${runner}"
	echo "  Args: ${bench_args}"

	# Parse runner command into array
	read -r -a runner_cmd <<<"${runner}"

	# Parse bench_args string into array
	read -r -a bench_args_array <<<"${bench_args}"

	# Infer backend for log filename
	backend="$(infer_backend "${runner}")"
	args_slug="$(sanitize_for_filename "${bench_args}")"

	# Build log filename: YYYYMMDD_HHMMSS_backend-quantization-args.log
	log_timestamp="$(date +%Y%m%d_%H%M%S)"
	log_file="${model_dir}/logs/${log_timestamp}_${backend}-${quant}-${args_slug}.log"

	# Log the command being run
	{
		echo "Command: ${runner_cmd[*]} ${bench_args_array[*]} -m ${model}"
		echo "Timestamp: $(date -Iseconds)"
		echo "---"
	} >> "${log_file}"

	# Get GTT baseline and start monitor
	gtt_baseline="$(get_gtt_baseline)"
	gtt_pid_file="${model_dir}/.gtt_monitor_$$"
	gtt_result_file="${model_dir}/.gtt_result_$$"
	touch "${gtt_pid_file}"

	# Start GTT monitor in background
	(monitor_gtt_memory "${gtt_pid_file}" "${gtt_baseline}" > "${gtt_result_file}") &
	gtt_monitor_pid=$!

	# Run benchmark and capture output (with timing)
	start_time=$(date +%s)
	start_time_str="$(date +%Y%m%d_%H%M%S)"
	bench_output="$("${runner_cmd[@]}" "${bench_args_array[@]}" -m "${model}" 2>&1)" || true
	end_time=$(date +%s)
	duration=$((end_time - start_time))

	echo "${bench_output}" >> "${log_file}"
	echo "---------------------------------------------------------------------" >> "${log_file}"
	echo "Duration: ${duration}s" >> "${log_file}"

	# Stop GTT monitor and get peak
	rm -f "${gtt_pid_file}"
	wait "${gtt_monitor_pid}" 2>/dev/null || true
	gtt_peak="$(cat "${gtt_result_file}" 2>/dev/null || echo "0")"
	rm -f "${gtt_result_file}"

	# Parse results
	llama_cpp_ver="$(extract_build_version "${bench_output}")"
	built_type="$(infer_build_type "${runner}")"

	# Extract performance values
	read -r pp_type tg_type <<< "$(detect_test_types "${bench_output}")"
	pp_size="${pp_type#pp}"  # "pp2048" -> "2048"
	tg_size="${tg_type#tg}"  # "tg512" -> "512"
	pp_val="$(extract_perf_value "${bench_output}" "${pp_type}")"
	tg_val="$(extract_perf_value "${bench_output}" "${tg_type}")"

	# Build notes for issues
	notes=""

	# If test failed, prepend log file path to notes
	if [[ "${pp_val}" == "UNKNOWN" || "${tg_val}" == "UNKNOWN" ]]; then
		notes+="${log_file} ; "
		failed=1
	fi

	[[ "${llama_cpp_ver}" == "unknown" ]] && notes+="build unknown; "
	[[ "${backend}" == "unknown" ]] && notes+="backend unknown; "
	[[ "${quant}" == "unknown" ]] && notes+="quant unknown; "
	[[ "${pp_val}" == "UNKNOWN" ]] && notes+="no ${pp_type}; "
	[[ "${tg_val}" == "UNKNOWN" ]] && notes+="no ${tg_type}; "
	notes="${notes%; }"

	# Write to CSV immediately (survives interruption/power loss)
	append_csv_row "${csv_file}" \
		"${sys_os}" "${sys_kernel}" "${sys_firmware}" \
		"${start_time_str}" \
		"${llama_cpp_ver}" "${built_type}" "${backend}" \
		"${model_name}" "${quant}" \
		"${pp_size:-512}" "${pp_val}" \
		"${tg_size:-128}" "${tg_val}" \
		"${gtt_peak:-0}" \
		"${duration}" \
		"${bench_args}" "${notes}"

	# Show progress and track failures
	if [[ "${failed}" -eq 0 ]]; then
		echo "  -> ${pp_type}: ${pp_val} t/s, ${tg_type}: ${tg_val} t/s (${duration}s)"
	else
		echo "  -> FAILED or no results (${duration}s)"
		append_to_rerun_csv "${model}" "${runner}" "${bench_args}"
	fi
	echo ""
}

# Run benchmarks from rerun CSV file
run_from_rerun_csv() {
	local rerun_file="$1"
	local line_num=0
	local model runner bench_args

	echo "Running from rerun file: ${rerun_file}"
	echo ""

	while IFS=, read -r model runner bench_args || [[ -n "${model}" ]]; do
		((line_num++))

		# Strip BOM from first line if present
		if [[ "${line_num}" -eq 1 ]]; then
			model="${model#﻿}"  # Remove UTF-8 BOM
		fi

		# Skip header row
		if [[ "${line_num}" -eq 1 && "${model}" == "model" ]]; then
			continue
		fi

		# Skip empty lines
		[[ -z "${model}" ]] && continue

		# Remove surrounding quotes if present (from csv_escape)
		model="${model#\"}"
		model="${model%\"}"
		runner="${runner#\"}"
		runner="${runner%\"}"
		bench_args="${bench_args#\"}"
		bench_args="${bench_args%\"}"

		# Run benchmark (continue even if it fails)
		run_single_benchmark "${model}" "${runner}" "${bench_args}" || true
	done < "${rerun_file}"
}

# Run full benchmark matrix (models × runners × bench_arg_sets)
run_full_matrix() {
	local model runner bench_args

	for model in "${models[@]}"; do
		for runner in "${runners[@]}"; do
			for bench_args in "${bench_arg_sets[@]}"; do
				run_single_benchmark "${model}" "${runner}" "${bench_args}"
			done
		done
	done
}

main() {
	local rerun_file="${1:-}"

	# Collect system info once (global for run_single_benchmark)
	sys_os="$(get_os_info)"
	sys_kernel="$(get_kernel_version)"
	sys_firmware="$(get_linux_firmware_version)"

	# Setup results directory (global for run_single_benchmark)
	results_dir="results"
	mkdir -p "${results_dir}"

	echo "=============================================="
	echo "LLM Benchmark Run: $(date +%Y%m%d_%H%M%S)"
	echo "=============================================="
	echo ""

	if [[ -n "${rerun_file}" && -f "${rerun_file}" ]]; then
		# Rerun mode: run specific tuples from CSV
		run_from_rerun_csv "${rerun_file}"
	else
		# Full mode: run Cartesian product
		run_full_matrix
	fi

	echo "=============================================="
	echo "Benchmark complete. Results in: ${results_dir}/"
	if [[ "${FAILED_COUNT}" -gt 0 ]]; then
		echo "Failures: ${FAILED_COUNT} (rerun with: $0 ${RERUN_CSV})"
	else
		echo "All benchmarks passed!"
	fi
	echo "=============================================="
}

# Run main
main "$@"
