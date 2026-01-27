# LLM Quantization Quality Testing Framework

**Purpose**: Measure intelligence/quality degradation from quantization using KL divergence

## Motivation

Traditional metrics miss critical degradation:
- **Accuracy**: Can stay identical while 5% of answers change ("flips")
- **Perplexity**: Only measures probability of correct token (misses distribution shifts)
- **KL Divergence**: Captures entire probability distribution, correlates 0.96-0.97 with answer flips

### Why KLD is Superior

```
Example token probabilities:
Original (FP16):  mat: 40%, rug: 30%, floor: 10%, table: 5%
Quantized:        mat: 35%, rug: 34%, floor: 5%, table: 4%

Perplexity: Sees minimal change (40%→35% on correct token)
KLD: Captures the entire distribution shift (rug: 30%→34% changes reasoning)
```

**Sources**:
- [Why We're Measuring LLM Compression Wrong](https://huggingface.co/blog/rishiraj/kld-guided-quantization)
- [llama.cpp KL Divergence Discussion #4110](https://github.com/ggml-org/llama.cpp/discussions/4110)
- [llama.cpp Perplexity Tool Documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)

## State of the Art (2026)

Recent research has transformed understanding of quantization degradation beyond simple perplexity metrics:

### 1. Multi-Dimensional Evaluation Framework

**Breakthrough**: No single metric captures quantization impact. State-of-the-art evaluation requires simultaneous measurement of:

- **Performance**: TTFT (Time to First Token), TPOT (Time per Output Token) at various percentiles
- **Energy**: Joules per token using GPU telemetry
- **Quality**: Task-specific accuracy, KLD, semantic similarity

**Key Finding**: "No single quantization method dominates across all three metrics" - a method optimal for energy may degrade quality significantly on reasoning tasks.

**Source**: [Systematic Characterization of LLM Quantization](https://arxiv.org/html/2508.16712v1)

### 2. Quantization-Induced Degradation (QiD) Scaling Laws

**NEW (2025)**: Predictive scaling laws for quantization degradation across 7B-405B models trained on up to 100T tokens:

- **Low-bit quantization (Q3-Q4)** works well for **undertrained LLMs** (≤10¹² training tokens)
- Becomes problematic for **fully-trained LLMs** (100T+ tokens), especially smaller models
- Can predict QiD before running expensive evaluations

**Implication**: Q4 may be acceptable for a 7B model trained on 1T tokens but risky for the same architecture trained on 15T tokens.

**Source**: [Low-Bit Quantization Favors Undertrained LLMs](https://aclanthology.org/2025.acl-long.1555.pdf)

### 3. Task-Specific Degradation Patterns

**Red Hat's 500K+ evaluation study** revealed dramatic task sensitivity:

| Task Type | Degradation Range |
|-----------|-------------------|
| Simple QA | <1% accuracy loss |
| Text generation | Most resilient |
| STEM reasoning | Highest sensitivity |
| Coding tasks | Up to 92% drop in extreme cases |

**Model size effects**:
- **8B models**: Noticeable variability with Q4
- **70B/405B models**: Negligible degradation even at Q4 (99%+ accuracy recovery)

**Key insight**: Larger models maintain quality better under quantization. Smaller models need higher precision (Q6/Q5) for reasoning tasks.

**Source**: [Red Hat: Half Million Evaluations](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)

### 4. Activation Outliers are the Root Cause

**Critical discovery**: Most quantization degradation stems from activation spikes (outliers 10-100× larger than typical activations).

**State-of-the-art mitigation strategies**:

1. **OSP (Outlier-Safe Pre-Training)**: Prevent outliers during training
   - Uses Muon optimizer + Single-Scale RMSNorm
   - Achieves 35.7 average score on 10 benchmarks with 4-bit quantization

2. **AWQ (Activation-Aware Weight Quantization)**: Protect salient weights
   - Identifies channels with large activation magnitudes
   - Preserves these weights at higher precision
   - Significantly outperforms uniform quantization

3. **RoLoRA (EMNLP 2024)**: Rotation-based outlier elimination
   - 29.5% absolute accuracy gain with 4-bit W-A quantization on LLaMA2-13B

**Sources**:
- [Outlier-Safe Pre-Training](https://arxiv.org/html/2506.19697v1)
- [AWQ: Activation-aware Weight Quantization](https://dl.acm.org/doi/10.1145/3714983.3714987)

### 5. Beyond Standard KL Divergence

**Problem**: Standard KLD can be "dominated by uninformative classes with negligible probabilities" due to long-tail distributions.

**Advanced approaches**:
- **Top-K KLD**: Only measure divergence on K most probable tokens (reduces noise)
- **Modified KLD**: Weight by token importance/frequency
- **Distribution-aware metrics**: Account for model-specific probability patterns

**Practical recommendation**: Use mean KLD as "ground-truth metric" but consider Top-K variants for noisy distributions.

**Source**: [Comprehensive Study on Quantization Techniques](https://arxiv.org/html/2411.02530v1)

### 6. Optimizer Impact on Quantization Robustness

**Recent finding**: Training optimizer affects post-quantization quality:

- **Muon-optimized models**: Degrade less when quantized
- **Adam-optimized models**: Show larger accuracy drops during quantization
- Models trained with better optimizers are inherently more quantization-friendly

**Implication**: If training models from scratch, optimizer choice affects downstream quantization quality.

### 7. Real-World Evaluation Frameworks

Modern evaluation goes beyond academic benchmarks:

**Comprehensive framework dimensions**:
1. **Knowledge & Capacity**: MMLU, ARC, HellaSwag
2. **Alignment**: Human preference, instruction following
3. **Efficiency**: Latency, memory, throughput
4. **Real-world performance**: Arena-Hard-Auto, MT-Bench

**Semantic similarity metrics** (ROUGE, BERTScore, STS) capture quality preservation better than token-level accuracy alone.

**Source**: [IJCAI 2025: Quantization Methods and Task Difficulty](https://www.ijcai.org/proceedings/2025/0902.pdf)

### Summary: Best Practices (2026)

1. **Multi-metric evaluation**: Never rely on accuracy or perplexity alone
   - KLD for distribution shift
   - Task-stratified benchmarks (test reasoning separately from QA)
   - Semantic similarity metrics

2. **Consider training history**:
   - Undertrained models: More forgiving to aggressive quantization
   - Fully-trained models: Need careful quantization, especially at smaller sizes

3. **Task-aware quantization**:
   - Coding/STEM: Use Q6/Q5 minimum
   - General chat: Q5/Q4 acceptable
   - Simple QA: Q4/Q3 may suffice

4. **Activation outlier management**:
   - Profile activation distributions
   - Use activation-aware quantization (AWQ) if available
   - Consider per-channel scaling

5. **Model size matters**:
   - <10B: Conservative quantization (Q6-Q4)
   - 70B+: Aggressive quantization often safe (Q4-Q3)

## Constraint: Can't Run Unquantized Models

**Solution**: Use Q8_0 as pseudo-baseline
- Q8_0 is "almost indistinguishable from FP16"
- Typical KLD from original: <0.001
- Fits in ~1.25x the VRAM of Q6_K

All quality measurements are relative to Q8_0.

## Datasets

| Dataset | Size | Tokens | Runtime | Use Case |
|---------|------|--------|---------|----------|
| **WikiText-2** | ~4MB | 6.7K | 30-60s | Standard benchmark, fast |
| **C4 validation** | ~300MB | - | 10-15min | Comprehensive, publication-grade |
| **Penn Treebank** | ~1MB | - | <30s | Quick validation |

**Recommendation**: Start with WikiText-2

### Auto-Download
```bash
# WikiText-2
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
```

## Architecture

```
run-kld-quality-tests.py
├── Config
│   ├── Model groups (baseline + test quantizations)
│   ├── Runners (toolbox containers, local builds)
│   └── Datasets (WikiText-2, C4, etc.)
│
├── DatasetManager
│   ├── Check if dataset exists locally
│   ├── Auto-download if missing
│   └── Validate checksums
│
├── BenchmarkRunner
│   ├── Execute llama-perplexity via subprocess
│   ├── Generate baseline KLD files (--kl-divergence-base)
│   ├── Test quantizations against baseline (--kl-divergence)
│   └── Capture stdout/stderr
│
├── ResultsParser
│   ├── Extract perplexity values
│   ├── Extract KL divergence scores
│   ├── Extract build version
│   └── Calculate model file sizes
│
├── ResultsAnalyzer
│   ├── Calculate PPL ratios
│   ├── Classify quality levels (excellent/good/acceptable/poor)
│   ├── Generate recommendations by use case
│   └── Identify Pareto frontier
│
├── Visualizer (matplotlib)
│   ├── Quality/Size tradeoff (Pareto chart)
│   ├── Perplexity comparison (bar chart)
│   ├── KLD heatmap (multi-backend)
│   └── Recommendation matrix
│
└── ReportGenerator
    ├── Executive summary (markdown)
    ├── Raw data CSV
    ├── Recommendations JSON
    └── Combine plots into report
```

## Output Structure

```
results/kld_YYYYMMDD_HHMMSS/
├── summary.md                      # ⭐ Executive summary for practitioners
├── results.csv                     # Raw data (PPL, KLD, size, etc.)
├── recommendations.json            # Structured recommendations
├── plots/
│   ├── kld_vs_size.png            # Quality/size Pareto frontier
│   ├── perplexity_comparison.png  # Bar chart: PPL by quantization
│   ├── kld_heatmap.png            # Multi-backend comparison
│   └── recommendation_matrix.png   # Visual guide: when to use what
└── logs/
    ├── baseline_Q8_0_rocm-7.1.1.log
    └── test_Q6_K_rocm-7.1.1.log
```

## CSV Schema

### Phase 1-2: Core Metrics
```csv
OS,Kernel,linux-firmware,llama-cpp,backend,model,baseline_quant,test_quant,dataset,baseline_ppl,test_ppl,ppl_ratio,kld,model_size_gb,duration_sec,notes
```

**Example Row**:
```csv
Fedora 43,6.18.3-200.fc43.x86_64,20250108-161.fc43,b7816,rocm-7.1.1,GLM-4.7-Flash (Unsloth),Q8_0,Q6_K,wiki.test.raw,5.234,5.241,1.0013,0.0021,3.8,45,
```

### Phase 3: Extended with Task-Specific Metrics
```csv
...,kld,kld_top_k,model_size_gb,humaneval_pass1,arc_accuracy,mmlu_5shot,bertscore_f1,answer_flips_pct,duration_sec,notes
```

**Rationale**:
- `kld_top_k`: Top-K KLD (reduces noise from long-tail)
- `humaneval_pass1`: Code generation accuracy (most sensitive to quantization)
- `arc_accuracy`: Reasoning task performance
- `mmlu_5shot`: General knowledge (5-shot to match research standards)
- `bertscore_f1`: Semantic similarity (catches drift accuracy misses)
- `answer_flips_pct`: % of answers that changed vs baseline (direct behavioral measure)

## Practitioner-Focused Summary Report

### Format: Markdown

```markdown
# {Model Name} Quantization Quality Report
**Generated**: {timestamp}
**Backend**: {backend}
**Dataset**: {dataset} ({token_count} tokens)

## Executive Summary
✅ **Recommended for Production**: Q6_K, Q5_K_M
⚠️  **Use with Caution**: Q4_K_M
❌ **Not Recommended**: Q3_K_M, IQ3_XXS

## Quality Metrics

| Quantization | Size   | PPL    | PPL Δ   | KLD     | Status | Use Case |
|-------------|--------|--------|---------|---------|--------|----------|
| Q8_0        | 5.0GB  | 5.234  | -       | 0.000   | ⭐ Baseline | Maximum quality |
| Q6_K        | 3.8GB  | 5.241  | +0.13%  | 0.0021  | ✅ Excellent | Production (recommended) |
| Q5_K_M      | 3.3GB  | 5.256  | +0.42%  | 0.0053  | ✅ Very Good | Balanced choice |
| Q4_K_M      | 2.8GB  | 5.312  | +1.49%  | 0.0187  | ⚠️ Acceptable | Size-constrained deploys |
| UD-Q4_K_XL  | 2.9GB  | 5.298  | +1.22%  | 0.0142  | ⚠️ Good | Better Q4 alternative |
| Q3_K_M      | 2.2GB  | 5.521  | +5.48%  | 0.0523  | ❌ Poor | High degradation |
| IQ3_XXS     | 1.9GB  | 5.743  | +9.72%  | 0.1024  | ❌ Not Viable | Severe degradation |

## Interpretation

**KLD Thresholds** (empirical guidelines from research):
- `< 0.005`: Negligible degradation, safe for production
- `0.005-0.020`: Minor degradation, acceptable for most uses
- `0.020-0.050`: Noticeable degradation, test thoroughly on your use case
- `> 0.050`: Significant degradation, not recommended

**Your Results**:
- Q6_K and Q5_K_M are virtually indistinguishable from Q8_0
- Q4_K_M crosses into "noticeable" territory but may be acceptable
- Q3_K_M shows significant reasoning degradation (5%+ PPL increase)

## Deployment Recommendations

### By Use Case

**General Chat/Completion**:
- **Best**: Q6_K (24% smaller, <0.003 KLD)
- **Budget**: Q5_K_M (34% smaller, excellent quality)
- **Minimum**: Q4_K_M (44% smaller, test first)

**Reasoning Tasks** (math, logic, code):
- **Best**: Q6_K
- **Acceptable**: Q5_K_M
- **Not recommended**: Q4_K_M and below

**VRAM-Constrained** (<8GB):
- Q4_K_M may be necessary, but expect 1-2% accuracy drops
- Consider UD-Q4_K_XL variants if available (better quality at similar size)

### Size/Quality Pareto Frontier

```
Q8_0 (5.0GB, baseline) → Q6_K (3.8GB, -24%, 0.002 KLD) →
Q5_K_M (3.3GB, -34%, 0.005 KLD) → UD-Q4_K_XL (2.9GB, -42%, 0.014 KLD)
```

**Trade-off sweet spot**: Q6_K or Q5_K_M

---
*Generated by llm-benchmarks KLD Quality Testing*
```

## Key Visualizations

### 1. Quality/Size Tradeoff (Pareto Chart)

```
KLD Score
    ^
0.10|                                  X Q3_K_M
    |                                 X IQ3_XXS
0.05|                           X Q4_K_M
    |                      X UD-Q4_K_XL
0.02|----------------Acceptable Threshold-----------
    |                 X Q5_K_M
    |            X Q6_K
0.00|       X Q8_0
    +---------------------------------> Model Size (GB)
        1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0

Legend:
- Green zone (< 0.005 KLD): Production ready
- Yellow zone (0.005-0.02): Acceptable
- Red zone (> 0.02): Use with caution
```

### 2. Perplexity Degradation (Bar Chart)

```
PPL
6.0 |                                               ▓▓▓
    |                                           ▓▓▓ ▓▓▓
5.5 |                                       ▓▓▓ ▓▓▓ ▓▓▓
    |                                   ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓
5.3 |               ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓
    |           ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓
5.0 +-------Q8--Q6--Q5--Q4--UDQ4-Q3--IQ3---------
           +0% +.1% +.4% 1.5% 1.2% 5.5% 9.7%
```

### 3. Backend Consistency Heatmap

If testing multiple backends (ROCm 6.4, 7.1.1, Vulkan), show KLD consistency:

```
             Q8_0  Q6_K  Q5_K  Q4_K  Q3_K
ROCm 6.4.4    0    0.002 0.005 0.018 0.052
ROCm 7.1.1    0    0.002 0.005 0.019 0.053
Vulkan-radv   0    0.003 0.006 0.020 0.055

Color: Green (< 0.01) → Yellow (0.01-0.03) → Red (> 0.03)
```

### 4. Recommendation Matrix

```
VRAM Budget vs Quality Requirement

Quality
  High  | Q8_0  Q8_0  Q6_K  Q6_K  Q5_K
Medium  | Q8_0  Q6_K  Q5_K  Q4_K  Q4_K
  Low   | Q6_K  Q5_K  Q4_K  Q3_K  Q3_K
        +--------------------------------
        4GB   6GB   8GB   12GB  16GB+
                    VRAM Budget
```

## Implementation Plan

### Phase 1: Core Functionality (KLD + Perplexity)
- [x] Design architecture
- [ ] Implement BenchmarkRunner (subprocess llama-perplexity)
- [ ] Implement ResultsParser (regex extraction: PPL, KLD, build version)
- [ ] Generate CSV output
- [ ] Basic plot: KLD vs Size (Pareto frontier)

**Deliverable**: Working script that generates CSV + 1 plot

**Metrics**: PPL, KLD (vs Q8_0 baseline), model size

### Phase 2: Practitioner Features + Multi-Dimensional Analysis
- [ ] ResultsAnalyzer with quality classifications
  - [ ] KLD thresholds: <0.005 (excellent), 0.005-0.02 (acceptable), >0.05 (poor)
  - [ ] Task-specific recommendations (chat vs reasoning vs code)
  - [ ] Model-size-aware guidance (8B vs 70B have different tolerances)
- [ ] Auto-recommendations considering:
  - [ ] VRAM budget
  - [ ] Use case (general/reasoning/coding)
  - [ ] Quality requirements
- [ ] Markdown summary report generator
- [ ] All 4 core visualizations
- [ ] Model size calculation
- [ ] Integration with throughput data (quality-adjusted throughput metric)

**Deliverable**: Production-ready reports for deployment decisions

**Metrics**: PPL, KLD, model size, throughput (from existing benchmarks)

### Phase 3: Task-Specific Evaluation (SOTA 2026)
- [ ] Implement task-stratified benchmarks:
  - [ ] Coding: HumanEval (most sensitive to quantization)
  - [ ] Reasoning: ARC-Challenge
  - [ ] General knowledge: MMLU 5-shot
  - [ ] Semantic similarity: BERTScore, ROUGE
- [ ] Real-world evaluation:
  - [ ] Arena-Hard-Auto or MT-Bench integration
  - [ ] Answer flip tracking (compare exact outputs Q8 vs Qx)
- [ ] Advanced KLD metrics:
  - [ ] Top-K KLD (reduce noise from tail)
  - [ ] Per-layer KLD sensitivity analysis
- [ ] Task difficulty correlation analysis

**Deliverable**: Research-grade quantization impact analysis

**Metrics**: PPL, KLD, task-specific accuracy, semantic similarity, answer flips

### Phase 4: Advanced Features
- [ ] DatasetManager with auto-download (WikiText-2, C4, HumanEval)
- [ ] Multi-backend comparison mode (ROCm versions, Vulkan)
- [ ] Custom calibration datasets
- [ ] Activation outlier detection:
  - [ ] Profile activation distributions
  - [ ] Identify outlier channels
  - [ ] Correlate with quantization sensitivity
- [ ] Mixed-precision exploration:
  - [ ] Layer-wise quantization sensitivity
  - [ ] Optimal bit allocation per layer
- [ ] Parallel execution (test multiple quants simultaneously)
- [ ] Rich progress bars and live updates
- [ ] JSON export for automation
- [ ] Integration with qMeter-style online characterization

**Deliverable**: Enterprise-grade quality testing framework with cutting-edge features

### Phase 5: Scaling Laws & Predictive Models (Future Research)
- [ ] Collect comprehensive dataset across:
  - [ ] Model sizes (1B-405B)
  - [ ] Training compute (undertrained vs overtrained)
  - [ ] Quantization levels (Q2-Q8)
  - [ ] Task types (QA, reasoning, code, etc.)
- [ ] Build QiD prediction model
- [ ] Validate: Can we predict Q4 quality from Q8 measurements?
- [ ] Training-aware recommendations (adjust based on model training tokens)

**Deliverable**: Predictive quantization quality framework

## Usage Example

```python
# run-kld-quality-tests.py

# Configuration
models = {
    "GLM-4.7-Flash": {
        "baseline": "/path/to/GLM-4.7-Flash-Q8_0.gguf",
        "tests": [
            "/path/to/GLM-4.7-Flash-Q6_K.gguf",
            "/path/to/GLM-4.7-Flash-Q4_K_M.gguf",
            "/path/to/GLM-4.7-Flash-UD-Q4_K_XL.gguf",
        ]
    }
}

runners = [
    "toolbox run -c llama-rocm-7.1.1 -- llama-perplexity",
    "toolbox run -c llama-vulkan-radv -- llama-perplexity",
]

datasets = [
    "/path/to/wikitext-2/wiki.test.raw"
]

# Run tests
results = run_kld_tests(models, runners, datasets)

# Generate practitioner report
generate_summary_report(results, output_dir="results/kld_20260126/")

# Output:
# ✅ summary.md (executive summary)
# ✅ results.csv (raw data)
# ✅ plots/ (4 visualizations)
# ✅ recommendations.json
```

## Expected Quantization Degradation

From empirical research and llama.cpp community data:

| Quantization | Typical KLD | Quality | Size vs FP16 |
|-------------|-------------|---------|--------------|
| Q8_0        | <0.001      | ~FP16   | ~50%         |
| Q6_K        | 0.002-0.004 | Excellent | ~38%       |
| Q5_K_M      | 0.005-0.008 | Very Good | ~33%       |
| Q4_K_M      | 0.015-0.025 | Acceptable | ~28%      |
| Q3_K_M      | 0.050-0.100 | Poor    | ~22%         |
| IQ3_XXS     | 0.100-0.200 | Very Poor | ~19%       |

**Note**: Reasoning tasks (math, code, logic) degrade faster than factual recall.

## Integration with Throughput Benchmarks

Future: Combine KLD quality results with throughput data to show:

```
Quality-Adjusted Throughput = tokens/sec × (1 - KLD)

Example:
Q4_K_M: 150 t/s, KLD 0.020 → Adjusted: 147 t/s
Q6_K:   120 t/s, KLD 0.003 → Adjusted: 119.6 t/s

Winner for quality-adjusted throughput: Q6_K
```

This helps answer: "Should I use faster Q4 or slower Q6?"

## Recommended Evaluation Stack for GLM-4.7-Flash

Based on 2026 state-of-the-art best practices:

### Tier 1: Essential (Phase 1-2)
1. **KLD baseline**: Q8_0 vs Q6/Q5/Q4 on WikiText-2
   - Fast (~30-60s per model)
   - Establishes distribution shift baseline
   - Sufficient for initial go/no-go decisions

2. **Integration with throughput benchmarks**:
   - Quality-adjusted throughput = tokens/sec × (1 - KLD)
   - Answers: "Should I deploy faster Q4 or slower Q6?"

### Tier 2: Recommended (Phase 3)
3. **Task-stratified evaluation**:
   - **Coding**: HumanEval (STEM tasks degrade most)
   - **Reasoning**: ARC-Challenge or GSM8K
   - **General**: MMLU 5-shot subset

4. **Semantic similarity**:
   - BERTScore or ROUGE on sample outputs
   - Catches semantic drift that perplexity misses

### Tier 3: Advanced (Phase 4-5)
5. **Answer flip tracking**: Compare Q8_0 vs Qx exact outputs on 100-500 diverse prompts
6. **Real-world proxy**: Arena-Hard-Auto or MT-Bench
7. **Activation profiling**: Identify if GLM-4.7-Flash has outlier issues

**Time budget recommendation**:
- Tier 1: ~4-6 hours (automate and run overnight)
- Tier 2: +8-12 hours (weekend run)
- Tier 3: +1-2 days (research-grade analysis)

**Minimal viable**: Tier 1 alone provides 80% of deployment decision value.

## References

### Core KL Divergence & Methodology
- [HuggingFace: KLD-Guided Quantization](https://huggingface.co/blog/rishiraj/kld-guided-quantization) - "Accuracy is Not All You Need"
- [llama.cpp: Perplexity Tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) - Official documentation
- [llama.cpp: KL Divergence Discussion #4110](https://github.com/ggml-org/llama.cpp/discussions/4110) - Community insights

### State-of-the-Art Research (2024-2026)
- [Systematic Characterization of LLM Quantization](https://arxiv.org/html/2508.16712v1) - Multi-dimensional evaluation framework
- [Low-Bit Quantization Favors Undertrained LLMs](https://aclanthology.org/2025.acl-long.1555.pdf) - QiD scaling laws (ACL 2025)
- [Red Hat: Half Million Evaluations](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms) - Task-specific degradation patterns
- [Comprehensive Study on Quantization Techniques](https://arxiv.org/html/2411.02530v1) - Modified KLD approaches
- [Comprehensive Evaluation Framework](https://arxiv.org/html/2507.17417) - Multi-dimensional metrics (2025)
- [IJCAI 2025: Quantization Methods and Task Difficulty](https://www.ijcai.org/proceedings/2025/0902.pdf) - Task correlation analysis

### Activation Outliers & Advanced Techniques
- [Outlier-Safe Pre-Training](https://arxiv.org/html/2506.19697v1) - OSP methodology (2025)
- [AWQ: Activation-aware Weight Quantization](https://dl.acm.org/doi/10.1145/3714983.3714987) - Protect salient weights
- [OWQ: Outlier-Aware Weight Quantization](https://arxiv.org/html/2306.02272v4) - Fine-tuning with outliers (AAAI 2024)
- [Mitigating Activation Spikes in GLU-Based LLMs](https://www.mdpi.com/1999-5903/17/4/185) - Gated Linear Units

### Community Resources
- [Comparing Quantized Performance in Llama Models](https://www.lesswrong.com/posts/qmPXQbyYA66DuJbht/comparing-quantized-performance-in-llama-models) - Practical comparisons
- [Awesome LLM Quantization](https://github.com/pprp/Awesome-LLM-Quantization) - Curated paper list

## Future Directions

### Immediate (Align with 2026 SOTA)
1. **Task-specific benchmarks**: Math (GSM8K), Code (HumanEval), Reasoning (ARC)
   - Research shows coding/STEM degrade most (up to 92% drops in extreme cases)
   - Essential for comprehensive evaluation
2. **Answer flip analysis**: Track exact questions where answers change
   - 0.96-0.97 correlation with KLD (ground truth for behavioral changes)
3. **Multi-dimensional metrics**: Track performance, energy, quality simultaneously
   - No single quantization dominates all three dimensions

### Medium-term (Novel Contributions)
4. **Mixed-precision exploration**: Quantize different layers differently
   - Attention: 6-8 bits (high sensitivity)
   - MoE FFN experts: 2-3 bits (low per-token sensitivity)
5. **Calibration dataset optimization**: Which calibration data minimizes KLD?
   - Current practice: Generic datasets (WikiText-2, C4)
   - Opportunity: Domain-specific calibration
6. **Activation outlier profiling**:
   - Identify models with severe outlier issues
   - Correlate outlier prevalence with quantization sensitivity
   - Guide quantization method selection (uniform vs AWQ)

### Long-term (Research Frontiers)
7. **QiD Scaling Laws for GLM Family**:
   - Build predictive model: training_tokens × model_size → QiD
   - Validate: Can Q8 tests predict Q4 quality without running Q4?
8. **Training-time quantization friendliness**:
   - Compare Muon vs Adam optimized models
   - Identify architectural choices that improve quantization robustness
9. **Real-world task correlation**: How well does KLD predict production performance?
   - A/B testing quantized models in production
   - Correlate KLD with user satisfaction, task success rate
10. **Automatic quantization selection**:
    - Input: Model, task, VRAM budget, quality threshold
    - Output: Optimal quantization level with confidence intervals
