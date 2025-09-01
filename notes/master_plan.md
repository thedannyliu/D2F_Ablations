Master Plan: Deconstructing the Success of Discrete Diffusion Forcing (D2F)
Document Status: v2.0 - UPDATED BASED ON CURRENT CODEBASE
Date: 2025-01-18
Previous Version: v1.0 (2025-08-27)

NOTICE: UPDATED IMPLEMENTATION GUIDE
This document has been updated to reflect the current state of the D2F codebase and provides a practical implementation roadmap. The foundational strategy and experimental blueprint remain consistent with the original vision, but now includes specific implementation details, existing code analysis, and practical execution steps based on the actual repository structure.

## Table of Contents

1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Current Repository Status & Analysis](#2-current-repository-status--analysis)
3. [Technical Deep Dive: Core Components of D2F](#3-technical-deep-dive-core-components-of-d2f)
4. [Updated Phased Experimental Plan](#4-updated-phased-experimental-plan)
   - [Phase 0: Environment Setup & Baseline Verification](#phase-0-environment-setup--baseline-verification)
   - [Phase 1: D2F Training Pipeline Execution](#phase-1-d2f-training-pipeline-execution)
   - [Phase 2: Core Ablation Studies - Isolating the "Why"](#phase-2-core-ablation-studies---isolating-the-why)
   - [Phase 3: Synthesis, Reporting, and Future Work](#phase-3-synthesis-reporting-and-future-work)
5. [Implementation Guide & Current Code Analysis](#5-implementation-guide--current-code-analysis)
6. [Practical Execution Steps](#6-practical-execution-steps)

---

## 2. Current Repository Status & Analysis

### 2.1. Repository Structure Overview

The D2F repository contains a complete implementation with the following key components:

```
Discrete-Diffusion-Forcing/
├── D2F-train/                    # Training pipeline (COMPLETE)
│   ├── train.py                  # Main training script with Accelerate support
│   ├── train.sh                  # Training launch scripts
│   ├── config/                   # Configuration files for different models
│   │   ├── dream_eagle.yaml      # Dream-Base-7B configuration
│   │   ├── llada.yaml           # LLaDA-8B configuration
│   │   └── acc_config           # Accelerate configuration
│   ├── model/                    # Model implementations
│   │   ├── modeling_llada.py     # LLaDA model architecture
│   │   └── configuration_llada.py
│   └── utils/                    # Training utilities
│       ├── data.py              # Data loading and preprocessing
│       ├── model.py             # Model initialization with LoRA
│       ├── loss.py              # Loss computation (distillation)
│       ├── generation.py        # Generation utilities
│       └── util.py              # Helper functions
├── D2F-eval/                     # Evaluation pipeline (COMPLETE)
│   ├── eval_dream.py            # Dream model evaluation
│   ├── eval_llada.py            # LLaDA model evaluation
│   ├── eval_*_vllm.py           # vLLM integration evaluations
│   ├── postprocess_code.py      # HumanEval post-processing
│   └── *.sh                     # Evaluation scripts
├── docs/                         # Documentation and assets
├── notes/                        # Project documentation
│   ├── master_plan.md           # This document
│   └── retrain_tutorial.md      # Detailed training tutorial
└── requirements.txt              # Dependencies
```

### 2.2. Implementation Status Assessment

#### ✅ **COMPLETE COMPONENTS**

1. **Training Pipeline**:
   - Full asymmetric distillation implementation
   - Block-wise causal attention masks
   - LoRA fine-tuning setup
   - Multi-GPU support via Accelerate
   - Checkpointing and resuming
   - WandB integration

2. **Model Support**:
   - Dream-Base-7B integration
   - LLaDA-8B integration
   - Configurable via YAML files

3. **Data Pipeline**:
   - BS17K dataset loading
   - Configurable data preprocessing
   - Tokenization with proper masking

4. **Evaluation Framework**:
   - Complete benchmark suite (GSM8K, MBPP, HumanEval, Math)
   - Performance metrics collection (TPS + Accuracy)
   - vLLM integration for accelerated inference

#### ⚠️ **CONFIGURATION REQUIRED**

1. **Model Paths**: Need to be set to actual model locations
2. **Dataset Paths**: Need to be configured for local/HuggingFace datasets
3. **Output Directories**: Need to be set for experiment management
4. **Hardware Configuration**: Accelerate config needs GPU specification

#### 🔄 **IMPLEMENTATION GAPS FOR ABLATION STUDIES**

The current codebase provides a solid foundation but needs specific modifications for systematic ablation studies:

1. **Cache-only Mode**: Need to implement serial block-by-block decoding
2. **Inference Algorithm Isolation**: Need original model + D2F inference
3. **Component Measurement**: Need separate timing for architecture vs algorithm

### 2.3. Key Insights from Code Analysis

1. **Asymmetric Distillation**: Implemented via `self_align=True` in configuration, using teacher model (without adapter) to guide student
2. **Block-wise Attention**: Implemented via `build_custom_float_attention_mask()` function
3. **Monotonic Noise**: Implemented via `forward_process_length()` with block-aware masking
4. **LoRA Integration**: All fine-tuning uses LoRA to reduce training cost
5. **Flexible Configuration**: YAML-based config system supports different experiments

---

## 1. Project Overview & Objectives
1.1. Context
Autoregressive (AR) Large Language Models (LLMs) have long dominated text generation but suffer from a fundamental limitation: sequential, token-by-token decoding, which creates an inference bottleneck. Diffusion Large Language Models (dLLMs) offer a promising alternative by enabling parallel decoding of multiple tokens. However, until the work "Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing" (D2F), no open-source dLLM had demonstrably surpassed the inference speed of comparably-sized AR models. The D2F framework represents a significant breakthrough.

1.2. Primary Goal
The primary objective of this project is to systematically deconstruct the D2F framework to rigorously quantify the contribution of each of its core components to the final performance gains in both inference speed (Tokens/Second) and generation quality (Score).

1.3. Core Research Question
Is the success of D2F attributable to:
a) The architectural change (Block-wise Causal Attention) enabling KV Caching?
b) The specialized training strategy (Asymmetric Distillation with Monotonic Noise)?
c) The sophisticated inference algorithm (Pipelined Parallel Decoding)?
d) An inseparable synergy between all three components?

1.4. Proof-of-Concept (PoC) Scope
To ensure focus and rapid initial progress, this project will begin with a tightly defined scope:

Base Model: Dream-Base-7B

Dataset & Task: GSM8K (Mathematical Reasoning)

Primary Metrics: Tokens/Second (TPS) and Accuracy Score.

Expansion to other models (e.g., LLaDA) and datasets will be considered in a subsequent project phase, building upon the modular framework established here.

## 3. Technical Deep Dive: Core Components of D2F

To structure our investigation, we dissect D2F into three mutually exclusive and collectively exhaustive (MECE) components, now with specific implementation references from the codebase.

### 3.1. Component 1: Architecture (Block-wise Causal Attention)

**Description**: D2F modifies the standard bidirectional self-attention of dLLMs. Attention is bidirectional within a block of tokens but causal (unidirectional) between blocks.

**Implementation Location**: 
- `D2F-eval/eval_dream.py` - `create_full_block_attention_mask()` function
- `D2F-train/utils/loss.py` - `build_custom_float_attention_mask()` function

**Function**: This structural constraint is the key enabler for standard KV Caching. Once a block is fully generated, its key and value states can be cached and reused for subsequent blocks, drastically reducing redundant computation.

**Code Implementation Details**:
```python
# Block-wise attention mask creation
def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    # Initialize mask with -inf (no attention)
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    
    # Block 0: Prompt (can see itself)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    
    # Each regular block can see prompt + all previous blocks + itself
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        # ... (causal attention implementation)
```

**Hypothesis**: This component is the primary source of raw computational savings and provides a foundational speedup over vanilla dLLMs.

### 3.2. Component 2: Training Strategy (Asymmetric Distillation)

**Description**: D2F is trained via distillation. A pre-trained, bidirectional dLLM (Teacher) provides supervision to the D2F model (Student). The Teacher has a global view of a noisy sequence, while the Student is trained with its causal view.

**Implementation Location**:
- `D2F-train/utils/loss.py` - `compute_loss()` and `compute_llada_loss()` functions
- `D2F-train/utils/util.py` - `forward_process_length()` function
- Configuration: `self_align: true` in YAML configs

**Function**: This strategy explicitly trains the model to handle the exact state it will encounter during pipelined inference.

**Code Implementation Details**:
```python
# Asymmetric distillation implementation
if self_align:
    with torch.no_grad():
        with denoiser.disable_adapter():  # Teacher model (no LoRA)
            ref_logits = denoiser(noisy_batch, attention_mask=bidirectional_mask).logits
            ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
    # Student loss against teacher
    token_loss = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none')
```

**Monotonic Noise Implementation**:
```python
# Block-aware noise scheduling
noisy_batch, masked_indices, p_mask = forward_process_length(
    input_ids, mask_id=mask_id, prompt_lengths=question_length, 
    block_size=block_size, eos_id=eos_id
)
```

**Hypothesis**: This specialized training is essential for maintaining high generation quality. Without it, the model would produce incoherent text when faced with the partially-generated contexts of the pipelined decoding algorithm.

### 3.3. Component 3: Inference Algorithm (Pipelined Parallel Decoding)

**Description**: A sophisticated decoding algorithm that maintains a sliding window of "active" blocks. It uses a dual-state mechanism (semi-activated vs. fully-activated) to balance aggressive parallel decoding with generation stability.

**Implementation Location**:
- `D2F-eval/eval_dream.py` - Complete implementation in the evaluation pipeline
- `D2F-train/utils/generation.py` - Generation utilities

**Function**: To maximize the number of tokens generated per forward pass, thereby boosting throughput beyond what KV Caching alone can provide.

**Code Implementation Features**:
- Block-based parallel generation
- KV cache utilization
- Confidence-based token selection
- Pipelined processing across multiple blocks

**Current Status**: ✅ Fully implemented in evaluation pipeline

**Hypothesis**: This algorithm provides a significant multiplier on top of the speedup from KV Caching, but its effectiveness is critically dependent on the model having undergone the specialized D2F training (Component 2).

## 4. Updated Phased Experimental Plan

This plan has been updated to reflect the current state of the codebase. Since the implementation is largely complete, we focus on configuration, training execution, and systematic ablation studies.

### Phase 0: Environment Setup & Baseline Verification

**Updated Goal**: Configure the existing codebase and verify baseline performance of the original Dream-Base-7B model using the implemented evaluation pipeline.

**Tasks**:

1. **Environment Configuration**: 
   - ✅ Dependencies already specified in `requirements.txt`
   - Set up environment using `uv sync` or conda
   - Configure GPU access via Accelerate

2. **Model and Data Setup**:
   - Download Dream-Base-7B model to local directory
   - Configure paths in `D2F-train/config/dream_eagle.yaml`
   - Verify dataset access (BS17K will auto-download)

3. **Baseline Verification**:
   - Use existing `D2F-eval/eval_dream.py` with original Dream-Base-7B
   - Run on GSM8K to establish baseline TPS and Score
   - Compare against paper results (Table 2)

4. **Infrastructure Testing**:
   - Test multi-GPU setup with Accelerate
   - Verify WandB logging (optional)
   - Test checkpointing and resuming

**Expected Timeline**: 1-2 days

**Exit Criteria**: 
- ✅ Baseline Dream-Base-7B performance verified within 5% of paper results
- ✅ Training pipeline successfully launches without errors
- ✅ Evaluation pipeline produces consistent results

**Command to Execute**:
```bash
cd D2F-eval
python eval_dream.py --model_path ../models/Dream-Base-7B --tasks gsm8k --output_path baseline_results.json
```

### Phase 1: D2F Training Pipeline Execution

**Updated Goal**: Execute the complete D2F training pipeline using the existing implementation to produce a trained D2F-Dream-Base-7B model.

**Tasks**:

1. **Training Configuration**:
   - ✅ Block-wise Causal Attention: Already implemented in `utils/loss.py`
   - ✅ Asymmetric Distillation: Already implemented with `self_align=True`
   - ✅ Monotonic Noise Scheduler: Already implemented in `forward_process_length()`
   - Configure training parameters in `config/dream_custom.yaml`

2. **Model Training Execution**:
   - Run the training pipeline using existing `train.py`
   - Monitor training progress via WandB/logs
   - Implement robust checkpointing (every 2000 steps)
   - Expected duration: 24-48 hours on 4x A100

3. **Training Monitoring**:
   - Track loss convergence
   - Monitor GPU utilization
   - Verify gradient flow and learning dynamics

4. **Model Validation**:
   - Periodic evaluation during training (every 1000 steps)
   - Early stopping if performance plateaus

**Expected Timeline**: 3-5 days (including training time)

**Exit Criteria**: 
- ✅ Training completes successfully with converged loss
- ✅ Trained model checkpoints saved and validated
- ✅ Training logs and metrics properly recorded

**Commands to Execute**:
```bash
cd D2F-train

# Configure paths in config/dream_custom.yaml
# Then launch training
accelerate launch --config_file config/acc_config --num_processes 4 \
  train.py --config config/dream_custom.yaml
```

**Key Implementation Note**: Unlike the original plan, we don't need to implement the components from scratch - they're already complete. The focus is on proper configuration and execution.

### Phase 2: Core Ablation Studies - Isolating the "Why"

**Updated Goal**: Systematically measure the contribution of each D2F component using the existing evaluation framework with specific modifications.

**Implementation Strategy**: Modify the existing evaluation scripts to create controlled ablations while maintaining the same experimental conditions.

#### Experiment 2.1: Isolate the Inference Algorithm

**Setup**: Use the vanilla Dream-Base-7B model (from Phase 0) with the D2F Pipelined Parallel Decoding algorithm.

**Implementation Approach**:
```python
# Modify eval_dream.py to use original model with D2F inference
model_path = "../models/Dream-Base-7B"  # Original model
use_d2f_inference = True  # But use D2F inference algorithm
use_kv_cache = True  # Enable KV caching
```

**Goal**: Test if the algorithm alone provides benefits without the specialized training.

**Expected Outcome**: Moderate speedup but potentially degraded quality score, demonstrating the necessity of the D2F training strategy.

**Implementation Files**:
- Create `eval_dream_ablation_exp21.py` based on `eval_dream.py`
- Modify inference to use D2F algorithm with original model

#### Experiment 2.2: Isolate the KV Cache / Architecture

**Setup**: Use the trained D2F-Dream-Base-7B model (from Phase 1) but replace the Pipelined Parallel Decoding with simple, serial block-by-block decoding.

**Implementation Approach**:
```python
# Modify eval_dream.py for cache-only mode
model_path = "../experiments/trained_d2f_model"  # Trained D2F model
use_d2f_inference = False  # Disable parallel decoding
use_kv_cache = True  # But keep KV caching
serial_block_decoding = True  # Decode blocks sequentially
```

**Goal**: Quantify the speedup from KV Caching alone, without inter-block parallelism.

**Expected Outcome**: Significant speedup over baseline but slower than full D2F. Quality should remain high.

**Implementation Files**:
- Create `eval_dream_ablation_exp22.py` 
- Implement serial block-by-block decoding mode

#### Additional Ablation: Training Strategy Isolation

**New Experiment 2.3**: Test the effect of asymmetric distillation vs. standard fine-tuning.

**Setup**: Train a model with D2F architecture but without asymmetric distillation (`self_align=False`).

**Implementation**: Modify training config:
```yaml
train:
  self_align: false  # Disable asymmetric distillation
  # Keep all other D2F components
```

#### Systematic Ablation Matrix

| Experiment | Model | Training | Inference | KV Cache | Expected Result |
|------------|-------|----------|-----------|----------|-----------------|
| Baseline | Original | Standard | Standard | No | Baseline TPS/Score |
| Exp 2.1 | Original | Standard | D2F | Yes | ↑TPS, ↓Score |
| Exp 2.2 | D2F | D2F | Serial | Yes | ↑↑TPS, ↑Score |
| Exp 2.3 | D2F | Standard | D2F | Yes | ↑↑TPS, ↓Score |
| Full D2F | D2F | D2F | D2F | Yes | ↑↑↑TPS, ↑Score |

### Phase 3: Synthesis, Reporting, and Future Work

**Updated Goal**: Comprehensive analysis of ablation results and documentation of findings with actionable insights.

**Tasks**:

1. **Data Analysis & Visualization**:
   - Aggregate results from all ablation experiments
   - Create performance comparison tables and charts
   - Calculate component-wise contribution percentages
   - Statistical significance testing of results

2. **Component Contribution Analysis**:
   - Quantify speedup attribution: Architecture vs. Training vs. Inference
   - Analyze quality-speed trade-offs for each component
   - Identify synergistic effects between components

3. **Technical Report Generation**:
   - Executive summary answering the core research question
   - Detailed methodology and implementation notes
   - Results section with comprehensive ablation analysis
   - Discussion of implications for future dLLM development

4. **Implementation Documentation**:
   - Code documentation and commenting
   - Reproducibility guide with exact commands
   - Configuration files for each experiment
   - Docker/container setup for full reproducibility

**Expected Timeline**: 2-3 days

**Exit Criteria**: 
- ✅ Complete technical report answering all research questions
- ✅ Quantified contribution of each D2F component
- ✅ Reproducible experimental setup documented

**Deliverables**:
- Technical report (15-20 pages)
- Ablation results summary table
- Performance visualization plots
- Complete reproducibility package

---

## 5. Implementation Guide & Current Code Analysis

### 5.1. Key Configuration Files

**Primary Training Configuration**: `D2F-train/config/dream_custom.yaml`
```yaml
training_mode: 'dream'
paths:
  model: './models/Dream-Base-7B'
  experiment: './experiments'
  data:
    bs: 'Lansechen/bs17k_collection_filtered_hard_maxlength600'

train:
  exp_name: 'dream_d2f_retrain'
  lr: 5e-6
  num_iters: 50000
  block_size: 16
  self_align: true    # Key: enables asymmetric distillation
  enable_shift: true  # Key: enables logit shifting
```

**Accelerate Configuration**: `D2F-train/config/acc_config`
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
use_cpu: false
mixed_precision: fp16
```

### 5.2. Critical Implementation Details

**Block-wise Attention Mask**:
- Location: `D2F-eval/eval_dream.py:45-91`
- Function: `create_full_block_attention_mask()`
- Creates causal attention between blocks, bidirectional within blocks

**Asymmetric Distillation Loss**:
- Location: `D2F-train/utils/loss.py:35-79`
- Teacher: Original model (no LoRA adapter)
- Student: Model with LoRA adapter + block-wise attention
- Loss: KL divergence between teacher and student outputs

**Monotonic Noise Scheduling**:
- Location: `D2F-train/utils/util.py` (referenced in loss computation)
- Progressive masking across blocks
- Higher noise levels in later blocks

### 5.3. Evaluation Pipeline Integration

**Standard Evaluation**:
```bash
cd D2F-eval
python eval_dream.py --model_path <path> --tasks gsm8k,mbpp,humaneval --output_path results.json
```

**Performance Metrics Collected**:
- Tokens Per Second (TPS) 
- Task-specific accuracy scores
- Memory usage statistics
- Inference latency breakdown

---

## 6. Practical Execution Steps

### 6.1. Quick Start Guide

**Step 1: Environment Setup**
```bash
cd /home/hice1/zdu89/scratch/dannyliu/Projects/Discrete-Diffusion-Forcing
uv sync  # or pip install -r requirements.txt
```

**Step 2: Model Download & Configuration**
```bash
# Download models
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('Dream-org/Dream-v0-Base-7B', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Dream-org/Dream-v0-Base-7B', trust_remote_code=True)
model.save_pretrained('./models/Dream-Base-7B')
tokenizer.save_pretrained('./models/Dream-Base-7B')
"

# Update config paths
sed -i 's|Dream-org/Dream-v0-Base-7B|./models/Dream-Base-7B|g' D2F-train/config/dream_eagle.yaml
```

**Step 3: Baseline Verification**
```bash
cd D2F-eval
python eval_dream.py --model_path ../models/Dream-Base-7B --tasks gsm8k --output_path baseline_results.json
```

**Step 4: Training Execution**
```bash
cd D2F-train
accelerate launch --config_file config/acc_config --num_processes 4 train.py --config config/dream_eagle.yaml
```

**Step 5: Ablation Studies**
```bash
# Create ablation scripts based on eval_dream.py
# Run systematic experiments following Phase 2 plan
```

### 6.2. Expected Resource Requirements

**Compute**: 4x A100 40GB GPUs (minimum 24GB VRAM per GPU)
**Storage**: 500GB for models, datasets, and outputs
**Time**: 
- Phase 0: 4-8 hours
- Phase 1: 24-48 hours (training)
- Phase 2: 8-16 hours (ablations)
- Phase 3: 8-16 hours (analysis)

**Total Project Timeline**: 1-2 weeks

### 6.3. Success Metrics

**Technical Success**:
- ✅ Reproduce paper baseline within 5%
- ✅ Complete D2F training with converged loss
- ✅ Achieve expected speedup in ablation studies

**Research Success**:
- ✅ Quantify contribution of each D2F component
- ✅ Identify key factors for dLLM acceleration
- ✅ Provide actionable insights for future research

**Implementation Success**:
- ✅ Fully documented and reproducible experimental setup
- ✅ Modular code supporting future extensions
- ✅ Comprehensive technical report with clear findings

---

## Summary

This updated master plan reflects the current state of the D2F repository and provides a clear, actionable roadmap for systematic deconstruction of the D2F framework. The key changes from the original plan include:

### ✅ **Implementation Status**: 
- Training pipeline is **complete** and ready for execution
- Evaluation framework is **complete** with comprehensive benchmarks
- All core D2F components are **already implemented**

### 🔄 **Focus Shift**: 
From "implement everything" to "configure, execute, and analyze"

### 📊 **Enhanced Ablation Design**: 
Systematic ablation matrix with specific implementation approaches

### 🚀 **Practical Execution**:
Step-by-step commands and timeline for immediate project execution

### 📋 **Resource Planning**: 
Clear compute requirements and expected timelines

The repository is in excellent condition for conducting the planned research. The main tasks now involve proper configuration, systematic execution of the training and evaluation pipelines, and methodical ablation studies to quantify the contribution of each D2F component.

**Next Immediate Steps**:
1. Follow the detailed tutorial in `notes/retrain_tutorial.md`
2. Execute Phase 0 (baseline verification) 
3. Proceed with Phase 1 (D2F training)
4. Conduct systematic ablation studies (Phase 2)
5. Generate comprehensive analysis report (Phase 3)

This approach will provide definitive answers to the core research question: **"What makes D2F successful?"**