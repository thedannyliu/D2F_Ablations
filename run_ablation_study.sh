#!/bin/bash

# Complete Ablation Study Runner
# Executes all 5 experiments in the ablation study systematically

set -e  # Exit on any error

# Configuration
CACHE_DIR="/scratch/dannyliu/.cache/huggingface"
PROJECT_ROOT="/home/hice1/zdu89/scratch/dannyliu/Projects/Discrete-Diffusion-Forcing"
ORIGINAL_MODEL="${CACHE_DIR}/hub/models--Dream-org--Dream-v0-Base-7B"
RESULTS_DIR="${PROJECT_ROOT}/experiments/ablation_results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Set environment variables
export HF_HOME="${CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "=== Starting Complete D2F Ablation Study ==="
echo "Original Model: ${ORIGINAL_MODEL}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=================================================="

# Function to run training with error handling
run_training() {
    local config_file=$1
    local exp_name=$2
    
    echo ""
    echo "🚀 Starting Training: ${exp_name}"
    echo "Config: ${config_file}"
    echo "----------------------------------------"
    
    cd "${PROJECT_ROOT}/D2F-train"
    
    # Check if config file exists
    if [[ ! -f "${config_file}" ]]; then
        echo "❌ Error: Config file not found: ${config_file}"
        return 1
    fi
    
    # Run training
    accelerate launch \
        --config_file config/acc_config \
        --num_processes 1 \
        --main_process_port 29500 \
        train.py \
        --config "${config_file}"
        
    if [[ $? -eq 0 ]]; then
        echo "✅ Training completed: ${exp_name}"
    else
        echo "❌ Training failed: ${exp_name}"
        return 1
    fi
}

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local output_file=$2
    local exp_name=$3
    local eval_script=$4
    local additional_args=$5
    
    echo ""
    echo "📊 Starting Evaluation: ${exp_name}"
    echo "Model: ${model_path}"
    echo "Output: ${output_file}"
    echo "----------------------------------------"
    
    cd "${PROJECT_ROOT}/D2F-eval"
    
    # Check if model exists
    if [[ ! -d "${model_path}" && ! -f "${model_path}" ]]; then
        echo "❌ Error: Model not found: ${model_path}"
        return 1
    fi
    
    # Run evaluation
    python "${eval_script}" \
        --model_path "${model_path}" \
        --tasks "gsm8k" \
        --batch_size 1 \
        --device cuda \
        --output_path "${output_file}" \
        ${additional_args}
        
    if [[ $? -eq 0 ]]; then
        echo "✅ Evaluation completed: ${exp_name}"
    else
        echo "❌ Evaluation failed: ${exp_name}"
        return 1
    fi
}

# =============================================================================
# STEP 1: BASELINE EXPERIMENT
# =============================================================================
echo ""
echo "🎯 STEP 1: BASELINE (Original Dream Model)"
echo "============================================="

# Check if baseline training is needed
BASELINE_MODEL="${PROJECT_ROOT}/experiments/ablation/baseline_original_dream"
if [[ ! -d "${BASELINE_MODEL}" ]]; then
    echo "Training baseline model..."
    run_training "config/exp_baseline.yaml" "Baseline"
else
    echo "✅ Baseline model already exists, skipping training"
fi

# Find latest baseline checkpoint
BASELINE_CHECKPOINT=$(find "${BASELINE_MODEL}" -name "Decoder-baseline_original_dream-*" -type d | sort -V | tail -1)
if [[ -z "${BASELINE_CHECKPOINT}" ]]; then
    echo "❌ No baseline checkpoint found!"
    exit 1
fi

echo "Using baseline checkpoint: ${BASELINE_CHECKPOINT}"

# Evaluate baseline
run_evaluation \
    "${BASELINE_CHECKPOINT}" \
    "${RESULTS_DIR}/baseline_results.json" \
    "Baseline" \
    "eval_dream.py" \
    ""

# =============================================================================
# STEP 2: EXPERIMENT 2.1 (Original Model + D2F Inference)
# =============================================================================
echo ""
echo "🎯 STEP 2: EXPERIMENT 2.1 (Original + D2F Inference)"
echo "===================================================="

# Use original model with D2F inference
run_evaluation \
    "${ORIGINAL_MODEL}" \
    "${RESULTS_DIR}/exp_2_1_results.json" \
    "Exp 2.1" \
    "eval_ablation_exp_2_1.py" \
    "--block_size 16"

# =============================================================================
# STEP 3: EXPERIMENT 2.3 (D2F Architecture + Standard Training)
# =============================================================================
echo ""
echo "🎯 STEP 3: EXPERIMENT 2.3 (D2F Architecture + Standard Training)"
echo "=================================================================="

# Train D2F architecture without asymmetric distillation
EXP_2_3_MODEL="${PROJECT_ROOT}/experiments/ablation/exp_2_3_d2f_standard_training"
if [[ ! -d "${EXP_2_3_MODEL}" ]]; then
    echo "Training Exp 2.3 model..."
    run_training "config/exp_2_3_d2f_standard_training.yaml" "Exp 2.3"
else
    echo "✅ Exp 2.3 model already exists, skipping training"
fi

# Find latest Exp 2.3 checkpoint
EXP_2_3_CHECKPOINT=$(find "${EXP_2_3_MODEL}" -name "Decoder-exp_2_3_d2f_standard_training-*" -type d | sort -V | tail -1)
if [[ -z "${EXP_2_3_CHECKPOINT}" ]]; then
    echo "❌ No Exp 2.3 checkpoint found!"
    exit 1
fi

echo "Using Exp 2.3 checkpoint: ${EXP_2_3_CHECKPOINT}"

# Evaluate Exp 2.3
run_evaluation \
    "${EXP_2_3_CHECKPOINT}" \
    "${RESULTS_DIR}/exp_2_3_results.json" \
    "Exp 2.3" \
    "eval_dream.py" \
    ""

# =============================================================================
# STEP 4: FULL D2F EXPERIMENT
# =============================================================================
echo ""
echo "🎯 STEP 4: FULL D2F (Complete Implementation)"
echo "=============================================="

# Train full D2F model
FULL_D2F_MODEL="${PROJECT_ROOT}/experiments/ablation/full_d2f_complete"
if [[ ! -d "${FULL_D2F_MODEL}" ]]; then
    echo "Training Full D2F model..."
    run_training "config/exp_full_d2f.yaml" "Full D2F"
else
    echo "✅ Full D2F model already exists, skipping training"
fi

# Find latest Full D2F checkpoint
FULL_D2F_CHECKPOINT=$(find "${FULL_D2F_MODEL}" -name "Decoder-full_d2f_complete-*" -type d | sort -V | tail -1)
if [[ -z "${FULL_D2F_CHECKPOINT}" ]]; then
    echo "❌ No Full D2F checkpoint found!"
    exit 1
fi

echo "Using Full D2F checkpoint: ${FULL_D2F_CHECKPOINT}"

# Evaluate Full D2F
run_evaluation \
    "${FULL_D2F_CHECKPOINT}" \
    "${RESULTS_DIR}/full_d2f_results.json" \
    "Full D2F" \
    "eval_dream.py" \
    ""

# =============================================================================
# STEP 5: EXPERIMENT 2.2 (D2F Model + Cache-Only Inference)
# =============================================================================
echo ""
echo "🎯 STEP 5: EXPERIMENT 2.2 (D2F Model + Cache-Only)"
echo "=================================================="

# Use trained D2F model with cache-only inference
run_evaluation \
    "${FULL_D2F_CHECKPOINT}" \
    "${RESULTS_DIR}/exp_2_2_results.json" \
    "Exp 2.2" \
    "eval_ablation_exp_2_2.py" \
    "--block_size 16"

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
echo ""
echo "🎯 STEP 6: RESULTS ANALYSIS"
echo "============================"

cd "${PROJECT_ROOT}"

# Create analysis script
cat > analyze_ablation_results.py << 'EOF'
import json
import pandas as pd
from pathlib import Path

def analyze_results():
    results_dir = Path("experiments/ablation_results")
    
    experiments = {
        "Baseline": results_dir / "baseline_results.json",
        "Exp 2.1 (Original + D2F Inference)": results_dir / "exp_2_1_results.json",
        "Exp 2.2 (D2F + Cache-Only)": results_dir / "exp_2_2_results.json", 
        "Exp 2.3 (D2F + Standard Training)": results_dir / "exp_2_3_results.json",
        "Full D2F": results_dir / "full_d2f_results.json"
    }
    
    results = {}
    for name, path in experiments.items():
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                results[name] = data
            print(f"✅ Loaded: {name}")
        else:
            print(f"❌ Missing: {name}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    
    for exp_name, data in results.items():
        print(f"\n{exp_name}:")
        print("-" * len(exp_name))
        
        if "results" in data and "gsm8k" in data["results"]:
            gsm8k_results = data["results"]["gsm8k"]
            score = gsm8k_results.get("acc", "N/A")
            print(f"  GSM8K Accuracy: {score}")
        
        if "experiment_info" in data:
            exp_info = data["experiment_info"]
            if "total_evaluation_time" in exp_info:
                time = exp_info["total_evaluation_time"]
                print(f"  Evaluation Time: {time:.2f}s")
    
    print(f"\nDetailed results saved in: {results_dir}")
    print("Analysis complete! 🎉")

if __name__ == "__main__":
    analyze_results()
EOF

# Run analysis
python analyze_ablation_results.py

echo ""
echo "🎉 ABLATION STUDY COMPLETE!"
echo "============================"
echo "All results saved in: ${RESULTS_DIR}"
echo ""
echo "Summary of experiments conducted:"
echo "1. ✅ Baseline (Original Dream)"
echo "2. ✅ Exp 2.1 (Original + D2F Inference)"
echo "3. ✅ Exp 2.2 (D2F + Cache-Only)"
echo "4. ✅ Exp 2.3 (D2F + Standard Training)"
echo "5. ✅ Full D2F (Complete)"
echo ""
echo "Check the results files for detailed metrics and analysis!"
