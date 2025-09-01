# D2F 完整重訓練教學指南
## Discrete Diffusion Forcing (D2F) Complete Retraining Tutorial

**文檔版本**: v1.0  
**更新日期**: 2025-01-18  
**適用對象**: 研究人員、工程師  
**前置知識**: Python, PyTorch, Transformers, Linux命令行  

---

## 目錄

1. [環境設置與依賴安裝](#1-環境設置與依賴安裝)
2. [數據準備與下載](#2-數據準備與下載)
3. [模型準備與配置](#3-模型準備與配置)
4. [訓練管道詳解](#4-訓練管道詳解)
5. [評估與測試](#5-評估與測試)
6. [Ablation Study 設計](#6-ablation-study-設計)
7. [故障排除與調試](#7-故障排除與調試)
8. [高級配置與優化](#8-高級配置與優化)

---

## 1. 環境設置與依賴安裝

### 1.1 系統需求

**硬體需求**:
- GPU: NVIDIA GPU with >= 24GB VRAM (推薦 A100/H100)
- CPU: >= 16 cores
- RAM: >= 64GB
- 儲存空間: >= 500GB (用於模型、數據集、輸出)

**軟體需求**:
- Ubuntu 18.04+ / CentOS 7+
- Python 3.10+
- CUDA 11.8+ / 12.0+
- Git

### 1.2 創建虛擬環境

#### 方法1: 使用 UV (推薦)

```bash
# 確保已安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 進入專案目錄
cd /home/hice1/zdu89/scratch/dannyliu/Projects/Discrete-Diffusion-Forcing

# 使用 uv 同步環境
uv sync
```

#### 方法2: 使用 Conda

```bash
# 創建新的 conda 環境
conda create -n d2f python=3.10 -y
conda activate d2f

# 安裝基礎套件
pip install -r requirements.txt

# 安裝額外的必要套件
pip install wandb  # 用於實驗追蹤
pip install tensorboard  # 可選，用於監控
```

### 1.3 驗證安裝

```bash
# 驗證 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 驗證 Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 驗證 Accelerate
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
```

---

## 2. 數據準備與下載

### 2.1 數據集概述

D2F 訓練需要以下數據集：

1. **BS17K**: 數學推理數據集 (主要訓練集)
2. **GSM8K**: 數學問題評估集
3. **MBPP**: 程式碼生成評估集  
4. **HumanEval**: 程式碼評估集
5. **Math**: 數學推理評估集

### 2.2 下載訓練數據集

#### BS17K 數據集下載

```bash
# 方法1: 直接從 HuggingFace 下載 (推薦)
# 這個會在第一次訓練時自動下載
# 無需手動操作，但確保網路連接正常

# 方法2: 手動下載並配置
# 創建數據目錄
mkdir -p ./data/bs17k

# 使用 Python 腳本下載
python -c "
from datasets import load_dataset
dataset = load_dataset('Lansechen/bs17k_collection_filtered_hard_maxlength600', split='train')
dataset.save_to_disk('./data/bs17k')
print(f'Downloaded {len(dataset)} samples')
"
```

#### 評估數據集下載

```bash
# 創建評估數據目錄
mkdir -p ./data/eval

# GSM8K
python -c "
from datasets import load_dataset
dataset = load_dataset('gsm8k', 'main', split='test')
dataset.save_to_disk('./data/eval/gsm8k')
"

# MBPP
python -c "
from datasets import load_dataset
dataset = load_dataset('mbpp', split='test')
dataset.save_to_disk('./data/eval/mbpp')
"

# HumanEval
python -c "
from datasets import load_dataset
dataset = load_dataset('openai_humaneval', split='test')
dataset.save_to_disk('./data/eval/humaneval')
"
```

### 2.3 數據集檢查

```bash
# 檢查數據集是否正確下載
python -c "
import os
from datasets import load_from_disk

# 檢查訓練數據
if os.path.exists('./data/bs17k'):
    dataset = load_from_disk('./data/bs17k')
    print(f'BS17K training samples: {len(dataset)}')
    print('Sample:', dataset[0])
else:
    print('BS17K dataset not found. Will download during training.')

# 檢查評估數據
for name in ['gsm8k', 'mbpp', 'humaneval']:
    path = f'./data/eval/{name}'
    if os.path.exists(path):
        dataset = load_from_disk(path)
        print(f'{name} eval samples: {len(dataset)}')
"
```

---

## 3. 模型準備與配置

### 3.1 下載預訓練模型

#### Dream-Base-7B 模型

```bash
# 創建模型目錄
mkdir -p ./models

# 下載 Dream-Base-7B
python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_path = './models/Dream-Base-7B'
os.makedirs(model_path, exist_ok=True)

# 下載模型和 tokenizer
model = AutoModel.from_pretrained('Dream-org/Dream-v0-Base-7B', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Dream-org/Dream-v0-Base-7B', trust_remote_code=True)

# 保存到本地
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f'Model saved to {model_path}')
"
```

#### LLaDA-8B 模型 (可選)

```bash
# 如果要訓練 LLaDA 模型
python -c "
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig
import os

model_path = './models/LLaDA-8B'
os.makedirs(model_path, exist_ok=True)

# 注意: 需要有權限訪問 LLaDA 模型
# 這裡提供框架，實際路徑需要根據你的 LLaDA 模型位置調整
print('LLaDA model setup - modify path in config file')
"
```

### 3.2 配置文件設置

#### 修改 Dream 配置

```bash
# 複製並修改配置文件
cp D2F-train/config/dream_eagle.yaml D2F-train/config/dream_custom.yaml
```

編輯 `D2F-train/config/dream_custom.yaml`:

```yaml
# Training mode configuration
training_mode: 'dream'

# Model and data path configuration  
paths:
  model: './models/Dream-Base-7B'  # 更新為你的本地路徑
  experiment: './experiments'       # 實驗輸出目錄
  data:
    bs: 'Lansechen/bs17k_collection_filtered_hard_maxlength600'  # 或本地路徑
    bs_easy: 'Lansechen/bs17k_collection_filtered_easy_maxlength600'

denoiser:
  encoder:
    name: 'dream'
    mask_id: 151666

  decoder:
    wiinit: true
    name: 'eagle_rope'
    num_blocks: 1
    seq_len: &seq_len 512
    input_dim: 3584
    hidden_dim: &dim 3584
    vocab_size: 152064
    block:
      seq_len: *seq_len
      hidden_dim: *dim
      num_heads: 32

train:
  decoder_resume_path:        # 如果從 checkpoint 恢復，設置路徑
  head_resume_path:
  skipped_keys:
  global_step:
  exp_name: &exp_name 'dream_d2f_retrain'
  wandb_proj: *exp_name
  output_dir: 'dream_d2f_output'
  logging_dir: 'logs'
  mixed_precision: 'fp16'
  gradient_accumulation_steps: 8    # 根據你的 GPU 記憶體調整
  report_to: 'wandb'                # 或 'no' 如果不使用 wandb
  block_size: 16 
  
  lr: 5e-6
  num_iters: 50000
  eval_every: 1000              # 更頻繁的評估
  save_every: 2000              # 更頻繁的保存

  enable_shift: true
  share_steps: 2
  self_align: true
  feature_align: false
  self_step: true

data:
  name: 'bs17k'
  batch_size: 1                 # 根據 GPU 記憶體調整
  max_length: *seq_len
```

### 3.3 創建實驗目錄結構

```bash
# 創建完整的實驗目錄結構
mkdir -p experiments/logs
mkdir -p experiments/checkpoints
mkdir -p experiments/results
mkdir -p experiments/ablation_studies

# 設置權限
chmod -R 755 experiments/
```

---

## 4. 訓練管道詳解

### 4.1 理解 D2F 訓練原理

D2F 訓練包含三個核心元件：

1. **Block-wise Causal Attention**: 修改注意力機制
2. **Asymmetric Distillation**: 不對稱蒸餾訓練策略
3. **Monotonic Noise Schedule**: 單調噪聲排程

### 4.2 單 GPU 訓練

```bash
cd D2F-train

# 設置環境變數
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 運行訓練
accelerate launch \
    --config_file config/acc_config \
    --num_processes 1 \
    --main_process_port 29500 \
    train.py \
    --config config/dream_custom.yaml
```

### 4.3 多 GPU 訓練

```bash
# 配置 accelerate
accelerate config

# 設置多 GPU 訓練
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 運行多 GPU 訓練
accelerate launch \
    --config_file config/acc_config \
    --num_processes 4 \
    --main_process_port 29500 \
    train.py \
    --config config/dream_custom.yaml
```

### 4.4 在 SLURM 集群上運行

創建 SLURM 腳本 `train_d2f.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=d2f_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 載入模組
module load cuda/11.8
module load python/3.10

# 激活環境
source activate d2f

# 設置環境變數
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="d2f_retraining"

# 進入工作目錄
cd $SLURM_SUBMIT_DIR/D2F-train

# 運行訓練
accelerate launch \
    --config_file config/acc_config \
    --num_processes 4 \
    --main_process_port 29500 \
    train.py \
    --config config/dream_custom.yaml
```

提交作業:

```bash
sbatch train_d2f.slurm
```

### 4.5 監控訓練進度

#### 使用 WandB

```bash
# 登入 WandB (首次使用)
wandb login

# 查看訓練進度
wandb online  # 確保在線模式
```

#### 使用終端監控

```bash
# 實時查看日誌
tail -f logs/train_*.out

# 查看 GPU 使用情況
watch -n 1 nvidia-smi

# 檢查檢查點
ls -la experiments/checkpoints/
```

### 4.6 從檢查點恢復訓練

如果訓練中斷，修改配置文件恢復：

```yaml
train:
  decoder_resume_path: './experiments/checkpoints/Decoder-dream_d2f_retrain-10k'
  global_step: 10000  # 設置為檢查點的步數
```

---

## 5. 評估與測試

### 5.1 評估 Dream 模型

```bash
cd D2F-eval

# 設置模型路徑
export MODEL_PATH="../experiments/checkpoints/Decoder-dream_d2f_retrain-50k"

# 運行完整評估
bash eval_dream.sh
```

### 5.2 單獨評估各個基準

#### GSM8K 評估

```bash
python eval_dream.py \
    --model_path $MODEL_PATH \
    --tasks gsm8k \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results/gsm8k_results.json
```

#### MBPP 評估

```bash
python eval_dream.py \
    --model_path $MODEL_PATH \
    --tasks mbpp \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results/mbpp_results.json
```

#### HumanEval 評估

```bash
# 運行 HumanEval
python eval_dream.py \
    --model_path $MODEL_PATH \
    --tasks humaneval \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results/humaneval_results.json

# 後處理 (計算 pass@1)
python postprocess_code.py ./results/samples_humaneval_*.jsonl
```

### 5.3 性能基準測試

創建 `benchmark_speed.py`:

```python
import time
import torch
from transformers import AutoTokenizer, AutoModel

def benchmark_model(model_path, num_samples=100):
    """基準測試模型速度"""
    # 載入模型
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model.cuda()
    model.eval()
    
    # 準備測試數據
    test_prompt = "Solve the following math problem: What is 25 + 37?"
    inputs = tokenizer(test_prompt, return_tensors="pt").cuda()
    
    # 預熱
    with torch.no_grad():
        for _ in range(10):
            _ = model.generate(**inputs, max_length=512)
    
    # 實際測試
    start_time = time.time()
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(**inputs, max_length=512)
            total_tokens += outputs.shape[1]
    
    end_time = time.time()
    
    # 計算 TPS
    total_time = end_time - start_time
    tps = total_tokens / total_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second (TPS): {tps:.2f}")
    
    return tps

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../experiments/checkpoints/latest"
    benchmark_model(model_path)
```

運行基準測試:

```bash
python benchmark_speed.py $MODEL_PATH
```

---

## 6. Ablation Study 設計

### 6.1 Ablation Study 概述

根據 master plan，我們需要進行以下消融實驗：

1. **實驗 2.1**: 隔離推理算法的效果
2. **實驗 2.2**: 隔離 KV Cache/架構的效果  
3. **完整對比**: 比較所有元件的貢獻

### 6.2 實驗 2.1: 推理算法隔離

測試原版 Dream 模型 + D2F 推理算法：

```bash
# 創建實驗配置
mkdir -p experiments/ablation/exp2_1

# 修改評估腳本，使用原版模型但 D2F 推理
python eval_dream.py \
    --model_path "./models/Dream-Base-7B" \
    --use_d2f_inference \
    --tasks gsm8k \
    --output_path ./experiments/ablation/exp2_1/results.json
```

### 6.3 實驗 2.2: 架構/KV Cache 隔離

使用訓練好的 D2F 模型，但簡單的串行解碼：

```bash
# 創建實驗配置
mkdir -p experiments/ablation/exp2_2

# 使用 cache-only 模式
python eval_dream.py \
    --model_path $MODEL_PATH \
    --cache_only_mode \
    --tasks gsm8k \
    --output_path ./experiments/ablation/exp2_2/results.json
```

### 6.4 完整對比實驗

創建對比腳本 `run_ablation_study.sh`:

```bash
#!/bin/bash

# 設置路徑
ORIGINAL_MODEL="./models/Dream-Base-7B"
D2F_MODEL="../experiments/checkpoints/Decoder-dream_d2f_retrain-50k"
OUTPUT_DIR="./experiments/ablation"

mkdir -p $OUTPUT_DIR

echo "Running complete ablation study..."

# 基線：原版 Dream 模型
echo "1. Running baseline (Original Dream)..."
python eval_dream.py \
    --model_path $ORIGINAL_MODEL \
    --tasks gsm8k,mbpp,humaneval \
    --output_path $OUTPUT_DIR/baseline_results.json

# 實驗 2.1：原版模型 + D2F 推理
echo "2. Running Exp 2.1 (Original + D2F Inference)..."
python eval_dream.py \
    --model_path $ORIGINAL_MODEL \
    --use_d2f_inference \
    --tasks gsm8k,mbpp,humaneval \
    --output_path $OUTPUT_DIR/exp2_1_results.json

# 實驗 2.2：D2F 模型 + Cache-only
echo "3. Running Exp 2.2 (D2F + Cache-only)..."
python eval_dream.py \
    --model_path $D2F_MODEL \
    --cache_only_mode \
    --tasks gsm8k,mbpp,humaneval \
    --output_path $OUTPUT_DIR/exp2_2_results.json

# 完整 D2F
echo "4. Running Full D2F..."
python eval_dream.py \
    --model_path $D2F_MODEL \
    --tasks gsm8k,mbpp,humaneval \
    --output_path $OUTPUT_DIR/full_d2f_results.json

echo "Ablation study completed!"
```

運行消融實驗:

```bash
chmod +x run_ablation_study.sh
./run_ablation_study.sh
```

### 6.5 結果分析

創建分析腳本 `analyze_ablation.py`:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

def analyze_ablation_results():
    """分析消融實驗結果"""
    
    # 讀取結果
    experiments = {
        'Baseline (Original Dream)': 'experiments/ablation/baseline_results.json',
        'Exp 2.1 (Original + D2F Inference)': 'experiments/ablation/exp2_1_results.json', 
        'Exp 2.2 (D2F + Cache-only)': 'experiments/ablation/exp2_2_results.json',
        'Full D2F': 'experiments/ablation/full_d2f_results.json'
    }
    
    results = {}
    for name, path in experiments.items():
        try:
            with open(path, 'r') as f:
                results[name] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            continue
    
    # 創建對比表
    metrics = ['gsm8k_score', 'gsm8k_tps', 'mbpp_score', 'mbpp_tps', 'humaneval_score', 'humaneval_tps']
    df = pd.DataFrame(index=experiments.keys(), columns=metrics)
    
    for exp_name, data in results.items():
        for metric in metrics:
            if metric in data:
                df.loc[exp_name, metric] = data[metric]
    
    # 保存表格
    df.to_csv('experiments/ablation/ablation_summary.csv')
    print("Ablation Summary:")
    print(df)
    
    # 計算貢獻度
    if 'Full D2F' in results and 'Baseline (Original Dream)' in results:
        baseline_tps = results['Baseline (Original Dream)'].get('gsm8k_tps', 0)
        full_tps = results['Full D2F'].get('gsm8k_tps', 0)
        speedup = full_tps / baseline_tps if baseline_tps > 0 else 0
        print(f"\nOverall Speedup: {speedup:.2f}x")
        
        # 分析各元件貢獻
        if 'Exp 2.2 (D2F + Cache-only)' in results:
            cache_tps = results['Exp 2.2 (D2F + Cache-only)'].get('gsm8k_tps', 0)
            cache_contribution = cache_tps / baseline_tps if baseline_tps > 0 else 0
            inference_contribution = full_tps / cache_tps if cache_tps > 0 else 0
            
            print(f"Cache/Architecture contribution: {cache_contribution:.2f}x")
            print(f"Inference algorithm contribution: {inference_contribution:.2f}x")

if __name__ == "__main__":
    analyze_ablation_results()
```

運行分析:

```bash
python analyze_ablation.py
```

---

## 7. 故障排除與調試

### 7.1 常見問題與解決方案

#### 記憶體不足 (CUDA OOM)

```bash
# 解決方案 1: 減少批次大小
# 修改配置文件中的 batch_size: 1

# 解決方案 2: 使用梯度累積
# 修改 gradient_accumulation_steps: 16

# 解決方案 3: 使用混合精度
# 設置 mixed_precision: 'fp16'

# 解決方案 4: 啟用梯度檢查點
# 在模型配置中添加 gradient_checkpointing: true
```

#### 模型載入錯誤

```bash
# 檢查模型文件完整性
python -c "
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained('./models/Dream-Base-7B', trust_remote_code=True)
    print('Model loaded successfully')
except Exception as e:
    print(f'Error: {e}')
"

# 清除快取重新下載
rm -rf ~/.cache/huggingface/
```

#### 訓練停滯或發散

```bash
# 檢查學習率
# 降低學習率: lr: 1e-6

# 檢查梯度裁剪
# 確保 gradient_clipping: 1.0

# 檢查資料質量
python -c "
from datasets import load_dataset
dataset = load_dataset('Lansechen/bs17k_collection_filtered_hard_maxlength600', split='train')
print('Sample data:', dataset[0])
print('Dataset size:', len(dataset))
"
```

### 7.2 調試工具

#### 啟用詳細日誌

```python
# 在訓練腳本開始添加
import logging
logging.basicConfig(level=logging.DEBUG)

# 啟用 transformers 日誌
import transformers
transformers.logging.set_verbosity_debug()
```

#### 檢查模型輸出

```python
# 添加到訓練循環中進行調試
print(f"Input shape: {input_ids.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss value: {loss.item()}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
```

#### 監控梯度

```python
# 檢查梯度範數
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm}")
```

---

## 8. 高級配置與優化

### 8.1 多節點分散式訓練

#### 配置 torchrun

```bash
# 節點 0 (主節點)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py --config config/dream_custom.yaml

# 節點 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py --config config/dream_custom.yaml
```

### 8.2 模型並行化

```python
# 在大型模型上使用模型並行
from torch.nn.parallel import DistributedDataParallel as DDP

# 配置模型並行
model = DDP(model, device_ids=[local_rank])
```

### 8.3 數據流水線優化

```python
# 優化數據載入
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,           # 增加工作進程
    pin_memory=True,         # 固定記憶體
    prefetch_factor=4,       # 預取數據
    persistent_workers=True  # 持久工作進程
)
```

### 8.4 檢查點策略

```yaml
# 高級檢查點配置
train:
  save_every: 1000
  keep_last_n_checkpoints: 5    # 只保留最近 5 個檢查點
  save_optimizer_state: true    # 保存優化器狀態
  save_scheduler_state: true    # 保存調度器狀態
```

### 8.5 實驗追蹤進階配置

```python
# WandB 進階配置
import wandb

wandb.init(
    project="d2f_retraining",
    name=f"dream_d2f_{timestamp}",
    config={
        "model": "Dream-Base-7B",
        "dataset": "BS17K", 
        "batch_size": config.data.batch_size,
        "learning_rate": config.train.lr,
        "block_size": config.train.block_size,
    },
    tags=["d2f", "ablation", "dream"],
    notes="Complete D2F retraining with ablation studies"
)

# 記錄模型架構
wandb.watch(model, log="all")
```

---

## 總結

這個完整的重訓練教學涵蓋了：

1. ✅ **環境設置**: 完整的依賴安裝和環境配置
2. ✅ **數據準備**: 詳細的數據集下載和預處理
3. ✅ **模型配置**: 模型下載和配置文件設置
4. ✅ **訓練流程**: 單/多GPU、SLURM集群訓練
5. ✅ **評估測試**: 完整的評估管道和基準測試
6. ✅ **消融實驗**: 系統性的 ablation study 設計
7. ✅ **故障排除**: 常見問題的解決方案
8. ✅ **高級優化**: 分散式訓練和性能優化

**重要提醒**:
- 確保有足夠的計算資源 (24GB+ GPU 記憶體)
- 監控訓練過程，及時調整超參數  
- 定期保存檢查點，防止訓練中斷
- 詳細記錄實驗設置，確保可重現性

**下一步建議**:
1. 先在小規模數據上測試整個流程
2. 逐步擴展到完整數據集
3. 系統性地進行消融實驗
4. 詳細分析各元件的貢獻度

如有任何問題，請參考故障排除章節或查看相關日誌文件。
