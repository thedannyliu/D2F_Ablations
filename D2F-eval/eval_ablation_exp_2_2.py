#!/usr/bin/env python3
"""
Ablation Experiment 2.2: D2F Model + Cache-Only (Serial Block Decoding)
Tests the contribution of KV caching and architecture without parallel inference.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModel, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Import D2F specific functions
from eval_dream import DreamD2F, create_full_block_attention_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class D2FModelCacheOnlyInference(HFLM):
    """
    D2F trained model using cache-only inference (serial block-by-block decoding)
    This tests the contribution of architecture/KV caching without parallel inference
    """
    
    def __init__(self, 
                 pretrained: str,
                 lora_path: str = None,
                 max_length: int = 512,
                 block_size: int = 16,
                 **kwargs):
        # Initialize with D2F model
        super().__init__(pretrained=pretrained, **kwargs)
        
        # Load LoRA weights if provided
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info(f"Loaded LoRA weights from {lora_path}")
        
        self.max_length = max_length
        self.block_size = block_size
        self.mask_token_id = 151666  # Dream mask token
        
        logger.info(f"Initialized D2F Model with Cache-Only Inference")
        logger.info(f"Model: {pretrained}")
        logger.info(f"Block size: {block_size}")
        logger.info(f"Max length: {max_length}")
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Generate using serial block-by-block decoding with KV caching
        This maintains the block-wise attention structure but decodes blocks sequentially
        """
        input_ids = context
        batch_size = input_ids.shape[0]
        
        # Start timing
        start_time = time.time()
        
        prompt_length = input_ids.shape[1]
        full_length = min(prompt_length + max_length, self.max_length)
        gen_length = full_length - prompt_length
        
        # Calculate number of blocks to generate
        num_blocks = (gen_length + self.block_size - 1) // self.block_size
        
        # Start with the prompt
        current_sequence = input_ids.clone()
        
        # Generate blocks sequentially (cache-only mode)
        with torch.no_grad():
            for block_idx in range(num_blocks):
                # Calculate current block range
                block_start = prompt_length + block_idx * self.block_size
                block_end = min(prompt_length + (block_idx + 1) * self.block_size, full_length)
                current_block_size = block_end - block_start
                
                if current_block_size <= 0:
                    break
                
                # Add mask tokens for current block
                mask_tokens = torch.full((batch_size, current_block_size), self.mask_token_id,
                                       device=input_ids.device, dtype=input_ids.dtype)
                current_sequence = torch.cat([current_sequence, mask_tokens], dim=1)
                
                # Create attention mask for current sequence length
                current_length = current_sequence.shape[1]
                attention_mask = create_full_block_attention_mask(
                    prompt_length=prompt_length,
                    max_length=current_length,
                    block_size=self.block_size,
                    device=input_ids.device,
                    dtype=torch.bfloat16
                )
                
                # Generate tokens for current block sequentially
                for pos_in_block in range(current_block_size):
                    current_pos = block_start + pos_in_block
                    
                    # Forward pass with KV caching
                    outputs = self.model(
                        input_ids=current_sequence,
                        attention_mask=attention_mask,
                        use_cache=True  # Key: Enable KV caching
                    )
                    
                    logits = outputs.logits
                    
                    # Get logits for the current mask position
                    current_logits = logits[0, current_pos, :]
                    
                    # Simple argmax for deterministic generation
                    predicted_token = torch.argmax(current_logits, dim=-1)
                    
                    # Update the current position
                    current_sequence[0, current_pos] = predicted_token
                    
                    # Check for stop conditions
                    if stop:
                        decoded_token = self.tokenizer.decode([predicted_token]).strip()
                        if any(decoded_token == s for s in stop):
                            # Truncate and return
                            return current_sequence[:, :current_pos+1]
                
                logger.info(f"Completed block {block_idx + 1}/{num_blocks}")
        
        # Calculate generation time
        generation_time = time.time() - start_time
        generated_tokens = current_sequence.shape[1] - prompt_length
        
        # Log performance metrics
        tps = generated_tokens / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s (TPS: {tps:.2f})")
        logger.info("Used cache-only (serial block) decoding")
        
        return current_sequence

def main():
    parser = argparse.ArgumentParser(description="Ablation Exp 2.2: D2F Model + Cache-Only Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained D2F model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights")
    parser.add_argument("--tasks", type=str, default="gsm8k",
                        help="Evaluation tasks (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--block_size", type=int, default=16,
                        help="Block size for attention")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation")
    
    args = parser.parse_args()
    
    # Setup model
    logger.info("Setting up D2F model with cache-only inference...")
    
    model = D2FModelCacheOnlyInference(
        pretrained=args.model_path,
        lora_path=args.lora_path,
        max_length=args.max_length,
        block_size=args.block_size,
        device=args.device,
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    
    # Run evaluation
    logger.info(f"Running evaluation on tasks: {tasks}")
    start_time = time.time()
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=args.batch_size,
        log_samples=True,
        write_out=True
    )
    
    total_time = time.time() - start_time
    
    # Add experiment metadata
    results["experiment_info"] = {
        "experiment_name": "Exp_2_2_D2F_Model_Cache_Only",
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "tasks": tasks,
        "block_size": args.block_size,
        "max_length": args.max_length,
        "total_evaluation_time": total_time,
        "description": "D2F trained model with cache-only (serial block) inference"
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_path}")
    logger.info(f"Total evaluation time: {total_time:.2f}s")
    
    # Print summary
    for task in tasks:
        if task in results["results"]:
            score = results["results"][task].get("acc", "N/A")
            logger.info(f"{task}: {score}")

if __name__ == "__main__":
    main()
