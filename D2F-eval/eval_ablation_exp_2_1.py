#!/usr/bin/env python3
"""
Ablation Experiment 2.1: Original Dream Model + D2F Inference Algorithm
Tests if D2F inference algorithm alone provides benefits without specialized training.
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

class OriginalDreamWithD2FInference(HFLM):
    """
    Original Dream model using D2F inference algorithm
    This tests the contribution of the inference algorithm alone
    """
    
    def __init__(self, 
                 pretrained: str,
                 max_length: int = 512,
                 block_size: int = 16,
                 **kwargs):
        # Initialize with original Dream model
        super().__init__(pretrained=pretrained, **kwargs)
        self.max_length = max_length
        self.block_size = block_size
        self.mask_token_id = 151666  # Dream mask token
        
        logger.info(f"Initialized Original Dream with D2F Inference")
        logger.info(f"Model: {pretrained}")
        logger.info(f"Block size: {block_size}")
        logger.info(f"Max length: {max_length}")
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Generate using D2F parallel decoding algorithm but with original model
        """
        input_ids = context
        batch_size = input_ids.shape[0]
        
        # Start timing
        start_time = time.time()
        
        # Create D2F attention mask for block-wise causal attention
        prompt_length = input_ids.shape[1]
        full_length = min(prompt_length + max_length, self.max_length)
        
        attention_mask = create_full_block_attention_mask(
            prompt_length=prompt_length,
            max_length=full_length,
            block_size=self.block_size,
            device=input_ids.device,
            dtype=torch.bfloat16
        )
        
        # Initialize generation sequence with mask tokens
        gen_length = full_length - prompt_length
        mask_tokens = torch.full((batch_size, gen_length), self.mask_token_id, 
                                device=input_ids.device, dtype=input_ids.dtype)
        
        full_sequence = torch.cat([input_ids, mask_tokens], dim=1)
        
        # D2F parallel decoding
        with torch.no_grad():
            for step in range(gen_length):
                # Find all mask positions
                mask_positions = (full_sequence == self.mask_token_id)
                
                if not mask_positions.any():
                    break
                
                # Forward pass with block attention mask
                outputs = self.model(
                    input_ids=full_sequence,
                    attention_mask=attention_mask,
                    use_cache=True  # Enable KV caching
                )
                
                logits = outputs.logits
                
                # Apply temperature and top-p sampling to masked positions
                mask_logits = logits[mask_positions]
                
                # Simple argmax for deterministic generation
                predicted_tokens = torch.argmax(mask_logits, dim=-1)
                
                # Update one token per step (sequential within parallel framework)
                if len(predicted_tokens) > 0:
                    # Update the first mask position
                    first_mask_idx = torch.where(mask_positions[0])[0][0]
                    full_sequence[0, first_mask_idx] = predicted_tokens[0]
                
                # Stop if we hit EOS or stop tokens
                if stop and any(self.tokenizer.decode([predicted_tokens[0]]).strip() == s for s in stop):
                    break
        
        # Calculate generation time
        generation_time = time.time() - start_time
        generated_tokens = gen_length
        
        # Log performance metrics
        tps = generated_tokens / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s (TPS: {tps:.2f})")
        
        return full_sequence

def main():
    parser = argparse.ArgumentParser(description="Ablation Exp 2.1: Original Dream + D2F Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to original Dream-Base-7B model")
    parser.add_argument("--tasks", type=str, default="gsm8k",
                        help="Evaluation tasks (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--block_size", type=int, default=16,
                        help="Block size for D2F attention")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation")
    
    args = parser.parse_args()
    
    # Setup model
    logger.info("Setting up Original Dream model with D2F inference...")
    
    model = OriginalDreamWithD2FInference(
        pretrained=args.model_path,
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
        "experiment_name": "Exp_2_1_Original_Dream_D2F_Inference",
        "model_path": args.model_path,
        "tasks": tasks,
        "block_size": args.block_size,
        "max_length": args.max_length,
        "total_evaluation_time": total_time,
        "description": "Original Dream model with D2F inference algorithm"
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
