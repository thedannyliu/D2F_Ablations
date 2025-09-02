import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig

def get_model_by_config(config):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode == 'dream':
        return get_model(config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/home/wx/data/model/Dream-org/Dream-v0-Base-7B"
    
    # Memory-optimized model loading for large models
    import torch
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_map = {"": 0}  # Force to GPU 0 if available
        print("Loading model on GPU...")
    else:
        device_map = "cpu"    # Fallback to CPU for debugging
        print("Loading model on CPU (no GPU available)...")
    
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        low_cpu_mem_usage=True,       # Enable low CPU memory usage
        device_map=device_map         # Adaptive device mapping
    )
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

def get_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    # Memory-optimized loading for LLaDA
    import torch
    
    # Check if CUDA is available for LLaDA too
    if torch.cuda.is_available():
        device_map = {"": 0}  # Force to GPU 0 if available
        print("Loading LLaDA model on GPU...")
    else:
        device_map = "cpu"    # Fallback to CPU for debugging
        print("Loading LLaDA model on CPU (no GPU available)...")
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(
        model_path,
        config=config_obj,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        low_cpu_mem_usage=True,       # Enable low CPU memory usage
        device_map=device_map         # Adaptive device mapping
    )
    # print(model.named_modules())
    # print(model,"model
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "attn_out"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).
