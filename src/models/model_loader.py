"""
Model Loading Utilities
=======================

Functions for loading and configuring T5-based models for summarization.
"""

import logging
from typing import Dict, Tuple, Optional, Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "google/flan-t5-small": {
        "params": "80M",
        "type": "encoder-decoder",
        "description": "Flan-T5 Small - Instruction-tuned T5",
    },
    "google/flan-t5-base": {
        "params": "250M",
        "type": "encoder-decoder",
        "description": "Flan-T5 Base - Instruction-tuned T5",
    },
    "google/flan-t5-large": {
        "params": "780M",
        "type": "encoder-decoder",
        "description": "Flan-T5 Large - Instruction-tuned T5",
    },
    "google/flan-t5-xl": {
        "params": "3B",
        "type": "encoder-decoder",
        "description": "Flan-T5 XL - Instruction-tuned T5",
    },
}


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
    use_peft: bool = False,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy ("auto", "cuda", "cpu")
        torch_dtype: Data type for model weights
        load_in_8bit: Load model in 8-bit precision (for QLoRA)
        load_in_4bit: Load model in 4-bit precision (for QLoRA)
        trust_remote_code: Trust remote code in model
        cache_dir: Directory to cache model files
        use_peft: Whether model will be used with PEFT (affects dtype choice)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = None
    if load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # for PEFT methods, float32 is more stable; for full fine-tuning, we can use fp16/bf16
    if torch_dtype is None:
        if use_peft:
            # using float32 for PEFT to avoid gradient issues
            torch_dtype = torch.float32
            logger.info("Using float32 for PEFT compatibility")
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    model_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    
    # only use device_map for larger models or when not using PEFT with fp32
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
    elif use_peft and torch_dtype == torch.float32:
        # for PEFT with fp32, load to single GPU without device_map for stability
        model_kwargs["torch_dtype"] = torch_dtype
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = device_map
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    # move to GPU if not using device_map
    if "device_map" not in model_kwargs and torch.cuda.is_available():
        model = model.cuda()
    
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    logger.info(f"Model loaded successfully:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {next(model.parameters()).device}")
    logger.info(f"  Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: torch.nn.Module, model_name: str = "") -> Dict[str, Any]:
    """
    Get detailed information about a model.
    
    Args:
        model: The model to analyze
        model_name: Optional model name for display
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    
    info = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        "frozen_parameters": total_params - trainable_params,
        "estimated_memory_mb": param_bytes / (1024 * 1024),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }
    
    return info


def print_model_info(model: torch.nn.Module, model_name: str = "") -> None:
    """Print formatted model information."""
    info = get_model_info(model, model_name)
    
    print("\n" + "=" * 60)
    print(f"MODEL INFORMATION: {info['model_name']}")
    print("=" * 60)
    print(f"Total Parameters:     {info['total_parameters']:>15,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:>15,}")
    print(f"Frozen Parameters:    {info['frozen_parameters']:>15,}")
    print(f"Trainable Percentage: {info['trainable_percentage']:>14.4f}%")
    print(f"Estimated Memory:     {info['estimated_memory_mb']:>14.2f} MB")
    print(f"Device:               {info['device']:>15}")
    print(f"Dtype:                {info['dtype']:>15}")
    print("=" * 60 + "\n")


def prepare_model_for_training(
    model: torch.nn.Module,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    model.train()
    
    return model


def freeze_model_layers(
    model: torch.nn.Module,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    freeze_embeddings: bool = False,
    freeze_layer_indices: Optional[list] = None,
) -> torch.nn.Module:
    # freeze embeddings
    if freeze_embeddings:
        if hasattr(model, 'shared'):
            for param in model.shared.parameters():
                param.requires_grad = False
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'embed_tokens'):
            for param in model.encoder.embed_tokens.parameters():
                param.requires_grad = False
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'embed_tokens'):
            for param in model.decoder.embed_tokens.parameters():
                param.requires_grad = False
        logger.info("Embeddings frozen")
    
    # freeze encoder
    if freeze_encoder and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    # freeze decoder
    if freeze_decoder and hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")
    
    # freeze specific layers
    if freeze_layer_indices is not None:
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
            for idx in freeze_layer_indices:
                if idx < len(model.encoder.block):
                    for param in model.encoder.block[idx].parameters():
                        param.requires_grad = False
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
            for idx in freeze_layer_indices:
                if idx < len(model.decoder.block):
                    for param in model.decoder.block[idx].parameters():
                        param.requires_grad = False
        logger.info(f"Layers {freeze_layer_indices} frozen")
    
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model loading...")
    model, tokenizer = load_model_and_tokenizer(
        "google/flan-t5-small",
        device_map="auto",
    )
    
    print_model_info(model, "google/flan-t5-small")
    test_input = "summarize: This is a test medical article about clinical trials."
    tokens = tokenizer(test_input, return_tensors="pt")
    print(f"Test tokenization: {tokens['input_ids'].shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            tokens["input_ids"].to(model.device),
            max_new_tokens=50,
            num_beams=4,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test generation: {decoded}")