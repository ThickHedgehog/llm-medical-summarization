"""
LoRA (Low-Rank Adaptation) Wrapper
==================================

Functions for configuring and applying LoRA to models.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM,
    modules_to_save: Optional[List[str]] = None,
    use_rslora: bool = False,
) -> LoraConfig:
    """
    Create a LoRA configuration.
    
    Args:
        r: Rank of the low-rank matrices
        lora_alpha: Scaling factor (alpha/r is the actual scaling)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias handling ("none", "all", "lora_only")
        task_type: Type of task (SEQ_2_SEQ_LM for summarization)
        modules_to_save: Additional modules to save besides LoRA
        use_rslora: Use Rank-Stabilized LoRA
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q", "v"]  # query and value projections
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        modules_to_save=modules_to_save,
        use_rslora=use_rslora,
    )
    
    logger.info(f"LoRA config created:")
    logger.info(f"  Rank (r): {r}")
    logger.info(f"  Alpha: {lora_alpha}")
    logger.info(f"  Dropout: {lora_dropout}")
    logger.info(f"  Target modules: {target_modules}")
    logger.info(f"  Bias: {bias}")
    
    return config


def setup_lora(
    model: torch.nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM,
    modules_to_save: Optional[List[str]] = None,
    is_quantized: bool = False,
) -> PeftModel:
    logger.info("Setting up LoRA...")
    
    # model for k-bit training if quantized
    if is_quantized:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)
    
    # create LoRA config
    lora_config = get_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        modules_to_save=modules_to_save,
    )
    
    model = get_peft_model(model, lora_config)
    
    # log parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    
    logger.info(f"LoRA applied successfully:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {all_params:,}")
    logger.info(f"  Trainable percentage: {trainable_percentage:.4f}%")
    
    return model


def print_lora_info(model: PeftModel) -> None:
    """Print detailed information about LoRA configuration."""
    print("\n" + "=" * 60)
    print("LORA MODEL INFORMATION")
    print("=" * 60)
    
    model.print_trainable_parameters()
    
    if hasattr(model, 'peft_config'):
        for adapter_name, config in model.peft_config.items():
            print(f"\nAdapter: {adapter_name}")
            print(f"  Rank (r): {config.r}")
            print(f"  Alpha: {config.lora_alpha}")
            print(f"  Scaling: {config.lora_alpha / config.r}")
            print(f"  Dropout: {config.lora_dropout}")
            print(f"  Target modules: {config.target_modules}")
            print(f"  Bias: {config.bias}")
    
    print("=" * 60 + "\n")


def get_lora_layers_info(model: PeftModel) -> Dict[str, Any]:
    lora_layers = []
    
    for name, module in model.named_modules():
        if "lora" in name.lower():
            lora_layers.append({
                "name": name,
                "type": type(module).__name__,
            })
    
    return {
        "num_lora_layers": len(lora_layers),
        "layers": lora_layers,
    }


def merge_lora_weights(model: PeftModel) -> torch.nn.Module:
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    logger.info("LoRA weights merged successfully")
    return merged_model


def save_lora_weights(model: PeftModel, save_path: str) -> None:
    logger.info(f"Saving LoRA adapter to {save_path}")
    model.save_pretrained(save_path)
    logger.info("LoRA adapter saved successfully")


def load_lora_weights(
    base_model: torch.nn.Module,
    adapter_path: str,
    is_trainable: bool = False,
) -> PeftModel:
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    )
    logger.info("LoRA adapter loaded successfully")
    return model


LORA_PRESETS = {
    "minimal": {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["q", "v"],
        "description": "Minimal LoRA - very few parameters",
    },
    "standard": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q", "v"],
        "description": "Standard LoRA - balanced performance",
    },
    "comprehensive": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q", "k", "v", "o"],
        "description": "Comprehensive LoRA - more expressive",
    },
    "high_rank": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q", "k", "v", "o"],
        "description": "High-rank LoRA - maximum expressiveness",
    },
}


def get_lora_preset(preset_name: str) -> Dict[str, Any]:
    """Get a predefined LoRA configuration preset."""
    if preset_name not in LORA_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(LORA_PRESETS.keys())}")
    return LORA_PRESETS[preset_name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from model_loader import load_model_and_tokenizer
    
    print("Testing LoRA setup...")
    model, tokenizer = load_model_and_tokenizer(
        "google/flan-t5-small",
        device_map="auto",
    )
    
    print(f"\nBefore LoRA:")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    lora_model = setup_lora(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
    )
    
    print(f"\nAfter LoRA:")
    print_lora_info(lora_model)
    
    print("\nAvailable presets:")
    for name, preset in LORA_PRESETS.items():
        print(f"  {name}: {preset['description']}")
