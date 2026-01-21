"""
Prompt Tuning Wrapper
=====================

Functions for configuring and applying Prompt Tuning to models.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import torch
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    PeftModel,
)

logger = logging.getLogger(__name__)


def get_prompt_tuning_config(
    num_virtual_tokens: int = 20,
    prompt_tuning_init: Union[str, PromptTuningInit] = PromptTuningInit.TEXT,
    prompt_tuning_init_text: str = "Summarize the following medical article:",
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM,
    tokenizer_name_or_path: Optional[str] = None,
) -> PromptTuningConfig:
    """
    Create a Prompt Tuning configuration.
    
    Args:
        num_virtual_tokens: Number of soft prompt tokens
        prompt_tuning_init: Initialization method (TEXT or RANDOM)
        prompt_tuning_init_text: Text for initialization if using TEXT init
        task_type: Type of task
        tokenizer_name_or_path: Path to tokenizer (for TEXT init)
        
    Returns:
        PromptTuningConfig object
    """
    if isinstance(prompt_tuning_init, str):
        prompt_tuning_init = PromptTuningInit(prompt_tuning_init)
    
    config_kwargs = {
        "task_type": task_type,
        "num_virtual_tokens": num_virtual_tokens,
        "prompt_tuning_init": prompt_tuning_init,
    }
    
    if prompt_tuning_init == PromptTuningInit.TEXT:
        config_kwargs["prompt_tuning_init_text"] = prompt_tuning_init_text
        if tokenizer_name_or_path:
            config_kwargs["tokenizer_name_or_path"] = tokenizer_name_or_path
    
    config = PromptTuningConfig(**config_kwargs)
    
    logger.info(f"Prompt Tuning config created:")
    logger.info(f"  Num virtual tokens: {num_virtual_tokens}")
    logger.info(f"  Initialization: {prompt_tuning_init}")
    if prompt_tuning_init == PromptTuningInit.TEXT:
        logger.info(f"  Init text: '{prompt_tuning_init_text}'")
    
    return config


def setup_prompt_tuning(
    model: torch.nn.Module,
    tokenizer,
    num_virtual_tokens: int = 20,
    prompt_tuning_init: Union[str, PromptTuningInit] = PromptTuningInit.TEXT,
    prompt_tuning_init_text: str = "Summarize the following medical article:",
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM,
) -> PeftModel:
    """
    Apply Prompt Tuning to a model.
    
    Args:
        model: Base model to apply prompt tuning to
        tokenizer: Tokenizer for the model
        num_virtual_tokens: Number of soft prompt tokens
        prompt_tuning_init: Initialization method
        prompt_tuning_init_text: Text for initialization
        task_type: Task type
        
    Returns:
        PeftModel with prompt tuning applied
    """
    logger.info("Setting up Prompt Tuning...")
    
    if isinstance(prompt_tuning_init, str):
        prompt_tuning_init = PromptTuningInit(prompt_tuning_init)
    
    peft_config = get_prompt_tuning_config(
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init=prompt_tuning_init,
        prompt_tuning_init_text=prompt_tuning_init_text,
        task_type=task_type,
        tokenizer_name_or_path=tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else None,
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    
    logger.info(f"Prompt Tuning applied successfully:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {all_params:,}")
    logger.info(f"  Trainable percentage: {trainable_percentage:.6f}%")
    
    embedding_dim = model.base_model.config.d_model if hasattr(model.base_model.config, 'd_model') else model.base_model.config.hidden_size
    soft_prompt_params = num_virtual_tokens * embedding_dim
    logger.info(f"  Soft prompt parameters: {soft_prompt_params:,} ({num_virtual_tokens} tokens × {embedding_dim} dim)")
    
    return model


def print_prompt_tuning_info(model: PeftModel) -> None:
    """Print detailed information about Prompt Tuning configuration."""
    print("\n" + "=" * 60)
    print("PROMPT TUNING MODEL INFORMATION")
    print("=" * 60)
    
    model.print_trainable_parameters()
    if hasattr(model, 'peft_config'):
        for adapter_name, config in model.peft_config.items():
            print(f"\nAdapter: {adapter_name}")
            print(f"  Num virtual tokens: {config.num_virtual_tokens}")
            print(f"  Initialization: {config.prompt_tuning_init}")
            if config.prompt_tuning_init == PromptTuningInit.TEXT:
                init_text = getattr(config, 'prompt_tuning_init_text', 'N/A')
                print(f"  Init text: '{init_text}'")
    
    for name, param in model.named_parameters():
        if "prompt" in name.lower() and param.requires_grad:
            print(f"\nSoft prompt embedding:")
            print(f"  Name: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Parameters: {param.numel():,}")
    
    print("=" * 60 + "\n")


def get_soft_prompt_embeddings(model: PeftModel) -> torch.Tensor:
    for name, param in model.named_parameters():
        if "prompt" in name.lower() and param.requires_grad:
            return param.data.clone()
    
    raise ValueError("No soft prompt embeddings found in model")


def analyze_soft_prompts(model: PeftModel, tokenizer) -> Dict[str, Any]:
    embeddings = get_soft_prompt_embeddings(model)
    
    if hasattr(model.base_model, 'shared'):
        word_embeddings = model.base_model.shared.weight
    elif hasattr(model.base_model, 'encoder') and hasattr(model.base_model.encoder, 'embed_tokens'):
        word_embeddings = model.base_model.encoder.embed_tokens.weight
    else:
        logger.warning("Could not find word embeddings for analysis")
        return {"embeddings_shape": embeddings.shape}
    
    nearest_tokens = []
    for i in range(embeddings.shape[0]):
        soft_emb = embeddings[i].unsqueeze(0)
        
        # cosine sim
        similarities = torch.nn.functional.cosine_similarity(
            soft_emb,
            word_embeddings,
            dim=1
        )
        
        # get top-5 nearest tokens
        top_indices = torch.topk(similarities, k=5).indices
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        nearest_tokens.append({
            "position": i,
            "nearest_tokens": top_tokens,
        })
    
    return {
        "embeddings_shape": embeddings.shape,
        "embedding_norm_mean": embeddings.norm(dim=-1).mean().item(),
        "embedding_norm_std": embeddings.norm(dim=-1).std().item(),
        "nearest_tokens": nearest_tokens,
    }


def save_prompt_tuning_weights(model: PeftModel, save_path: str) -> None:
    """
    Save prompt tuning adapter weights.
    
    Args:
        model: PeftModel with trained prompt tuning
        save_path: Path to save the adapter
    """
    logger.info(f"Saving Prompt Tuning adapter to {save_path}")
    model.save_pretrained(save_path)
    logger.info("Prompt Tuning adapter saved successfully")


def load_prompt_tuning_weights(
    base_model: torch.nn.Module,
    adapter_path: str,
    is_trainable: bool = False,
) -> PeftModel:
    """
    Load prompt tuning adapter weights onto a base model.
    
    Args:
        base_model: Base model to load adapter onto
        adapter_path: Path to saved adapter
        is_trainable: Whether to make adapter trainable
        
    Returns:
        PeftModel with loaded adapter
    """
    logger.info(f"Loading Prompt Tuning adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    )
    logger.info("Prompt Tuning adapter loaded successfully")
    return model


PROMPT_TUNING_PRESETS = {
    "minimal": {
        "num_virtual_tokens": 8,
        "prompt_tuning_init": PromptTuningInit.TEXT,
        "prompt_tuning_init_text": "Summarize:",
        "description": "Minimal - 8 virtual tokens",
    },
    "standard": {
        "num_virtual_tokens": 20,
        "prompt_tuning_init": PromptTuningInit.TEXT,
        "prompt_tuning_init_text": "Summarize the following medical article:",
        "description": "Standard - 20 virtual tokens",
    },
    "extended": {
        "num_virtual_tokens": 50,
        "prompt_tuning_init": PromptTuningInit.TEXT,
        "prompt_tuning_init_text": "Generate a concise summary of the following medical research article, highlighting key findings:",
        "description": "Extended - 50 virtual tokens",
    },
    "random": {
        "num_virtual_tokens": 20,
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "prompt_tuning_init_text": "",
        "description": "Random initialization - 20 virtual tokens",
    },
}


def get_prompt_tuning_preset(preset_name: str) -> Dict[str, Any]:
    """Get a predefined Prompt Tuning configuration preset."""
    if preset_name not in PROMPT_TUNING_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PROMPT_TUNING_PRESETS.keys())}")
    return PROMPT_TUNING_PRESETS[preset_name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from model_loader import load_model_and_tokenizer
    
    print("Testing Prompt Tuning setup...")
    model, tokenizer = load_model_and_tokenizer(
        "google/flan-t5-small",
        device_map="auto",
    )
    
    print(f"\nBefore Prompt Tuning:")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    pt_model = setup_prompt_tuning(
        model,
        tokenizer,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Summarize the following medical article:",
    )
    
    print(f"\nAfter Prompt Tuning:")
    print_prompt_tuning_info(pt_model)
    
    print("\nAvailable presets:")
    for name, preset in PROMPT_TUNING_PRESETS.items():
        print(f"  {name}: {preset['description']}")
