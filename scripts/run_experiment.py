#!/usr/bin/env python3
"""
Main Experiment Runner
======================

Run training experiments for different adaptation methods.

Usage:
    python scripts/run_experiment.py --model_name google/flan-t5-base --method lora
    python scripts/run_experiment.py --model_name google/flan-t5-base --method prompt_tuning
    python scripts/run_experiment.py --model_name google/flan-t5-base --method full_ft
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from peft import TaskType

from data.dataset_loader import load_pubmed_dataset, print_dataset_info
from data.preprocessing import prepare_dataset_for_training
from models.model_loader import (
    load_model_and_tokenizer,
    print_model_info,
    count_parameters,
    count_trainable_parameters,
)
from models.lora_wrapper import setup_lora, print_lora_info
from models.prompt_tuning_wrapper import setup_prompt_tuning, print_prompt_tuning_info
from training.trainer import TrainingConfig, run_training
from evaluation.metrics import create_compute_metrics_fn
from evaluation.evaluator import evaluate_model
from utils.helpers import (
    set_seed,
    load_config,
    save_config,
    get_experiment_name,
    print_experiment_header,
)
from utils.logging_utils import setup_logging, log_system_info
from utils.memory_tracker import get_gpu_memory_info, print_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run medical summarization experiments"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        choices=[
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        help="Model to use",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["full_ft", "lora", "prompt_tuning"],
        help="Adaptation method",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to method-specific config file (will be merged with base config)",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to base config file (default: configs/base_config.yaml)",
    )
    
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Number of training samples (None for full dataset)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=None,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Number of test samples",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use small dataset for debugging",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="Number of virtual tokens for prompt tuning",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip final evaluation",
    )
    
    return parser.parse_args()


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge two configs, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config(method: str, model_name: str) -> dict:
    """Get default configuration for a method."""
    config = {
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "fp16": True,
            "logging_steps": 100,
            "eval_steps": 500,
            "save_steps": 500,
        },
        "dataset": {
            "max_input_length": 1024,
            "max_target_length": 256,
        },
        "generation": {
            "max_length": 256,
            "num_beams": 4,
        },
    }
    
    if method == "lora":
        config["training"]["learning_rate"] = 3e-4
        config["training"]["num_train_epochs"] = 5
        config["training"]["per_device_train_batch_size"] = 8
        config["training"]["gradient_accumulation_steps"] = 2
        config["lora"] = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q", "v"],
        }
    
    elif method == "prompt_tuning":
        config["training"]["learning_rate"] = 3e-2
        config["training"]["num_train_epochs"] = 10
        config["training"]["per_device_train_batch_size"] = 16
        config["training"]["gradient_accumulation_steps"] = 1
        config["prompt_tuning"] = {
            "num_virtual_tokens": 20,
            "prompt_tuning_init": "TEXT",
            "prompt_tuning_init_text": "Summarize the following medical article:",
        }
    
    elif method == "full_ft":
        config["training"]["learning_rate"] = 5e-5
        config["training"]["num_train_epochs"] = 3
        config["training"]["gradient_checkpointing"] = True
        
        if "large" in model_name:
            config["training"]["per_device_train_batch_size"] = 2
            config["training"]["gradient_accumulation_steps"] = 8
        elif "base" in model_name:
            config["training"]["per_device_train_batch_size"] = 4
            config["training"]["gradient_accumulation_steps"] = 4
        else:
            config["training"]["per_device_train_batch_size"] = 8
            config["training"]["gradient_accumulation_steps"] = 2
    
    return config


def main():
    """Main function."""
    args = parse_args()
    
    setup_logging(log_level="INFO", log_dir=os.path.join(args.output_dir, "logs"))
    logger = logging.getLogger(__name__)
    
    log_system_info()
    set_seed(args.seed)
    
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = get_experiment_name(args.model_name, args.method)
    
    if args.base_config:
        base_config_path = Path(args.base_config)
    else:
        base_config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    
    if base_config_path.exists():
        base_config = load_config(str(base_config_path))
        logger.info(f"Loaded base config from {base_config_path}")
    else:
        base_config = {}
        logger.info("No base config found, using defaults")
    
    if args.config:
        method_config = load_config(args.config)
        logger.info(f"Loaded method config from {args.config}")
        # base <- method_config (method overrides base)
        config = merge_configs(base_config, method_config)
    else:
        # use hardcoded defaults
        default_config = get_default_config(args.method, args.model_name)
        config = merge_configs(base_config, default_config)
    
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.gradient_checkpointing:
        config["training"]["gradient_checkpointing"] = True
    if args.lora_r and args.method == "lora":
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha and args.method == "lora":
        config["lora"]["lora_alpha"] = args.lora_alpha
    if args.num_virtual_tokens and args.method == "prompt_tuning":
        config["prompt_tuning"]["num_virtual_tokens"] = args.num_virtual_tokens
    
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print_experiment_header(
        experiment_name=run_name,
        model_name=args.model_name,
        method=args.method,
        config=config["training"],
    )
    
    save_config(config, os.path.join(output_dir, "config.yaml"))
    
    logger.info("Loading dataset...")
    
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    
    if args.debug:
        logger.info("Debug mode: using small dataset")
        train_size = train_size or 500
        val_size = val_size or 100
        test_size = test_size or 100
    
    dataset = load_pubmed_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=args.seed,
    )
    
    print_dataset_info(dataset)
    
    logger.info(f"Loading model: {args.model_name}")
    
    use_peft = args.method in ["lora", "prompt_tuning"]
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device_map="auto" if not use_peft else None,  # let the function handle this
        use_peft=use_peft,
    )
    
    print_model_info(model, args.model_name)
    print_gpu_memory("After model load: ")
    
    logger.info("Preprocessing dataset...")
    
    tokenized_dataset = prepare_dataset_for_training(
        dataset=dataset,
        tokenizer=tokenizer,
        max_input_length=config["dataset"]["max_input_length"],
        max_target_length=config["dataset"]["max_target_length"],
        prefix="summarize: ",
    )
    
    logger.info(f"Applying adaptation method: {args.method}")
    
    if args.method == "lora":
        model = setup_lora(
            model=model,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
        )
        print_lora_info(model)
    
    elif args.method == "prompt_tuning":
        model = setup_prompt_tuning(
            model=model,
            tokenizer=tokenizer,
            num_virtual_tokens=config["prompt_tuning"]["num_virtual_tokens"],
            prompt_tuning_init=config["prompt_tuning"]["prompt_tuning_init"],
            prompt_tuning_init_text=config["prompt_tuning"]["prompt_tuning_init_text"],
        )
        print_prompt_tuning_info(model)
    
    elif args.method == "full_ft":
        if config["training"].get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    trainable_pct = 100 * trainable_params / total_params
    
    logger.info(f"Total: {total_params:,}, Trainable: {trainable_params:,} ({trainable_pct:.4f}%)")
    print_gpu_memory("After adaptation: ")
    
    compute_metrics = create_compute_metrics_fn(
        tokenizer=tokenizer,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        compute_bertscore_flag=False,
    )
    
    logger.info("Starting training...")
    
    # for PEFT methods disable fp16 to avoid gradient issues
    use_fp16 = config["training"].get("fp16", True)
    if args.method in ["lora", "prompt_tuning"]:
        use_fp16 = False
        logger.info("Disabled fp16 for PEFT method (using fp32 for stability)")
    
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        fp16=use_fp16,
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        generation_max_length=config["generation"]["max_length"],
        generation_num_beams=config["generation"]["num_beams"],
        seed=args.seed,
    )
    
    training_results = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        config=training_config,
        compute_metrics=compute_metrics,
        run_name=run_name,
    )
    
    if not args.no_eval:
        logger.info("Running final evaluation on test set...")
        
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_dataset=tokenized_dataset["test"],
            batch_size=config["training"]["per_device_eval_batch_size"],
            max_length=config["generation"]["max_length"],
            num_beams=config["generation"]["num_beams"],
            compute_bertscore=True,
            num_qualitative_examples=10,
            output_dir=os.path.join(output_dir, "evaluation"),
        )
        
        training_results["test_metrics"] = eval_results["metrics"]
    
    training_results["model_info"] = {
        "model_name": args.model_name,
        "method": args.method,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_pct,
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in value.items()
                    if not callable(v)
                }
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Experiment: {run_name}")
    print(f"Model: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.4f}%)")
    
    if "test_metrics" in training_results:
        print("\nTest Set Metrics:")
        for metric, value in training_results["test_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.2f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70 + "\n")
    
    return training_results


if __name__ == "__main__":
    main()