"""
Training Utilities
==================

Custom trainer and training loop implementations.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union

import torch
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./results"
    
    # training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # optimizer
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    
    # mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # memory optimization
    gradient_checkpointing: bool = False
    
    # logging-evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    
    # best model
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rouge1"
    greater_is_better: bool = True
    
    # early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # generation settings
    predict_with_generate: bool = True
    generation_max_length: int = 256
    generation_num_beams: int = 4
    
    # other
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    label_smoothing_factor: float = 0.0
    
    # reporting
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class MemoryTrackingCallback(TrainerCallback):
    def __init__(self):
        self.memory_log = []
        
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            self.memory_log.append({
                "step": state.global_step,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
            })
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.memory_log:
            return {"peak_allocated_gb": 0, "peak_reserved_gb": 0}
        
        return {
            "peak_allocated_gb": max(m["memory_allocated_gb"] for m in self.memory_log),
            "peak_reserved_gb": max(m["memory_reserved_gb"] for m in self.memory_log),
        }


class TimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self.epoch_start = None
        self.training_start = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start = time.time()
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start:
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)


def create_training_arguments(
    config: Union[TrainingConfig, Dict[str, Any]],
    run_name: Optional[str] = None,
) -> Seq2SeqTrainingArguments:
    if isinstance(config, dict):
        config = TrainingConfig(**config)
    
    # create output directory with run name
    output_dir = config.output_dir
    if run_name:
        output_dir = os.path.join(output_dir, run_name)
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # training
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        
        # mixed precision
        fp16=config.fp16,
        bf16=config.bf16,
        
        # memory
        gradient_checkpointing=config.gradient_checkpointing,
        
        # logging-evaluation
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        
        # best model
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # generation
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
        
        # other
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=config.remove_unused_columns,
        label_smoothing_factor=config.label_smoothing_factor,
        
        # reporting
        report_to=config.report_to,
        run_name=run_name,
    )
    
    return args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: Seq2SeqTrainingArguments,
    compute_metrics: Optional[Callable] = None,
    data_collator=None,
    callbacks: Optional[List[TrainerCallback]] = None,
) -> Seq2SeqTrainer:
    """
    Create a Seq2SeqTrainer instance.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        compute_metrics: Function to compute metrics
        data_collator: Data collator
        callbacks: List of callbacks
        
    Returns:
        Configured Seq2SeqTrainer
    """
    # create data collator if not provided
    if data_collator is None:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )
    
    if callbacks is None:
        callbacks = []
    
    # add memory and timing callbacks
    memory_callback = MemoryTrackingCallback()
    timing_callback = TimingCallback()
    callbacks.extend([memory_callback, timing_callback])
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # store custom callbacks for later access
    trainer.memory_callback = memory_callback
    trainer.timing_callback = timing_callback
    
    return trainer


def run_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Union[TrainingConfig, Dict[str, Any]],
    compute_metrics: Optional[Callable] = None,
    run_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:

    if isinstance(config, dict):
        config = TrainingConfig(**config)
    
    logger.info(f"Starting training: {run_name or 'unnamed'}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    training_args = create_training_arguments(config, run_name)
    callbacks = []
    if config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )
    
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    if resume_from_checkpoint is None:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Found checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    
    memory_stats = trainer.memory_callback.get_peak_memory()
    timing_stats = trainer.timing_callback.get_timing_stats()
    
    trainer.save_model()
    trainer.save_state()
    
    results = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "memory_stats": memory_stats,
        "timing_stats": timing_stats,
        "run_name": run_name,
        "config": config.to_dict(),
    }
    
    logger.info(f"Training complete!")
    logger.info(f"  Final train loss: {train_metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info(f"  Peak memory: {memory_stats.get('peak_allocated_gb', 0):.2f} GB")
    logger.info(f"  Total time: {timing_stats.get('total_training_time_minutes', 0):.2f} minutes")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = TrainingConfig(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    print("Training config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    args = create_training_arguments(config, run_name="test_run")
    print(f"\nTraining args created: {args.output_dir}")
