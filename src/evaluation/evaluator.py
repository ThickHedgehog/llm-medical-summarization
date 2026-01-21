"""
Model Evaluator
===============

Functions for evaluating trained models and generating predictions.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        max_length: int = 256,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def generate_predictions(
        self,
        dataset,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> Tuple[List[str], List[str]]:
        predictions = []
        references = []
        
        def collate_fn(batch):
            """Collate batch into tensors."""
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            labels = torch.tensor([item["labels"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        iterator = tqdm(dataloader, desc="Generating predictions") if show_progress else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )
                
                decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend([pred.strip() for pred in decoded_preds])
                
                labels = labels.numpy()
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                references.extend([ref.strip() for ref in decoded_refs])
        
        return predictions, references
    
    def evaluate(
        self,
        dataset,
        batch_size: int = 8,
        compute_bertscore: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        logger.info("Generating predictions...")
        predictions, references = self.generate_predictions(
            dataset,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        
        logger.info("Computing metrics...")
        metrics = compute_metrics(
            predictions,
            references,
            compute_bertscore_flag=compute_bertscore,
        )
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
            "num_samples": len(predictions),
        }
    
    def get_qualitative_examples(
        self,
        dataset,
        num_examples: int = 10,
        indices: Optional[List[int]] = None,
    ) -> List[Dict[str, str]]:
        if indices is None:
            indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
        
        examples = []
        
        for idx in indices:
            sample = dataset[int(idx)]
            
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )
            
            input_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            labels = np.array(sample["labels"])
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            reference = self.tokenizer.decode(labels, skip_special_tokens=True)
            
            examples.append({
                "index": int(idx),
                "input": input_text[:1000] + "..." if len(input_text) > 1000 else input_text,
                "prediction": prediction,
                "reference": reference,
            })
        
        return examples


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    batch_size: int = 8,
    max_length: int = 256,
    num_beams: int = 4,
    compute_bertscore: bool = True,
    num_qualitative_examples: int = 10,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        num_beams=num_beams,
    )
    
    results = evaluator.evaluate(
        test_dataset,
        batch_size=batch_size,
        compute_bertscore=compute_bertscore,
    )
    
    examples = evaluator.get_qualitative_examples(
        test_dataset,
        num_examples=num_qualitative_examples,
    )
    results["qualitative_examples"] = examples
    
    print(format_metrics(results["metrics"]))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(results["metrics"], f, indent=2)
        
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump({
                "predictions": results["predictions"],
                "references": results["references"],
            }, f, indent=2)
        
        with open(os.path.join(output_dir, "examples.json"), "w") as f:
            json.dump(results["qualitative_examples"], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    return results


def generate_predictions(
    model,
    tokenizer,
    dataset,
    batch_size: int = 8,
    max_length: int = 256,
    num_beams: int = 4,
) -> Tuple[List[str], List[str]]:
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        num_beams=num_beams,
    )
    
    return evaluator.generate_predictions(dataset, batch_size=batch_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Evaluator module ready for use.")
    print("use evaluate_model() for full evaluation pipeline.")
