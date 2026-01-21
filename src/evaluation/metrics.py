"""
Evaluation Metrics
==================

Functions for computing evaluation metrics for summarization.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Tuple

import numpy as np
import nltk
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        rouge_types: Types of ROUGE to compute
        use_stemmer: Whether to use Porter stemmer
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)
    
    scores = {rouge_type: [] for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)
    
    results = {}
    for rouge_type in rouge_types:
        results[rouge_type] = np.mean(scores[rouge_type]) * 100  # to percentage
    
    return results


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute BERTScore.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        model_type: Model to use for BERTScore
        lang: Language
        batch_size: Batch size for computation
        device: Device to use
        
    Returns:
        Dictionary with precision, recall, and F1 BERTScores
    """
    try:
        from bert_score import score
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore computation.")
        return {"bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0}
    
    P, R, F1 = score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    
    return {
        "bertscore_precision": P.mean().item() * 100,
        "bertscore_recall": R.mean().item() * 100,
        "bertscore_f1": F1.mean().item() * 100,
    }


def create_compute_metrics_fn(
    tokenizer,
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"],
    compute_bertscore_flag: bool = False,
) -> Callable:
    def compute_metrics(eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        
        # handle tuple output (predictions, decoder_hidden_states, cross_attentions)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # ensure predictions are valid token IDs
        # replace any negative values or values >= vocab_size with pad_token_id
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        predictions = np.array(predictions)
        predictions = np.where(predictions < 0, pad_token_id, predictions)
        predictions = np.where(predictions >= vocab_size, pad_token_id, predictions)
        predictions = predictions.astype(np.int64)
        
        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error decoding predictions: {e}")
            decoded_preds = ["" for _ in range(len(predictions))]
        
        # replace -100 in labels (padding) with pad token id
        labels = np.array(labels)
        labels = np.where(labels != -100, labels, pad_token_id)
        labels = np.where(labels < 0, pad_token_id, labels)
        labels = np.where(labels >= vocab_size, pad_token_id, labels)
        labels = labels.astype(np.int64)
        
        try:
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error decoding labels: {e}")
            decoded_labels = ["" for _ in range(len(labels))]
        
        # clean up predictions and labels
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # filter out empty predictions/labels for ROUGE computation
        valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p and l]
        
        if not valid_pairs:
            logger.warning("No valid prediction-label pairs for metric computation")
            return {rouge_type: 0.0 for rouge_type in rouge_types + ["rougeLsum", "gen_len"]}
        
        valid_preds, valid_labels = zip(*valid_pairs)
        
        # add newlines for rougeLsum
        decoded_preds_newline = ["\n".join(nltk.sent_tokenize(pred)) for pred in valid_preds]
        decoded_labels_newline = ["\n".join(nltk.sent_tokenize(label)) for label in valid_labels]
        
        # compute ROUGE
        results = compute_rouge_scores(
            decoded_preds_newline,
            decoded_labels_newline,
            rouge_types=rouge_types + ["rougeLsum"],
        )
        
        # optionally compute BERTScore
        if compute_bertscore_flag:
            bertscore_results = compute_bertscore(list(valid_preds), list(valid_labels))
            results.update(bertscore_results)
        
        results["gen_len"] = np.mean([len(pred.split()) for pred in valid_preds])
        
        return results
    
    return compute_metrics


def compute_metrics(
    predictions: List[str],
    references: List[str],
    compute_bertscore_flag: bool = True,
) -> Dict[str, float]:
    predictions_newline = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    references_newline = ["\n".join(nltk.sent_tokenize(ref)) for ref in references]
    
    results = compute_rouge_scores(predictions_newline, references_newline)
    
    if compute_bertscore_flag:
        bertscore_results = compute_bertscore(predictions, references)
        results.update(bertscore_results)
    
    results["avg_pred_length"] = np.mean([len(pred.split()) for pred in predictions])
    results["avg_ref_length"] = np.mean([len(ref.split()) for ref in references])
    
    return results


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = ["=" * 50, "EVALUATION METRICS", "=" * 50]
    
    # ROUGE scores
    rouge_keys = [k for k in metrics if k.startswith("rouge")]
    if rouge_keys:
        lines.append("\nROUGE Scores:")
        for key in sorted(rouge_keys):
            lines.append(f"  {key}: {metrics[key]:.2f}")
    
    # BERTScore
    bert_keys = [k for k in metrics if k.startswith("bertscore")]
    if bert_keys:
        lines.append("\nBERTScore:")
        for key in sorted(bert_keys):
            lines.append(f"  {key}: {metrics[key]:.2f}")
    
    # length stats
    if "avg_pred_length" in metrics:
        lines.append(f"\nAverage prediction length: {metrics['avg_pred_length']:.1f} words")
    if "avg_ref_length" in metrics:
        lines.append(f"Average reference length: {metrics['avg_ref_length']:.1f} words")
    if "gen_len" in metrics:
        lines.append(f"Average generation length: {metrics['gen_len']:.1f} words")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictions = [
        "The study found that the treatment was effective.",
        "Results showed significant improvement in patients.",
    ]
    
    references = [
        "The clinical trial demonstrated the treatment's efficacy.",
        "The research revealed notable patient improvement.",
    ]
    
    print("Testing metrics computation...")
    
    rouge_results = compute_rouge_scores(predictions, references)
    print("\nROUGE scores:")
    for key, value in rouge_results.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nComputing all metrics (including BERTScore)...")
    all_results = compute_metrics(predictions, references, compute_bertscore_flag=False)
    print(format_metrics(all_results))