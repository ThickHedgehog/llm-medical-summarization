"""
Results Analysis
================

Functions for comparing and analyzing experimental results.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> Dict[str, Any]:
    results = {}
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)
    
    predictions_path = os.path.join(results_dir, "predictions.json")
    if os.path.exists(predictions_path):
        with open(predictions_path) as f:
            results["predictions"] = json.load(f)
    
    examples_path = os.path.join(results_dir, "examples.json")
    if os.path.exists(examples_path):
        with open(examples_path) as f:
            results["examples"] = json.load(f)
    
    training_stats_path = os.path.join(results_dir, "training_stats.json")
    if os.path.exists(training_stats_path):
        with open(training_stats_path) as f:
            results["training_stats"] = json.load(f)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    if "metrics" in results:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(results["metrics"], f, indent=2)
    
    if "predictions" in results:
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(results["predictions"], f, indent=2)
    
    if "examples" in results:
        with open(os.path.join(output_dir, "examples.json"), "w") as f:
            json.dump(results["examples"], f, indent=2)
    
    if "training_stats" in results:
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(results["training_stats"], f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


def compare_results(
    results_dict: Dict[str, Dict[str, Any]],
    metrics_to_compare: Optional[List[str]] = None,
) -> pd.DataFrame:
    if metrics_to_compare is None:
        metrics_to_compare = [
            "rouge1", "rouge2", "rougeL", "rougeLsum",
            "bertscore_f1",
        ]
    
    comparison_data = []
    
    for exp_name, results in results_dict.items():
        metrics = results.get("metrics", {})
        training_stats = results.get("training_stats", {})
        
        row = {"experiment": exp_name}
        
        for metric in metrics_to_compare:
            row[metric] = metrics.get(metric, np.nan)
        
        if training_stats:
            row["trainable_params"] = training_stats.get("trainable_parameters", np.nan)
            row["trainable_pct"] = training_stats.get("trainable_percentage", np.nan)
            row["training_time_min"] = training_stats.get("training_time_minutes", np.nan)
            row["peak_memory_gb"] = training_stats.get("peak_memory_gb", np.nan)
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index("experiment")
    
    return df


def create_comparison_table(
    results_dict: Dict[str, Dict[str, Any]],
    output_format: str = "markdown",
) -> str:
    df = compare_results(results_dict)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    if output_format == "markdown":
        return df.to_markdown()
    elif output_format == "latex":
        return df.to_latex()
    elif output_format == "html":
        return df.to_html()
    else:
        return df.to_string()


def analyze_experiment_set(
    base_results_dir: str,
    experiment_pattern: Optional[str] = None,
) -> pd.DataFrame:
    results_dict = {}
    
    base_path = Path(base_results_dir)
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            if experiment_pattern is None or experiment_pattern in exp_dir.name:
                results = load_results(str(exp_dir))
                if results:
                    results_dict[exp_dir.name] = results
    
    if not results_dict:
        logger.warning(f"No results found in {base_results_dir}")
        return pd.DataFrame()
    
    return compare_results(results_dict)


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    
    metric_cols = [c for c in df.columns if c.startswith(("rouge", "bertscore"))]
    
    summary["best_by_metric"] = {}
    for col in metric_cols:
        if col in df.columns:
            best_idx = df[col].idxmax()
            summary["best_by_metric"][col] = {
                "experiment": best_idx,
                "value": df.loc[best_idx, col],
            }
    
    if "trainable_params" in df.columns:
        summary["most_efficient"] = df["trainable_params"].idxmin()
        summary["least_efficient"] = df["trainable_params"].idxmax()
    
    if "training_time_min" in df.columns:
        summary["fastest_training"] = df["training_time_min"].idxmin()
        summary["slowest_training"] = df["training_time_min"].idxmax()
    
    return summary


def generate_report_section(
    results_dict: Dict[str, Dict[str, Any]],
    section_title: str = "Experimental Results",
) -> str:
    lines = [
        f"## {section_title}",
        "",
        "### Quantitative Results",
        "",
    ]
    
    df = compare_results(results_dict)
    if not df.empty:
        lines.append(df.to_markdown())
        lines.append("")
    
    summary = create_summary_statistics(df)
    
    if "best_by_metric" in summary:
        lines.append("### Best Performing Models")
        lines.append("")
        for metric, info in summary["best_by_metric"].items():
            lines.append(f"- **{metric}**: {info['experiment']} ({info['value']:.2f})")
        lines.append("")
    
    if "most_efficient" in summary:
        lines.append("### Efficiency Analysis")
        lines.append("")
        lines.append(f"- Most parameter-efficient: **{summary['most_efficient']}**")
        if "fastest_training" in summary:
            lines.append(f"- Fastest training: **{summary['fastest_training']}**")
        lines.append("")
    
    return "\n".join(lines)


def export_for_visualization(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    df = compare_results(results_dict)
    
    csv_path = output_path.replace(".json", ".csv") if output_path.endswith(".json") else output_path + ".csv"
    df.to_csv(csv_path)
    
    json_path = output_path.replace(".csv", ".json") if output_path.endswith(".csv") else output_path + ".json"
    
    export_data = {
        "experiments": df.reset_index().to_dict(orient="records"),
        "metrics": list(df.columns),
    }
    
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported visualization data to {csv_path} and {json_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sample_results = {
        "flan-t5-base-lora": {
            "metrics": {
                "rouge1": 45.2,
                "rouge2": 22.1,
                "rougeL": 38.4,
                "bertscore_f1": 68.3,
            },
            "training_stats": {
                "trainable_parameters": 589824,
                "trainable_percentage": 0.24,
                "training_time_minutes": 45,
                "peak_memory_gb": 12.3,
            }
        },
        "flan-t5-base-prompt": {
            "metrics": {
                "rouge1": 42.8,
                "rouge2": 20.5,
                "rougeL": 36.2,
                "bertscore_f1": 66.1,
            },
            "training_stats": {
                "trainable_parameters": 15360,
                "trainable_percentage": 0.006,
                "training_time_minutes": 30,
                "peak_memory_gb": 8.5,
            }
        },
        "flan-t5-base-full": {
            "metrics": {
                "rouge1": 46.5,
                "rouge2": 23.2,
                "rougeL": 39.8,
                "bertscore_f1": 69.5,
            },
            "training_stats": {
                "trainable_parameters": 247577856,
                "trainable_percentage": 100.0,
                "training_time_minutes": 180,
                "peak_memory_gb": 24.5,
            }
        },
    }
    
    print("Comparison Table:")
    print(create_comparison_table(sample_results))
    
    print("\n" + "=" * 60 + "\n")
    
    print("Report Section:")
    print(generate_report_section(sample_results))
