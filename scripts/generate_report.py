#!/usr/bin/env python3
"""
Generate Comparison Report
==========================

Generate a comprehensive comparison report from experiment results.

Usage:
    python scripts/generate_report.py --results_dir ./results --output ./results/report.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd


def load_experiment_results(results_dir: str) -> dict:
    """Load all experiment results from a directory."""
    results = {}
    
    results_path = Path(results_dir)
    
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results[exp_dir.name] = json.load(f)
    
    return results


def create_metrics_dataframe(results: dict) -> pd.DataFrame:
    """Create a DataFrame from experiment results."""
    rows = []
    
    for exp_name, exp_data in results.items():
        row = {"experiment": exp_name}
        
        parts = exp_name.split("_")
        if len(parts) >= 2:
            row["model"] = parts[0]
            row["method"] = "_".join(parts[1:])
        
        if "model_info" in exp_data:
            row["total_params"] = exp_data["model_info"].get("total_parameters", 0)
            row["trainable_params"] = exp_data["model_info"].get("trainable_parameters", 0)
            row["trainable_pct"] = exp_data["model_info"].get("trainable_percentage", 0)
        
        if "test_metrics" in exp_data:
            for metric, value in exp_data["test_metrics"].items():
                if isinstance(value, (int, float)):
                    row[metric] = value
        
        if "timing_stats" in exp_data:
            row["training_time_min"] = exp_data["timing_stats"].get(
                "total_training_time_minutes", 0
            )
        
        if "memory_stats" in exp_data:
            row["peak_memory_gb"] = exp_data["memory_stats"].get(
                "peak_allocated_gb", 0
            )
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_markdown_report(df: pd.DataFrame, output_path: str) -> None:
    """Generate a Markdown report."""
    lines = [
        "# Parameter-Efficient Fine-Tuning Comparison Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Overview",
        "",
        "This report compares different adaptation methods (LoRA, Prompt Tuning, Full Fine-Tuning) ",
        "across multiple Flan-T5 model sizes for medical text summarization.",
        "",
        "---",
        "",
        "## Results Summary",
        "",
    ]
    
    if not df.empty:
        display_cols = ["experiment", "method"]
        metric_cols = [c for c in df.columns if c.startswith(("rouge", "bertscore"))]
        param_cols = ["trainable_params", "trainable_pct"]
        efficiency_cols = ["training_time_min", "peak_memory_gb"]
        
        available_cols = display_cols + [c for c in metric_cols + param_cols + efficiency_cols if c in df.columns]
        display_df = df[available_cols].copy()
        
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].round(2)
        
        lines.append("### Performance Metrics")
        lines.append("")
        lines.append(display_df.to_markdown(index=False))
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Best Performing Models")
    lines.append("")
    
    metric_cols = [c for c in df.columns if c.startswith(("rouge", "bertscore"))]
    for metric in metric_cols:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_exp = df.loc[best_idx, "experiment"]
            best_val = df.loc[best_idx, metric]
            lines.append(f"- **{metric}**: {best_exp} ({best_val:.2f})")
    
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Efficiency Analysis")
    lines.append("")
    
    if "trainable_params" in df.columns:
        method_stats = df.groupby("method").agg({
            "trainable_params": "mean",
            "trainable_pct": "mean",
        }).round(2)
        
        lines.append("### Trainable Parameters by Method")
        lines.append("")
        lines.append(method_stats.to_markdown())
        lines.append("")
    
    if "training_time_min" in df.columns:
        lines.append("### Training Time Comparison")
        lines.append("")
        
        time_stats = df.groupby("method")["training_time_min"].mean().round(2)
        for method, time in time_stats.items():
            lines.append(f"- **{method}**: {time:.1f} minutes (average)")
        lines.append("")
    
    if "peak_memory_gb" in df.columns:
        lines.append("### Peak GPU Memory Usage")
        lines.append("")
        
        mem_stats = df.groupby("method")["peak_memory_gb"].mean().round(2)
        for method, mem in mem_stats.items():
            lines.append(f"- **{method}**: {mem:.1f} GB (average)")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("Based on the experimental results:")
    lines.append("")
    
    if "rouge1" in df.columns and "trainable_pct" in df.columns:
        df_sorted = df.sort_values("rouge1", ascending=False)
        best_overall = df_sorted.iloc[0]["experiment"]
        
        lora_results = df[df["method"].str.contains("lora", case=False)]
        if not lora_results.empty:
            best_lora = lora_results.sort_values("rouge1", ascending=False).iloc[0]
            lines.append(f"1. **LoRA** achieves {best_lora['rouge1']:.1f} ROUGE-1 with only "
                        f"{best_lora['trainable_pct']:.2f}% trainable parameters")
        
        pt_results = df[df["method"].str.contains("prompt", case=False)]
        if not pt_results.empty:
            best_pt = pt_results.sort_values("rouge1", ascending=False).iloc[0]
            lines.append(f"2. **Prompt Tuning** is the most parameter-efficient with "
                        f"{best_pt['trainable_pct']:.4f}% trainable parameters")
        
        full_results = df[df["method"].str.contains("full", case=False)]
        if not full_results.empty:
            best_full = full_results.sort_values("rouge1", ascending=False).iloc[0]
            lines.append(f"3. **Full Fine-Tuning** achieves the highest performance "
                        f"({best_full['rouge1']:.1f} ROUGE-1) but requires all parameters")
    
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} experiments")
    
    df = create_metrics_dataframe(results)
    
    generate_markdown_report(df, args.output)
    
    csv_path = args.output.replace(".md", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
