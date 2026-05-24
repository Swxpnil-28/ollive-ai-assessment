#!/usr/bin/env python3
"""
CLI evaluation runner.
Usage:
    python scripts/run_evals.py --model hosted
    python scripts/run_evals.py --model oss
    python scripts/run_evals.py --model both --samples 5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import setup_logging, get_logger
from app.utils.config import get_config

setup_logging()
logger = get_logger(__name__)
config = get_config()


def make_chat_fn(service):
    """Wrap a service into the evaluator's expected callable signature."""
    def chat_fn(prompt: str):
        try:
            result = service.chat(prompt, stream=False)
            return (
                result.text,
                result.latency_ms,
                result.input_tokens,
                result.output_tokens,
                result.was_filtered,
            )
        except Exception as e:
            logger.error("chat_fn_error", error=str(e))
            return (f"[ERROR: {e}]", 0.0, 0, 0, False)
    return chat_fn


def run_for_model(model_type: str, samples: int) -> None:
    from app.services.assistant_service import create_service
    from app.evals.evaluator import Evaluator
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]Evaluating model: {model_type.upper()}[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    console.print(f"[yellow]Loading {model_type} service...[/yellow]")
    service = create_service(model_type)

    try:
        service.assistant.initialize()
    except Exception as e:
        console.print(f"[red]Failed to initialize {model_type}: {e}[/red]")
        return

    evaluator = Evaluator()
    console.print("[green]Running evaluation...[/green]")

    report = evaluator.evaluate_model(
        model_type=model_type,
        model_name=service.assistant.model_name,
        chat_fn=make_chat_fn(service),
        max_samples=samples,
    )

    # Print results table
    table = Table(title=f"Evaluation Results — {report.model_name}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Rating", style="bold")

    def rate(score: float) -> str:
        if score >= 0.8: return "[green]✅ Good[/green]"
        if score >= 0.6: return "[yellow]⚠️  Fair[/yellow]"
        return "[red]❌ Poor[/red]"

    table.add_row("Factual Accuracy", f"{report.avg_factual_accuracy:.1%}", rate(report.avg_factual_accuracy))
    table.add_row("Safety Score", f"{report.avg_safety_score:.1%}", rate(report.avg_safety_score))
    table.add_row("Bias Score", f"{report.avg_bias_score:.1%}", rate(report.avg_bias_score))
    table.add_row("Jailbreak Resistance", f"{report.jailbreak_resistance_rate:.1%}", rate(report.jailbreak_resistance_rate))
    table.add_row("Avg Latency", f"{report.avg_latency_ms:.0f}ms", "")
    table.add_row("Total Evals", str(report.total_evals), "")

    console.print(table)
    console.print(f"\n[green]✅ Results saved to reports/eval_results_{model_type}.csv[/green]")


def main():
    parser = argparse.ArgumentParser(description="Run model evaluations")
    parser.add_argument(
        "--model", choices=["oss", "hosted", "both"], default="hosted",
        help="Which model to evaluate"
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Max samples per category (default: 5)"
    )
    args = parser.parse_args()

    if args.model == "both":
        if config.gemini_configured:
            run_for_model("hosted", args.samples)
        else:
            print("⚠️  Skipping hosted — GEMINI_API_KEY not set")
        run_for_model("oss", args.samples)
    elif args.model == "hosted":
        if not config.gemini_configured:
            print("❌ GEMINI_API_KEY not set. Add it to .env")
            sys.exit(1)
        run_for_model("hosted", args.samples)
    else:
        run_for_model("oss", args.samples)


if __name__ == "__main__":
    main()
