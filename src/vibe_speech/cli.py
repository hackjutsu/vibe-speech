from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .logging import setup_logging
from .runtime import SpeechRuntime

console = Console()


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
@click.option("--log-level", default=None, help="Override log level (INFO, DEBUG, ...)")
@click.pass_context
def app(ctx: click.Context, config_path: Optional[Path], log_level: Optional[str]) -> None:
    cfg = AppConfig.from_file(config_path)
    if log_level:
        cfg.log_level = log_level
    setup_logging(cfg.log_level)
    ctx.obj = cfg


@app.command()
@click.option("--dry-run", is_flag=True, help="Do not type; log only.")
@click.pass_obj
def serve(cfg: AppConfig, dry_run: bool) -> None:
    """Start the voice-to-text loop (currently stubbed)."""
    cfg.output.dry_run = dry_run or cfg.output.dry_run
    runtime = SpeechRuntime(cfg)
    console.print("[yellow]Starting vibe-speech (audio/Whisper not yet wired). Press Ctrl+C to exit.[/yellow]")
    runtime.block_forever()


@app.command()
@click.pass_obj
def doctor(cfg: AppConfig) -> None:
    """Show current configuration."""
    table = Table(title="vibe-speech configuration")
    table.add_column("Section")
    table.add_column("Value")
    table.add_row("audio.sample_rate", str(cfg.audio.sample_rate))
    table.add_row("audio.chunk_seconds", str(cfg.audio.chunk_seconds))
    table.add_row("audio.device_name", str(cfg.audio.device_name))
    table.add_row("whisper.model_size", cfg.whisper.model_size)
    table.add_row("whisper.compute_type", cfg.whisper.compute_type)
    table.add_row("processing.mode", cfg.processing.mode)
    table.add_row("processing.max_chars", str(cfg.processing.max_chars))
    table.add_row("output.dry_run", str(cfg.output.dry_run))
    table.add_row("output.typing_delay", str(cfg.output.typing_delay))
    table.add_row("output.focus_target", str(cfg.output.focus_target))
    table.add_row("hotkey.toggle", cfg.hotkey.toggle)
    table.add_row("log_level", cfg.log_level)
    console.print(table)


if __name__ == "__main__":
    app()

