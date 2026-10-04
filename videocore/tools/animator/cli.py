"""CLI command interface for Real <-> Animation Stylization tool."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from videocore.tools.animator.engine import (
    AnimationConfig,
    AnimatorEngine,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from videocore.tools.animator.models import ensure_all_anime_models
from videocore.tools.base import BaseTool

console = Console()


class AnimatorTool(BaseTool):
    """Tool for bidirectional transformation between real video/photos and animation/anime."""

    @property
    def name(self) -> str:
        return "animate"

    @property
    def description(self) -> str:
        return "Transform real videos into anime/animation and convert animation back into photorealistic video."

    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            self.name,
            aliases=["stylize", "anime"],
            help=self.description,
            description=(
                "VideoCore Real <-> Animation Studio:\n"
                "Transform real video/photos into anime (Ghibli Hayao, Shinkai, Paprika, Cel-Shading, Comic)\n"
                "or convert 2D animation back into photorealistic video with film grain and face restoration."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        input_group = parser.add_argument_group("Input & Output")
        input_group.add_argument(
            "-i", "--input",
            dest="input_path",
            type=Path,
            help="Path to the input image or video file.",
        )
        input_group.add_argument(
            "-o", "--output",
            dest="output_path",
            type=Path,
            help="Path to save the transformed output file.",
        )

        mode_group = parser.add_argument_group("Transformation Options")
        mode_group.add_argument(
            "-d", "--direction",
            dest="direction",
            choices=["real_to_anim", "anim_to_real"],
            default="real_to_anim",
            help="Direction of transformation (default: real_to_anim).",
        )
        mode_group.add_argument(
            "-s", "--style",
            dest="style",
            choices=["hayao", "shinkai", "paprika", "cartoon", "comic", "watercolor"],
            default="hayao",
            help="Animation style for real_to_anim (default: hayao).",
        )
        mode_group.add_argument(
            "--strength",
            dest="strength",
            type=float,
            default=1.0,
            help="Effect blend strength between 0.0 and 1.0 (default: 1.0).",
        )
        mode_group.add_argument(
            "--line-weight",
            dest="line_weight",
            type=float,
            default=1.0,
            help="Line art thickness multiplier (default: 1.0).",
        )
        mode_group.add_argument(
            "--saturation",
            dest="color_saturation",
            type=float,
            default=1.1,
            help="Color saturation multiplier (default: 1.1).",
        )

        realism_group = parser.add_argument_group("Animation -> Real Photorealism Options")
        realism_group.add_argument(
            "--edge-soften",
            dest="edge_soften",
            type=float,
            default=0.7,
            help="Line delineation and contour softening factor (default: 0.7).",
        )
        realism_group.add_argument(
            "--film-grain",
            dest="film_grain",
            type=float,
            default=0.15,
            help="35mm physical film grain intensity (default: 0.15).",
        )
        realism_group.add_argument(
            "--face-realism",
            dest="face_realism",
            action="store_true",
            help="Apply AI facial reconstruction (GFPGAN) to synthesize photorealistic human faces.",
        )

        codec_group = parser.add_argument_group("Hardware & Codec")
        codec_group.add_argument(
            "--codec",
            dest="video_codec",
            default="h264_nvenc",
            help="FFmpeg video encoder codec (default: h264_nvenc).",
        )
        codec_group.add_argument(
            "--download-models",
            dest="download_models",
            action="store_true",
            help="Pre-download all AnimeGAN neural models and exit.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        if args.download_models:
            console.print("[cyan]Downloading all AnimeGANv2 neural models...[/cyan]")
            ensure_all_anime_models()
            console.print("[bold green]All models downloaded successfully![/bold green]")
            return 0

        if not args.input_path:
            console.print("[bold red]Error:[/bold red] Input path is required. Use -i / --input <file>.")
            return 1

        inp = Path(args.input_path).resolve()
        if not inp.exists():
            console.print(f"[bold red]Error:[/bold red] Input file does not exist: {inp}")
            return 1

        config = AnimationConfig(
            input_path=inp,
            output_path=args.output_path,
            direction=args.direction,
            style=args.style,
            strength=args.strength,
            line_weight=args.line_weight,
            color_saturation=args.color_saturation,
            edge_soften=args.edge_soften,
            film_grain=args.film_grain,
            face_realism=args.face_realism,
            video_codec=args.video_codec,
        )

        direction_label = "Real -> Animation" if config.direction == "real_to_anim" else "Animation -> Real"
        console.print(
            Panel.fit(
                f"[bold cyan]VideoCore Studio - Animation & Stylizer[/bold cyan]\n"
                f"Mode: [bold magenta]{direction_label}[/bold magenta] * Style: [bold yellow]{config.style.title()}[/bold yellow]\n"
                f"Input: [white]{inp.name}[/white]",
                border_style="cyan",
            )
        )

        engine = AnimatorEngine(config)
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=100)

            def update_cb(stage: str, pct: float):
                progress.update(task, description=f"[cyan]{stage}[/cyan]", completed=pct)

            try:
                res = engine.execute(stage_callback=update_cb)
            except Exception as e:
                console.print(f"\n[bold red]Transformation failed:[/bold red] {e}")
                return 1

        elapsed = time.time() - start_time
        console.print(f"\n[bold green]Processing completed in {elapsed:.2f} seconds![/bold green]")
        console.print(f"Output saved to: [bold white]{res.output_path}[/bold white]\n")

        table = Table(title="Transformation Summary", show_header=True)
        table.add_column("Property", style="dim")
        table.add_column("Details", style="cyan")
        table.add_row("Direction", direction_label)
        table.add_row("Style", config.style.title())
        table.add_row("Input Resolution", f"{res.original_resolution[0]}x{res.original_resolution[1]}")
        table.add_row("Output Resolution", f"{res.processed_resolution[0]}x{res.processed_resolution[1]}")
        if res.is_video and res.duration:
            table.add_row("Duration", f"{res.duration:.2f}s")
            table.add_row("Framerate", f"{res.original_fps:.1f} fps")
        table.add_row("Output File", str(res.output_path))
        console.print(table)

        return 0
