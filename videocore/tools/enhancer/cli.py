"""CLI command interface for AI Media Enhancement tool."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from videocore.tools.base import BaseTool
from videocore.tools.enhancer.engine import EnhanceConfig, EnhancerEngine, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from videocore.tools.enhancer.models import ensure_all_models

console = Console()


class EnhancerTool(BaseTool):
    """Tool for AI-powered image and video enhancement, upscaling, and restoration."""

    @property
    def name(self) -> str:
        return "enhance"

    @property
    def description(self) -> str:
        return "Enhance, deblur, upscale, restore faces, and smooth pictures and videos using AI."

    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            self.name,
            help=self.description,
            description=(
                "VideoCore AI Media Enhancer:\n"
                "Super-resolution upscaling (Real-ESRGAN), facial restoration (GFPGAN),\n"
                "and video framerate smoothing (RIFE) with RTX GPU hardware acceleration."
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
            help="Path to save the enhanced output file. If omitted, saves beside input with '_enhanced'.",
        )

        enhance_group = parser.add_argument_group("Enhancement Options")
        enhance_group.add_argument(
            "-s", "--scale",
            dest="scale",
            type=int,
            choices=[1, 2, 3, 4],
            default=None,
            help="Upscale factor (1=no upscale, 2=2x, 4=4x). Defaults to 4 for images, 2 for videos.",
        )
        enhance_group.add_argument(
            "-m", "--model",
            dest="model",
            type=str,
            default="realesrgan-x4plus",
            help="Upscaling model preset ('realesrgan-x4plus', 'realesrgan-x4plus-anime', etc.).",
        )
        enhance_group.add_argument(
            "--face-restore",
            dest="face_restore",
            action="store_true",
            help="Enable AI facial reconstruction for blurry or low-res portraits and people.",
        )
        enhance_group.add_argument(
            "--face-fidelity",
            dest="face_fidelity",
            type=float,
            default=0.7,
            help="Face restoration fidelity slider (0.0 to 1.0, default: 0.7). Higher preserves more identity.",
        )

        video_group = parser.add_argument_group("Video Specific Options")
        video_group.add_argument(
            "--interpolate",
            dest="interpolate",
            action="store_true",
            help="Smooth video motion by interpolating intermediate frames (RIFE).",
        )
        video_group.add_argument(
            "--target-fps",
            dest="target_fps",
            type=float,
            default=None,
            help="Target video framerate for motion smoothing (e.g. 60).",
        )
        video_group.add_argument(
            "--codec",
            dest="codec",
            type=str,
            default="h264_nvenc",
            help="Video re-encoding codec ('h264_nvenc', 'hevc_nvenc', 'libx264'). Default: h264_nvenc.",
        )

        perf_group = parser.add_argument_group("Performance & GPU")
        perf_group.add_argument(
            "--tile",
            dest="tile_size",
            type=int,
            default=400,
            help="Tile size for GPU inference (default 400, prevents out-of-memory errors on 8GB GPUs).",
        )
        perf_group.add_argument(
            "--gpu",
            dest="gpu_id",
            type=int,
            default=0,
            help="GPU device id to use (default: 0).",
        )
        perf_group.add_argument(
            "--download-models",
            dest="download_models",
            action="store_true",
            help="Pre-download all AI models and binaries without running an enhancement task.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        if args.download_models:
            ensure_all_models()
            return 0

        if not args.input_path:
            console.print("[bold red]Error:[/bold red] Input path is required. Use -i / --input <file>.")
            return 1

        inp: Path = args.input_path.resolve()
        if not inp.exists():
            console.print(f"[bold red]Error:[/bold red] Input file not found: {inp}")
            return 1

        ext = inp.suffix.lower()
        is_image = ext in IMAGE_EXTENSIONS
        is_video = ext in VIDEO_EXTENSIONS

        if not is_image and not is_video:
            console.print(f"[bold red]Error:[/bold red] Unsupported media extension '{ext}'.")
            return 1

        # Determine default scale
        scale = args.scale
        if scale is None:
            scale = 4 if is_image else 2

        config = EnhanceConfig(
            input_path=inp,
            output_path=args.output_path.resolve() if args.output_path else None,
            scale=scale,
            model_name=args.model,
            face_restore=args.face_restore,
            face_fidelity=args.face_fidelity,
            interpolate=args.interpolate or (args.target_fps is not None),
            target_fps=args.target_fps,
            tile_size=args.tile_size,
            gpu_id=args.gpu_id,
            video_codec=args.codec,
        )

        # Print Pre-flight summary table
        table = Table(title="[bold cyan]AI Media Enhancement Job[/bold cyan]", box=None)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Input File", str(inp.name))
        table.add_row("Media Type", "Image" if is_image else "Video")
        table.add_row("Upscale Factor", f"{scale}x ({config.model_name})" if scale > 1 else "Off (1x)")
        table.add_row("Face Restoration", f"[green]Enabled[/green] (fidelity: {config.face_fidelity})" if config.face_restore else "[dim]Disabled[/dim]")
        if is_video:
            table.add_row("Motion Smoothing", f"[green]Enabled[/green] (target: {config.target_fps or '2x'} fps)" if config.interpolate else "[dim]Disabled[/dim]")
            table.add_row("Video Encoder", f"[green]{config.video_codec}[/green] (NVIDIA NVENC Accelerated)")
        table.add_row("GPU Device", f"Device {config.gpu_id} (Tile size: {config.tile_size})")

        console.print(table)
        console.print()

        engine = EnhancerEngine(config)
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Initializing AI Pipeline...", total=100.0)

            def stage_updater(stage_name: str, pct: float):
                progress.update(main_task, description=stage_name, completed=pct)

            try:
                result = engine.execute(stage_callback=stage_updater)
                progress.update(main_task, description="Enhancement Complete", completed=100.0)
            except Exception as e:
                console.print(f"\n[bold red]Enhancement failed:[/bold red] {e}")
                return 1

        elapsed = time.time() - start_time

        # Results summary panel
        summary_lines = [
            f"[bold green][OK] Media Enhancement Succeeded![/bold green]",
            f"* Output File: [bold cyan]{result.output_path}[/bold cyan]",
            f"* Resolution: {result.original_resolution[0]}x{result.original_resolution[1]} -> [bold green]{result.enhanced_resolution[0]}x{result.enhanced_resolution[1]}[/bold green]",
        ]
        if result.is_video:
            summary_lines.append(f"* Frame Rate: {result.original_fps:.2f} fps -> [bold green]{result.enhanced_fps:.2f} fps[/bold green]")
        if result.faces_restored:
            summary_lines.append("* Faces Restored: [bold green]Yes (GFPGAN)[/bold green]")
        summary_lines.append(f"* Processing Time: [bold yellow]{elapsed:.2f} seconds[/bold yellow]")

        console.print()
        console.print(Panel("\n".join(summary_lines), title="[bold green]Success[/bold green]", border_style="green"))
        return 0
