"""CLI command interface for Face Analysis & Face Replacement tool."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from videocore.tools.base import BaseTool
from videocore.tools.face_studio.engine import (
    FaceStudioConfig,
    FaceStudioEngine,
)
from videocore.tools.face_studio.models import ensure_face_models

console = Console()


class FaceStudioTool(BaseTool):
    """Tool for analyzing faces in video and replacing/updating faces with new pictures."""

    @property
    def name(self) -> str:
        return "faces"

    @property
    def description(self) -> str:
        return "Analyze faces in video, cluster unique individuals, and replace/swap faces with new pictures."

    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            self.name,
            aliases=["face-studio", "faceswap"],
            help=self.description,
            description=(
                "VideoCore Face Studio:\n"
                "Scan videos to detect all unique individuals, inspect their appearances,\n"
                "and update/replace any detected face with custom pictures using AI restoration."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "-i", "--input",
            dest="input_path",
            type=Path,
            help="Path to the input video or photo.",
        )
        parser.add_argument(
            "-o", "--output",
            dest="output_path",
            type=Path,
            help="Path to save the swapped output video/photo.",
        )
        parser.add_argument(
            "--scan-only",
            action="store_true",
            help="Scan the video and display all detected individuals without swapping.",
        )
        parser.add_argument(
            "--target",
            dest="target_person",
            default="person_1",
            help="Target person ID to replace (default: person_1).",
        )
        parser.add_argument(
            "--new-pic",
            dest="new_pic",
            type=Path,
            help="Path to the new replacement photo.",
        )
        parser.add_argument(
            "--interval",
            dest="interval",
            type=float,
            default=0.5,
            help="Frame scan sampling interval in seconds (default: 0.5).",
        )
        parser.add_argument(
            "--no-gfpgan",
            action="store_true",
            help="Disable GFPGAN neural facial enhancement on replaced faces.",
        )
        parser.add_argument(
            "--codec",
            dest="codec",
            default="h264_nvenc",
            help="Output video codec (default: h264_nvenc).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        if not args.input_path:
            console.print("[bold red]Error:[/bold red] Input path is required. Use -i / --input <file>.")
            return 1

        inp = Path(args.input_path).resolve()
        if not inp.exists():
            console.print(f"[bold red]Error:[/bold red] Input file does not exist: {inp}")
            return 1

        ensure_face_models()

        config = FaceStudioConfig(
            input_path=inp,
            output_path=args.output_path,
            video_codec=args.codec,
            enable_gfpgan=not args.no_gfpgan,
        )
        engine = FaceStudioEngine(config)

        console.print(
            Panel.fit(
                f"[bold cyan]VideoCore Studio - Face Analyzer & Swapper[/bold cyan]\n"
                f"Input: [white]{inp.name}[/white]",
                border_style="cyan",
            )
        )

        # 1. Scan and detect faces
        with Progress(
            TextColumn("[cyan]*[/cyan]"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing faces...", total=100)

            def scan_cb(desc: str, pct: float):
                progress.update(task, description=f"[cyan]{desc}[/cyan]", completed=pct)

            clusters = engine.analyze_faces(sample_interval=args.interval, progress_cb=scan_cb)

        console.print(f"\n[bold green]Scan complete! Detected {len(clusters)} unique individuals:[/bold green]\n")

        table = Table(title="Detected Faces", show_header=True)
        table.add_column("Person ID", style="bold yellow")
        table.add_column("Name / Label", style="cyan")
        table.add_column("Appearances", style="green")
        table.add_column("First Seen", style="dim")

        for c in clusters:
            table.add_row(
                c.person_id,
                c.label,
                f"{c.appearance_count} frames",
                f"{c.first_seen_sec:.1f}s",
            )
        console.print(table)

        if args.scan_only or not args.new_pic:
            if not args.scan_only:
                console.print("\n[dim]To replace a face, specify --new-pic <path> --target <person_id> -o <output_path>[/dim]")
            return 0

        new_pic_path = Path(args.new_pic).resolve()
        if not new_pic_path.is_file():
            console.print(f"[bold red]Error:[/bold red] Replacement photo not found: {new_pic_path}")
            return 1

        config.replacement_map = {args.target_person: new_pic_path}

        # 2. Execute swap
        with Progress(
            TextColumn("[cyan]*[/cyan]"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task2 = progress.add_task("Rendering replaced faces...", total=100)

            def swap_cb(desc: str, pct: float):
                progress.update(task2, description=f"[cyan]{desc}[/cyan]", completed=pct)

            res = engine.execute_swap(clusters, stage_callback=swap_cb)

        console.print("\n[bold green]Face swap completed![/bold green]")
        console.print(f"Output saved to: [bold white]{res.output_path}[/bold white]\n")
        return 0
