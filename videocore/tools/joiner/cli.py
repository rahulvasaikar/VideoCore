"""CLI handler for video joining command."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from videocore.core.constants import VIDEO_EXTENSIONS
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from videocore.core.probe import probe_video
from videocore.tools.base import BaseTool
from videocore.tools.joiner.engine import JoinConfig, JoinMode, VideoJoiner


console = Console()

RESOLUTION_PRESETS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "4k": (3840, 2160),
    "480p": (854, 480),
    "vertical_1080p": (1080, 1920),
    "vertical_720p": (720, 1280),
}


def parse_resolution(res_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not res_str:
        return None
    res_str = res_str.lower().strip()
    if res_str in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[res_str]
    if "x" in res_str:
        w, h = res_str.split("x")
        return int(w), int(h)
    raise ValueError(f"Invalid resolution format '{res_str}'. Use 'WIDTHxHEIGHT' or preset (1080p, 720p, 4k).")


def parse_filelist(filelist_path: Path) -> List[Path]:
    if not filelist_path.is_file():
        raise FileNotFoundError(f"File list not found: {filelist_path}")

    paths = []
    base_dir = filelist_path.parent
    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            p = Path(clean)
            if not p.is_absolute():
                candidate_file_rel = (base_dir / p).resolve()
                candidate_cwd_rel = (Path.cwd() / p).resolve()
                if candidate_file_rel.exists():
                    p = candidate_file_rel
                elif candidate_cwd_rel.exists():
                    p = candidate_cwd_rel
                else:
                    p = candidate_file_rel
            paths.append(p)
    return paths


def natural_sort_key(name: str) -> list:
    """Sorts strings with embedded numbers naturally (e.g. clip2 before clip10, MOV00021 before MOV00022)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


def discover_folder_videos(folder: Path, exclude_output: Optional[Path] = None) -> List[Path]:
    """Finds and naturally sorts all video files in a folder."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")

    videos = []
    for item in folder.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        lower_name = item.name.lower()
        # Exclude temporary or previous joined outputs to prevent recursive loop
        if "_joined" in lower_name or "joined_output" in lower_name:
            continue
        if exclude_output and item.resolve() == exclude_output.resolve():
            continue
        videos.append(item)

    videos.sort(key=lambda p: natural_sort_key(p.name))
    return videos


class JoinerTool(BaseTool):
    """Tool for concatenating multiple videos in order."""

    @property
    def name(self) -> str:
        return "join"

    @property
    def description(self) -> str:
        return "Join multiple videos together in a specified sequence order."

    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            self.name,
            help=self.description,
            description="Concatenates multiple video files in user-specified order with automatic format normalization.",
        )

        input_group = parser.add_argument_group("Input Videos")
        input_group.add_argument(
            "-i", "--inputs",
            nargs="*",
            help="List of video file paths in the desired order, or directory path containing videos.",
        )
        input_group.add_argument(
            "-d", "--dir", "--folder",
            dest="folder",
            type=Path,
            help="Directory path containing videos to join. Videos will be discovered and joined in natural filename order.",
        )
        input_group.add_argument(
            "-l", "--filelist",
            type=Path,
            help="Path to a text file containing video paths in order (one per line).",
        )
        input_group.add_argument(
            "positional_inputs",
            nargs="*",
            help="Optional positional video file paths or directory path.",
        )
        input_group.add_argument(
            "--order",
            help="Reorder inputs using comma-separated 1-based indices, e.g. '3,1,2'.",
        )

        output_group = parser.add_argument_group("Output Options")
        output_group.add_argument(
            "-o", "--output",
            type=Path,
            default=None,
            help="Path for the output joined video (defaults to matching input format, e.g. <folder>_joined.avi or joined_output.mp4).",
        )
        output_group.add_argument(
            "-y", "--overwrite",
            action="store_true",
            default=True,
            help="Overwrite output file if it already exists (default: True).",
        )

        codec_group = parser.add_argument_group("Joining & Encoding Settings")
        codec_group.add_argument(
            "-m", "--mode",
            choices=["auto", "fast", "normalize"],
            default="auto",
            help="Joining mode: 'auto' (choose fastest safe mode), 'fast' (stream copy without re-encoding), 'normalize' (standardize canvas & re-encode).",
        )
        codec_group.add_argument(
            "-r", "--resolution",
            help="Target resolution (e.g. 1920x1080, 1280x720, 1080p, 720p, 4k). Defaults to first video's resolution.",
        )
        codec_group.add_argument(
            "--fps",
            type=float,
            help="Target framerate (e.g. 30, 60). Defaults to first video's framerate.",
        )
        codec_group.add_argument(
            "--crf",
            type=int,
            default=20,
            help="Constant Rate Factor for video quality when normalizing (0-51, lower is better, default 20).",
        )
        codec_group.add_argument(
            "--preset",
            default="medium",
            choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
            help="x264 encoding preset speed/efficiency (default: medium).",
        )
        codec_group.add_argument(
            "-g", "--gpu",
            choices=["auto", "nvenc", "qsv", "amf", "cpu"],
            default="auto",
            help="Hardware GPU acceleration for video re-encoding (default: auto). Uses NVIDIA NVENC if an NVIDIA GPU is available.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        # Determine if folder mode is being used
        target_folder: Optional[Path] = None
        if args.folder:
            target_folder = args.folder
        elif args.inputs and len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
            target_folder = Path(args.inputs[0])
        elif args.positional_inputs and len(args.positional_inputs) == 1 and Path(args.positional_inputs[0]).is_dir():
            target_folder = Path(args.positional_inputs[0])

        raw_inputs: List[Path] = []
        if target_folder:
            target_folder = target_folder.resolve()
            raw_inputs = discover_folder_videos(target_folder, exclude_output=args.output)
            if not raw_inputs:
                console.print(f"[bold red]Error:[/bold red] No supported video files found in folder: '{target_folder}'.")
                return 1
            console.print(f"[bold green]Discovered {len(raw_inputs)} videos in folder:[/bold green] {target_folder}")
        else:
            if args.inputs:
                raw_inputs.extend([Path(p) for p in args.inputs])
            if args.positional_inputs:
                raw_inputs.extend([Path(p) for p in args.positional_inputs])
            if args.filelist:
                raw_inputs.extend(parse_filelist(args.filelist))
            
            seen = set()
            unique_inputs = []
            for p in raw_inputs:
                resolved = p.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    unique_inputs.append(p)
            raw_inputs = unique_inputs


        if not raw_inputs:
            console.print("[bold red]Error:[/bold red] Please provide input videos using -i/--inputs, -d/--folder, or -l/--filelist.")
            return 1

        # Handle custom ordering if specified
        if args.order:
            try:
                indices = [int(x.strip()) - 1 for x in args.order.split(",")]
                if sorted(indices) != list(range(len(raw_inputs))):
                    console.print(
                        f"[bold red]Error:[/bold red] --order must contain each index from 1 to {len(raw_inputs)} exactly once."
                    )
                    return 1
                ordered_inputs = [raw_inputs[idx] for idx in indices]
            except Exception as e:
                console.print(f"[bold red]Error parsing --order:[/bold red] {e}")
                return 1
        else:
            ordered_inputs = raw_inputs

        # Parse target resolution
        try:
            target_res = parse_resolution(args.resolution)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return 1

        output_path = args.output
        input_ext = ordered_inputs[0].suffix if ordered_inputs[0].suffix else ".mp4"

        if target_folder:
            if output_path:
                if not output_path.is_absolute():
                    output_path = (target_folder / output_path).resolve()
            else:
                # Default output in the same folder with folder name
                folder_clean_name = target_folder.name.strip()
                output_path = (target_folder / f"{folder_clean_name}_joined{input_ext}").resolve()
        else:
            if not output_path:
                output_path = Path(f"joined_output{input_ext}")

        config = JoinConfig(
            inputs=ordered_inputs,
            output=output_path,
            mode=JoinMode(args.mode),
            target_resolution=target_res,
            target_fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            gpu=args.gpu,
            overwrite=args.overwrite,
        )

        joiner = VideoJoiner(config)

        # Inspect and print input table
        console.print("\n[bold cyan]VideoCore Joiner[/bold cyan] - Input Sequence:")
        table = Table(title="Videos to Join (in order)", header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Filename", style="bold white")
        table.add_column("Resolution", style="cyan")
        table.add_column("FPS", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Codec", style="blue")
        table.add_column("Audio", style="magenta")

        metas = []
        total_duration = 0.0

        for idx, path in enumerate(ordered_inputs, start=1):
            meta = probe_video(path)
            metas.append(meta)
            total_duration += meta.duration
            pv = meta.primary_video
            pa = meta.primary_audio
            res_str = f"{pv.width}x{pv.height}" if pv else "N/A"
            fps_str = f"{pv.fps:.2f}" if pv else "N/A"
            codec_str = pv.codec_name if pv else "N/A"
            audio_str = f"{pa.codec_name} ({pa.sample_rate}Hz)" if pa else "None"
            dur_str = f"{meta.duration:.1f}s"
            table.add_row(str(idx), path.name, res_str, fps_str, dur_str, codec_str, audio_str)

        console.print(table)
        console.print(f"Total Combined Duration: [bold yellow]{total_duration:.1f}s[/bold yellow]")

        mode_to_use, reason = joiner.determine_mode(metas)
        mode_badge = (
            "[bold black on green] FAST STREAM COPY [/bold black on green]"
            if mode_to_use == JoinMode.FAST
            else "[bold white on blue] NORMALIZED RE-ENCODE [/bold white on blue]"
        )
        console.print(f"\nExecution Mode: {mode_badge} ({reason})\n")

        # Progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Joining {len(ordered_inputs)} videos...", total=100)

            def progress_callback(fraction: float, speed: str, fps: str, time_str: str) -> None:
                pct = fraction * 100
                desc = f"Joining videos... [dim](time: {time_str}, fps: {fps}, speed: {speed})[/dim]"
                progress.update(task, completed=pct, description=desc)

            result = joiner.join(on_progress=progress_callback)

        output_size_mb = result.output_size_bytes / (1024 * 1024)
        console.print(
            Panel.fit(
                f"[bold green]Successfully joined {result.inputs_count} videos![/bold green]\n\n"
                f"[bold]Output File:[/bold] [underline]{result.output_path.resolve()}[/underline]\n"
                f"[bold]Output Size:[/bold] {output_size_mb:.2f} MB\n"
                f"[bold]Duration:[/bold] {result.total_duration:.1f} seconds\n"
                f"[bold]Mode Used:[/bold] {result.mode_used.value.upper()}\n"
                f"[bold]Encoder:[/bold] {result.encoder_used}",
                title="Join Complete",
                border_style="green",
            )
        )
        return 0
