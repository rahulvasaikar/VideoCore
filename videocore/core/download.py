"""Shared file downloading utility."""

import shutil
import subprocess
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TimeRemainingColumn

console = Console()

def download_file(url: str, dest_path: Path, description: str = "Downloading") -> Path:
    """Downloads a file with rich progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    # Try curl first for high speed and resume support
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-L", url, "-o", str(temp_path)]
        with console.status(f"[cyan]{description}...[/cyan]"):
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return dest_path

    # Fallback to urllib with progress bar
    import urllib.request
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=None)

        def hook(count, block_size, total_size):
            if total_size > 0:
                progress.update(task_id, total=total_size, completed=count * block_size)

        urllib.request.urlretrieve(url, temp_path, reporthook=hook)

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
    return dest_path
