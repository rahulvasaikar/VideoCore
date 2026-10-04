"""CLI command to launch the VideoCore Studio interactive web application."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from videocore.tools.base import BaseTool

console = Console()


class UITool(BaseTool):
    """Tool to launch the VideoCore Studio graphical interface."""

    @property
    def name(self) -> str:
        return "ui"

    @property
    def description(self) -> str:
        return "Launch the interactive VideoCore Studio graphical user interface."

    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            self.name,
            aliases=["app", "studio"],
            help=self.description,
            description="Launches the VideoCore Studio web dashboard in your browser.",
        )
        parser.add_argument(
            "-p", "--port",
            type=int,
            default=8501,
            help="Port to run the application server on (default: 8501).",
        )
        parser.add_argument(
            "--no-browser",
            action="store_true",
            help="Do not open the browser automatically upon launch.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        app_path = Path(__file__).resolve().parent.parent / "api" / "server.py"
        if not app_path.is_file():
            console.print(f"[bold red]Error:[/bold red] App entry point not found at: {app_path}")
            return 1

        # Free target port if already occupied by previous instances
        if sys.platform == "win32":
            try:
                ps_cmd = (
                    f'Get-NetTCPConnection -LocalPort {args.port} -ErrorAction SilentlyContinue | '
                    'ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }'
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, timeout=5)
            except Exception:
                pass

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "videocore.api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--reload",
        ]

        console.print("\n[bold cyan]Launching VideoCore Studio...[/bold cyan]")
        console.print(f"Server URL: [bold green]http://localhost:{args.port}[/bold green]\n")

        import webbrowser
        import threading
        import time

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{args.port}")

        if not args.no_browser:
            threading.Thread(target=open_browser, daemon=True).start()

        try:
            return subprocess.call(cmd)
        except KeyboardInterrupt:
            console.print("\n[dim]VideoCore Studio stopped.[/dim]")
            return 0

