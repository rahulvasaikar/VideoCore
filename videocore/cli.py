"""Main CLI entrypoint for VideoCore."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from videocore import __version__
from videocore.tools import TOOLS


console = Console()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videocore",
        description="VideoCore: High-performance, modular video processing toolkit.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"VideoCore v{__version__}",
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True,
        help="Video tool to execute",
    )

    # Register each tool's CLI subparser
    for tool in TOOLS:
        tool.register_subcommand(subparsers)

    return parser


def main() -> int:
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Find the matching tool
    tool_map = {tool.name: tool for tool in TOOLS}
    tool = tool_map.get(args.command)
    if not tool:
        console.print(f"[bold red]Unknown command:[/bold red] {args.command}")
        return 1

    try:
        return tool.execute(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
