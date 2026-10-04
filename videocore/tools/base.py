"""Base interfaces for VideoCore tools."""

from abc import ABC, abstractmethod
import argparse


class BaseTool(ABC):
    """Abstract base class for all VideoCore tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The command name used in CLI (e.g. 'join')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A short description of what the tool does."""
        pass

    @abstractmethod
    def register_subcommand(self, subparser: argparse._SubParsersAction) -> None:
        """Register CLI arguments for this tool."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the tool with parsed arguments."""
        pass
