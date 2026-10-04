"""Tests for VideoCore UI components and tool registration."""

import argparse
from pathlib import Path
import unittest

from videocore.tools import TOOLS
from videocore.tools.ui import UITool


class TestUI(unittest.TestCase):
    def test_all_tools_registered(self):
        tool_names = [t.name for t in TOOLS]
        self.assertIn("join", tool_names)
        self.assertIn("enhance", tool_names)
        self.assertIn("animate", tool_names)
        self.assertIn("faces", tool_names)
        self.assertIn("ui", tool_names)

    def test_ui_tool_registered(self):
        ui_tools = [t for t in TOOLS if t.name == "ui"]
        self.assertEqual(len(ui_tools), 1)
        tool = ui_tools[0]
        self.assertIsInstance(tool, UITool)

    def test_ui_tool_arguments(self):
        tool = UITool()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        tool.register_subcommand(subparsers)

        args = parser.parse_args(["ui", "--port", "9000", "--no-browser"])
        self.assertEqual(args.command, "ui")
        self.assertEqual(args.port, 9000)
        self.assertTrue(args.no_browser)

    def test_app_script_exists(self):
        app_path = Path(__file__).resolve().parent.parent / "videocore" / "ui" / "app.py"
        self.assertTrue(app_path.is_file())

    def test_dialogs_module(self):
        from videocore.ui import dialogs
        self.assertTrue(hasattr(dialogs, "select_folder"))
        self.assertTrue(hasattr(dialogs, "select_video_files"))
        self.assertTrue(hasattr(dialogs, "select_media_file"))
        self.assertTrue(hasattr(dialogs, "select_save_file"))
        self.assertTrue(hasattr(dialogs, "open_in_file_explorer"))
        self.assertTrue(hasattr(dialogs, "play_video"))


    def test_resolution_presets_mapping(self):
        from videocore.tools.joiner.cli import RESOLUTION_PRESETS
        # Test preset lookups match expected resolutions
        self.assertEqual(RESOLUTION_PRESETS.get("1080p"), (1920, 1080))
        self.assertEqual(RESOLUTION_PRESETS.get("720p"), (1280, 720))
        self.assertEqual(RESOLUTION_PRESETS.get("4k"), (3840, 2160))
        self.assertEqual(RESOLUTION_PRESETS.get("vertical_1080p"), (1080, 1920))
        self.assertEqual(RESOLUTION_PRESETS.get("vertical_720p"), (720, 1280))

    def test_format_extensions(self):
        video_formats = [".mp4", ".mkv", ".mov", ".avi", ".webm"]
        image_formats = [".png", ".jpg", ".webp"]
        test_path = Path("G:/Videos/sample.avi")
        for fmt in video_formats:
            self.assertEqual(test_path.with_suffix(fmt).suffix, fmt)
        for fmt in image_formats:
            self.assertEqual(test_path.with_suffix(fmt).suffix, fmt)


if __name__ == "__main__":
    unittest.main()

