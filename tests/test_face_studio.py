"""Tests for Face Studio tool, analyzer, and face swapper."""

import argparse
from pathlib import Path
import unittest
import numpy as np

from videocore.tools import TOOLS
from videocore.tools.face_studio.cli import FaceStudioTool
from videocore.tools.face_studio.engine import FaceStudioConfig, FaceStudioEngine
from videocore.tools.face_studio.models import get_sface_model, get_yunet_model


class TestFaceStudio(unittest.TestCase):
    def test_tool_registered(self):
        tools = [t for t in TOOLS if t.name == "faces"]
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertIsInstance(tool, FaceStudioTool)

    def test_cli_arguments(self):
        tool = FaceStudioTool()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        tool.register_subcommand(subparsers)

        args = parser.parse_args([
            "faces",
            "-i", "samples/test_face.jpg",
            "-o", "samples/swapped.mp4",
            "--target", "person_1",
            "--new-pic", "samples/test_hayao_engine.png",
            "--interval", "1.0",
        ])
        self.assertEqual(args.command, "faces")
        self.assertEqual(args.input_path, Path("samples/test_face.jpg"))
        self.assertEqual(args.output_path, Path("samples/swapped.mp4"))
        self.assertEqual(args.target_person, "person_1")
        self.assertEqual(args.new_pic, Path("samples/test_hayao_engine.png"))
        self.assertEqual(args.interval, 1.0)

    def test_models_exist(self):
        yunet = get_yunet_model()
        self.assertIsNotNone(yunet)
        self.assertTrue(yunet.is_file())

        sface = get_sface_model()
        self.assertIsNotNone(sface)
        self.assertTrue(sface.is_file())

    def test_image_face_analysis(self):
        test_img_path = Path("samples/test_face.jpg")
        if not test_img_path.is_file():
            self.skipTest("Sample test_face.jpg not available")

        cfg = FaceStudioConfig(input_path=test_img_path)
        engine = FaceStudioEngine(cfg)
        faces = engine.analyze_faces()

        self.assertGreaterEqual(len(faces), 1)
        face0 = faces[0]
        self.assertEqual(face0.person_id, "person_1")
        self.assertGreater(face0.thumbnail_bgr.shape[0], 0)
        self.assertEqual(face0.embedding.shape, (1, 128))


if __name__ == "__main__":
    unittest.main()
