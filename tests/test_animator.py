"""Tests for VideoCore Animator and Stylizer tool."""

import argparse
from pathlib import Path
import unittest
import numpy as np

from videocore.tools import TOOLS
from videocore.tools.animator.cli import AnimatorTool
from videocore.tools.animator.engine import AnimationConfig, AnimatorEngine
from videocore.tools.animator.filters import (
    apply_cartoon_filter,
    apply_comic_filter,
    apply_de_cartoon_filter,
    apply_watercolor_filter,
    blend_with_original,
)
from videocore.tools.animator.models import ANIMEGAN_MODELS, get_animegan_model_path


class TestAnimatorTool(unittest.TestCase):
    def test_tool_registered(self):
        anim_tools = [t for t in TOOLS if t.name == "animate"]
        self.assertEqual(len(anim_tools), 1)
        tool = anim_tools[0]
        self.assertIsInstance(tool, AnimatorTool)

    def test_cli_arguments(self):
        tool = AnimatorTool()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        tool.register_subcommand(subparsers)

        args = parser.parse_args([
            "animate",
            "-i", "samples/test_face.jpg",
            "-o", "samples/out.mp4",
            "-d", "real_to_anim",
            "-s", "hayao",
            "--strength", "0.8",
            "--line-weight", "1.2",
            "--saturation", "1.3",
        ])
        self.assertEqual(args.command, "animate")
        self.assertEqual(args.input_path, Path("samples/test_face.jpg"))
        self.assertEqual(args.output_path, Path("samples/out.mp4"))
        self.assertEqual(args.direction, "real_to_anim")
        self.assertEqual(args.style, "hayao")
        self.assertAlmostEqual(args.strength, 0.8)
        self.assertAlmostEqual(args.line_weight, 1.2)
        self.assertAlmostEqual(args.color_saturation, 1.3)

    def test_models_metadata(self):
        self.assertIn("hayao", ANIMEGAN_MODELS)
        self.assertIn("shinkai", ANIMEGAN_MODELS)
        self.assertIn("paprika", ANIMEGAN_MODELS)
        # Hayao model should be downloaded and present
        hayao_path = get_animegan_model_path("hayao")
        self.assertIsNotNone(hayao_path)
        self.assertTrue(hayao_path.is_file())

    def test_filters_execution(self):
        # Create a synthetic 128x128 BGR image
        test_img = np.zeros((128, 128, 3), dtype=np.uint8)
        test_img[:, :] = (120, 150, 180)
        # Add a high-contrast circle
        import cv2
        cv2.circle(test_img, (64, 64), 30, (30, 40, 200), -1)

        # 1. Cartoon filter
        cartoon = apply_cartoon_filter(test_img, line_weight=1.0)
        self.assertEqual(cartoon.shape, test_img.shape)

        # 2. Comic filter
        comic = apply_comic_filter(test_img, line_weight=1.0)
        self.assertEqual(comic.shape, test_img.shape)

        # 3. Watercolor filter
        watercolor = apply_watercolor_filter(test_img, line_weight=1.0)
        self.assertEqual(watercolor.shape, test_img.shape)

        # 4. De-cartoon filter (Animation -> Real)
        real = apply_de_cartoon_filter(test_img, edge_soften=0.7, grain_amount=0.15)
        self.assertEqual(real.shape, test_img.shape)

        # 5. Blend helper
        blended = blend_with_original(test_img, cartoon, strength=0.5)
        self.assertEqual(blended.shape, test_img.shape)

    def test_engine_transform_frame(self):
        # Create a synthetic 64x64 BGR frame
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 150

        # Real -> Anim (algorithmic filter)
        cfg_art = AnimationConfig(
            input_path=Path("dummy.png"),
            direction="real_to_anim",
            style="cartoon",
        )
        engine_art = AnimatorEngine(cfg_art)
        out_art = engine_art.transform_frame(frame)
        self.assertEqual(out_art.shape, (64, 64, 3))

        # Anim -> Real (photoreal filter)
        cfg_real = AnimationConfig(
            input_path=Path("dummy.png"),
            direction="anim_to_real",
            film_grain=0.1,
        )
        engine_real = AnimatorEngine(cfg_real)
        out_real = engine_real.transform_frame(frame)
        self.assertEqual(out_real.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
