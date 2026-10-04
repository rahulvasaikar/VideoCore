"""Tests for VideoCore probe and joiner tools."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from videocore.core.ffmpeg import get_ffmpeg_path
from videocore.core.probe import probe_video, are_streams_compatible
from videocore.tools.joiner.engine import JoinConfig, JoinMode, VideoJoiner


class TestVideoJoiner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = get_ffmpeg_path()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Generate sample clip 1: 1280x720 30fps with test sine audio, 2 seconds
        cls.clip1 = cls.temp_path / "clip1.mp4"
        subprocess.run([
            cls.ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(cls.clip1),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Generate sample clip 2: 1280x720 30fps with test sine audio, 2 seconds (identical params to clip1)
        cls.clip2 = cls.temp_path / "clip2.mp4"
        subprocess.run([
            cls.ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(cls.clip2),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Generate sample clip 3: 640x360 24fps NO AUDIO, 1.5 seconds (disparate resolution/fps/no-audio)
        cls.clip3_mismatch = cls.temp_path / "clip3_mismatch.mp4"
        subprocess.run([
            cls.ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1.5:size=640x360:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(cls.clip3_mismatch),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_probe_clip(self):
        meta = probe_video(self.clip1)
        self.assertAlmostEqual(meta.duration, 2.0, delta=0.2)
        self.assertIsNotNone(meta.primary_video)
        self.assertEqual(meta.primary_video.width, 1280)
        self.assertEqual(meta.primary_video.height, 720)
        self.assertEqual(meta.primary_video.fps, 30.0)
        self.assertTrue(meta.has_audio)

    def test_stream_compatibility(self):
        m1 = probe_video(self.clip1)
        m2 = probe_video(self.clip2)
        m3 = probe_video(self.clip3_mismatch)

        compat_1_2, reason_1_2 = are_streams_compatible([m1, m2])
        self.assertTrue(compat_1_2, f"Expected clip1 and clip2 to be compatible: {reason_1_2}")

        compat_1_3, reason_1_3 = are_streams_compatible([m1, m3])
        self.assertFalse(compat_1_3, "Expected clip1 and clip3 to be incompatible")

    def test_fast_join_identical_clips(self):
        out_path = self.temp_path / "fast_joined.mp4"
        config = JoinConfig(
            inputs=[self.clip1, self.clip2],
            output=out_path,
            mode=JoinMode.FAST,
            overwrite=True,
        )
        joiner = VideoJoiner(config)
        result = joiner.join()

        self.assertTrue(out_path.exists())
        self.assertEqual(result.mode_used, JoinMode.FAST)
        self.assertEqual(result.inputs_count, 2)
        
        meta = probe_video(out_path)
        self.assertAlmostEqual(meta.duration, 4.0, delta=0.3)
        self.assertEqual(meta.primary_video.width, 1280)
        self.assertEqual(meta.primary_video.height, 720)

    def test_normalize_join_mismatched_clips(self):
        out_path = self.temp_path / "normalize_joined.mp4"
        config = JoinConfig(
            inputs=[self.clip1, self.clip3_mismatch],
            output=out_path,
            mode=JoinMode.AUTO,
            overwrite=True,
        )
        joiner = VideoJoiner(config)
        result = joiner.join()

        self.assertTrue(out_path.exists())
        self.assertEqual(result.mode_used, JoinMode.NORMALIZE)
        
        meta = probe_video(out_path)
        # 2s + 1.5s = 3.5s
        self.assertAlmostEqual(meta.duration, 3.5, delta=0.4)
        # Scaled to first video (1280x720)
        self.assertEqual(meta.primary_video.width, 1280)
        self.assertEqual(meta.primary_video.height, 720)
        # Both output video and audio should be present and valid
        self.assertTrue(meta.has_audio)

    def test_join_order(self):
        # Join in order [clip2, clip1]
        out_path = self.temp_path / "order_2_1.mp4"
        config = JoinConfig(
            inputs=[self.clip2, self.clip1],
            output=out_path,
            mode=JoinMode.AUTO,
            overwrite=True,
        )
        joiner = VideoJoiner(config)
        result = joiner.join()
        self.assertTrue(out_path.exists())
        self.assertEqual(result.inputs_count, 2)

    def test_discover_folder_and_natural_sort(self):
        from videocore.tools.joiner.cli import discover_folder_videos
        import shutil

        test_dir = self.temp_path / "subfolder"
        test_dir.mkdir()
        # Create clips with numbers: clip1, clip2, clip10
        shutil.copy(self.clip1, test_dir / "clip1.mp4")
        shutil.copy(self.clip2, test_dir / "clip10.mp4")
        shutil.copy(self.clip1, test_dir / "clip2.mp4")
        # And a dummy non-video file
        (test_dir / "notes.txt").write_text("hello")

        videos = discover_folder_videos(test_dir)
        names = [v.name for v in videos]
        # Verify natural order: clip1, clip2, clip10 (not clip1, clip10, clip2)
        self.assertEqual(names, ["clip1.mp4", "clip2.mp4", "clip10.mp4"])


if __name__ == "__main__":
    unittest.main()

