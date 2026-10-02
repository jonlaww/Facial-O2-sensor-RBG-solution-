"""Unit sampling checks and FaceMesh integration on real, licensed photographs."""
import hashlib
import json
from pathlib import Path
import unittest

import cv2
import mediapipe as mp
import numpy as np

from roi_smoke_test import rgb_patch_stats, roi_stats

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    bgr = cv2.imread(str(FIXTURES / name))
    if bgr is None:
        raise ValueError(f"Cannot decode {name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PhotographTests(unittest.TestCase):
    def test_fixture_integrity(self):
        for item in json.loads((FIXTURES / "sources.json").read_text()):
            with self.subTest(image=item["name"]):
                self.assertEqual(hashlib.sha256((FIXTURES / item["name"]).read_bytes()).hexdigest(),
                                 item["sha256"])

    def test_medical_patch_channel_means(self):
        # Independent per-channel integer summation on a selected central rectangle.
        # This rectangle contains lesion/skin; it is not a healthy-skin annotation.
        for name in ("ISIC_0000000.jpg", "ISIC_0000001.jpg", "ISIC_0000002.jpg"):
            with self.subTest(image=name):
                rgb = load(name)
                h, w = rgb.shape[:2]
                rect = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                actual_rect, means, count = rgb_patch_stats(rgb, rect)
                x0, y0, x1, y1 = rect
                expected = [int(rgb[y0:y1, x0:x1, i].sum(dtype=np.uint64)) /
                            ((x1 - x0) * (y1 - y0)) for i in range(3)]
                self.assertEqual(actual_rect, rect)
                self.assertEqual(count, (x1 - x0) * (y1 - y0))
                np.testing.assert_allclose(means, expected, atol=1e-12, rtol=0)
                self.assertGreater(abs(means[0] - means[2]), 1)

    def test_medical_patch_channel_permutation(self):
        rgb = load("ISIC_0000000.jpg")
        _, original, _ = rgb_patch_stats(rgb, (40, 40, 200, 200))
        _, swapped, _ = rgb_patch_stats(rgb[..., ::-1], (40, 40, 200, 200))
        np.testing.assert_array_equal(swapped, original[::-1])

    def test_clipped_medical_patch(self):
        rgb = load("ISIC_0000001.jpg")
        rect, means, count = rgb_patch_stats(rgb, (-20, -30, 100, 120))
        self.assertEqual(rect, (0, 0, 100, 120))
        self.assertEqual(count, 12000)
        np.testing.assert_allclose(means, rgb[:120, :100].mean(axis=(0, 1)))

    def test_invalid_patch_rejection(self):
        rgb = load("ISIC_0000002.jpg")
        for rect in ((10, 10, 10, 20), (-40, -40, -1, -1),
                     (0, 0, float("nan"), 20), (20, 20, 10, 10)):
            with self.subTest(rect=rect):
                self.assertIsNone(rgb_patch_stats(rgb, rect))

    def test_no_face_in_dermoscopy(self):
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                           max_num_faces=1, min_detection_confidence=0.5) as mesh:
            for name in ("ISIC_0000000.jpg", "ISIC_0000001.jpg", "ISIC_0000002.jpg"):
                with self.subTest(image=name):
                    self.assertFalse(mesh.process(load(name)).multi_face_landmarks)

    def test_face_photo_variants(self):
        rgb = load("astronaut.png")
        variants = {"original": rgb, "mirror": cv2.flip(rgb, 1),
                    "half_resolution": cv2.resize(rgb, (256, 256)),
                    "darker_75_percent": (rgb.astype(float) * .75).astype(np.uint8)}
        # Variants are transformations of ONE real photograph, not new participants.
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                           max_num_faces=1, min_detection_confidence=0.5) as mesh:
            for name, image in variants.items():
                with self.subTest(variant=name):
                    faces = mesh.process(image).multi_face_landmarks
                    self.assertTrue(faces, "Expected a face in the photograph")
                    patches = roi_stats(image, faces[0].landmark)
                    self.assertEqual(set(patches), {"forehead", "cheek_a", "cheek_b"})
                    for rect, means, count in patches.values():
                        self.assertGreater(count, 0)
                        self.assertTrue(np.isfinite(means).all())
                        x0, y0, x1, y1 = rect
                        self.assertTrue(0 <= x0 < x1 <= image.shape[1])
                        self.assertTrue(0 <= y0 < y1 <= image.shape[0])


if __name__ == "__main__":
    unittest.main()
