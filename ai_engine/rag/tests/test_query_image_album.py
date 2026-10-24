import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_engine.rag.scripts.query_image_album import (
    MODEL_ID,
    build_annotation_index,
    build_output,
    discover_image_paths,
    matching_annotations,
    serialize_output,
    serialize_results,
)


class QueryImageAlbumTest(unittest.TestCase):
    def test_annotation_index_matches_case_insensitive_ground_truth(self):
        annotations = [
            {
                "query_en": "silver heart-shaped bracelet",
                "ground_truth": ["IMG_3726.JPG", "IMG_3726.JPG"],
                "Source": "V",
            },
            {
                "query_en": "spring flowers blooming",
                "ground_truth": ["nested/IMG_3726.jpg", "IMG_0085.JPG"],
                "Source": "M",
            },
        ]

        index = build_annotation_index(annotations)

        self.assertEqual(
            [entry["query_en"] for entry in matching_annotations("IMG_3726.jpg", index)],
            ["silver heart-shaped bracelet", "spring flowers blooming"],
        )
        self.assertEqual(
            [entry["query_en"] for entry in matching_annotations("IMG_0085.JPG", index)],
            ["spring flowers blooming"],
        )
        self.assertEqual(matching_annotations("missing.jpg", index), [])

    def test_image_discovery_is_recursive_and_deterministic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z" / "nested").mkdir(parents=True)
            (root / "a.JPG").write_bytes(b"placeholder")
            (root / "z" / "nested" / "b.png").write_bytes(b"placeholder")
            (root / "ignore.txt").write_bytes(b"not an image")

            resolved_root = root.resolve()
            first = [
                path.relative_to(resolved_root).as_posix()
                for path in discover_image_paths(root)
            ]
            second = [
                path.relative_to(resolved_root).as_posix()
                for path in discover_image_paths(root)
            ]

        self.assertEqual(first, ["a.JPG", "z/nested/b.png"])
        self.assertEqual(first, second)

    def test_result_serialization_is_stable_and_json_serializable(self):
        results = [
            {
                "filename": "zeta.jpg",
                "score": 0.75,
                "path": Path("/album/zeta.jpg"),
                "annotations": [{"query_en": "zeta", "ground_truth": ["zeta.jpg"]}],
            },
            {
                "filename": "alpha.jpg",
                "score": 0.75,
                "path": "/album/alpha.jpg",
                "annotations": [],
            },
        ]

        canonical = serialize_results(results)
        payload = build_output(
            query="test query",
            album_dir=Path("/album"),
            top_k=5,
            results=canonical,
        )
        first = serialize_output(payload)
        second = serialize_output(payload)
        decoded = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual([result["filename"] for result in decoded["results"]], [
            "alpha.jpg",
            "zeta.jpg",
        ])
        self.assertEqual(decoded["model"], MODEL_ID)
        self.assertTrue(decoded["raw_media_used"])
        self.assertEqual(decoded["top_k"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
