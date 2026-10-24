import hashlib
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_engine.rag.scripts.prepare_album3_audit import (
    DEFAULT_ALBUM_DIR,
    NEGATIVE_CONTROL_ANSWER,
    OutputConflict,
    build_preview,
    build_question_set,
    sha256_file,
    serialize_json,
    write_artifacts,
)


class PrepareAlbum3AuditTest(unittest.TestCase):
    def test_deterministic_ids_and_counts(self):
        with TemporaryDirectory() as output_directory:
            first = build_question_set(DEFAULT_ALBUM_DIR)
            second = build_question_set(DEFAULT_ALBUM_DIR)
            first_preview = build_preview(first)
            second_preview = build_preview(second)

            self.assertEqual(first, second)
            self.assertEqual(
                [question["id"] for question in first["questions"]],
                [f"L10-A3-{index:03d}" for index in range(1, 21)],
            )
            self.assertEqual(first["query_count"], 20)
            self.assertEqual(first["supported_query_count"], 18)
            self.assertEqual(first["negative_control_count"], 2)
            self.assertEqual(first["image_count"], 69)
            self.assertEqual(first_preview, second_preview)
            self.assertEqual(len(first_preview["examples"]), 20)

    def test_real_album1_and_album2_packets_are_data_aligned(self):
        cases = (
            ("album1", "l10-album1-v1", "L10-A1-", 64, 18, 2),
            ("album2", "l10-album2-v1", "L10-A2-", 58, 17, 3),
        )
        available_cases = [
            case for case in cases if (DEFAULT_ALBUM_DIR.parent / case[0]).is_dir()
        ]
        if not available_cases:
            self.skipTest("copied L10 album1/album2 directories are unavailable")

        for (
            album_name,
            packet_version,
            id_prefix,
            image_count,
            supported_count,
            negative_count,
        ) in available_cases:
            with self.subTest(album=album_name):
                question_set = build_question_set(DEFAULT_ALBUM_DIR.parent / album_name)
                self.assertEqual(question_set["packet_version"], packet_version)
                self.assertEqual(question_set["query_count"], 20)
                self.assertEqual(question_set["image_count"], image_count)
                self.assertEqual(question_set["supported_query_count"], supported_count)
                self.assertEqual(question_set["negative_control_count"], negative_count)
                self.assertEqual(
                    [question["id"] for question in question_set["questions"]],
                    [f"{id_prefix}{index:03d}" for index in range(1, 21)],
                )
                preview = build_preview(question_set)
                self.assertEqual(preview["packet_version"], packet_version)
                self.assertEqual(len(preview["examples"]), 20)

    def test_empty_ground_truth_entries_are_explicit_negative_controls(self):
        question_set = build_question_set(DEFAULT_ALBUM_DIR)
        negative_questions = [
            question
            for question in question_set["questions"]
            if question["negative_control"]
        ]

        self.assertEqual([question["id"] for question in negative_questions], [
            "L10-A3-014",
            "L10-A3-017",
        ])
        for question in negative_questions:
            self.assertEqual(question["ground_truth"], [])
            self.assertEqual(question["expected_answer"], NEGATIVE_CONTROL_ANSWER)
            self.assertEqual(question["media"], [])
            self.assertEqual(question["relative_media_paths"], [])
            self.assertFalse(question["raw_media_available"])

        preview = build_preview(question_set)
        rows_by_id = {row["id"]: row for row in preview["examples"]}
        for question in negative_questions:
            row = rows_by_id[question["id"]]
            self.assertIn("DESIGN/REFERENCE DATA", row["reference_context"])
            self.assertIn("not a retrieval result or generated answer", row["reference_context"])
            self.assertEqual(row["answer"], "")
            self.assertIn("no model inference run", row["answer_role"])

    def test_media_path_resolution_is_relative_and_deterministic(self):
        with TemporaryDirectory() as directory:
            album = Path(directory) / "album3"
            image_directory = album / "images" / "nested"
            image_directory.mkdir(parents=True)
            source = json.loads((DEFAULT_ALBUM_DIR / "query.json").read_text(encoding="utf-8"))
            for index, entry in enumerate(source):
                entry["ground_truth"] = []
                entry["ground_truth_count"] = 0
                entry["query_en"] = f"temporary query {index}"
            source[0]["ground_truth"] = ["nested/PHOTO.JPG", "missing.jpg"]
            source[0]["ground_truth_count"] = 2
            source[1]["ground_truth"] = ["nested/photo2.png"]
            source[1]["ground_truth_count"] = 1
            image_directory.joinpath("PHOTO.JPG").write_bytes(b"one")
            (album / "images" / "photo2.png").write_bytes(b"two")
            (album / "images" / "ignore.txt").write_bytes(b"not an image")
            (album / "query.json").write_text(
                json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            question_set = build_question_set(album)
            first = question_set["questions"][0]
            second = question_set["questions"][1]

            self.assertEqual(question_set["image_count"], 2)
            self.assertEqual(first["relative_media_paths"], [
                "images/nested/PHOTO.JPG",
                None,
            ])
            self.assertEqual(first["media"][0]["filename"], "nested/PHOTO.JPG")
            self.assertTrue(first["media"][0]["raw_media_available"])
            self.assertFalse(first["media"][1]["raw_media_available"])
            self.assertEqual(second["relative_media_paths"], ["images/photo2.png"])

    def test_query_hash_changes_when_source_bytes_change(self):
        with TemporaryDirectory() as directory:
            source_album = Path(directory) / "album3"
            source_album.mkdir()
            (source_album / "images").mkdir()
            shutil.copy2(DEFAULT_ALBUM_DIR / "query.json", source_album / "query.json")
            original_hash = sha256_file(source_album / "query.json")
            first = build_question_set(source_album)

            payload = json.loads((source_album / "query.json").read_text(encoding="utf-8"))
            payload[0]["query_en"] = "changed exact query"
            (source_album / "query.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_hash = sha256_file(source_album / "query.json")
            second = build_question_set(source_album)

            self.assertEqual(original_hash, first["query_json_sha256"])
            self.assertNotEqual(original_hash, changed_hash)
            self.assertEqual(changed_hash, second["query_json_sha256"])
            self.assertNotEqual(first["query_json_sha256"], second["query_json_sha256"])
            self.assertNotEqual(
                first["questions"][0]["query_en"], second["questions"][0]["query_en"]
            )

    def test_stable_json_output_and_non_identical_conflict(self):
        with TemporaryDirectory() as directory:
            output_directory = Path(directory) / "audit-output"
            question_set = build_question_set(DEFAULT_ALBUM_DIR)
            preview = build_preview(question_set)
            first_question_set_json = serialize_json(question_set)
            second_question_set_json = serialize_json(question_set)
            self.assertEqual(first_question_set_json, second_question_set_json)
            self.assertEqual(
                hashlib.sha256(first_question_set_json.encode("utf-8")).hexdigest(),
                hashlib.sha256(second_question_set_json.encode("utf-8")).hexdigest(),
            )

            written = write_artifacts(question_set, preview, output_directory)
            self.assertEqual(
                [path.name for path in written],
                ["question_set.json", "question_preview.json"],
            )
            first_bytes = {
                path.name: path.read_bytes()
                for path in written
            }
            write_artifacts(question_set, preview, output_directory)
            self.assertEqual(
                first_bytes,
                {path.name: path.read_bytes() for path in written},
            )

            changed_question_set = dict(question_set)
            changed_question_set["query_count"] = 999
            with self.assertRaises(OutputConflict):
                write_artifacts(changed_question_set, preview, output_directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
