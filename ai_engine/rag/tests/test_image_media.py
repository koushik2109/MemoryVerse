import hashlib
import json
import unittest
import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ai_engine.rag.image_media import (
    SUPPORTED_IMAGE_EXTENSIONS,
    build_image_media_payload,
    load_image_bgr,
    resolve_media_path,
)
from ai_engine.rag.scripts.prepare_image_audit_v2 import (
    _build_multimodal_messages,
    _generate_answers,
    _media_payload,
)


class ImageMediaTest(unittest.TestCase):
    def test_missing_media_returns_none_and_loader_raises_clearly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "PhotoBench" / "images" / "missing.jpg"

            self.assertIsNone(resolve_media_path("PhotoBench/images/missing.jpg", root))
            with self.assertRaisesRegex(FileNotFoundError, "Image file not found"):
                load_image_bgr(missing)

    def test_absolute_and_relative_paths_resolve_under_rag_and_repo_roots(self):
        with TemporaryDirectory() as directory:
            repo_root = Path(directory) / "repo"
            rag_root = repo_root / "ai_engine" / "rag"
            image_path = rag_root / "PhotoBench" / "images" / "example.JPG"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"temporary image bytes")

            self.assertEqual(
                resolve_media_path(
                    "PhotoBench/images/example.JPG", repo_root, rag_root
                ),
                image_path.resolve(),
            )
            self.assertEqual(
                resolve_media_path(str(image_path), repo_root, rag_root),
                image_path.resolve(),
            )

    def test_unsupported_extension_is_rejected_even_when_file_exists(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            unsupported = root / "PhotoBench" / "images" / "not-an-image.txt"
            unsupported.parent.mkdir(parents=True)
            unsupported.write_bytes(b"not an image")

            self.assertIsNone(resolve_media_path(str(unsupported), root))
            self.assertIn(".jpg", SUPPORTED_IMAGE_EXTENSIONS)
            self.assertNotIn(".txt", SUPPORTED_IMAGE_EXTENSIONS)

    def test_payload_identity_is_stable_and_content_addressed(self):
        media_bytes = b"stable raw image bytes"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.jpg"
            second_path = root / "copy.jpg"
            first_path.write_bytes(media_bytes)
            second_path.write_bytes(media_bytes)
            record = {
                "item_id": "PhotoBench:example.jpg",
                "modality": "image",
                "file_path": "PhotoBench/images/example.jpg",
            }

            first = build_image_media_payload(record, first_path)
            second = build_image_media_payload(
                {**record, "file_path": "another/location.jpg"}, second_path
            )

            expected_digest = hashlib.sha256(media_bytes).hexdigest()
            self.assertEqual(first["media_id"], expected_digest)
            self.assertEqual(second["media_id"], expected_digest)
            self.assertTrue(first["raw_media_available"])
            self.assertEqual(first["item_id"], record["item_id"])
            self.assertEqual(first["modality"], "image")
            self.assertEqual(first["file_path"], record["file_path"])
            self.assertEqual(first["resolved_media_path"], str(first_path.resolve()))
            json.dumps(first)

    def test_audit_media_payload_is_explicit_when_raw_file_is_missing(self):
        with TemporaryDirectory() as directory:
            record = {
                "item_id": "IMG_8266",
                "modality": "image",
                "file_path": "PhotoBench/images/IMG_8266.jpg",
            }
            payload = _media_payload(record, Path(directory))

        self.assertFalse(payload["raw_media_available"])
        self.assertIsNone(payload["resolved_media_path"])
        self.assertEqual(payload["file_path"], record["file_path"])

    def test_available_audit_media_is_attached_as_a_vision_input(self):
        with TemporaryDirectory() as directory:
            media_root = Path(directory)
            image_path = media_root / "PhotoBench" / "images" / "IMG_8266.jpg"
            image_path.parent.mkdir(parents=True)
            image_bytes = b"synthetic image payload"
            image_path.write_bytes(image_bytes)
            record = {
                "item_id": "IMG_8266",
                "modality": "image",
                "file_path": "PhotoBench/images/IMG_8266.jpg",
            }

            payload = _media_payload(record, media_root)
            messages, attached = _build_multimodal_messages("Answer from evidence.", [payload])

        self.assertEqual(attached, 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("Answer from evidence.", content[0]["text"])
        self.assertIn("actual image pixels", content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        encoded_url = content[1]["image_url"]["url"]
        encoded = encoded_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), image_bytes)
        self.assertEqual(payload["resolved_media_path"], str(image_path.resolve()))
        self.assertEqual(payload["item_id"], record["item_id"])
        self.assertTrue(payload["raw_media_available"])

    def test_audit_media_resolves_documented_raw_album_layout(self):
        with TemporaryDirectory() as directory:
            media_root = Path(directory)
            image_path = media_root / "album2" / "IMG_8266.JPG"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"album-layout image")
            record = {
                "item_id": "IMG_8266",
                "modality": "image",
                "file_path": "PhotoBench/images/IMG_8266.jpg",
                "metadata": {"album": "album2"},
            }

            payload = _media_payload(record, media_root)
            resolved_bytes = Path(payload["resolved_media_path"]).read_bytes()

        self.assertTrue(payload["raw_media_available"])
        self.assertEqual(resolved_bytes, b"album-layout image")

    def test_generation_sends_retrieved_media_to_the_vision_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.jsonl"
            media_path = root / "PhotoBench" / "images" / "IMG_8266.jpg"
            media_path.parent.mkdir(parents=True)
            media_path.write_bytes(b"synthetic image bytes")
            record = {
                "item_id": "IMG_8266",
                "modality": "image",
                "file_path": "PhotoBench/images/IMG_8266.jpg",
                "text_representation": (
                    "marathon volunteers handing water December 17, 2023 photos"
                ),
            }
            corpus_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            questions = [
                {
                    "id": f"Q-{index:02d}",
                    "query": "What date is associated with the image?",
                    "target_keyword": "December 17, 2023",
                }
                for index in range(25)
            ]
            question_set = {
                "questions": questions,
                "scope_counts": {"question_count": 25},
            }
            calls = []

            def create_completion(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="December 17, 2023")
                        )
                    ]
                )

            client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=create_completion
                    )
                )
            )
            runtime = {
                "CANDIDATE_DEPTH": 30,
                "RRF_DEPTH": 100,
                "RRF_K": 60,
                "RERANK_DENSE_WEIGHT": 0.65,
                "build_answer_prompt": lambda query, contexts: f"{query} {contexts[0]}",
                "get_openai_client": lambda: client,
                "rank_candidates": lambda *args, **kwargs: [],
                "tokenize": lambda text: text.split(),
            }

            with (
                patch(
                    "ai_engine.rag.scripts.prepare_image_audit_v2._load_runtime_helpers",
                    return_value=runtime,
                ),
                patch(
                    "ai_engine.rag.scripts.prepare_image_audit_v2._build_or_get_image_index",
                    return_value={"placeholder": True},
                ),
                patch(
                    "ai_engine.rag.scripts.prepare_image_audit_v2._retrieve_documents",
                    return_value=[record],
                ),
            ):
                packet = _generate_answers(
                    question_set,
                    "question-set-hash",
                    corpus_path,
                    root / "output",
                    root,
                )

        example = packet["examples"][0]
        self.assertEqual(example["media_attached_count"], 1)
        self.assertTrue(example["retrieved_media"][0]["raw_media_available"])
        self.assertEqual(packet["generation"]["raw_media_attached"], 25)
        first_content = calls[0]["messages"][0]["content"]
        self.assertEqual(first_content[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main(verbosity=2)
