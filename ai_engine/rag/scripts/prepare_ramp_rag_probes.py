"""
Prepare Strict Dev/Test Partition and Expanded Scene-Text / Document Probes
===========================================================================
- 50/50 partition by Image/Document ID (Seed 42).
- Zero target leakage between Dev and Test.
- 30 Dev probes targeting Dev items (15 photo scene-text, 15 doc).
- 55 Test probes targeting Test items (30 photo scene-text, 25 doc).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_DATA_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "ocr_extracted_corpus.json"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "ramp_rag_probes_split.json"

RANDOM_SEED = 42


def main():
    with open(OCR_DATA_PATH, encoding="utf-8") as f:
        corpus = json.load(f)["records"]

    text_photos = [r for r in corpus if r.get("has_text") and "photo" in r["id"]]
    non_text_photos = [r for r in corpus if not r.get("has_text") and "photo" in r["id"]]
    docs = [r for r in corpus if "doc" in r["id"]]

    rng = np.random.default_rng(RANDOM_SEED)

    # Deterministic permutation for 50/50 split
    tp_perm = rng.permutation(len(text_photos))
    dev_tp = [text_photos[i] for i in tp_perm[: len(text_photos) // 2]]
    test_tp = [text_photos[i] for i in tp_perm[len(text_photos) // 2 :]]

    ntp_perm = rng.permutation(len(non_text_photos))
    dev_ntp = [non_text_photos[i] for i in ntp_perm[: len(non_text_photos) // 2]]
    test_ntp = [non_text_photos[i] for i in ntp_perm[len(non_text_photos) // 2 :]]

    # 1 doc in Dev, 1 doc in Test
    dev_docs = [docs[0]]   # ocr_doc_IEEE_Paper_MemoryVerse_p1
    test_docs = [docs[1]]  # ocr_doc_Enhancing_Interactive_Querying..._p1

    dev_records = dev_tp + dev_ntp + dev_docs
    test_records = test_tp + test_ntp + test_docs

    dev_ids = {r["id"] for r in dev_records}
    test_ids = {r["id"] for r in test_records}

    assert len(dev_ids & test_ids) == 0, "Leakage detected: IDs overlap between Dev and Test!"
    print(f"Partitioned corpus into Dev ({len(dev_records)} items) and Test ({len(test_records)} items).")

    # ─── 30 DEV PROBES (Strictly targeting Dev items) ─────────────────────────
    dev_probes_data = [
        # 15 Scene-Text Photo Probes from Dev Photos
        ("Baotu Spring cultural stone inscription 泉突约 趵突泉", {"ocr_photo_IMG_1192"}),
        ("Shandong modern history museum anti-invasion 半殖民地化", {"ocr_photo_IMG_1386"}),
        ("where is china on the horizon 天涯何处是神州", {"ocr_photo_IMG_1386"}),
        ("nation should wake up today 国民知醒宜今醒", {"ocr_photo_IMG_1390"}),
        ("Boxer movement anti-imperialist patriotic 义和团 辛亥革命", {"ocr_photo_IMG_1390"}),
        ("rising sun red flag flutters 旭日东升 红旗飘扬 齐鲁大地", {"ocr_photo_IMG_1400"}),
        ("stand-up comedy show SOOGLE COMEDY 松果 脱口秀", {"ocr_photo_IMG_6222"}),
        ("stand-up comedy show talk show 松果 脱口秀", {"ocr_photo_IMG_4146"}),
        ("Yang family barbecue 1995 杨家烤肉", {"ocr_photo_IMG_4540"}),
        ("all living things are grass唯你是青山 众生皆草木", {"ocr_photo_IMG_6022"}),
        ("firmly loved is better than anything 被坚定的爱着", {"ocr_photo_IMG_6023"}),
        ("Japanese thick cut fried steak 日式厚切体統 3F13-07", {"ocr_photo_IMG_7955"}),
        ("Fuying leadership summit 2018福婴 领袖高峰会", {"ocr_photo_2018-04-01 170511"}),
        ("National forest park visitor service 国家森林公园 游客服务点", {"ocr_photo_2018-04-15 073828"}),
        ("Changshou Road police station alarm 22048084 长寿路派出所 江宁路", {"ocr_photo_IMG_20220623_212642"}),

        # 15 Document Probes from Dev Doc (IEEE_Paper_MemoryVerse_p1)
        ("Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Candidate Reranking and Multi-Hop Query Bridging", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("National Institute of Technology Warangal Computer Science", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Anurag Krishna Vennela Sai Balaji Koushik", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Otsu binarization and Hough transform skew correction", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Dual-Signal Candidate Reranking Overlap", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Tesseract OCR engine ICDAR benchmark", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Reciprocal Rank Fusion RRF k=60", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Whisper ASR speech recognition acoustic", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Proxy consistency benchmark HitRate evaluation", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Sentence-BERT Siamese network cosine similarity", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("convex score fusion alpha parameter", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Confidence-Triggered Cascade Ranking cross-encoder ms-marco", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("heterogeneous media retrieval exact lexical cues", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Apple Vision framework OCR pipeline text extraction", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
    ]

    # Verify all Dev targets belong to Dev
    for q, targets in dev_probes_data:
        assert all(t in dev_ids for t in targets), f"Dev probe target not in Dev corpus: {targets}"

    # ─── 55 TEST PROBES (Strictly targeting Test items) ───────────────────────
    test_probes_data = [
        # 30 Scene-Text Photo Probes from Test Photos
        ("fresh probiotic flavored fermented milk 新鮮 益生菌 风味发酵乳", {"ocr_photo_IMG_20220502_163720"}),
        ("room 406 hotel room 严想篡州", {"ocr_photo_IMG_5241"}),
        ("ice pop 5支起售 淡素 图片仅供参考", {"ocr_photo_IMG_20220624_151900"}),
        ("T MIND brand exhibition 04.18 06.29", {"ocr_photo_IMG_7996"}),
        ("XULAITAI chicken cat grass cat treats 全价主食猫条", {"ocr_photo_IMG_5159"}),
        ("ticket booth scenic entrance 售票处 捡票处", {"ocr_photo_IMG_1191"}),
        ("mall emergency exit opening hours 10-22:00 EXI", {"ocr_photo_IMG_7954"}),
        ("flambe glazed porcelain vase 窑变釉瓶 馆藏胡清官金", {"ocr_photo_IMG_1478"}),
        ("poet Yuan Haowen stone inscription 元好问 1190-1257", {"ocr_photo_2019-01-01 164619"}),
        ("thick cut fried yogurt dessert 12.8元 四块体验装 厚切炒酸奶", {"ocr_photo_IMG_20220624_151905"}),
        ("Sino-Japanese war general Zuo Baogui 中日甲午战争 左宝贵", {"ocr_photo_IMG_1388"}),
        ("Weihaiwei Jiaozhou Bay Qingdao concessions 英租威海 德占胶澳 日占青岛", {"ocr_photo_IMG_1388"}),
        ("vintage postal stamp label code T.L.19-9-5", {"ocr_photo_2017-11-05 152200"}),
        ("flambe glaze antique porcelain jar 密变釉瓶 体北産", {"ocr_photo_IMG_1483 3"}),
        ("studio workshop door sign 工作巫", {"ocr_photo_IMG_4022(1)"}),
        ("bakery store signboard BAKESHOP 1OP BO", {"ocr_photo_IMG_1472"}),
        ("Yintai department store mall 银泰 辛好", {"ocr_photo_IMG_9610"}),
        ("ONYENAN museum souvenir gift shop ONYENAN", {"ocr_photo_IMG_1013"}),
        ("chemical reagent batch code 希脯黒磷 18/2128620", {"ocr_photo_IMG_1570 2"}),
        ("PP tiger cartoon mascot logo PP虎", {"ocr_photo_IMG_0555"}),
        ("pet nutrition chicken cat grass 90% meat 全价主食猫条", {"ocr_photo_IMG_5156"}),
        ("XU.AITAS pet nutrition chicken cat grass", {"ocr_photo_IMG_5156"}),
        ("fried yogurt sweet dessert 颜值高 厚订炒酸如 調額間食", {"ocr_photo_IMG_20220730_181336"}),
        ("gourmet wild food snacks 野食準口 钓欲玫跳婴釉花盐", {"ocr_photo_IMG_1019"}),
        ("market fruit stall signboard 产宙瓜 公主州 请每", {"ocr_photo_IMG_9616"}),
        ("summer festival exhibition poster SET9O 枸南 04.18-06.29", {"ocr_photo_IMG_7999"}),
        ("traditional handicraft heritage 延术豆金", {"ocr_photo_IMG_1018"}),
        ("phosphorus chemical container label 希脯黒磷 18/2128620", {"ocr_photo_IMG_1570"}),
        ("La Jeunesse dawn light torch 新青年 黎明曙光 大道新火", {"ocr_photo_IMG_1396"}),
        ("red flag of the future world 试看将来的环球 必是赤旗的世界", {"ocr_photo_IMG_1396"}),

        # 25 Document Probes from Test Doc (Enhancing_Interactive_Querying..._p1)
        ("Enhancing Interactive Querying with a Multimodal RAG System", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Integrating Text Video and Document Analysis via LLaMA3", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Anjali Patel Rajpurohit Shivani Bangalore Institute of Technology", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("N.V.Usha Dr. Shruthiba A BIT Bangalore", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Department of Artificial Intelligence and Machine Learning BIT Bangalore", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("International Conference on Emerging Technologies in Computing and Communication ETCC", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("streamline interaction and information retrieval across diverse data formats", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Retrieval-Augmented Generation RAG system across readable and non-readable sources", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("video frame chunking audio transcription and document analysis via LLaMA3", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Milvus vector database storing ImageBind multimodal embeddings", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("heterogeneous data formats documents videos images and audio", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("video processing frame extraction and visual context capture", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("audio processing Whisper speech-to-text conversion", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("text embedding generation for document chunks via LLaMA3", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("combining visual text and acoustic embeddings in unified vector space", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("cross-modal semantic alignment using ImageBind model", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("similarity search in Milvus vector database using cosine distance", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("LLaMA3 generation augmented with multimodal retrieved context", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("comprehensive query answering across video transcripts and PDF pages", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("evaluation metrics precision recall latency in multimodal RAG", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("audio feature extraction acoustic representations ETCC 2025", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("dense vector retrieval over multi-format knowledge repositories", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("interactive querying system text video audio document integration", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Patel Shivani Usha Shruthiba multimodal RAG framework", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("enterprise scale multimodal information retrieval LLaMA3 architecture", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
    ]

    # Verify all Test targets belong to Test
    for q, targets in test_probes_data:
        assert all(t in test_ids for t in targets), f"Test probe target not in Test corpus: {targets}"

    payload = {
        "random_seed": RANDOM_SEED,
        "dev_corpus_size": len(dev_records),
        "test_corpus_size": len(test_records),
        "dev_record_ids": sorted(list(dev_ids)),
        "test_record_ids": sorted(list(test_ids)),
        "dev_probes": [{"query": q, "ground_truth": list(gt)} for q, gt in dev_probes_data],
        "test_probes": [{"query": q, "ground_truth": list(gt)} for q, gt in test_probes_data],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(dev_probes_data)} Dev probes and {len(test_probes_data)} Test probes.")
    print(f"Saved probe configuration to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
