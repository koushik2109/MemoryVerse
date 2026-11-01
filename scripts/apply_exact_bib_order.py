import re
from pathlib import Path

p = Path("paper_v3/ijcds_body.tex")
text = p.read_text(encoding="utf-8")

exact_order = [
    "jing2024llmvectordb",
    "lewis2020rag",
    "abootorabi2025multimodal",
    "patel2025multimodal",
    "joshi2024multimodel",
    "faysse2024colpali",
    "kantor2000confusion",
    "mcnamee2004character",
    "apple2020vision",
    "smith2007tesseract",
    "jaidedai2020easyocr",
    "faysse2024colqwen2",
    "mathew2021docvqa",
    "karpathy2015deep",
    "lin2014coco",
    "zhai2023siglip",
    "radford2021clip",
    "girdhar2023imagebind",
    "robertson2009bm25",
    "robertson1995okapi",
    "cho2024m3docrag",
    "wang2024qwen2vl",
    "fateh2024ocr",
    "otsu1979",
    "hough1962",
    "reimers2019sbert",
    "xiao2023bge",
    "chen2024bgem3",
    "formal2021splade",
    "santhanam2022colbertv2",
    "khattab2021baleen",
    "wang2011cascade",
    "gallagher2019cascade",
    "bates2021conformal",
    "angelopoulos2024conformal",
    "angelopoulos2021gentle",
    "li2022blip",
    "li2023blip2",
    "xiao2024florence2",
    "radford2023whisper",
    "wu2023clap",
    "cormack2009rrf",
    "nogueira2019passage",
    "huang2019sroie",
    "jaume2019funsd",
    "openai2024gpt4o",
    "vaessen2024jiwer",
    "zheng2023judging",
    "piczak2015esc50"
]

m = re.search(r"(\\begin\{thebibliography\}\{00\})(.*?)(\\end\{thebibliography\})", text, re.DOTALL)
bib_text = m.group(2)

bib_entries = {}
parts = re.split(r"(?=\\bibitem\{)", bib_text)
for part in parts:
    part = part.strip()
    if not part:
        continue
    bm = re.match(r"\\bibitem\{([^}]+)\}(.*)", part, re.DOTALL)
    if bm:
        key = bm.group(1).strip()
        bib_entries[key] = part

ordered_bib_parts = [bib_entries[k] for k in exact_order if k in bib_entries]
print(f"Ordered {len(ordered_bib_parts)} bibitems")

new_bib_text = "\n\\footnotesize\n" + "\n".join(ordered_bib_parts) + "\n"
new_text = text[:m.start(2)] + new_bib_text + text[m.end(2):]

p.write_text(new_text, encoding="utf-8")
print("Saved ijcds_body.tex with exact order!")
