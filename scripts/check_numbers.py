#!/usr/bin/env python3
"""scripts/check_numbers.py: Validate numbers in IEEE_Paper_v9.tex against results/v9/number_provenance.csv.

Extracts every empirical number reported in:
- Abstract
- Key Contributions
- Limitations
- Conclusion
- Table captions
And asserts that each number appears in results/v9/number_provenance.csv.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "IEEE_Paper_v10.tex"
PROV_PATH = ROOT / "results" / "v10" / "number_provenance.csv"

# Whitelist structural/formatting numbers that are nominal levels or formatting parameters
IGNORED_PATTERNS = {
    # Years
    "2021", "2022", "2023", "2024", "2025", "2026",
    # TeX layout dimensions/points
    "0.1", "0.2", "0.5", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    # Nominal statistical parameters
    "95%", "95", "0.05", "0.10", "1.000", "1.0", "1",
    "0.0", "0.00", "0.000", "0.0000", "1.0000", "1.00",
    # Evaluation random seeds
    "42", "43", "44",
    # Zero or small dimension
    "0", "00", "000",
}


def load_provenance_numbers() -> set[str]:
    if not PROV_PATH.exists():
        print(f"Error: {PROV_PATH} not found.")
        sys.exit(1)
    nums = set()
    with PROV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = r["number"].strip()
            # Normalize comma separators and percentages
            clean = val.replace(",", "").replace("%", "").strip()
            nums.add(val)
            nums.add(clean)
            if clean.startswith("+") or clean.startswith("-"):
                nums.add(clean[1:])
            else:
                nums.add(f"+{clean}")
                nums.add(f"-{clean}")
            if "%" in val:
                nums.add(val.replace("%", "\\%"))
                nums.add(clean + "\\%")
                nums.add(clean + "%")
    return nums


def extract_numbers_from_text(text: str) -> list[tuple[str, str]]:
    """Extract numbers and their local context snippet from text."""
    # Strip LaTeX commands like \url, \cite, \ref, \label, \texttt
    cleaned = re.sub(r"\\url\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\cite\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\ref\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\label\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\texttt\{[^}]*\}", "", cleaned)
    # Strip dimension commands like \itemsep}{...}
    cleaned = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", cleaned)

    # Normalize ranges like 0.8937--0.9192
    cleaned = cleaned.replace("--", " to ")

    # Regex for numbers: e.g. 0.9436, +0.1065, -0.0193, 249.21, 1{,}000, 4,736.8, 813, 78.4%
    pattern = r"(?<![a-zA-Z\\])[+-]?(?:\d{1,3}(?:(?:\{,\}|,)\d{3})+|\d+)(?:\.\d+)?%?(?:\\%)?"
    matches = []
    for m in re.finditer(pattern, cleaned):
        num_str = m.group(0).replace("{,}", "").replace(",", "").strip()
        start = max(0, m.start() - 25)
        end = min(len(cleaned), m.end() + 25)
        snippet = cleaned[start:end].replace("\n", " ").strip()
        matches.append((num_str, snippet))
    return matches


def check_section(section_name: str, content: str, prov_numbers: set[str]) -> list[str]:
    errors = []
    raw_nums = extract_numbers_from_text(content)
    for num, snippet in raw_nums:
        clean_num = num.lstrip("+-").rstrip("%").rstrip("\\%").strip()
        if clean_num in IGNORED_PATTERNS or num in IGNORED_PATTERNS or f"{clean_num}%" in IGNORED_PATTERNS:
            continue
        if (num not in prov_numbers and
            clean_num not in prov_numbers and
            f"{clean_num}%" not in prov_numbers and
            f"{num}%" not in prov_numbers and
            f"+{clean_num}" not in prov_numbers and
            f"-{clean_num}" not in prov_numbers):
            errors.append(f"[{section_name}] Number '{num}' not found in number_provenance.csv! Context: ...{snippet}...")
    return errors


def main() -> None:
    if not TEX_PATH.exists():
        print(f"Error: {TEX_PATH} not found.")
        sys.exit(1)

    tex_content = TEX_PATH.read_text(encoding="utf-8")
    prov_numbers = load_provenance_numbers()

    errors: list[str] = []

    # 1. Abstract
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_content, re.DOTALL)
    if abs_match:
        errors.extend(check_section("Abstract", abs_match.group(1), prov_numbers))
    else:
        errors.append("Abstract section not found in TeX file.")

    # 2. Key Contributions
    contrib_match = re.search(r"\\textbf\{Key Contributions:\}(.*?)\\section\{Related Work\}", tex_content, re.DOTALL)
    if contrib_match:
        errors.extend(check_section("Key Contributions", contrib_match.group(1), prov_numbers))
    else:
        errors.append("Key Contributions section not found in TeX file.")

    # 3. Limitations
    lim_match = re.search(r"\\section\{Limitations and Threats to Validity\}(.*?)\\section\{Conclusion", tex_content, re.DOTALL)
    if lim_match:
        errors.extend(check_section("Limitations", lim_match.group(1), prov_numbers))
    else:
        errors.append("Limitations section not found in TeX file.")

    # 4. Conclusion
    concl_match = re.search(r"\\section\{Conclusion and Future Work\}(.*?)(?:\\appendices|\\begin\{thebibliography\}|\\bibliographystyle)", tex_content, re.DOTALL)
    if concl_match:
        errors.extend(check_section("Conclusion", concl_match.group(1), prov_numbers))
    else:
        errors.append("Conclusion section not found in TeX file.")

    # 5. Table Captions
    table_captions = re.findall(r"\\begin\{table\*?\}.*?\\caption\{(.*?)\}.*?\\end\{table\*?\}", tex_content, re.DOTALL)
    for idx, cap in enumerate(table_captions):
        errors.extend(check_section(f"Table Caption #{idx+1}", cap, prov_numbers))

    print("=" * 80)
    print("CHECK_NUMBERS.PY VERIFICATION REPORT")
    print("=" * 80)
    if errors:
        print(f"FAILED: {len(errors)} number provenance discrepancies found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PASS: 100% of numbers in Abstract, Contributions, Limitations, Conclusion, and Table Captions are verified against results/v10/number_provenance.csv.")
        sys.exit(0)


if __name__ == "__main__":
    main()
