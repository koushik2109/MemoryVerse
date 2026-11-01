import fitz, re
from pathlib import Path

p = Path("paper_v3/ijcds_body.tex")
text = p.read_text(encoding="utf-8")

# Extract bibitems
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

# Map current keys to numbers
bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", bib_text)

# Read PDF to get visual appearance order
doc = fitz.open("paper_v3/main.pdf")
seen_nums = []
for page_idx in range(14):
    blocks = doc[page_idx].get_text("blocks")
    for b in blocks:
        txt = b[4]
        matches = re.findall(r"\[([0-9,\s–-]+)\]", txt)
        for match in matches:
            tokens = re.split(r"[,–-]", match)
            for token in tokens:
                token = token.strip()
                if token.isdigit():
                    num = int(token)
                    if 1 <= num <= len(bib_keys) and num not in seen_nums:
                        seen_nums.append(num)

keys_in_visual_order = [bib_keys[num - 1] for num in seen_nums]
print("Visual order count:", len(keys_in_visual_order))

# Check any missing keys
for k in bib_keys:
    if k not in keys_in_visual_order:
        keys_in_visual_order.append(k)

# Reassemble
ordered_bib_parts = [bib_entries[k] for k in keys_in_visual_order if k in bib_entries]
new_bib_text = "\n\\footnotesize\n" + "\n".join(ordered_bib_parts) + "\n"
new_text = text[:m.start(2)] + new_bib_text + text[m.end(2):]

p.write_text(new_text, encoding="utf-8")
print("Reassembled bibliography in visual reading order!")

