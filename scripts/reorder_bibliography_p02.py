import re
from pathlib import Path

p = Path("paper_v3/ijcds_body.tex")
text = p.read_text(encoding="utf-8")

# Extract citations in appearance order
cites = re.findall(r"\\cite\{([^}]+)\}", text)
seen = []
for c in cites:
    for item in c.split(","):
        item = item.strip()
        if item not in seen:
            seen.append(item)

print(f"Total unique citations: {len(seen)}")

# Extract thebibliography block
m = re.search(r"(\\begin\{thebibliography\}\{00\})(.*?)(\\end\{thebibliography\})", text, re.DOTALL)
if not m:
    print("Could not find thebibliography block!")
    exit(1)

pre_bib = m.group(1)
bib_text = m.group(2)
post_bib = m.group(3)

# Parse bibitems
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

# Update wang2011cascade (P-01)
bib_entries["wang2011cascade"] = r"""\bibitem{wang2011cascade}
L.~Wang, J.~Lin, and D.~Metzler, ``A cascade ranking model for efficient ranked retrieval,'' in \emph{Proc. 34th Int. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR)}, pp. 105--114, 2011."""

# Update gallagher2019cascade (P-01)
bib_entries["gallagher2019cascade"] = r"""\bibitem{gallagher2019cascade}
L.~Gallagher, R.-C. Chen, R.~Blanco, and J.~S. Culpepper, ``Joint optimization of cascade ranking models,'' in \emph{Proc. 12th ACM Int. Conf. on Web Search and Data Mining (WSDM)}, pp. 15--23, 2019."""

# Add piczak2015esc50 (P-04 / B-17)
bib_entries["piczak2015esc50"] = r"""\bibitem{piczak2015esc50}
K.~J. Piczak, ``ESC: Dataset for environmental sound classification,'' in \emph{Proc. 23rd ACM Int. Conf. on Multimedia (ACM MM)}, pp. 1015--1018, 2015."""

# Reassemble in exact appearance order
ordered_bib_parts = []
for key in seen:
    if key in bib_entries:
        ordered_bib_parts.append(bib_entries[key])
    else:
        print(f"Warning: citation {key} has no bibitem entry!")

new_bib_text = "\n\\footnotesize\n" + "\n".join(ordered_bib_parts) + "\n"
new_text = text[:m.start(2)] + new_bib_text + text[m.end(2):]

p.write_text(new_text, encoding="utf-8")
print(f"Successfully ordered {len(ordered_bib_parts)} bibitems in appearance order!")

