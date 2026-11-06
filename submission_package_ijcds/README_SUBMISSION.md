# IJCDS Manuscript Submission Package

**Manuscript Title:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Journal Target:** *International Journal of Computing and Digital Systems (IJCDS)*  
**Article Type:** Regular Paper  

---

## Authors & Affiliations
1. **Anurag Krishna Vennela** (Corresponding Author)  
   Department of Computer Science and Engineering  
   National Institute of Technology Warangal, Telangana 506004, India  
   Email: `anuragkrishnavennela@gmail.com`  

2. **G. Sai Balaji Koushik** (Co-Author)  
   Department of Computer Science and Engineering  
   National Institute of Technology Warangal, Telangana 506004, India  
   Email: `gummadavellikoushik2109@gmail.com`  

---

## Package Contents

```text
submission_package_ijcds/
├── Manuscript_MemoryVerse_IJCDS.pdf   # Publication-ready compiled PDF manuscript
├── Cover_Letter_IJCDS.docx            # Official Cover Letter (Word format)
├── Cover_Letter_IJCDS.md              # Official Cover Letter (Markdown / Plaintext)
├── README_SUBMISSION.md               # Package manifest and build documentation
└── source/                            # Self-contained LaTeX compilation package
    ├── main.tex                       # Primary LaTeX driver file
    ├── ijcds_header.tex               # Header configurations, styling, and metadata
    ├── ijcds_body.tex                 # Core manuscript body
    ├── references.bib                 # Complete BibTeX bibliography
    ├── IEEEtran.cls                   # IEEE journal document class
    ├── IEEEtran.bst                   # IEEE bibliography style
    ├── report.cls / report.sty        # Auxiliary formatting styles
    ├── widetext.sty                   # Multi-column balancing style
    ├── ub_logo.png                    # University of Bahrain / IJCDS header logo
    └── figures/                       # High-resolution (300 DPI) publication figures
        ├── fig_architecture.png
        ├── fig_mrr_regime_systems.png
        ├── fig_dose_response_dual_v6.png
        ├── fig_mechanism_composite.png
        ├── fig_conformal_pareto.png
        └── fig_coco_native_vs_proxy.png
```

---

## Verification & Compilation Instructions

The source directory is completely self-contained with zero external dependencies.

### Compilation with Tectonic (Recommended)
```bash
cd source
tectonic main.tex
```

### Compilation with standard TeXLive / PDFLaTeX
```bash
cd source
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Output PDF will be produced as `main.pdf`.

