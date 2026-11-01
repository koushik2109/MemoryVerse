# R-03 Deliverable: FUNSD EasyOCR Optical Detection Failure Audit

This deliverable analyzes the near-random retrieval performance (MRR 0.02 - 0.05) observed under EasyOCR 1.7.2 on blurred FUNSD administrative forms.

### Empirical Audit Findings
1. **Extraction Failure vs IR Failure:** On pristine forms, EasyOCR extracts an average of 779.7 characters per form (min 172). Under optical blur degradation (Gaussian blur radius 1.5, 50% scale, rotation, JPEG Q20), EasyOCR extracts an average of only 3.9 characters per form.
2. **Empty Index Rate:** 135 out of 199 forms (67.8%) returned exactly 0 characters. 189 out of 199 forms (95.0%) returned under 20 characters.
3. **Detection Mechanism:** EasyOCR relies on CRAFT (Character Region Awareness for Text Detection) for bounding-box proposals. When small administrative form fonts undergo Gaussian blur and downscaling, CRAFT heatmap confidence scores fail to reach the region threshold, yielding zero text bounding boxes.
4. **Comparison with Tesseract:** Tesseract 5.5 LSTM uses line-finding heuristics rather than CRAFT heatmaps, extracting noisy text fragments (mean 342.2 chars) without collapsing to zero, enabling partial character n-gram matching (MRR 0.4725).
5. **Scientific Reporting Recommendation:** In Phase 4, explicitly report EasyOCR's MRR 0.02 - 0.05 as an **optical text detection threshold collapse** resulting in empty text indices, rather than presenting it as an intrinsic IR ranking robustness finding.
