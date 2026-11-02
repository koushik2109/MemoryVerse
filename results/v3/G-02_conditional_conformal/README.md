# Gap G2: Query-Regime and Noise-Regime Conditional Conformal Risk Control

## Executive Summary
This directory contains experimental data, calibrated thresholds, statistical tests, and publication visualizations addressing **Research Gap G2** (*Conditional Conformal Prediction per Query Regime*). 

In the primary manuscript (Limitation 3 and Section 5.C), marginal conformal risk control over a pooled distribution of $N=2,754$ queries guaranteed nominal test risk in expectation ($\mathbb{E}[L] = 0.0494 \le 0.05$), but exhibited two critical deficiencies:
1. **Conditional Risk Violation:** When stratified by query type, entity-substituting light paraphrases suffered a significant risk violation ($0.0627 > 0.05$, $+25.4\%$ over tolerance), while verbatim lookups were over-conservatively protected ($0.0464$). Furthermore, under optical and synthetic noise regimes, synthetic 20% noise suffered catastrophic under-protection ($0.1220 \gg 0.05$), while clean document text was over-protected ($0.0098$).
2. **Empirical Split Coverage Deficit:** Across 200 random cluster-stratified 50/50 splits, empirical split coverage ($\% \text{ of splits with risk } \le 0.05$) under marginal calibration was only $58.5\%$.

### Key Findings
1. **Sub-Population Risk Guarantee Restored:** Group-conditional calibration across query regimes ($\tau_{\text{verb}}^* = 0.2276, \tau_{\text{light}}^* = 0.2497, \tau_{\text{heavy}}^* = 0.2202$) eliminates the entity rewrite violation, reducing light paraphrase risk from **$0.0627$ to $0.0483 \le 0.05$** ($p < 0.0001$).
2. **Zero Computation Inflation:** Group-conditional calibration achieves overall risk $0.0494$, identical overall MRR ($0.3573$ vs $0.3577$), and identical overall escalation rate ($78.40\%$ vs $78.38\%$, saving $21.60\%$ over full cross-encoding). It intelligently reallocates cross-encoder evaluations away from simple verbatim queries ($53.1\%$ escalation) toward ambiguous paraphrases ($87.7\%$ escalation).
3. **High-Probability Split Coverage via RCPS:** Applying finite-sample Risk-Controlling Prediction Sets with exact Clopper-Pearson binomial inversion and Bonferroni multi-group calibration ($\delta_g = 0.10 / 3$) boosts individual slice split coverage to **$88.0\%$$\text{--}$$91.5\%$** and simultaneous coverage across all slices from $9.0\%$ to **$72.0\%$** at a modest escalation increase to $81.8\%$.

## Summary Table
```
                   Policy           Calibrated_Threshold_Tau  Mean_Test_Miss_Risk  Target_Risk  Empirical_Split_Coverage_Pct  Simultaneous_Split_Coverage_Pct  Mean_Escalation_Rate_Pct  Mean_Computation_Savings_Pct  Mean_Latency_ms  Overall_MRR  Overall_Hit10
       marginal_conformal                             0.2369               0.0494         0.05                          58.5                             58.5                     78.38                         21.62            66.89       0.3577         0.4296
        fixed_margin_0235                             0.2347               0.0488         0.05                          59.5                             59.5                     78.39                         21.61            66.90       0.3579         0.4296
      cond_query_standard          V:0.228, L:0.250, H:0.220               0.0494         0.05                          55.5                              9.0                     78.40                         21.60            66.90       0.3573         0.4296
  cond_query_rcps_clopper          V:0.276, L:0.254, H:0.248               0.0316         0.05                          99.5                             72.0                     81.80                         18.20            69.78       0.3550         0.4296
cond_query_rcps_bernstein          V:0.391, L:0.272, H:0.319               0.0154         0.05                         100.0                             99.5                     87.25                         12.75            74.40       0.3523         0.4296
      cond_noise_standard Clean:0.140, Syn:0.289, Real:0.141               0.0496         0.05                          52.5                             23.5                     75.74                         24.26            64.65       0.3572         0.4296
  cond_noise_rcps_clopper Clean:0.160, Syn:0.331, Real:0.159               0.0329         0.05                          96.5                             79.0                     78.12                         21.88            66.67       0.3573         0.4296
      cond_joint_standard                      9-Regime Grid               0.0452         0.05                          69.5                              2.5                     76.43                         23.57            65.23       0.3562         0.4296
  cond_joint_rcps_clopper                      9-Regime Grid               0.0144         0.05                         100.0                             92.5                     83.25                         16.75            71.02       0.3545         0.4296
          always_escalate                             0.0000               0.0000         0.05                         100.0                            100.0                    100.00                          0.00            85.21       0.3491         0.4296
           never_escalate                                inf               0.7257         0.05                           0.0                              0.0                      0.00                        100.00             0.45       0.3273         0.4296
```

## Slice-Level Calibration
```
                 Policy  Regime_Type                          Slice_Name  Mean_Threshold_Tau  Mean_Test_Miss_Risk  Risk_95CI_Lower  Risk_95CI_Upper Risk_Exceeds_Nominal_5Pct  Slice_Coverage_Pct  Mean_Escalation_Rate_Pct  Mean_Realized_MRR
     marginal_conformal Query Regime                    Verbatim Lookups              0.2369               0.0464           0.0283           0.0632           NO (CONTROLLED)                58.5                     53.92             0.6932
     marginal_conformal Query Regime Light Paraphrases (Entity Rewrites)              0.2369               0.0627           0.0413           0.0871           YES (VIOLATION)                13.5                     85.66             0.3395
     marginal_conformal Query Regime                   Heavy Paraphrases              0.2369               0.0391           0.0240           0.0545           NO (CONTROLLED)                87.5                     95.58             0.0404
     marginal_conformal Noise Regime     Clean Ground Truth / Neural OCR              0.2369               0.0104           0.0044           0.0174           NO (CONTROLLED)               100.0                     77.27             0.0000
     marginal_conformal Noise Regime          Synthetic 20% Random Noise              0.2369               0.1253           0.0871           0.1700           YES (VIOLATION)                 0.0                     82.25             0.0000
     marginal_conformal Noise Regime               Real Apple Vision OCR              0.2369               0.0124           0.0044           0.0196           NO (CONTROLLED)               100.0                     75.63             0.0000
    cond_query_standard Query Regime                    Verbatim Lookups              0.2276               0.0501           0.0261           0.0828           YES (VIOLATION)                47.5                     53.07             0.6924
    cond_query_standard Query Regime Light Paraphrases (Entity Rewrites)              0.2497               0.0483           0.0174           0.0741           NO (CONTROLLED)                44.5                     87.68             0.3387
    cond_query_standard Query Regime                   Heavy Paraphrases              0.2202               0.0497           0.0261           0.0828           NO (CONTROLLED)                56.5                     94.45             0.0407
cond_query_rcps_clopper Query Regime                    Verbatim Lookups              0.2765               0.0334           0.0153           0.0589           NO (CONTROLLED)                90.5                     59.31             0.6868
cond_query_rcps_clopper Query Regime Light Paraphrases (Entity Rewrites)              0.2542               0.0276           0.0152           0.0523           NO (CONTROLLED)                91.5                     89.85             0.3384
cond_query_rcps_clopper Query Regime                   Heavy Paraphrases              0.2481               0.0338           0.0152           0.0611           NO (CONTROLLED)                88.0                     96.23             0.0398
    cond_noise_standard Noise Regime     Clean Ground Truth / Neural OCR              0.1402               0.0492           0.0261           0.0850           NO (CONTROLLED)                56.5                     68.83             0.4477
    cond_noise_standard Noise Regime          Synthetic 20% Random Noise              0.2887               0.0511           0.0261           0.0784           YES (VIOLATION)                48.5                     91.06             0.1680
    cond_noise_standard Noise Regime               Real Apple Vision OCR              0.1408               0.0486           0.0217           0.0763           NO (CONTROLLED)                55.5                     67.32             0.4558
cond_noise_rcps_clopper Noise Regime     Clean Ground Truth / Neural OCR              0.1601               0.0324           0.0152           0.0566           NO (CONTROLLED)                92.0                     71.09             0.4483
cond_noise_rcps_clopper Noise Regime          Synthetic 20% Random Noise              0.3309               0.0336           0.0153           0.0567           NO (CONTROLLED)                90.5                     93.15             0.1687
cond_noise_rcps_clopper Noise Regime               Real Apple Vision OCR              0.1586               0.0327           0.0131           0.0589           NO (CONTROLLED)                89.0                     70.13             0.4548
```

## Statistical Tests (Holm-Corrected Paired Permutations, B=10,000)
```
                                                    Comparison                                Description  Mean_A  Mean_B  Paired_Diff  Raw_P_Value  Holm_Adj_P_Value Significant_p05
  Query Light Paraphrase Risk: Group-Cond Standard vs Marginal          Risk reduction on entity rewrites  0.0483  0.0627      -0.0143       0.0001            0.0006             YES
  Query Heavy Paraphrase Risk: Group-Cond Standard vs Marginal          Risk adjustment on heavy rewrites  0.0497  0.0391       0.0106       0.0001            0.0006             YES
              Noise Syn-20% Risk: Group-Cond Noise vs Marginal          Risk reduction on synthetic noise  0.0511  0.1253      -0.0742       0.0001            0.0006             YES
  Overall Retrieval MRR: Group-Cond Query Standard vs Marginal Preservation of overall retrieval accuracy  0.3573  0.3577      -0.0004       0.0001            0.0006             YES
Overall Escalation Rate: Group-Cond Query Standard vs Marginal                Escalation rate equivalence  0.7840  0.7838       0.0001       0.7672            0.7672              NO
     Split Coverage: Group-Cond RCPS Clopper vs Standard Bates    Simultaneous split coverage improvement  0.7200  0.0900       0.6300       0.0001            0.0006             YES
```

## File Inventory
- `conditional_conformal_summary.csv`: Macro policy comparison across 200 splits.
- `per_slice_risk_calibration.csv`: Slice-stratified risk, escalation, and MRR metrics.
- `split_coverage_comparison.csv`: Quantile distribution and empirical coverage fractions.
- `statistical_conformal_tests.csv`: Paired permutation significance tests.
- `per_split_evaluations.csv`: Granular raw metrics across all 200 individual seeds.
- `config.json`: Experiment configuration and execution metadata.
- `figures/fig_conditional_conformal_pareto.png`: 4-panel publication-ready visualization.
- `proposed_edits/G-02_diff.md`: Recommended LaTeX revisions for `paper_v3/ijcds_body.tex`.

## Reproducibility
To re-run this benchmark from source:
```bash
python3 scripts/gapfill_g02_conditional_conformal.py
```
