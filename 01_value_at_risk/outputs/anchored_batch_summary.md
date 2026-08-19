# Anchored-NN VaR — batch summary (ranked by pinball; DM vs unanchored; MCS)

Split: TRAIN ≤ 2021-12-31 · VAL ≤ 2023-06-30 · TEST after.

Architectures: SimpleQuantileNeuron.

**Disclosure —** specifications evaluated: **912**, test-set evaluations: **528**.

## H1 / H4 — anchored vs unanchored

A claim of improvement requires BOTH a Diebold-Mariano rejection at 5% and a `detectable` seed-noise verdict (pre-registration amendment 2026-08-16). Both H4 readings are reported; neither is used alone.

| ticker | alpha | anchored | chosen_weight | edge | dm_p_anchored_better | edge_exceeds_anchored_iqr | edge_exceeds_baseline_iqr | seed_noise_verdict | claim_supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ^GSPC | 0.05000 | Anchor param | 10.00000 | -0.00001 | 0.51534 | · | · | not detectable | · |
| ^GSPC | 0.05000 | Anchor hist | 1.00000 | 0.00001 | 0.20751 | ✓ | · | ambiguous | · |
| ^GSPC | 0.01000 | Anchor param | 5.00000 | -0.00002 | 0.67415 | · | · | not detectable | · |
| ^GSPC | 0.01000 | Anchor hist | 1.00000 | -0.00001 | 0.55646 | · | · | not detectable | · |
| BTC-USD | 0.05000 | Anchor param | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| BTC-USD | 0.05000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| BTC-USD | 0.01000 | Anchor param | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| BTC-USD | 0.01000 | Anchor hist | 1.00000 | -0.00002 | 0.93447 | · | · | not detectable | · |
| TSLA | 0.05000 | Anchor param | 5.00000 | -0.00017 | 0.99806 | · | · | not detectable | · |
| TSLA | 0.05000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| TSLA | 0.01000 | Anchor param | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| TSLA | 0.01000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| NVDA | 0.05000 | Anchor param | 1.00000 | -0.00000 | 0.52562 | · | · | not detectable | · |
| NVDA | 0.05000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| NVDA | 0.01000 | Anchor param | 1.00000 | 0.00003 | 0.15579 | ✓ | ✓ | detectable | · |
| NVDA | 0.01000 | Anchor hist | 1.00000 | -0.00003 | 0.80938 | · | · | not detectable | · |
| SQM | 0.05000 | Anchor param | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| SQM | 0.05000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| SQM | 0.01000 | Anchor param | 1.00000 | 0.00008 | 0.00467 | ✓ | ✓ | detectable | ✓ |
| SQM | 0.01000 | Anchor hist | 10.00000 | 0.00007 | 0.00011 | ✓ | ✓ | detectable | ✓ |
| CLP=X | 0.05000 | Anchor param | 1.00000 | 0.00000 | 0.30108 | · | · | not detectable | · |
| CLP=X | 0.05000 | Anchor hist | 1.00000 | 0.00001 | 0.02010 | ✓ | · | ambiguous | · |
| CLP=X | 0.01000 | Anchor param | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| CLP=X | 0.01000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| HG=F | 0.05000 | Anchor param | 1.00000 | -0.00000 | 0.54511 | · | · | not detectable | · |
| HG=F | 0.05000 | Anchor hist | 0.00000 | 0.00000 | 0.50000 | · | · | anchor disabled by VAL | · |
| HG=F | 0.01000 | Anchor param | 1.00000 | 0.00002 | 0.17363 | ✓ | ✓ | detectable | · |
| HG=F | 0.01000 | Anchor hist | 1.00000 | 0.00004 | 0.05630 | ✓ | ✓ | detectable | · |
| CL=F | 0.05000 | Anchor param | 1.00000 | 0.00001 | 0.34597 | ✓ | · | ambiguous | · |
| CL=F | 0.05000 | Anchor hist | 5.00000 | -0.00010 | 0.96206 | · | · | not detectable | · |
| CL=F | 0.01000 | Anchor param | 1.00000 | 0.00001 | 0.44148 | ✓ | · | ambiguous | · |
| CL=F | 0.01000 | Anchor hist | 10.00000 | 0.00006 | 0.22660 | ✓ | ✓ | detectable | · |

**Panel result:** the anchoring claim is supported in **2 of 32** ticker-alpha comparisons.

**Limitation —** VAL selected a grid endpoint in 3 case(s), so the VAL optimum lies outside the pre-registered weight grid. The grid was deliberately NOT widened after seeing results.

### BTC-USD — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00085 | 0.00375 | ✓ | 0.18100 | 0.00649 | 0.21635 | 0.76217 | ✓ |
| Historical | 0.00087 | 0.06126 | ✓ | 0.18100 | 0.01484 | 0.13603 | 0.48726 | ✓ |
| Unanchored | 0.00097 | — | ✓ | 0.18100 | 0.00186 | 0.00096 | 0.93125 | · |
| Anchor param | 0.00097 | 0.50000 | ✓ | 0.18100 | 0.00186 | 0.00096 | 0.93125 | · |
| Parametric-Normal | 0.00098 | 0.56076 | ✓ | 0.18100 | 0.02968 | 0.00000 | 0.95885 | · |
| Anchor hist | 0.00099 | 0.93447 | ✓ | 0.18100 | 0.00186 | 0.00096 | 0.93125 | · |

### BTC-USD — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical | 0.00284 | 0.28190 | ✓ | 0.85000 | 0.04917 | 0.89965 | 0.05465 | ✓ |
| GARCH(1,1)-t | 0.00284 | 0.24992 | ✓ | 0.85000 | 0.05844 | 0.21497 | 0.00306 | · |
| Parametric-Normal | 0.00288 | 0.48523 | ✓ | 0.85000 | 0.05659 | 0.33073 | 0.18356 | ✓ |
| Unanchored | 0.00288 | — | ✓ | 0.85000 | 0.03525 | 0.01923 | 0.75006 | · |
| Anchor hist | 0.00288 | 0.50000 | ✓ | 0.85000 | 0.03525 | 0.01923 | 0.75006 | · |
| Anchor param | 0.00288 | 0.50000 | ✓ | 0.85000 | 0.03525 | 0.01923 | 0.75006 | · |

### CL=F — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00080 | 0.10437 | ✓ | 0.29000 | 0.01203 | 0.58832 | 0.09159 | ✓ |
| Anchor hist | 0.00087 | 0.22660 | ✓ | 0.29000 | 0.00668 | 0.33219 | 0.79518 | ✓ |
| Anchor param | 0.00092 | 0.44148 | ✓ | 0.29000 | 0.02139 | 0.00656 | 0.34717 | · |
| Unanchored | 0.00093 | — | ✓ | 0.29000 | 0.00668 | 0.33219 | 0.79518 | ✓ |
| Parametric-Normal | 0.00100 | 0.75165 | ✓ | 0.29000 | 0.02674 | 0.00014 | 0.55821 | · |
| Historical | 0.00105 | 0.80606 | ✓ | 0.29000 | 0.01604 | 0.12675 | 0.18069 | ✓ |

### CL=F — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00272 | 0.04014 | ✓ | 0.15600 | 0.06016 | 0.21599 | 0.85394 | ✓ |
| Anchor param | 0.00282 | 0.34597 | ✓ | 0.15600 | 0.04144 | 0.26915 | 0.54127 | ✓ |
| Unanchored | 0.00283 | — | ✓ | 0.15600 | 0.04144 | 0.26915 | 0.54127 | ✓ |
| Parametric-Normal | 0.00286 | 0.64407 | ✓ | 0.15600 | 0.05749 | 0.35832 | 0.74110 | ✓ |
| Anchor hist | 0.00294 | 0.96206 | ✓ | 0.15600 | 0.03743 | 0.09949 | 0.10063 | ✓ |
| Historical | 0.00307 | 0.97206 | ✓ | 0.15600 | 0.05615 | 0.44877 | 0.29859 | ✓ |

### CLP=X — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanchored | 0.00030 | — | ✓ | 0.68300 | 0.01429 | 0.26143 | 0.57205 | ✓ |
| Anchor param | 0.00030 | 0.50000 | ✓ | 0.68300 | 0.01429 | 0.26143 | 0.57205 | ✓ |
| Anchor hist | 0.00030 | 0.50000 | ✓ | 0.68300 | 0.01429 | 0.26143 | 0.57205 | ✓ |
| Historical | 0.00030 | 0.53555 | ✓ | 0.68300 | 0.01299 | 0.42580 | 0.60772 | ✓ |
| GARCH(1,1)-t | 0.00030 | 0.54474 | ✓ | 0.68300 | 0.00519 | 0.13995 | 0.83794 | ✓ |
| Parametric-Normal | 0.00032 | 0.76939 | ✓ | 0.68300 | 0.02078 | 0.00865 | 0.40959 | · |

### CLP=X — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor hist | 0.00102 | 0.02010 | ✓ | 0.38800 | 0.04805 | 0.80291 | 0.05306 | ✓ |
| Anchor param | 0.00102 | 0.30108 | ✓ | 0.38800 | 0.04545 | 0.55695 | 0.06765 | ✓ |
| Unanchored | 0.00103 | — | ✓ | 0.38800 | 0.04545 | 0.55695 | 0.06765 | ✓ |
| Historical | 0.00104 | 0.79043 | ✓ | 0.38800 | 0.04545 | 0.55695 | 0.06765 | ✓ |
| Parametric-Normal | 0.00107 | 0.97710 | · | 0.08400 | 0.05455 | 0.56818 | 0.02756 | · |
| GARCH(1,1)-t | 0.00109 | 0.99192 | · | 0.08400 | 0.04935 | 0.93397 | 0.04680 | · |

### HG=F — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor hist | 0.00083 | 0.05630 | ✓ | 0.82700 | 0.01471 | 0.22672 | 0.00795 | · |
| Parametric-Normal | 0.00085 | 0.21422 | ✓ | 0.82700 | 0.01471 | 0.22672 | 0.14744 | ✓ |
| Anchor param | 0.00085 | 0.17363 | ✓ | 0.82700 | 0.01070 | 0.85013 | 0.06900 | ✓ |
| Historical | 0.00087 | 0.39099 | ✓ | 0.82700 | 0.01738 | 0.06632 | 0.01705 | · |
| GARCH(1,1)-t | 0.00087 | 0.40198 | ✓ | 0.82700 | 0.01738 | 0.06632 | 0.01705 | · |
| Unanchored | 0.00088 | — | ✓ | 0.82700 | 0.01203 | 0.58832 | 0.09159 | ✓ |

### HG=F — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanchored | 0.00214 | — | ✓ | 0.71900 | 0.05615 | 0.44877 | 0.82800 | ✓ |
| Anchor hist | 0.00214 | 0.50000 | ✓ | 0.71900 | 0.05615 | 0.44877 | 0.82800 | ✓ |
| Anchor param | 0.00214 | 0.54511 | ✓ | 0.71900 | 0.05348 | 0.66608 | 0.94820 | ✓ |
| GARCH(1,1)-t | 0.00215 | 0.67584 | ✓ | 0.71900 | 0.05882 | 0.28081 | 0.76121 | ✓ |
| Historical | 0.00217 | 0.72096 | ✓ | 0.71900 | 0.06150 | 0.16304 | 0.60578 | ✓ |
| Parametric-Normal | 0.00219 | 0.86718 | ✓ | 0.71900 | 0.06016 | 0.21599 | 0.65845 | ✓ |

### NVDA — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor param | 0.00097 | 0.15579 | ✓ | 0.49000 | 0.01738 | 0.06632 | 0.49738 | ✓ |
| GARCH(1,1)-t | 0.00097 | 0.05217 | ✓ | 0.49000 | 0.00668 | 0.33219 | 0.02175 | · |
| Unanchored | 0.00100 | — | ✓ | 0.49000 | 0.00535 | 0.16057 | 0.83560 | ✓ |
| Parametric-Normal | 0.00101 | 0.64608 | ✓ | 0.49000 | 0.02540 | 0.00039 | 0.31929 | · |
| Anchor hist | 0.00103 | 0.80938 | ✓ | 0.49000 | 0.00802 | 0.57325 | 0.75526 | ✓ |
| Historical | 0.00111 | 0.96306 | · | 0.09400 | 0.01604 | 0.12675 | 0.53133 | ✓ |

### NVDA — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanchored | 0.00326 | — | ✓ | 0.52700 | 0.04947 | 0.94641 | 0.48218 | ✓ |
| Anchor hist | 0.00326 | 0.50000 | ✓ | 0.52700 | 0.04947 | 0.94641 | 0.48218 | ✓ |
| Anchor param | 0.00326 | 0.52562 | ✓ | 0.52700 | 0.05214 | 0.78974 | 0.39967 | ✓ |
| GARCH(1,1)-t | 0.00330 | 0.79369 | ✓ | 0.52700 | 0.05749 | 0.35832 | 0.73100 | ✓ |
| Parametric-Normal | 0.00334 | 0.92464 | ✓ | 0.52700 | 0.07086 | 0.01353 | 0.89511 | · |
| Historical | 0.00341 | 0.93020 | ✓ | 0.52700 | 0.05348 | 0.66608 | 0.91758 | ✓ |

### SQM — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor param | 0.00083 | 0.00467 | ✓ | 0.18200 | 0.01337 | 0.37848 | 0.60241 | ✓ |
| Anchor hist | 0.00084 | 0.00011 | ✓ | 0.18200 | 0.00668 | 0.33219 | 0.79518 | ✓ |
| GARCH(1,1)-t | 0.00085 | 0.04016 | ✓ | 0.18200 | 0.00802 | 0.57325 | 0.75526 | ✓ |
| Parametric-Normal | 0.00088 | 0.27548 | ✓ | 0.18200 | 0.02273 | 0.00271 | 0.37354 | · |
| Historical | 0.00088 | 0.29810 | ✓ | 0.18200 | 0.00802 | 0.57325 | 0.75526 | ✓ |
| Unanchored | 0.00091 | — | ✓ | 0.18200 | 0.00267 | 0.01673 | 0.91747 | · |

### SQM — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanchored | 0.00313 | — | ✓ | 0.14600 | 0.04947 | 0.94641 | 0.03799 | · |
| Anchor param | 0.00313 | 0.50000 | ✓ | 0.14600 | 0.04947 | 0.94641 | 0.03799 | · |
| Anchor hist | 0.00313 | 0.50000 | ✓ | 0.14600 | 0.04947 | 0.94641 | 0.03799 | · |
| Historical | 0.00314 | 0.58025 | ✓ | 0.14600 | 0.05214 | 0.78974 | 0.19329 | ✓ |
| GARCH(1,1)-t | 0.00317 | 0.84956 | ✓ | 0.14600 | 0.05348 | 0.66608 | 0.07291 | ✓ |
| Parametric-Normal | 0.00325 | 0.96456 | ✓ | 0.14600 | 0.05348 | 0.66608 | 0.55661 | ✓ |

### TSLA — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00128 | 0.00046 | ✓ | 0.14400 | 0.00802 | 0.57325 | 0.75526 | ✓ |
| Historical | 0.00130 | 0.08489 | ✓ | 0.14400 | 0.00936 | 0.85848 | 0.71592 | ✓ |
| Unanchored | 0.00137 | — | ✓ | 0.14400 | 0.00936 | 0.85848 | 0.71592 | ✓ |
| Anchor param | 0.00137 | 0.50000 | ✓ | 0.14400 | 0.00936 | 0.85848 | 0.71592 | ✓ |
| Anchor hist | 0.00137 | 0.50000 | ✓ | 0.14400 | 0.00936 | 0.85848 | 0.71592 | ✓ |
| Parametric-Normal | 0.00152 | 0.91158 | ✓ | 0.14400 | 0.02273 | 0.00271 | 0.37354 | · |

### TSLA — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00390 | 0.04850 | ✓ | 0.65100 | 0.04545 | 0.56264 | 0.62381 | ✓ |
| Unanchored | 0.00399 | — | ✓ | 0.65100 | 0.04144 | 0.26915 | 0.78452 | ✓ |
| Anchor hist | 0.00399 | 0.50000 | ✓ | 0.65100 | 0.04144 | 0.26915 | 0.78452 | ✓ |
| Historical | 0.00401 | 0.61066 | ✓ | 0.65100 | 0.04278 | 0.35343 | 0.09051 | ✓ |
| Anchor param | 0.00417 | 0.99806 | · | 0.00600 | 0.05080 | 0.92002 | 0.43970 | ✓ |
| Parametric-Normal | 0.00421 | 0.99853 | · | 0.00600 | 0.05348 | 0.66608 | 0.36211 | ✓ |

### ^GSPC — α = 0.01  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanchored | 0.00037 | — | ✓ | 0.52800 | 0.01070 | 0.85013 | 0.06900 | ✓ |
| Anchor hist | 0.00037 | 0.55646 | ✓ | 0.52800 | 0.00668 | 0.33219 | 0.02175 | · |
| Anchor param | 0.00038 | 0.67415 | ✓ | 0.52800 | 0.02273 | 0.00271 | 0.39634 | · |
| GARCH(1,1)-t | 0.00040 | 0.71198 | ✓ | 0.52800 | 0.01738 | 0.06632 | 0.21742 | ✓ |
| Historical | 0.00040 | 0.71857 | ✓ | 0.52800 | 0.01471 | 0.22672 | 0.00795 | · |
| Parametric-Normal | 0.00041 | 0.85671 | ✓ | 0.52800 | 0.02807 | 0.00005 | 0.61598 | · |

### ^GSPC — α = 0.05  (DM baseline: Unanchored)

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor hist | 0.00108 | 0.20751 | ✓ | 0.79300 | 0.03342 | 0.02724 | 0.25883 | · |
| Unanchored | 0.00110 | — | ✓ | 0.79300 | 0.03743 | 0.09949 | 0.38729 | ✓ |
| Anchor param | 0.00110 | 0.51534 | ✓ | 0.79300 | 0.06016 | 0.21599 | 0.63170 | ✓ |
| Parametric-Normal | 0.00111 | 0.76970 | ✓ | 0.79300 | 0.07353 | 0.00564 | 0.97877 | · |
| GARCH(1,1)-t | 0.00112 | 0.75834 | ✓ | 0.79300 | 0.07353 | 0.00564 | 0.97877 | · |
| Historical | 0.00113 | 0.73916 | ✓ | 0.79300 | 0.04947 | 0.94641 | 0.13897 | ✓ |
