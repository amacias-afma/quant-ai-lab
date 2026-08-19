# Anchored-NN VaR - batch summary (ranked by pinball; DM vs unanchored; MCS)

Split: TRAIN <= 2021-12-31 . VAL <= 2023-06-30 . TEST after.

Architectures: SimpleQuantileNeuron, QuantileMLP.

**Disclosure -** specifications evaluated: **435**, test-set evaluations: **315**.

## H1 / H4 - anchored vs unanchored

A claim of improvement requires BOTH a Diebold-Mariano rejection at 5% and a `detectable` seed-noise verdict (pre-registration amendment 2026-08-16). Both H4 readings are reported; neither is used alone.

| ticker | alpha | anchored | chosen_weight | edge | dm_p_anchored_better | edge_exceeds_anchored_iqr | edge_exceeds_baseline_iqr | seed_noise_verdict | claim_supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ^GSPC | 0.01000 | Anchor param [SimpleQuantileNeuron] | 0.50000 | -0.00000 | 0.37934 | . | . | not detectable | . |
| ^GSPC | 0.01000 | Anchor hist [SimpleQuantileNeuron] | 0.50000 | -0.00001 | 0.55676 | . | . | not detectable | . |
| ^GSPC | 0.01000 | Anchor param [QuantileMLP] | 0.50000 | 0.00007 | 0.13558 | OK | . | ambiguous | . |
| ^GSPC | 0.01000 | Anchor hist [QuantileMLP] | 0.00000 | 0.00000 | 0.50000 | . | . | anchor disabled by VAL | . |
| NVDA | 0.01000 | Anchor param [SimpleQuantileNeuron] | 0.50000 | 0.00004 | 0.03373 | OK | OK | detectable | OK |
| NVDA | 0.01000 | Anchor hist [SimpleQuantileNeuron] | 0.50000 | -0.00002 | 0.73366 | . | . | not detectable | . |
| NVDA | 0.01000 | Anchor param [QuantileMLP] | 0.00000 | 0.00000 | 0.50000 | . | . | anchor disabled by VAL | . |
| NVDA | 0.01000 | Anchor hist [QuantileMLP] | 0.50000 | -0.00000 | 0.56900 | . | . | not detectable | . |
| BTC-USD | 0.01000 | Anchor param [SimpleQuantileNeuron] | 0.00000 | 0.00000 | 0.50000 | . | . | anchor disabled by VAL | . |
| BTC-USD | 0.01000 | Anchor hist [SimpleQuantileNeuron] | 0.50000 | -0.00002 | 0.95512 | . | . | not detectable | . |
| BTC-USD | 0.01000 | Anchor param [QuantileMLP] | 0.00000 | 0.00000 | 0.50000 | . | . | anchor disabled by VAL | . |
| BTC-USD | 0.01000 | Anchor hist [QuantileMLP] | 0.50000 | -0.00007 | 0.92939 | . | . | not detectable | . |
| SQM | 0.01000 | Anchor param [SimpleQuantileNeuron] | 0.50000 | 0.00008 | 0.00041 | OK | OK | detectable | OK |
| SQM | 0.01000 | Anchor hist [SimpleQuantileNeuron] | 0.50000 | 0.00006 | 0.00000 | OK | OK | detectable | OK |
| SQM | 0.01000 | Anchor param [QuantileMLP] | 0.00000 | 0.00000 | 0.50000 | . | . | anchor disabled by VAL | . |
| SQM | 0.01000 | Anchor hist [QuantileMLP] | 0.50000 | 0.00007 | 0.03413 | OK | OK | detectable | OK |
| CL=F | 0.01000 | Anchor param [SimpleQuantileNeuron] | 0.50000 | 0.00002 | 0.32179 | OK | OK | detectable | . |
| CL=F | 0.01000 | Anchor hist [SimpleQuantileNeuron] | 0.50000 | 0.00006 | 0.19613 | OK | OK | detectable | . |
| CL=F | 0.01000 | Anchor param [QuantileMLP] | 0.50000 | 0.00012 | 0.00815 | OK | OK | detectable | OK |
| CL=F | 0.01000 | Anchor hist [QuantileMLP] | 0.50000 | 0.00011 | 0.09702 | OK | OK | detectable | . |

**Panel result:** the anchoring claim is supported in **5 of 20** ticker-alpha comparisons.

**Limitation -** VAL selected a grid endpoint in 15 case(s), so the VAL optimum lies outside the pre-registered weight grid. The grid was deliberately NOT widened after seeing results.

### BTC-USD - alpha = 0.01  (DM baseline: Unanchored [SimpleQuantileNeuron])

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00085 | 0.00375 | OK | 0.11000 | 0.00649 | 0.21635 | 0.76217 | OK |
| Historical | 0.00087 | 0.06126 | OK | 0.11000 | 0.01484 | 0.13603 | 0.48726 | OK |
| Anchor param [QuantileMLP] | 0.00092 | 0.05116 | OK | 0.11000 | 0.00371 | 0.01724 | 0.86289 | . |
| Unanchored [QuantileMLP] | 0.00092 | 0.05116 | OK | 0.11000 | 0.00371 | 0.01724 | 0.86289 | . |
| Anchor param [SimpleQuantileNeuron] | 0.00097 | 0.50000 | OK | 0.11000 | 0.00186 | 0.00096 | 0.93125 | . |
| Unanchored [SimpleQuantileNeuron] | 0.00097 | - | OK | 0.11000 | 0.00186 | 0.00096 | 0.93125 | . |
| Anchor hist [QuantileMLP] | 0.00098 | 0.67087 | OK | 0.11000 | 0.00186 | 0.00096 | 0.93125 | . |
| Parametric-Normal | 0.00098 | 0.56076 | OK | 0.11000 | 0.02968 | 0.00000 | 0.95885 | . |
| Anchor hist [SimpleQuantileNeuron] | 0.00099 | 0.95512 | OK | 0.11000 | 0.00186 | 0.00096 | 0.93125 | . |

### CL=F - alpha = 0.01  (DM baseline: Unanchored [SimpleQuantileNeuron])

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GARCH(1,1)-t | 0.00080 | 0.10437 | OK | 0.43500 | 0.01203 | 0.58832 | 0.09159 | OK |
| Anchor hist [SimpleQuantileNeuron] | 0.00087 | 0.19613 | OK | 0.43500 | 0.00802 | 0.57325 | 0.03420 | . |
| Anchor param [QuantileMLP] | 0.00089 | 0.34750 | OK | 0.43500 | 0.01738 | 0.06632 | 0.01705 | . |
| Anchor param [SimpleQuantileNeuron] | 0.00091 | 0.32179 | OK | 0.43500 | 0.01872 | 0.03256 | 0.25751 | . |
| Anchor hist [QuantileMLP] | 0.00091 | 0.42819 | OK | 0.43500 | 0.00802 | 0.57325 | 0.03420 | . |
| Unanchored [SimpleQuantileNeuron] | 0.00093 | - | OK | 0.43500 | 0.00668 | 0.33219 | 0.79518 | OK |
| Parametric-Normal | 0.00100 | 0.75165 | OK | 0.43500 | 0.02674 | 0.00014 | 0.55821 | . |
| Unanchored [QuantileMLP] | 0.00102 | 0.73315 | OK | 0.43500 | 0.01337 | 0.37848 | 0.11773 | OK |
| Historical | 0.00105 | 0.80606 | OK | 0.43500 | 0.01604 | 0.12675 | 0.18069 | OK |

### NVDA - alpha = 0.01  (DM baseline: Unanchored [SimpleQuantileNeuron])

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor param [SimpleQuantileNeuron] | 0.00095 | 0.03373 | OK | 0.15800 | 0.01203 | 0.58832 | 0.63941 | OK |
| GARCH(1,1)-t | 0.00097 | 0.05217 | OK | 0.15800 | 0.00668 | 0.33219 | 0.02175 | . |
| Unanchored [SimpleQuantileNeuron] | 0.00100 | - | OK | 0.15800 | 0.00535 | 0.16057 | 0.83560 | OK |
| Parametric-Normal | 0.00101 | 0.64608 | OK | 0.15800 | 0.02540 | 0.00039 | 0.31929 | . |
| Anchor hist [SimpleQuantileNeuron] | 0.00102 | 0.73366 | OK | 0.15800 | 0.00668 | 0.33219 | 0.79518 | OK |
| Unanchored [QuantileMLP] | 0.00104 | 0.77850 | OK | 0.15800 | 0.00936 | 0.85848 | 0.71592 | OK |
| Anchor param [QuantileMLP] | 0.00104 | 0.77850 | OK | 0.15800 | 0.00936 | 0.85848 | 0.71592 | OK |
| Anchor hist [QuantileMLP] | 0.00104 | 0.92054 | OK | 0.15800 | 0.00936 | 0.85848 | 0.71592 | OK |
| Historical | 0.00111 | 0.96306 | OK | 0.15800 | 0.01604 | 0.12675 | 0.53133 | OK |

### SQM - alpha = 0.01  (DM baseline: Unanchored [SimpleQuantileNeuron])

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor param [SimpleQuantileNeuron] | 0.00083 | 0.00041 | OK | 0.25500 | 0.01203 | 0.58832 | 0.63941 | OK |
| Anchor hist [SimpleQuantileNeuron] | 0.00085 | 0.00000 | OK | 0.25500 | 0.00401 | 0.06117 | 0.87639 | OK |
| Anchor hist [QuantileMLP] | 0.00085 | 0.01277 | OK | 0.25500 | 0.01070 | 0.85013 | 0.67727 | OK |
| GARCH(1,1)-t | 0.00085 | 0.04016 | OK | 0.25500 | 0.00802 | 0.57325 | 0.75526 | OK |
| Parametric-Normal | 0.00088 | 0.27548 | OK | 0.25500 | 0.02273 | 0.00271 | 0.37354 | . |
| Historical | 0.00088 | 0.29810 | OK | 0.25500 | 0.00802 | 0.57325 | 0.75526 | OK |
| Unanchored [SimpleQuantileNeuron] | 0.00091 | - | OK | 0.25500 | 0.00267 | 0.01673 | 0.91747 | . |
| Unanchored [QuantileMLP] | 0.00093 | 0.64878 | OK | 0.25500 | 0.01471 | 0.22672 | 0.56636 | OK |
| Anchor param [QuantileMLP] | 0.00093 | 0.64878 | OK | 0.25500 | 0.01471 | 0.22672 | 0.56636 | OK |

### ^GSPC - alpha = 0.01  (DM baseline: Unanchored [SimpleQuantileNeuron])

| model | pinball | dm_p_better_than_baseline | in_mcs | mcs_pvalue | breach_rate | kupiec_p | christoffersen_ind_p | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor param [SimpleQuantileNeuron] | 0.00036 | 0.37934 | OK | 0.65500 | 0.01337 | 0.37848 | 0.11773 | OK |
| Unanchored [SimpleQuantileNeuron] | 0.00037 | - | OK | 0.65500 | 0.01070 | 0.85013 | 0.06900 | OK |
| Anchor hist [SimpleQuantileNeuron] | 0.00037 | 0.55676 | OK | 0.65500 | 0.00668 | 0.33219 | 0.02175 | . |
| Anchor param [QuantileMLP] | 0.00038 | 0.61999 | OK | 0.65500 | 0.00535 | 0.16057 | 0.01237 | . |
| GARCH(1,1)-t | 0.00040 | 0.71198 | OK | 0.65500 | 0.01738 | 0.06632 | 0.21742 | OK |
| Historical | 0.00040 | 0.71857 | OK | 0.65500 | 0.01471 | 0.22672 | 0.00795 | . |
| Parametric-Normal | 0.00041 | 0.85671 | OK | 0.65500 | 0.02807 | 0.00005 | 0.61598 | . |
| Unanchored [QuantileMLP] | 0.00046 | 0.96004 | OK | 0.65500 | 0.00535 | 0.16057 | 0.01237 | . |
| Anchor hist [QuantileMLP] | 0.00046 | 0.96004 | OK | 0.65500 | 0.00535 | 0.16057 | 0.01237 | . |
