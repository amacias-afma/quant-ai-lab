# 🏆 Value-at-Risk Batch Backtesting Summary

| Ticker | Model | Breach Rate (%) | Kupiec p-value | Christoff p-value | Avg Capital Reserved | Max Capital Reserved | Min Capital Reserved | Responsiveness (Std) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | Neural Network | 0.00% | 0.0 | 1.0 | -10.45% | -14.36% | -10.14% | 0.0047 | ❌ FAIL |
| BTC-USD | GARCH(1,1) | 0.28% | 0.0129 | 0.0 | -22.96% | -442.96% | -3.98% | 0.3824 | ❌ FAIL |
| BTC-USD | Anchor NN | 1.21% | 0.4435 | 0.0 | -6.15% | -9.00% | -4.09% | 0.0091 | ✅ PASS |
| BTC-USD | Anchor Hist NN | 0.19% | 0.0031 | 0.0 | -8.04% | -9.22% | -6.45% | 0.0075 | ❌ FAIL |
| BTC-USD | Historical VaR | 1.39% | 0.2164 | 0.0 | -5.93% | -7.11% | -5.07% | 0.0061 | ✅ PASS |
| ^GSPC | Neural Network | 0.27% | 0.0407 | 0.0019 | -3.30% | -5.10% | -3.26% | 0.0011 | ❌ FAIL |
| ^GSPC | GARCH(1,1) | 1.60% | 0.0972 | 0.0 | -3.47% | -19.90% | -0.99% | 0.0281 | ✅ PASS |
| ^GSPC | Anchor NN | 1.07% | 0.8527 | 0.069 | -2.49% | -5.42% | -2.00% | 0.004 | ✅ PASS |
| ^GSPC | Anchor Hist NN | 0.94% | 1.0 | 0.0499 | -2.66% | -3.94% | -2.19% | 0.0039 | ✅ PASS |
| ^GSPC | Historical VaR | 1.34% | 0.3527 | 0.0051 | -2.27% | -3.02% | -1.48% | 0.0053 | ✅ PASS |

