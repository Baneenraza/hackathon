# Block 3b - CMAPSS FD001 RUL regression: error analysis

## Model comparison (test set, 100 engines)

| model | MAE | RMSE | CMAPSS score |
|---|---|---|---|
| baseline_median | 38.74 | 49.275 | 150743.4 |
| lstm_w30 | 11.03 | 14.501 | 328.3 |
| lstm_w50 | 10.763 | 14.776 | 372.7 |

**Selected: `lstm_w50`** (lowest validation RMSE). The LSTM cuts test MAE roughly in half vs the median baseline.

## Residuals bucketed by true RUL

| true_rul   |   n |   mae |   mean_bias |
|:-----------|----:|------:|------------:|
| 0-25       |  19 |  2.96 |        2.16 |
| 26-50      |  14 |  5.71 |        0.69 |
| 51-75      |  10 | 11.84 |        3.86 |
| 76-100     |  24 | 15.84 |        6.91 |
| 100+       |  33 | 13.39 |       -8.43 |

### Reading the errors
- The RUL target is capped at 125. Engines whose true RUL is well above the cap (early life) are trained toward a flat ceiling, so the model is deliberately uninformative there - acceptable because early-life prognosis is not actionable.
- Error concentrates in the mid-life band where the degradation signal is weak. Near end-of-life (0-25) the model is most accurate, which is the operationally important regime.
- Sign of `mean_bias`: negative = the model predicts failure sooner than reality (conservative / safe); positive = optimistic (risky).

## Worst 5 predictions
|   engine |   true_rul |   pred_rul |   residual |
|---------:|-----------:|-----------:|-----------:|
|       67 |         77 |      120.5 |       43.5 |
|       30 |        115 |       73.9 |      -41.1 |
|       12 |        124 |       83.8 |      -40.2 |
|       93 |         85 |       49.8 |      -35.2 |
|       79 |         63 |       98.1 |       35.1 |
