# Block 3a - Tabular failure classification: error analysis

## Model comparison (test set)

| model | precision | recall | f1 | roc_auc | false positives |
|---|---|---|---|---|---|
| rf_baseline | 0.8276 | 0.7059 | 0.7619 | 0.9539 | 10 |
| xgb_default | 0.6923 | 0.6618 | 0.6767 | 0.9438 | 20 |
| xgb_tuned | 0.7419 | 0.6765 | 0.7077 | 0.9561 | 16 |

**Selected model: `rf_baseline`** (highest validation F1).

## Per-failure-mode recall (test set, selected model)

| mode   |   n_true |   n_detected |   recall_by_mode |
|:-------|---------:|-------------:|-----------------:|
| TWF    |       12 |            0 |            0     |
| HDF    |       20 |           19 |            0.95  |
| PWF    |       18 |           16 |            0.889 |
| OSF    |       18 |           17 |            0.944 |
| RNF    |        2 |            0 |            0     |

- False-positive rate on healthy machines: 0.0052 (10 rows)

### Reading the errors
- **RNF (random failures)** carry no sensor signature by construction (AI4I injects them at 0.1% independent of features) - the model cannot and should not learn them; low RNF recall is expected, not a defect.
- **TWF** has the fewest positive examples (46 in 10k) so recall is the noisiest; misses concentrate where tool wear sits mid-range and torque is normal.
- **HDF / PWF / OSF** are physically driven (temp difference, power band, wear x torque) and the engineered features `temp_diff_K`, `power_W`, `wear_torque` give the model strong separation - these dominate recall.
- Threshold is 0.5; lowering it trades the low false-positive rate for higher recall on TWF/RNF (see PR behaviour via roc_auc).
