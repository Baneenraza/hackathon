# Block 5 - NLP on maintenance notes (450 synthetic records)

## Classification: TRAIN vs VALIDATION accuracy (overfitting check)

| task | classes | train acc | val acc | test acc | 5-fold CV acc | test macro-F1 | overfit gap (train-val) |
|---|---|---|---|---|---|---|---|
| category | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **0.0** |
| urgency | 3 | 0.8057 | 0.75 | 0.75 | 0.7356 | 0.7546 | **0.0557** |

### Overfitting assessment (not hidden)
- The corpus is 450 synthetic notes with templated phrasing, so TF-IDF picks up give-away tokens and train accuracy sits near 1.0.
- category: train-val gap = 0.000 with 5-fold CV = 1.000. A CV score of ~1.0 is NOT a success story - it means the synthetic templates put a unique give-away keyword in almost every note (e.g. 'breaker'->electrical, 'lubrication'->mechanical). On real maintenance notes this task would be materially harder; treat this model as a placeholder.
- urgency: train-val gap = 0.056, 5-fold CV = 0.736. This is the more realistic task - urgency is genuinely ambiguous in the text and the model sits well below ceiling, which is the honest expected behaviour.
- The 5-fold CV accuracy is the number to trust; the held-out test accuracy is on only ~68 notes so it is noisy.
- Mitigations applied: min_df=2 (drop hapax tokens), sublinear TF, L2-regularised LogisticRegression, class_weight balanced. Further work would need real, non-templated notes.

## Test-set classification reports

### category
```
              precision    recall  f1-score   support

  electrical       1.00      1.00      1.00        20
  mechanical       1.00      1.00      1.00        21
      safety       1.00      1.00      1.00        13
    software       1.00      1.00      1.00        14

    accuracy                           1.00        68
   macro avg       1.00      1.00      1.00        68
weighted avg       1.00      1.00      1.00        68
```

### urgency
```
              precision    recall  f1-score   support

        high       1.00      0.91      0.95        22
         low       0.59      0.65      0.62        20
      medium       0.69      0.69      0.69        26

    accuracy                           0.75        68
   macro avg       0.76      0.75      0.75        68
weighted avg       0.76      0.75      0.75        68
```

## Regex field extraction (all 450 notes)

| field | notes with >=1 hit |
|---|---|
| machine_id | 450 |
| section_ref | 10 |
| failure_mode (in free text) | 18 |
| reading | 134 |

- Machine-ID regex round-trips correctly on **450/450** notes (matches the structured `machine_id` column exactly).
- Only ~18 descriptions name a failure mode (TWF/HDF/PWF/OSF/RNF) in free text; 215 notes carry a structured `failure_type_hint`. The free text is intentionally narrative, so the structured column is the reliable source for mode - the regex is a fallback for pasted text.
- Extractor: `src/models/train_nlp.py:extract_fields(text)` -> dict of `machine_ids`, `section_refs`, `failure_modes`, `readings`.
