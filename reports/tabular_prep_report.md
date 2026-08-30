# Block 2 - AI4I Tabular Preparation Report

Raw shape: (10000, 14)
Class balance (raw): {0: 9661, 1: 339} (3.39% positive)

## 1. Injected data-quality issues (seeded, reproducible)
- Duplicated 180 random rows verbatim -> shape now (10180, 14)
- Inserted NaNs into ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Tool wear [min]'] at ~2.5% each
- Missing-value counts after injection:
```
Air temperature [K]        254
Process temperature [K]    254
Rotational speed [rpm]     254
Tool wear [min]            254
```

## 2. Duplicate handling
- Exact duplicate rows detected: 150
- Additional UDI-level duplicates removed: 30
- Shape after de-duplication: (10000, 14)

## 3. Leakage control & feature engineering
- Dropped identifiers: UDI, Product ID
- Held out (NOT features, used for Block 3 error analysis): ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
- Engineered features: ['temp_diff_K', 'power_W', 'wear_torque', 'type_L', 'type_M', 'type_H']
- Final feature matrix: 11 features

## 4. Stratified train / val / test split (60 / 20 / 20)
- train: 6000 rows, 203 positives (3.38%)
- val: 2000 rows, 68 positives (3.40%)
- test: 2000 rows, 68 positives (3.40%)

## 5. Missing-value imputation + scaling (fit on TRAIN only -> no leakage)
- Train medians used for imputation: {'air_temp_K': np.float64(300.1), 'process_temp_K': np.float64(310.1), 'rot_speed_rpm': np.float64(1502.0), 'torque_Nm': np.float64(40.2), 'tool_wear_min': np.float64(108.0), 'temp_diff_K': np.float64(9.8), 'power_W': np.float64(6276.965), 'wear_torque': np.float64(3996.15), 'type_L': np.float64(1.0), 'type_M': np.float64(0.0), 'type_H': np.float64(0.0)}
- NaNs remaining after impute: train=0

## 6. Artifacts written
- data\processed\tabular_train.csv
- data\processed\tabular_val.csv
- data\processed\tabular_test.csv
- models_registry\tabular_preprocessor.joblib
- data\processed\tabular_clean_unscaled.csv
