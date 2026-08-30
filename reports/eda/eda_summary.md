# AI4I EDA summary

- rows=10000, positive rate=3.39%
- failure modes (raw counts): {'TWF': 46, 'HDF': 115, 'PWF': 95, 'OSF': 98, 'RNF': 19}
- overlap (rows with >1 mode): 24
- failure rate by type: {'H': 2.09, 'L': 3.92, 'M': 2.77}
- strongest |corr| with target: Torque [Nm]=0.19, wear_torque=0.19, power_W=0.17, temp_diff_K=0.12, Tool wear [min]=0.10

Plots: 01_class_balance, 02_feature_dists, 03_correlation, 04_failure_by_type (PNG in this folder).
