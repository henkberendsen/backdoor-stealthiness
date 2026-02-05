import pandas as pd
import numpy as np
from io import StringIO

# ==========================================
# 1. DATA ENTRY (Extracted from Table V in PDF)
# ==========================================

# HIGH POISONING RATE (Approx 5%)
# Note: SSIM values are generally stable, but CDBI/TUP change with PR.
# SSIM values are taken from Table II (using test set if available).
csv_data_high_cdbi = """DATASET,ATTACK,SSIM,CDBI,TUP
CIFAR-10,BadNets,0.956,0.574,16.2
CIFAR-10,Blend,0.784,0.738,3.68
CIFAR-10,WaNet,0.978,0.641,9.30
CIFAR-10,BppAttack,0.992,0.640,21.0
CIFAR-10,Adap-Patch,0.956,1.29,8.38
CIFAR-10,Adap-Blend,0.783,1.66,3.34
CIFAR-10,DFST,0.719,0.650,3.70
CIFAR-10,Narcissus,0.532,0.767,2.15
CIFAR-10,Grond,0.934,0.868,1.87
CIFAR-10,DFBA,0.992,NaN,10.2
CIFAR-100,BadNets,0.955,0.424,27.7
CIFAR-100,Blend,0.765,0.620,10.2
CIFAR-100,WaNet,0.968,0.409,25.4
CIFAR-100,BppAttack,0.991,0.477,52.4
CIFAR-100,Adap-Patch,0.956,1.19,13.4
CIFAR-100,Adap-Blend,0.764,1.24,9.93
CIFAR-100,DFST,0.708,0.475,8.91
CIFAR-100,Narcissus,0.630,3.47,8.75
CIFAR-100,Grond,0.929,1.95,8.86
CIFAR-100,DFBA,0.993,NaN,10.6
Tiny-ImageNet,BadNets,0.96,0.646,10.227
Tiny-ImageNet,Blend,0.81,0.53,3.885
Tiny-ImageNet,WaNet,0.93,0.523,7.078
Tiny-ImageNet,BppAttack,0.99,0.584,34.275
Tiny-ImageNet,Adap-Patch,0.99,1.841,3.522
Tiny-ImageNet,Adap-Blend,0.79,0.916,3.209
Tiny-ImageNet,DFST,0.68,0.475,3.063
Tiny-ImageNet,Narcissus,0.36,1.107,3.192
Tiny-ImageNet,Grond,0.8,1.074,3.236
Tiny-ImageNet,DFBA,0.99,NaN,3.917
Imagenette,BadNets,0.981,0.418,18.4
Imagenette,Blend,0.697,3.07,3.30
Imagenette,WaNet,0.981,0.571,8.85
Imagenette,BppAttack,0.983,0.518,13.9
Imagenette,Adap-Patch,0.955,2.90,5.72
Imagenette,Adap-Blend,0.697,3.00,3.29
Imagenette,DFST,0.692,0.476,3.81
Imagenette,Narcissus,0.615,4.27,2.64
Imagenette,Grond,0.882,24.0,1.61
Imagenette,DFBA,0.999,NaN,13.5"""

# LOW POISONING RATE (Approx 0.3%)
csv_data_low_cdbi = """DATASET,ATTACK,SSIM,CDBI,TUP
CIFAR-10,BadNets,0.956,0.627,23.1
CIFAR-10,Blend,0.784,0.796,3.94
CIFAR-10,WaNet,0.978,0.644,8.56
CIFAR-10,BppAttack,0.992,0.519,10.9
CIFAR-10,Adap-Patch,0.956,3.42,10.2
CIFAR-10,Adap-Blend,0.783,5.19,3.63
CIFAR-10,DFST,0.719,0.550,4.47
CIFAR-10,Narcissus,0.532,5.92,2.13
CIFAR-10,Grond,0.934,5.81,1.78
CIFAR-10,DFBA,0.992,NaN,10.2
CIFAR-100,BadNets,0.955,0.534,29.6
CIFAR-100,Blend,0.765,0.911,10.8
CIFAR-100,WaNet,0.968,1.54,22.1
CIFAR-100,BppAttack,0.991,0.436,56.1
CIFAR-100,Adap-Patch,0.956,2.27,9.77
CIFAR-100,Adap-Blend,0.764,1.53,11.2
CIFAR-100,DFST,0.708,0.260,11.1
CIFAR-100,Narcissus,0.630,4.34,8.48
CIFAR-100,Grond,0.929,1.53,8.64
CIFAR-100,DFBA,0.993,NaN,10.6
Tiny-ImageNet,BadNets,0.96,0.624,3.895
Tiny-ImageNet,Blend,0.81,0.576,3.407
Tiny-ImageNet,WaNet,0.93,1.377,3.657
Tiny-ImageNet,BppAttack,0.99,0.637,3.578
Tiny-ImageNet,Adap-Patch,0.99,0.622,3.495
Tiny-ImageNet,Adap-Blend,0.79,0.497,3.451
Tiny-ImageNet,DFST,0.68,0.353,3.093
Tiny-ImageNet,Narcissus,0.36,1.434,3.25
Tiny-ImageNet,Grond,0.8,1.19,3.368
Tiny-ImageNet,DFBA,0.99,NaN,3.917
Imagenette,BadNets,0.981,0.871,19.3
Imagenette,Blend,0.697,2.20,2.77
Imagenette,WaNet,0.981,0.809,3.84
Imagenette,BppAttack,0.983,0.854,1.56
Imagenette,Adap-Patch,0.955,4.45,2.35
Imagenette,Adap-Blend,0.697,12.3,2.45
Imagenette,DFST,0.692,0.934,4.43
Imagenette,Narcissus,0.615,7.54,2.13
Imagenette,Grond,0.882,12.4,1.05
Imagenette,DFBA,0.999,NaN,13.5"""

# ==========================================
# 2. CALCULATION LOGIC (Using CDBI Formulation)
# ==========================================

def calculate_tri_space_cdbi_metrics(group):
    # --- Median Anchor Calibration ---
    # We calibrate lambda based on the median of the current group/dataset
    median_cdbi = group['CDBI'].median()
    median_tup = group['TUP'].median()
    
    # Calculate lambdas
    lambda_f = np.log(2) / median_cdbi if median_cdbi != 0 else 0
    lambda_p = np.log(2) / median_tup if median_tup != 0 else 0
    
    # --- 1. Input Space Score (S_in) ---
    group['S_in'] = group['SSIM']
    
    # --- 2. Feature Space Score (S_feat) - USING CDBI ---
    # Formula: 1 - exp(-lambda * CDBI)
    # Higher CDBI = Stealthier (closer to 1)
    group['S_feat'] = 1 - np.exp(-lambda_f * group['CDBI'])
    
    # --- 3. Parameter Space Score (S_param) ---
    # Formula: exp(-lambda * TUP)
    # Lower TUP = Stealthier (closer to 1)
    group['S_param'] = np.exp(-lambda_p * group['TUP'])
    
    # --- 4. Tri-Space Aggregation ---
    
    # Worst-Case (Min)
    group['s_min_tri'] = group[['S_in', 'S_feat', 'S_param']].min(axis=1)
    
    # Geometric Mean (Balanced)
    group['s_geo_tri'] = np.power(group['S_in'] * group['S_feat'] * group['S_param'], 1/3)
    
    return group

# ==========================================
# 3. EXECUTION
# ==========================================

# Process High PR (5%) with CDBI
df_high = pd.read_csv(StringIO(csv_data_high_cdbi))
results_high = df_high.groupby('DATASET', group_keys=False).apply(calculate_tri_space_cdbi_metrics)
results_high[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_csv('tri_space_metrics_cdbi_high.csv', index=False)

# Process Low PR (0.3%) with CDBI
df_low = pd.read_csv(StringIO(csv_data_low_cdbi))
results_low = df_low.groupby('DATASET', group_keys=False).apply(calculate_tri_space_cdbi_metrics)
results_low[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_csv('tri_space_metrics_cdbi_low.csv', index=False)

# Display Preview
print("--- CDBI RESULTS (High Poisoning Rate) ---")
print(results_high[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_string(index=False))

print("\n--- CDBI RESULTS (Low Poisoning Rate) ---")
print(results_low[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_string(index=False))