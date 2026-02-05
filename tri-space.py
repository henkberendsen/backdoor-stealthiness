import pandas as pd
import numpy as np
from io import StringIO

# ==========================================
# 1. DATA ENTRY (Extracted from Tables 2, 3, 5)
# ==========================================

# HIGH POISONING RATE (Approx 5%)
# Note: SSIM is generally constant across PR in Table 2, but TUP/SS change.
csv_data_high = """DATASET,ATTACK,SSIM,SS,TUP
CIFAR-10,BadNets,0.956,0.507,16.2
CIFAR-10,Blend,0.784,0.473,3.68
CIFAR-10,WaNet,0.978,0.382,9.30
CIFAR-10,BppAttack,0.992,0.500,21.0
CIFAR-10,Adap-Patch,0.956,0.254,8.38
CIFAR-10,Adap-Blend,0.783,0.231,3.34
CIFAR-10,DFST,0.719,0.519,3.70
CIFAR-10,Narcissus,0.532,0.491,2.15
CIFAR-10,Grond,0.934,0.435,1.87
CIFAR-10,DFBA,0.992,NaN,10.2
CIFAR-100,BadNets,0.955,0.511,27.7
CIFAR-100,Blend,0.765,0.435,10.2
CIFAR-100,WaNet,0.968,0.492,25.4
CIFAR-100,BppAttack,0.991,0.527,52.4
CIFAR-100,Adap-Patch,0.956,0.0600,13.4
CIFAR-100,Adap-Blend,0.764,0.0382,9.93
CIFAR-100,DFST,0.708,0.509,8.91
CIFAR-100,Narcissus,0.630,0.0743,8.75
CIFAR-100,Grond,0.929,0.233,8.86
CIFAR-100,DFBA,0.993,NaN,10.6
Tiny-ImageNet,BadNets,0.96,0.295,10.227
Tiny-ImageNet,Blend,0.81,0.339,3.885
Tiny-ImageNet,WaNet,0.93,0.342,7.078
Tiny-ImageNet,BppAttack,0.99,0.433,34.275
Tiny-ImageNet,Adap-Patch,0.99,0.067,3.522
Tiny-ImageNet,Adap-Blend,0.79,0.213,3.209
Tiny-ImageNet,DFST,0.68,0.485,3.063
Tiny-ImageNet,Narcissus,0.36,0.314,3.192
Tiny-ImageNet,Grond,0.8,0.327,3.236
Tiny-ImageNet,DFBA,0.99,NaN,3.917
Imagenette,BadNets,0.981,0.588,18.4
Imagenette,Blend,0.697,0.0216,3.30
Imagenette,WaNet,0.981,0.557,8.85
Imagenette,BppAttack,0.983,0.607,13.9
Imagenette,Adap-Patch,0.955,0.0822,5.72
Imagenette,Adap-Blend,0.697,0.0630,3.29
Imagenette,DFST,0.692,0.641,3.81
Imagenette,Narcissus,0.615,0.0431,2.64
Imagenette,Grond,0.882,0.0308,1.61
Imagenette,DFBA,0.999,NaN,13.5"""

# LOW POISONING RATE (Approx 0.3%)
csv_data_low = """DATASET,ATTACK,SSIM,SS,TUP
CIFAR-10,BadNets,0.956,0.355,23.1
CIFAR-10,Blend,0.784,0.276,3.94
CIFAR-10,WaNet,0.978,0.378,8.56
CIFAR-10,BppAttack,0.992,0.371,10.9
CIFAR-10,Adap-Patch,0.956,0.0500,10.2
CIFAR-10,Adap-Blend,0.783,0.149,3.63
CIFAR-10,DFST,0.719,0.339,4.47
CIFAR-10,Narcissus,0.532,-0.0070,2.13
CIFAR-10,Grond,0.934,-0.101,1.78
CIFAR-10,DFBA,0.992,NaN,10.2
CIFAR-100,BadNets,0.955,0.462,29.6
CIFAR-100,Blend,0.765,0.379,10.8
CIFAR-100,WaNet,0.968,0.217,22.1
CIFAR-100,BppAttack,0.991,0.540,56.1
CIFAR-100,Adap-Patch,0.956,0.0578,9.77
CIFAR-100,Adap-Blend,0.764,0.191,11.2
CIFAR-100,DFST,0.708,0.730,11.1
CIFAR-100,Narcissus,0.630,0.0620,8.48
CIFAR-100,Grond,0.929,0.208,8.64
CIFAR-100,DFBA,0.993,NaN,10.6
Tiny-ImageNet,BadNets,0.96,0.463,3.895
Tiny-ImageNet,Blend,0.81,0.423,3.407
Tiny-ImageNet,WaNet,0.93,0.303,3.657
Tiny-ImageNet,BppAttack,0.99,0.347,3.578
Tiny-ImageNet,Adap-Patch,0.99,0.245,3.495
Tiny-ImageNet,Adap-Blend,0.79,0.311,3.451
Tiny-ImageNet,DFST,0.68,0.63,3.093
Tiny-ImageNet,Narcissus,0.36,0.302,3.25
Tiny-ImageNet,Grond,0.8,0.355,3.368
Tiny-ImageNet,DFBA,0.99,NaN,3.917
Imagenette,BadNets,0.981,0.260,19.3
Imagenette,Blend,0.697,0.0975,2.77
Imagenette,WaNet,0.981,0.192,3.84
Imagenette,BppAttack,0.983,0.227,1.56
Imagenette,Adap-Patch,0.955,-0.0455,2.35
Imagenette,Adap-Blend,0.697,-0.0657,2.45
Imagenette,DFST,0.692,0.261,4.43
Imagenette,Narcissus,0.615,-0.0164,2.13
Imagenette,Grond,0.882,0.0546,1.05
Imagenette,DFBA,0.999,NaN,13.5"""

# ==========================================
# 2. CALCULATION LOGIC
# ==========================================

def calculate_tri_space_metrics(group):
    # --- Parameter Space Calibration (Median Anchor) ---
    median_tup = group['TUP'].median()
    
    # Calculate lambda for TUP (ln(2) / median)
    lambda_p = np.log(2) / median_tup if median_tup != 0 else 0
    
    # --- 1. Input Space Score (S_in) ---
    # SSIM is already 0-1, where 1 is perfect stealth.
    group['S_in'] = group['SSIM']
    
    # --- 2. Feature Space Score (S_feat) ---
    # Metric: Silhouette Score (SS) in range [-1, 1].
    # Logic: Lower SS is stealthier (more overlap).
    # Formula: (1 - SS) / 2
    #   If SS = -1 (Stealthy) -> (1 - (-1))/2 = 1.0 (High Score)
    #   If SS =  1 (Distinct) -> (1 - 1)/2    = 0.0 (Low Score)
    group['S_feat'] = (1 - group['SS']) / 2
    
    # --- 3. Parameter Space Score (S_param) ---
    # Metric: TUP (Lower is stealthier).
    # Logic: exp(-lambda * TUP)
    group['S_param'] = np.exp(-lambda_p * group['TUP'])
    
    # --- 4. Tri-Space Aggregation ---
    
    # Worst-Case (Min)
    group['s_min_tri'] = group[['S_in', 'S_feat', 'S_param']].min(axis=1)
    
    # Geometric Mean (Balanced)
    # Note: If any score is NaN (like DFBA Feature), this returns NaN, which is correct.
    group['s_geo_tri'] = np.power(group['S_in'] * group['S_feat'] * group['S_param'], 1/3)
    
    return group

# ==========================================
# 3. EXECUTION
# ==========================================

# Process High PR (5%)
df_high = pd.read_csv(StringIO(csv_data_high))
results_high = df_high.groupby('DATASET', group_keys=False).apply(calculate_tri_space_metrics)
results_high[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_csv('tri_space_metrics_high.csv', index=False)

# Process Low PR (0.3%)
df_low = pd.read_csv(StringIO(csv_data_low))
results_low = df_low.groupby('DATASET', group_keys=False).apply(calculate_tri_space_metrics)
results_low[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_csv('tri_space_metrics_low.csv', index=False)

# Display Preview of High PR Results
print("--- RESULTS (High Poisoning Rate) ---")
print(results_high[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_string(index=False))

print("\n--- RESULTS (Low Poisoning Rate) ---")
print(results_low[['DATASET', 'ATTACK', 's_min_tri', 's_geo_tri']].to_string(index=False))