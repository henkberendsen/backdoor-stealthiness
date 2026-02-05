import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. PRE-PROCESSING: ALIGN DIRECTIONS
# ==========================================
# Goal: Make HIGHER value always indicate HIGHER STEALTH.
# If a metric/defense is "Lower is Better/Stealthier", we flip it (multiply by -1).

df = pd.read_csv("dataframing_results_df.csv")

aligned_df = df.copy()

# A. METRICS TO FLIP (Lower Value = Higher Stealth)
# L2, LPIPS, IS, SAM -> Low is stealthy.
# DSWD, BMS -> Low is stealthy.
# UCLC, TAC, TUP -> Low is stealthy.
flip_metrics = [
    'l1', 'l2', 'l_inf', 'MSE', 'LPIPS', 'IS', 'SAM', # Input
    'DSWD', 'BMS',                                    # Feature
    'UCLC', 'TAC', 'TUP'                              # Parameter
]

# B. DEFENSES TO FLIP (Higher Value = Higher Detection/Lower Stealth)
# F1 Scores -> High means detected.
# Detection Counts (BTI, BAN) -> High means detected.
# TABOR Score -> High means anomaly detected.
# REMOVED: AC_F1 (Activation Clustering)
flip_defenses = [
    'SPECTRE_F1', 'STRIP_F1', 'SS_F1', 
    'BTI_Detected', 'BAN_Detected', 'TABOR_Score'
]

# C. METRICS/DEFENSES TO KEEP (Higher Value = Higher Stealth)
# PSNR, SSIM, pHash -> High is stealthy.
# SS (Silhouette) -> Lower SS implies greater stealthiness. So SS should be FLIPPED.
flip_metrics.append('SS') 

# CDBI -> Higher is stealthier. Keep.
# CLP_ASR, FeatureRE_ASR -> High ASR means defense failed (Stealthy). Keep.
# IBAU_ASR -> High ASR means defense failed (Stealthy). Keep.
# NC_Norm -> High Norm = Stealthy (Lower norm is less stealthy). Keep.

# --- APPLY FLIPS ---
for col in flip_metrics + flip_defenses:
    if col in aligned_df.columns:
        # We multiply by -1 so the correlation logic works (High = Stealth)
        aligned_df[col] = aligned_df[col] * -1

# ==========================================
# 2. RESIDUAL CORRELATION FUNCTION
# ==========================================
def calculate_residuals_heatmap(data, metric_cols, defense_cols):
    results = pd.DataFrame(index=defense_cols, columns=metric_cols)
    p_values = pd.DataFrame(index=defense_cols, columns=metric_cols)
    
    # CONTROL VARIABLES
    # Controlling for ASR_Before and Dataset.
    control_formula = "ASR_Before + C(Dataset)" 

    print(f"Controlling for: {control_formula}")

    for defense in defense_cols:
        # Filter for valid data
        sub = data.dropna(subset=[defense] + metric_cols + ['ASR_Before', 'Dataset'])
        
        # If defense has no variance (e.g., always 0.0), skip
        if sub[defense].std() == 0:
            print(f"Skipping {defense} due to zero variance.")
            continue

        # 1. Get Defense Residuals (Cleaned Defense Score)
        try:
            model_def = smf.ols(f"{defense} ~ {control_formula}", data=sub).fit()
            y_res = model_def.resid
        except Exception as e:
            print(f"Error modeling {defense}: {e}")
            continue

        for metric in metric_cols:
            # 2. Get Metric Residuals (Cleaned Metric Score)
            try:
                model_met = smf.ols(f"{metric} ~ {control_formula}", data=sub).fit()
                x_res = model_met.resid
                
                # 3. Spearman Correlation
                corr, p_val = spearmanr(x_res, y_res)
                results.loc[defense, metric] = corr
                p_values.loc[defense, metric] = p_val
            except:
                results.loc[defense, metric] = np.nan
                p_values.loc[defense, metric] = np.nan

    return results.astype(float), p_values.astype(float)

# ==========================================
# 3. PLOTTING FUNCTION WITH MASKING
# ==========================================
# Label mapping for proper LaTeX rendering (using mathbf for bold math font)
LABEL_MAP = {
    # ===== INPUT-SPACE METRICS =====
    'l1': r'$\mathbf{\ell_1}$',
    'l2': r'$\mathbf{\ell_2}$',
    'l_inf': r'$\mathbf{\ell_\infty}$',
    'MSE': r'$\mathbf{MSE}$',
    'PSNR': r'$\mathbf{PSNR}$',
    'SSIM': r'$\mathbf{SSIM}$',
    'LPIPS': r'$\mathbf{LPIPS}$',
    'IS': r'$\mathbf{IS}$',
    'pHash': r'$\mathbf{pHash}$',
    'SAM': r'$\mathbf{SAM}$',
    
    # ===== FEATURE-SPACE METRICS =====
    'SS': r'$\mathbf{SS}$',
    'DSWD': r'$\mathbf{DSWD}$',
    'CDBI': r'$\mathbf{CDBI}$',
    'BMS': r'$\mathbf{BMS}$',
    
    # ===== PARAMETER-SPACE METRICS =====
    'UCLC': r'$\mathbf{UCLC}$',
    'TAC': r'$\mathbf{TAC}$',
    'TUP': r'$\mathbf{TUP}$',
    
    # ===== DEFENSES =====
    'SPECTRE_F1': r'$\mathbf{SPECTRE}$ $(F_1)$',
    'STRIP_F1': r'$\mathbf{STRIP}$ $(F_1)$',
    'SS_F1': r'$\mathbf{SS}$ $(F_1)$',
    'IBAU_ASR': r'$\mathbf{IBAU}$ (ASR)',
    'CLP_ASR': r'$\mathbf{CLP}$ (ASR)',
    'FeatureRE_ASR': r'$\mathbf{FeatureRE}$ (ASR)',
    'NC_Norm': r'$\mathbf{NC}$ (Norm)',
    'TABOR_Score': r'$\mathbf{TABOR}$ (Score)',
    'BTI_Detected': r'$\mathbf{BTI}$ (Detected)',
    'BAN_Detected': r'$\mathbf{BAN}$ (Detected)',
}

def plot_significance_heatmap(corr_data, p_data, title, filename):
    # Set math font to STIX (similar to Times/Computer Modern - classic math look)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    plt.figure(figsize=(14, 8))
    
    # Create annotation labels: Show value only if p < 0.05
    # Add a star (*) for highly significant results
    annot_labels = corr_data.copy().astype(object)
    for r in corr_data.index:
        for c in corr_data.columns:
            val = corr_data.loc[r, c]
            p = p_data.loc[r, c]
            
            if pd.isna(val):
                annot_labels.loc[r, c] = ""
            elif p < 0.05:
                annot_labels.loc[r, c] = f"{val:.2f}*"
            else:
                # OPTION 1: Show insignificant values without star
                annot_labels.loc[r, c] = f"{val:.2f}"
                # OPTION 2: Hide insignificant values (Uncomment below to use)
                # annot_labels.loc[r, c] = "" 

    ax = sns.heatmap(
        corr_data, 
        annot=annot_labels, 
        fmt="", 
        cmap='RdBu_r', # Red = Positive (Good Metric), Blue = Negative (Bad Metric)
        center=0,
        vmin=-1, vmax=1,
        annot_kws={'fontweight': 'bold'}
    )
    
    # Apply label mapping for proper math symbols
    new_xticklabels = [LABEL_MAP.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
    new_yticklabels = [LABEL_MAP.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()]
    ax.set_xticklabels(new_xticklabels)
    ax.set_yticklabels(new_yticklabels)
    
    plt.title(title)
    plt.xlabel("Stealthiness Metrics (Aligned: Higher = Stealthier)")
    plt.ylabel("Defenses (Aligned: Higher = Defense Failed)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 4. RUN ANALYSIS (SUBSET)
# ==========================================
# Group metrics for cleaner plotting
input_mets = ['l2', 'SSIM', 'LPIPS', 'pHash', 'SAM'] 
feat_param_mets = ['SS', 'DSWD', 'CDBI', 'UCLC', 'TAC', 'TUP']

defense_list = [
    'STRIP_F1', 'SPECTRE_F1', 
    'CLP_ASR', 'IBAU_ASR', 'FeatureRE_ASR', 
    'NC_Norm', 'TABOR_Score',          
    'BTI_Detected', 'BAN_Detected'     
]

# Calculate
corr_matrix, p_matrix = calculate_residuals_heatmap(
    aligned_df, 
    input_mets + feat_param_mets, 
    defense_list
)

# Visualize
plot_significance_heatmap(
    corr_matrix, 
    p_matrix, 
    "Residual Correlation (Selected Subset)\n* indicates p < 0.05", 
    "residuals_heatmap.png"
)

# ==========================================
# 5. RUN ANALYSIS (FULL)
# ==========================================
all_metrics = [
    'l1', 'l2', 'l_inf', 'MSE', 'PSNR', 'SSIM', 'LPIPS', 'IS', 'pHash', 'SAM',
    'SS', 'DSWD', 'CDBI', 'UCLC', 'TAC', 'TUP'
]

all_defenses = [
    'SPECTRE_F1', 'STRIP_F1', 'SS_F1', 
    'IBAU_ASR', 'CLP_ASR', 'FeatureRE_ASR', 'NC_Norm', 'TABOR_Score',
    'BTI_Detected', 'BAN_Detected'
]

# Calculate
corr_matrix_all, p_matrix_all = calculate_residuals_heatmap(
    aligned_df, 
    all_metrics, 
    all_defenses
)

# Visualize
plt.figure(figsize=(18, 12)) # Slightly larger for full matrix
plot_significance_heatmap(
    corr_matrix_all, 
    p_matrix_all, 
    "Residual Correlations \n* indicates p < 0.05", 
    "residuals_heatmap_all.png"
)