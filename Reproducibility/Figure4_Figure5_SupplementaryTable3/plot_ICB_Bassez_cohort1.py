import re
import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu, kruskal
from pathlib import Path

# ── publication rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'pdf.fonttype': 42, 'figure.dpi': 300, 'savefig.dpi': 300,
})

# ── dataset settings ──────────────────────────────────────────────────────────
test_string       = "Bassez_cohort1_cd8_QCed_log1p"
str_title         = 'Bassez_cohort1'
cancer_type       = 'BC'
dense_function_flag = 'deltarho_30best'
suffix_           = dense_function_flag
Col_time          = 'timepoint'
pre_treat_name    = 'Pre'
timepoints        = ['Pre', 'On']
gene_names         = ['CXCL13', 'GZMB']

BASE = '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein'
in_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_0913_combine_6_models/'
study_log_path = f'{BASE}/data_collection/Bassez_BreastCancer_NatMed/cohort1_Su_out/Bassez_cohort1_cd8_QCed_log1p.h5ad'
out_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_20260630_combine_6_models_Uniform/'
os.makedirs(out_dir, exist_ok=True)

test_data_list = ['test_caushi', 'test_hanada', 'test_lowery', 'test_meng', 'test_oliveira', 'test_zheng']
ck_dic = {
    'test_caushi': [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_caushi/hvg2000_by_study_6_89_genes/regenerate_pt/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e3.pt'],
    'test_hanada': [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_hanada/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e25.pt'],
    'test_lowery': [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_lowery/hvg2000_by_study_6_89_genes/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e1.pt'],
    'test_meng':   [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_meng/hvg2000_by_study_6_89_genes_newCK_lessLayer/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e10.pt'],
    'test_oliveira': [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_oliveira/hvg2000_by_study_6_89_genes_newCK_lessLayer/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght1.0-SuContinue/model_e5.pt'],
    'test_zheng':  [f'{BASE}/jobs/classifier/six_studies/supconloss_simCLR/test_zheng/hvg2000_by_study_6_89_genes/regenerate_pt/save/dev_SixStudy_supconH_SimCLR_glenhvg2000_by_study_6_89_genes_unfix1_lr0.0001_lrseq0.0001_npos9_nneg120_t0.1_maxl100_simclr_wght0.3-SuContinue/model_e1.pt'],
}

set2_palette = sns.color_palette("Set2", n_colors=4)
custom_palette = {'Non_Response': set2_palette[2], 'Response': set2_palette[1]}
response_order = ['Non_Response', 'Response']

# ── load or recompute cell-level predictions ──────────────────────────────────
csv_path = out_dir + f'{test_string}_{dense_function_flag}_df_filtered.csv'

if os.path.exists(csv_path):
    df_filtered = pd.read_csv(csv_path, index_col=0)

else:
    # load predictions from 6 checkpoints
    df_list = []
    for test_da in test_data_list:
        pt_path = ck_dic[test_da][0]
        pt_dir  = os.path.dirname(pt_path)
        epc     = re.search(r'model_e(\d+).', pt_path).group(1)
        paths   = glob.glob(os.path.join(pt_dir, f"{test_string}_e{epc}-*.h5ad"))
        paths   = [p for p in paths if not p.endswith("_averagedPRJ.h5ad")]
        val_ad  = sc.read_h5ad(max(paths, key=os.path.getmtime))
        df_list.append(val_ad.obs[[f'pro_jenks_OT_{suffix_}', f'pro_jenks_OT_{suffix_}_score']])

    renamed = []
    for i, df in enumerate(df_list):
        renamed.append(df.rename(columns={
            f'pro_jenks_OT_{suffix_}':       f'model{i+1}_prediction',
            f'pro_jenks_OT_{suffix_}_score': f'model{i+1}_score',
        }))
    final_dataframe = pd.concat(renamed, axis=1)
    final_dataframe['predicts_sum'] = sum(
        final_dataframe[f'model{i+1}_prediction'].astype(int) for i in range(6)
    )
    for i in range(6):
        final_dataframe[f'model{i+1}_rank'] = final_dataframe[f'model{i+1}_score'].rank()
    final_dataframe['predicts_rank_sum'] = sum(
        final_dataframe[f'model{i+1}_rank'] for i in range(6)
    )
    final_dataframe.to_csv(out_dir + f'{test_string}_{dense_function_flag}_predicts_df.csv')
    #final_n_cells = final_dataframe.shape[0]

    # load metadata from first checkpoint obs
    pt_path = ck_dic[test_data_list[0]][0]
    pt_dir  = os.path.dirname(pt_path)
    epc     = re.search(r'model_e(\d+).', pt_path).group(1)
    paths   = glob.glob(os.path.join(pt_dir, f"{test_string}_e{epc}-*.h5ad"))
    val_ad  = sc.read_h5ad(min(paths, key=os.path.getmtime))
    df_study_meta = val_ad.obs.copy()

    df_study_meta['expansion'] = df_study_meta['expansion'].astype('category')
    df_study_meta = df_study_meta.copy()
    df_study_meta['expansion'] = df_study_meta['expansion'].cat.add_categories(['Unknown_Response'])
    mapping = {'E': 'Response', 'NE': 'Non_Response'}
    df_study_meta['Response'] = df_study_meta['expansion'].map(mapping).fillna('Unknown_Response')
    df_study_meta = df_study_meta[['sample', 'study', Col_time, 'Response']]

    df_meta_score   = df_study_meta.merge(final_dataframe, left_index=True, right_index=True, how='left')
    df_meta_score_f = df_meta_score[df_meta_score['Response'] != 'Unknown_Response']

    df_filtered = df_meta_score_f.copy()
    valid_samples = pd.read_csv(in_dir+f'{test_string}_{dense_function_flag}_valid_samples_forReUse.csv', index_col=None)['valid_samples'].tolist()
    df_filtered = df_filtered[df_filtered['sample'].isin(valid_samples)]
    # Rank should be updated again after removing some of the samples.
    df_filtered = df_filtered[df_filtered[Col_time] == pre_treat_name].copy()
    for i in range(6):
        df_filtered[f'model{i+1}_rank'] = df_filtered[f'model{i+1}_score'].rank()
    df_filtered['predicts_rank_sum'] = sum(
        df_filtered[f'model{i+1}_rank'] for i in range(6)
    )
    
    df_filtered['Response'] = pd.Categorical(df_filtered['Response'], categories=response_order, ordered=True)
    df_filtered = df_filtered.sort_values(by='Response')
    df_filtered = df_filtered[df_filtered['predicts_sum'] >= 0]
    df_filtered['predicts_rank_sum_normalized'] = df_filtered['predicts_rank_sum'] / (df_filtered.shape[0] * 6)

    df_filtered.to_csv(csv_path)

# ── helpers ───────────────────────────────────────────────────────────────────
category_map_binary = {'Non_Response': 1, 'Response': 2}

df_filtered_nonzero = df_filtered[df_filtered['predicts_sum'] > 0].copy()


def _mann_p_spearman(subset, y_col, group_col, cat_map):
    nr = subset[subset[group_col] == 'Non_Response'][y_col]
    re = subset[subset[group_col] == 'Response'][y_col]
    if nr.empty or re.empty:
        return None, None, None
    _, p = mannwhitneyu(nr, re, alternative='two-sided')
    subset = subset.copy()
    subset['_code'] = subset[group_col].map(cat_map)
    rho, rp = stats.spearmanr(subset['_code'], subset[y_col])
    return p, rho, rp


# ── plot 1: Median predicts_sum box plot ──────────────────────────────────────
for timepoint in [pre_treat_name]:
    subset = df_filtered_nonzero
    sample_medians = subset.groupby(['sample', Col_time, 'Response'])['predicts_sum'].median().dropna().reset_index()
    subset_t = sample_medians[sample_medians[Col_time] == timepoint]
    subset_t = subset_t.copy()
    if subset_t.empty:
        continue

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset_t, x='Response', y='predicts_sum', hue='Response',
                order=response_order, palette=custom_palette, ax=ax)
    jitter = np.random.uniform(-0.15, 0.15, size=len(subset_t))
    x_pos_map = {cat: idx for idx, cat in enumerate(response_order)}
    for j, (_, row) in enumerate(subset_t.iterrows()):
        ax.scatter(x=x_pos_map[row['Response']] + jitter[j], y=row['predicts_sum'],
                   color='red', zorder=3)

    p, rho, rp = _mann_p_spearman(subset_t, 'predicts_sum', 'Response', category_map_binary)
    if p is not None:
        ax.set_ylim(bottom=0.0, top=5.5)
        ax.text(0.5, 5.2, f"p={p:.3f}", ha='center', fontsize=14, color='red')
        ax.text(0.7, 0.05, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
                ha='center', fontsize=18, color='blue', transform=ax.transAxes)

    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel("Median Predicts Sum", fontsize=14)
    ax.set_xlabel("Response", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot.svg", dpi=300, bbox_inches='tight')
    # save the dataframe for reuse
    subset_t.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot.csv",index=True)
    plt.close()


# ── plot 2: Median predicts_rank_sum box plot ─────────────────────────────────
for timepoint in [pre_treat_name]:
    sample_medians = df_filtered_nonzero.groupby(['sample', Col_time, 'Response'])['predicts_rank_sum'].median().dropna().reset_index()
    subset_t = sample_medians[sample_medians[Col_time] == timepoint].copy()
    if subset_t.empty:
        continue

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset_t, x='Response', y='predicts_rank_sum', hue='Response',
                order=response_order, palette=custom_palette, ax=ax)
    jitter = np.random.uniform(-0.1, 0.1, size=len(subset_t))
    x_pos_map = {cat: idx for idx, cat in enumerate(response_order)}
    for j, (_, row) in enumerate(subset_t.iterrows()):
        ax.scatter(x=x_pos_map[row['Response']] + jitter[j], y=row['predicts_rank_sum'],
                   color='red', zorder=3)

    p, rho, rp = _mann_p_spearman(subset_t, 'predicts_rank_sum', 'Response', category_map_binary)
    if p is not None:
        ax.text(0.5, 0.97, f"p={p:.3f}", ha='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.text(0.68, 0.1, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
                ha='center', fontsize=18, color='blue', transform=ax.transAxes)

    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel("Median Predicts Rank Sum", fontsize=14)
    ax.set_xlabel("Response", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot.svg",
                dpi=300, bbox_inches='tight')
    # save the dataframe for reuse
    subset_t.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot.csv",index=True)
    plt.close()


# ── plot 3: predicts_rank_sum_normalized box plot ─────────────────────────────
for timepoint in [pre_treat_name]:
    sample_medians = df_filtered.groupby(['sample', Col_time, 'Response'])['predicts_rank_sum_normalized'].median().dropna().reset_index()
    subset_t = sample_medians[sample_medians[Col_time] == timepoint].copy()
    subset_t['_code'] = subset_t['Response'].map(category_map_binary)
    if subset_t.empty:
        continue

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset_t, x='Response', y='predicts_rank_sum_normalized', hue='Response',
                order=response_order, palette=custom_palette, ax=ax)
    jitter = np.random.uniform(-0.1, 0.1, size=len(subset_t))
    x_pos_map = {cat: idx for idx, cat in enumerate(response_order)}
    for j, (_, row) in enumerate(subset_t.iterrows()):
        ax.scatter(x=x_pos_map[row['Response']] + jitter[j], y=row['predicts_rank_sum_normalized'],
                   color='red', zorder=3)

    nr = subset_t[subset_t['Response'] == 'Non_Response']['predicts_rank_sum_normalized']
    re = subset_t[subset_t['Response'] == 'Response']['predicts_rank_sum_normalized']
    if not nr.empty and not re.empty:
        _, p = mannwhitneyu(nr, re, alternative='two-sided')
        rho, rp = stats.spearmanr(subset_t['_code'], subset_t['predicts_rank_sum_normalized'])
        ax.text(0.5, 0.97, f"p={p:.3f}", ha='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.text(0.69, 0.01, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
                ha='center', fontsize=18, color='blue', transform=ax.transAxes)

    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel("Median Predicts Rank Sum Normalized", fontsize=14)
    ax.set_xlabel("Response", fontsize=14)
    plt.tight_layout()
    subset_t['study_sample'] = f'{str_title}_' + subset_t['sample'].astype(str)
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized.svg",
                dpi=300, bbox_inches='tight')
    # save the dataframe for reuse
    subset_t.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized.csv",index=True)
    plt.close()


# ── plot 4: predict rate box plot (includes 0) ────────────────────────────────
cell_df = df_filtered.copy()
cell_df['predict'] = np.where(cell_df['predicts_sum'] > 4, 1, 0)
predict_rate = cell_df.groupby(['Response', 'sample', Col_time])['predict'].mean().reset_index()
predict_rate = predict_rate[predict_rate[Col_time] == pre_treat_name].copy()
predict_rate.rename(columns={'predict': 'predict_rate'}, inplace=True)
predict_rate.dropna(subset=['predict_rate'], inplace=True)
predict_rate['Response'] = pd.Categorical(predict_rate['Response'], categories=response_order, ordered=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.tick_params(axis='both', labelsize=14)
sns.boxplot(data=predict_rate, x='Response', y='predict_rate', linewidth=1.2,
            order=response_order, hue='Response', palette=custom_palette, ax=ax)
sns.stripplot(data=predict_rate, x='Response', y='predict_rate', color='red',
              alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

nr = predict_rate[predict_rate['Response'] == 'Non_Response']['predict_rate']
re = predict_rate[predict_rate['Response'] == 'Response']['predict_rate']
if not nr.empty and not re.empty:
    _, p = mannwhitneyu(nr, re, alternative='two-sided')
    y_max = max(nr.max(), re.max()) * 1.01
    x_mid = 0.5
    ax.set_ylim(bottom=-0.1)
    ax.text(x_mid, y_max, f"p={p:.3f}", ha='center', fontsize=14, color='red')
    predict_rate['_code'] = predict_rate['Response'].map(category_map_binary)
    rho, rp = stats.spearmanr(predict_rate['_code'], predict_rate['predict_rate'])
    ax.text(0.69, 0.01, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)

ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
ax.set_ylabel("Predict Rate", fontsize=14)
ax.set_xlabel("Response_detail", fontsize=14)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_predictRate_box_plot_include0.svg",
            dpi=300, bbox_inches='tight')
# save the dataframe for reuse
predict_rate.to_csv(f"{out_dir}/{str_title}_{cancer_type}_predictRate_box_plot_include0.csv",index=True)
plt.close()


# ── plot 5: CXCL13 expression rate box plot ───────────────────────────────────
for gene_name in gene_names:
    cxcl13_csv = in_dir + f'{test_string}_{dense_function_flag}_proportion_{gene_name}_positive.csv'
    if os.path.exists(cxcl13_csv):
        print(f"load {gene_name} exp from input_dir")
        proportion_per_sample = pd.read_csv(cxcl13_csv, index_col=0)
    else:
        print(f"Attention: regenerate {gene_name} exp!!!")
        study_log_ad = sc.read_h5ad(study_log_path)
        expr = study_log_ad[:, gene_name].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()
        study_log_ad.obs[f'express_{gene_name}'] = (expr > 0).astype(int)
        df_toexplore = study_log_ad.obs
        df_toexplore_pre = df_toexplore[df_toexplore[Col_time] == pre_treat_name]
        proportion_per_sample = df_toexplore_pre.groupby('sample')[f'express_{gene_name}'].mean()
        proportion_per_sample = proportion_per_sample.reset_index().rename(
            columns={f'express_{gene_name}': f'proportion_{gene_name}_positive'})
        proportion_per_sample = proportion_per_sample.dropna(subset=[f'proportion_{gene_name}_positive'])
        df_meta_ref = df_filtered[['sample', 'Response']].drop_duplicates()
        proportion_per_sample = proportion_per_sample.merge(df_meta_ref, on='sample', how='left')
        proportion_per_sample = proportion_per_sample.drop_duplicates()
        proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'] != 'Unknown_Response'].copy()
        proportion_per_sample['study_sample'] = f'{str_title}_' + proportion_per_sample['sample'].astype(str)
        proportion_per_sample.to_csv(cxcl13_csv)
    
    proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'] != 'Unknown_Response'].copy()
    response_rates     = proportion_per_sample[proportion_per_sample['Response'] == 'Response'][f'proportion_{gene_name}_positive']
    non_response_rates = proportion_per_sample[proportion_per_sample['Response'] == 'Non_Response'][f'proportion_{gene_name}_positive']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=proportion_per_sample, x='Response', y=f'proportion_{gene_name}_positive',
                linewidth=1.2, hue='Response', order=response_order, palette=custom_palette, ax=ax)
    sns.stripplot(data=proportion_per_sample, x='Response', y=f'proportion_{gene_name}_positive',
                  color='red', alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)
    
    if not response_rates.empty and not non_response_rates.empty:
        _, p = mannwhitneyu(response_rates, non_response_rates, alternative='two-sided')
        y_max = max(response_rates.max(), non_response_rates.max()) * 1.01
        x_mid = 0.5
        ax.set_ylim(bottom=-0.08)
        ax.text(x_mid, y_max,
                f"p={p:.3e}" if p < 0.05 else f"p={p:.3f}",
                ha='center', fontsize=14, color='red')
        proportion_per_sample['_code'] = proportion_per_sample['Response'].map(category_map_binary)
        rho, rp = stats.spearmanr(proportion_per_sample['_code'],
                                   proportion_per_sample[f'proportion_{gene_name}_positive'])
        ax.text(0.69, 0.01, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
                ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel(f"{gene_name} Expression Rate", fontsize=14)
    ax.set_xlabel("Response", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Pre_{gene_name}_ExpressionRate_box_plot.svg",
                dpi=300, bbox_inches='tight')
    plt.close()

"""
# ── assembly figure: 1 × 5 panel ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(30, 6))

plot_files = [
    f"{out_dir}/{str_title}_{cancer_type}_Median_box_plot.svg",
    f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot.svg",
    f"{out_dir}/{str_title}_{cancer_type}_predicts_rank_sum_normalized.svg",
    f"{out_dir}/{str_title}_{cancer_type}_Pre_predictRate_box_plot_include0.svg",
    f"{out_dir}/{str_title}_{cancer_type}_Pre_{gene_name}_ExpressionRate_box_plot.svg",
]
titles = ['Median predicts sum', 'Median rank sum', 'Rank sum normalized',
          'Predict rate', f'{gene_name} expression rate']

for ax_i, (ax, title) in enumerate(zip(axes, titles)):
    ax.set_title(f"{str_title}\n{title}", fontsize=12)
    ax.axis('off')

plt.suptitle(f"{str_title} | {cancer_type} — ICB Response Prediction Summary",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.pdf",
            dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.svg",
            dpi=300, bbox_inches='tight')
plt.close()

"""

print(f"Done. Outputs saved to: {out_dir}")
