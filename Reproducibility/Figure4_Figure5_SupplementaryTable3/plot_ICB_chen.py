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
import scikit_posthocs as sp
from pathlib import Path

# ── publication rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'pdf.fonttype': 42, 'figure.dpi': 300, 'savefig.dpi': 300,
})

# ── dataset settings ──────────────────────────────────────────────────────────
test_string       = "chen_CRC_cd8_log1p"
str_title         = 'Chen'
cancer_type       = 'CRC'
dense_function_flag = 'deltarho_30best'
suffix_           = dense_function_flag
Col_time          = 'Treatment'
pre_treat_name    = 'I'
gene_names         = ['CXCL13', 'GZMB']

BASE = '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein'
in_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_0913_combine_6_models/'
study_log_path = f'{BASE}/data_collection/chen_CRC_CancerCell/rawCount_adatas_cd8/chen_CRC_cd8_log1p.h5ad'
out_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_20260630_combine_6_models_Uniform/'
os.makedirs(out_dir, exist_ok=True)

# patient → response mapping (hardcoded from notebook)
pa_str = 'P01\tP02\tP03\tP04\tP05\tP08\tP09\tP11\tP12\tP14\tP15\tP16\tP17\tP18\tP19\tP20\tP21\tP22\tP23\tP24\tP25\tP26'
patient_list = pa_str.split('\t')
resp_str = 'CR\tSD\tPR\tCR\tPR\tCR\tCR\tCR\tCR\tPR\tPR\tSD\tPR\tCR\tPR\tCR\tCR\tCR\tCR\tPR\tSD\tCR'
resp_list = resp_str.split('\t')
response_dict       = dict(zip(patient_list, resp_list))
response_dict_coarse = {'CR': 'Response', 'SD': 'Non_Response', 'PR': 'Response'}

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
custom_palette_detail = {'SD': set2_palette[2], 'PR': set2_palette[3], 'CR': set2_palette[1]}
response_detail_order = ['SD', 'PR', 'CR']
category_map_detail   = {'SD': 1, 'PR': 2, 'CR': 3}

# ── load or recompute cell-level predictions ──────────────────────────────────
csv_path = out_dir + f'{test_string}_{dense_function_flag}_df_filtered.csv'

if os.path.exists(csv_path):
    df_filtered = pd.read_csv(csv_path, index_col=0)

else:
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
    

    # metadata from first checkpoint obs
    pt_path = ck_dic[test_data_list[0]][0]
    pt_dir  = os.path.dirname(pt_path)
    epc     = re.search(r'model_e(\d+).', pt_path).group(1)
    paths   = glob.glob(os.path.join(pt_dir, f"{test_string}_e{epc}-*.h5ad"))
    val_ad  = sc.read_h5ad(min(paths, key=os.path.getmtime))
    df_study_meta = val_ad.obs.copy()
    df_study_meta = df_study_meta[['sample', 'study', 'orig.ident', 'Ident', 'Patient', 'Treatment', 'Tissue']]
    df_study_meta['sample'] = df_study_meta['Ident']
    df_study_meta['Response_detail'] = df_study_meta['Patient'].map(response_dict)
    df_study_meta['Response'] = df_study_meta['Response_detail'].map(response_dict_coarse)
    df_study_meta['Response_detail'] = pd.Categorical(df_study_meta['Response_detail'],
                                                       categories=response_detail_order, ordered=True)
    df_study_meta = df_study_meta[['sample', 'study', Col_time, 'Response', 'Response_detail', 'Tissue']]

    df_meta_score   = df_study_meta.merge(final_dataframe, left_index=True, right_index=True, how='left')
    df_meta_score_f = df_meta_score[df_meta_score['Response'] != 'Unknown_Response']
    df_meta_score_f = df_meta_score_f.dropna(subset=['Response'])

    df_filtered = df_meta_score_f.copy()
    valid_samples_pa = in_dir+f'{test_string}_{dense_function_flag}_valid_samples_forReUse.csv'
    if os.path.exists(valid_samples_pa):
        valid_samples = pd.read_csv(valid_samples_pa, index_col=None)['valid_samples'].tolist()
    else:
        sample_counts = df_filtered['sample'].value_counts()
        valid_samples = sample_counts[sample_counts > 5].index
        valid_samples_list = list(valid_samples)
        pd.DataFrame(valid_samples_list, columns=["valid_samples"]).to_csv(out_dir+f'{test_string}_{dense_function_flag}_valid_samples_forReUse.csv', index=False)

    df_filtered = df_filtered[df_filtered['sample'].isin(valid_samples)]
    # Rank should be updated again after removing some of the samples.
    df_filtered = df_filtered[(df_filtered[Col_time] == 'I') & (df_filtered['Tissue'] == 'Tumor')]
    
    for i in range(6):
        df_filtered[f'model{i+1}_rank'] = df_filtered[f'model{i+1}_score'].rank()
    df_filtered['predicts_rank_sum'] = sum(
        df_filtered[f'model{i+1}_rank'] for i in range(6)
    )
    
    df_filtered['Response_detail'] = pd.Categorical(df_filtered['Response_detail'],
                                                      categories=response_detail_order, ordered=True)
    df_filtered = df_filtered[df_filtered['predicts_sum'] >= 0]
    df_filtered['predicts_rank_sum_normalized'] = df_filtered['predicts_rank_sum'] / (df_filtered.shape[0] * 6)

    df_filtered.to_csv(csv_path)
    # save meta for CXCL13 reuse
    df_filtered[['sample', 'Response', 'Response_detail', Col_time, 'Tissue']].drop_duplicates().to_csv(
        out_dir + f'{test_string}_{dense_function_flag}_df_study_meta_forReUse.csv')


df_filtered_nonzero = df_filtered[df_filtered['predicts_sum'] > 0].copy()



# ── plot 1: Median predicts_sum box plot (Tumor, timepoint I) ─────────────────
sample_medians = df_filtered_nonzero.groupby(
    ['sample', Col_time, 'Response_detail', 'Tissue'])['predicts_sum'].median().dropna().reset_index()
subset = sample_medians[(sample_medians[Col_time] == 'I') & (sample_medians['Tissue'] == 'Tumor')].copy()
subset['Response_detail'] = pd.Categorical(subset['Response_detail'], categories=response_detail_order, ordered=True)
subset['Response_detailCode'] = subset['Response_detail'].map(category_map_detail)

if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col='predicts_sum', group_col='Response_detail')
    print("check dunn_result here:")
    print(dunn_result)
    non_values = subset[subset['Response_detail'] == 'SD']['predicts_sum']
    pr_values  = subset[subset['Response_detail'] == 'PR']['predicts_sum']
    cr_values  = subset[subset['Response_detail'] == 'CR']['predicts_sum']
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    y_offset = 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y='predicts_sum', hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y='predicts_sum', color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    stat, p_kw = kruskal(non_values, pr_values, cr_values)
    if p_kw < 0.05 and dunn_result.loc['SD', 'CR'] < 0.05:
        p_val = dunn_result.loc['SD', 'CR']
        y_max = max(max(non_values.max(), cr_values.max()) * y_offset, 6.5)
        ax.text(1.0, y_max - 0.14, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
        x1, x2 = x_positions['SD'], x_positions['CR']
        ax.plot([x1, x1, x2, x2], [y_max-0.3, y_max-0.2, y_max-0.2, y_max-0.3], lw=0.8, color='black')

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset['predicts_sum'])
    ax.text(0.72, 0.02, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Median Predicts Sum", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot.svg", dpi=300, bbox_inches='tight')
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot.csv",index=True)
    plt.close()


# ── plot 2: Median predicts_rank_sum box plot (Tumor, timepoint I) ────────────
sample_medians = df_filtered_nonzero.groupby(
    ['sample', Col_time, 'Response_detail', 'Tissue'])['predicts_rank_sum'].median().dropna().reset_index()
subset = sample_medians[(sample_medians[Col_time] == 'I') & (sample_medians['Tissue'] == 'Tumor')].copy()
subset['Response_detail'] = pd.Categorical(subset['Response_detail'], categories=response_detail_order, ordered=True)
subset['Response_detailCode'] = subset['Response_detail'].map(category_map_detail)

if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col='predicts_rank_sum', group_col='Response_detail')
    print('check dunn_result:')
    print(dunn_result)
    non_values = subset[subset['Response_detail'] == 'SD']['predicts_rank_sum']
    cr_values  = subset[subset['Response_detail'] == 'CR']['predicts_rank_sum']
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    y_offset = 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y='predicts_rank_sum', hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y='predicts_rank_sum', color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    if dunn_result.loc['SD', 'CR'] < 0.05:
        p_val = dunn_result.loc['SD', 'CR']
        y_max = max(non_values.max(), cr_values.max()) 
        y_max_ = max(non_values.max(), cr_values.max()) * y_offset
        ax.text(1.0, y_max_, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
        x1, x2 = x_positions['SD'], x_positions['CR']
        
        ax.plot([x1, x1, x2, x2], [y_max_-1000, y_max_, y_max_, y_max_-1000], lw=0.8, color='black')

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset['predicts_rank_sum'])
    ax.text(0.65, 0.02, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_ylabel("Median Predicts Rank Sum", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot.svg", dpi=300, bbox_inches='tight')
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot.csv",index=True)
    plt.close()


# ── plot 3: predicts_rank_sum_normalized box plot (Tumor, timepoint I) ────────
y_col = 'predicts_rank_sum_normalized'
sample_medians = df_filtered.groupby(
    ['sample', Col_time, 'Response_detail', 'Tissue'])[y_col].median().dropna().reset_index()
subset = sample_medians[(sample_medians[Col_time] == 'I') & (sample_medians['Tissue'] == 'Tumor')].copy()
subset['Response_detail'] = pd.Categorical(subset['Response_detail'], categories=response_detail_order, ordered=True)
subset['Response_detailCode'] = subset['Response_detail'].map(category_map_detail)

if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col=y_col, group_col='Response_detail')
    print('predicts_rank_sum_normalized dunn_result:')
    print(dunn_result)
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y=y_col, hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y=y_col, color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    groups = subset['Response_detail'].unique()
    group_data = [subset[subset['Response_detail'] == g][y_col] for g in groups]
    stat, p_kw = kruskal(*group_data)
    print(f"predicts_rank_sum_normalized: kruskal pvalue: {p_kw}")
    if p_kw < 0.052:
        x1, x2 = 0, 2
        y_pos = max(subset[y_col]) * 1.0
        p_val = dunn_result.iloc[0, 2]  # SD vs CR
        ax.plot([x1, x1, x2, x2], [y_pos+0.01, y_pos+0.03, y_pos+0.03, y_pos+0.01], lw=0.8, color='black')
        ax.text(1, 0.95, f"p={p_val:.3f}", ha='center', fontsize=11, color='red')
    ax.text(0.5, 0.95, f"Kruskal-Wallis p={p_kw:.3f}", ha='center', fontsize=14, color='red', transform=ax.transAxes)

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset[y_col])
    ax.text(0.66, 0.01, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    #ax.set_ylim(bottom=0.35, top=1.0)
    ax.set_ylabel("Median Predicts Rank Sum Normalized", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    subset['study_sample'] = 'chen_CRC_' + subset['sample'].astype(str)
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized.svg", dpi=300, bbox_inches='tight')
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized.csv",index=True)
    plt.close()


# ── plot 4: predict rate box plot (includes 0, Tumor, timepoint I) ────────────
cell_df = df_filtered.copy()
cell_df['predict'] = np.where(cell_df['predicts_sum'] > 4, 1, 0)
cell_df_filtered = cell_df[(cell_df[Col_time] == 'I') & (cell_df['Tissue'] == 'Tumor')]
predict_rate = cell_df_filtered.groupby(['Response_detail', 'sample'])['predict'].mean().reset_index()
predict_rate.rename(columns={'predict': 'predict_rate'}, inplace=True)
predict_rate.dropna(subset=['predict_rate'], inplace=True)
predict_rate['Response_detail'] = pd.Categorical(predict_rate['Response_detail'],
                                                   categories=response_detail_order, ordered=True)
dunn_result = sp.posthoc_dunn(predict_rate, val_col='predict_rate', group_col='Response_detail')
print('dunn_result')
print(dunn_result)
groups_pr = predict_rate['Response_detail'].unique()
group_data_pr = [predict_rate[predict_rate['Response_detail'] == g]['predict_rate'] for g in groups_pr]
stat, p_kw = kruskal(*group_data_pr)
x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_ylim(bottom=-0.1, top=0.6)
ax.tick_params(axis='both', labelsize=14)
sns.boxplot(data=predict_rate, x='Response_detail', y='predict_rate', linewidth=1.2,
            hue='Response_detail', order=response_detail_order, palette=custom_palette_detail, ax=ax)
sns.stripplot(data=predict_rate, x='Response_detail', y='predict_rate', color='red',
              alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

if p_kw < 0.05:
    x1, x2 = 0, 2
    y_pos = max(predict_rate['predict_rate']) * 1.0
    p_val = dunn_result.iloc[0, 2]  # SD vs CR
    ax.plot([x1, x1, x2, x2], [y_pos+0.01, y_pos+0.02, y_pos+0.02, y_pos+0.01], lw=0.8, color='black')
    ax.text((x1+x2)/2, y_pos+0.03, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')

predict_rate['Response_detailCode'] = predict_rate['Response_detail'].map(category_map_detail)
rho, rp = stats.spearmanr(predict_rate['Response_detailCode'], predict_rate['predict_rate'])
ax.text(0.7, 0.02, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
        ha='center', fontsize=18, color='blue', transform=ax.transAxes)
ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
ax.set_ylabel("Predict Rate", fontsize=14)
ax.set_xlabel("Response_detail", fontsize=14)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_predictRate_box_plot_include0.svg", dpi=300, bbox_inches='tight')
predict_rate.to_csv(f"{out_dir}/{str_title}_{cancer_type}_predictRate_box_plot_include0.csv",index=True)
plt.close()


# ── plot 5: CXCL13 expression rate box plot (Tumor, timepoint I) ──────────────
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
        study_log_ad.obs['express_cxcl13'] = (expr > 0).astype(int)
        # filter to Tumor / Treatment I cells using cell_df index
        tumor_i_idx = list(cell_df[(cell_df[Col_time] == 'I') & (cell_df['Tissue'] == 'Tumor')].index)
        df_toexplore = study_log_ad.obs.loc[study_log_ad.obs.index.isin(tumor_i_idx)].copy()
        df_toexplore['sample'] = df_toexplore['Ident']
        proportion_per_sample = df_toexplore.groupby('sample')['express_cxcl13'].mean()
        proportion_per_sample = proportion_per_sample.reset_index().rename(
            columns={'express_cxcl13': 'proportion_cxcl13_positive'})
        proportion_per_sample = proportion_per_sample.dropna(subset=['proportion_cxcl13_positive'])
        # load saved meta for Response/Response_detail
        meta_csv = out_dir + f'{test_string}_{dense_function_flag}_df_study_meta_forReUse.csv'
        if os.path.exists(meta_csv):
            df_meta_ref = pd.read_csv(meta_csv, index_col=0)[['sample', 'Response', 'Response_detail']].drop_duplicates()
        else:
            df_meta_ref = df_filtered[['sample', 'Response', 'Response_detail']].drop_duplicates()
        proportion_per_sample = proportion_per_sample.merge(df_meta_ref, on='sample', how='left')
        proportion_per_sample = proportion_per_sample.drop_duplicates()
        proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'] != 'Unknown_Response'].copy()
        proportion_per_sample['study_sample'] = 'chen_CRC_' + proportion_per_sample['sample'].astype(str)
        proportion_per_sample.to_csv(cxcl13_csv)
    
    proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'].notna()].copy()
    proportion_per_sample['predict_rate'] = proportion_per_sample[f'proportion_{gene_name}_positive']
    proportion_per_sample['Response_detail'] = pd.Categorical(
        proportion_per_sample['Response_detail'], categories=response_detail_order, ordered=True)
    dunn_result = sp.posthoc_dunn(proportion_per_sample, val_col=f'proportion_{gene_name}_positive',
                                   group_col='Response_detail')
    print('dunn_result')
    print(dunn_result)
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(bottom=-0.08, top=0.62)
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=proportion_per_sample, x='Response_detail', y=f'proportion_{gene_name}_positive',
                linewidth=1.2, hue='Response_detail', order=response_detail_order,
                palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=proportion_per_sample, x='Response_detail', y=f'proportion_{gene_name}_positive',
                  color='red', alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)
    
    x1, x2 = 0, 2
    y_pos = max(proportion_per_sample['predict_rate']) * 1.0
    p_val = dunn_result.loc["SD", "CR"]
    ax.plot([x1, x1, x2, x2], [y_pos+0.01, y_pos+0.03, y_pos+0.03, y_pos+0.01], lw=0.8, color='black')
    ax.text((x1+x2)/2, y_pos+0.04, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
    
    x1, x2 = 1, 2
    y_pos2 = max(proportion_per_sample['predict_rate']) * 1.1
    p_val2 = dunn_result.loc["PR", "CR"]
    ax.plot([x1, x1, x2, x2], [y_pos2+0.01, y_pos2+0.025, y_pos2+0.025, y_pos2+0.01], lw=0.8, color='black')
    ax.text((x1+x2)/2, y_pos2+0.03, f"p={p_val2:.3f}", ha='center', fontsize=14, color='red')
    
    proportion_per_sample['Response_detailCode'] = proportion_per_sample['Response_detail'].map(category_map_detail)
    rho, rp = stats.spearmanr(proportion_per_sample['Response_detailCode'],
                               proportion_per_sample[f'proportion_{gene_name}_positive'])
    ax.text(0.7, 0.01, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel(f"{gene_name} Expression Rate", fontsize=14)
    ax.set_xlabel("Response", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Pre_{gene_name}_ExpressionRate_box_plot_by_responseDetails.svg", dpi=300, bbox_inches='tight')
    plt.close()

"""
# ── assembly figure: 1 × 5 panel ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(30, 6))
titles = ['Median predicts sum', 'Median rank sum', 'Rank sum normalized',
          'Predict rate', f'{gene_name} expression rate']
for ax, title in zip(axes, titles):
    ax.set_title(f"{str_title}\n{title}", fontsize=12)
    ax.axis('off')
plt.suptitle(f"{str_title} | {cancer_type} — ICB Response Prediction Summary (Tumor, Timepoint I)",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.svg", dpi=300, bbox_inches='tight')
plt.close()
"""

print(f"Done. Outputs saved to: {out_dir}")
