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

# ── publication rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'pdf.fonttype': 42, 'figure.dpi': 300, 'savefig.dpi': 300,
})

# ── dataset settings ──────────────────────────────────────────────────────────
test_string       = "chiffelle_15samples_CD8_log1p"
str_title         = 'chiffelle'
cancer_type       = 'melanoma'
dense_function_flag = 'deltarho_30best'
suffix_           = dense_function_flag
Col_time          = 'Timepoint'
pre_treat_name    = 'Pre'
gene_names         = ['CXCL13', 'GZMB']

BASE = '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein'
in_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_0913_combine_6_models/'
study_log_path = (f'{BASE}/data_collection/chiffelle_melanoma_Immunity/'
                  'rawCount_adatas_cd8/chiffelle_15samples_CD8_log1p.h5ad')
out_dir = f'{BASE}/results/classifier/case_study/case_study89_ICB_20260630_combine_6_models_Uniform/'
os.makedirs(out_dir, exist_ok=True)

# patient → response mapping (13 patients, from notebook cell 14-17)
detailed_response_labels = ['SD', 'CR', 'CR', 'SD', 'PD', 'PD', 'PR', 'PR', 'PR', 'SD', 'PD', 'PD', 'PR']
patient_ids = [f'chiffelle_Patient{i+1}' for i in range(len(detailed_response_labels))]
patient_response_dict = dict(zip(patient_ids, detailed_response_labels))
response_mapping = {'CR': 'Response', 'SD': 'Non_Response', 'PR': 'Response', 'PD': 'Non_Response'}
patient_coarse_response_dict = {pid: response_mapping[detail]
                                 for pid, detail in patient_response_dict.items()}

def get_patient_id(sample):
    return '_'.join(sample.split('_')[:2])

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
custom_palette_detail = {'PD': set2_palette[2], 'SD': set2_palette[0],
                          'PR': set2_palette[3], 'CR': set2_palette[1]}
response_detail_order = ['PD', 'SD', 'PR', 'CR']
category_map_detail   = {'PD': 1, 'SD': 2, 'PR': 3, 'CR': 4}

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

    # assign Response and Response_detail from sample name
    df_study_meta['Patient_ID']      = df_study_meta['sample'].apply(get_patient_id)
    df_study_meta['Response']        = df_study_meta['Patient_ID'].map(patient_coarse_response_dict)
    df_study_meta['Response_detail'] = df_study_meta['Patient_ID'].map(patient_response_dict)
    df_study_meta['Timepoint']       = 'Pre'

    df_study_meta = df_study_meta[['sample', 'Patient_ID', 'study', Col_time, 'Response', 'Response_detail']]
    df_study_meta.to_csv(out_dir + f'{test_string}_{dense_function_flag}_meta_forReUse.csv')

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
    df_filtered = df_filtered[df_filtered[Col_time] == pre_treat_name]
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


df_filtered_nonzero = df_filtered[df_filtered['predicts_sum'] > 0].copy()


# ── plot 1: Median predicts_sum box plot ──────────────────────────────────────
sample_medians = df_filtered_nonzero.groupby(
    ['sample', Col_time, 'Response'])['predicts_sum'].median().dropna().reset_index()
sample_medians['Response_detail'] = sample_medians['sample'].apply(get_patient_id).map(patient_response_dict)
sample_medians['Response_detail'] = pd.Categorical(sample_medians['Response_detail'],
                                                     categories=response_detail_order, ordered=True)
sample_medians['Response_detailCode'] = sample_medians['Response_detail'].map(category_map_detail)
subset = sample_medians

if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col='predicts_sum', group_col='Response_detail')
    print("check Median predicts_sum dunn_result:")
    print(dunn_result)
    pd_values  = subset[subset['Response_detail'] == 'PD']['predicts_sum']
    non_values = subset[subset['Response_detail'] == 'SD']['predicts_sum']
    pr_values  = subset[subset['Response_detail'] == 'PR']['predicts_sum']
    cr_values  = subset[subset['Response_detail'] == 'CR']['predicts_sum']
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    y_offset = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y='predicts_sum', hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y='predicts_sum', color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    if not pd_values.empty and not pr_values.empty:
        p_val = dunn_result.loc['PD', 'PR']
        if p_val < 0.05:
            y_max = max(max(pr_values.max(), pd_values.max()) * y_offset, 4.5)
            ax.text(1, y_max + 0.15, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
            ax.plot([x_positions['PD'], x_positions['PD'], x_positions['PR'], x_positions['PR']],
                    [y_max, y_max + 0.1, y_max + 0.1, y_max], lw=0.8, color='black')

    if not pd_values.empty and not cr_values.empty:
        p_val = dunn_result.loc['PD', 'CR']
        if p_val < 0.05:
            y_max = max(max(pd_values.max(), cr_values.max()) * y_offset, 4.25)
            ax.text(1.5, y_max + 0.15, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
            ax.plot([x_positions['PD'], x_positions['PD'], x_positions['CR'], x_positions['CR']],
                    [y_max, y_max + 0.1, y_max + 0.1, y_max], lw=0.8, color='black')

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset['predicts_sum'])
    ax.text(0.7, 0.1, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_ylabel("Median Predicts Sum", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot_ResponseDetails.svg", dpi=300, bbox_inches='tight')
    # save the dataframe for reuse
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_Predicts_Sum_box_plot_ResponseDetails.csv",index=True)
    plt.close()


# ── plot 2: Median predicts_rank_sum box plot ─────────────────────────────────
sample_medians = df_filtered_nonzero.groupby(
    ['sample', Col_time, 'Response'])['predicts_rank_sum'].median().dropna().reset_index()
sample_medians['Response_detail'] = sample_medians['sample'].apply(get_patient_id).map(patient_response_dict)
sample_medians['Response_detail'] = pd.Categorical(sample_medians['Response_detail'],
                                                     categories=response_detail_order, ordered=True)
sample_medians['Response_detailCode'] = sample_medians['Response_detail'].map(category_map_detail)
subset = sample_medians

if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col='predicts_rank_sum', group_col='Response_detail')
    print("check Median predicts_rank_sum dunn_result:")
    print(dunn_result)
    pd_values  = subset[subset['Response_detail'] == 'PD']['predicts_rank_sum']
    non_values = subset[subset['Response_detail'] == 'SD']['predicts_rank_sum']
    pr_values  = subset[subset['Response_detail'] == 'PR']['predicts_rank_sum']
    cr_values  = subset[subset['Response_detail'] == 'CR']['predicts_rank_sum']
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    y_offset = 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y='predicts_rank_sum', hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y='predicts_rank_sum', color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    if not pd_values.empty and not pr_values.empty:
        p_val = dunn_result.loc['PR', 'PD']
        if p_val < 0.05:
            ax.text(1.0, 80500, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
            ax.plot([x_positions['PD'], x_positions['PD'], x_positions['PR'], x_positions['PR']],
                    [80000, 80300, 80300, 80000], lw=0.8, color='black')

    if not non_values.empty and not cr_values.empty:
        p_val = dunn_result.loc['SD', 'CR']
        if p_val < 0.05:
            y_max = max(max(non_values.max(), cr_values.max()) * y_offset, 6)
            ax.text(1.0, y_max + 0.5, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
            ax.plot([x_positions['SD'], x_positions['CR']], [y_max + 0.4, y_max + 0.4], color='black', linewidth=1.5)

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset['predicts_rank_sum'])
    ax.text(0.7, 0.1, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_ylabel("Median Predicts Rank Sum", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot_ResponseDetails.svg", dpi=300, bbox_inches='tight')
    # save the dataframe for reuse
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_box_plot_ResponseDetails.csv",index=True)
    plt.close()


# ── plot 3: predicts_rank_sum_normalized box plot ─────────────────────────────
y_col = 'predicts_rank_sum_normalized'

sample_medians = df_filtered.groupby(
    ['sample', Col_time, 'Response_detail'])[y_col].median().dropna().reset_index()
sample_medians['Response_detail'] = pd.Categorical(sample_medians['Response_detail'],
                                                     categories=response_detail_order, ordered=True)
sample_medians['Response_detailCode'] = sample_medians['Response_detail'].map(category_map_detail)

subset = sample_medians[sample_medians[Col_time] == pre_treat_name].copy()
if not subset.empty:
    dunn_result = sp.posthoc_dunn(subset, val_col=y_col, group_col='Response_detail')
    print("check Median predicts_rank_sum_normalized dunn_result:")
    print(dunn_result)
    pd_values  = subset[subset['Response_detail'] == 'PD'][y_col]
    pr_values  = subset[subset['Response_detail'] == 'PR'][y_col]
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    y_offset = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=subset, x='Response_detail', y=y_col, hue='Response_detail',
                order=response_detail_order, palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=subset, x='Response_detail', y=y_col, color='red',
                  alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

    if not pd_values.empty and not pr_values.empty:
        p_val = dunn_result.loc['PR', 'PD']
        if p_val < 0.05:
            y_max = max(max(subset[y_col]) * y_offset, 0.78)
            ax.text(1.0, y_max + 0.03, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
            ax.plot([x_positions['PD'], x_positions['PD'], x_positions['PR'], x_positions['PR']],
                    [y_max, y_max + 0.02, y_max + 0.02, y_max], lw=0.8, color='black')

    rho, rp = stats.spearmanr(subset['Response_detailCode'], subset[y_col])
    ax.text(0.7, 0.07, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3e}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_ylabel("Median Predicts Rank Sum Normalized", fontsize=14)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
    plt.tight_layout()
    subset['study_sample'] = 'chiffelle_15samples_' + subset['sample'].astype(str)
    plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized_box_plot_byresponsedetail.svg", dpi=300, bbox_inches='tight')
    subset.to_csv(f"{out_dir}/{str_title}_{cancer_type}_Median_predicts_rank_sum_normalized_box_plot_byresponsedetail.csv",index=True)
    plt.close()


# ── plot 4: predict rate box plot (includes 0) ───────────────────────────────
cell_df = df_filtered.copy()
cell_df['predict'] = np.where(cell_df['predicts_sum'] > 4, 1, 0)
predict_rate = cell_df.groupby(['Response_detail', 'sample'])['predict'].mean().reset_index()
predict_rate.rename(columns={'predict': 'predict_rate'}, inplace=True)
predict_rate.dropna(subset=['predict_rate'], inplace=True)
predict_rate['Response_detail'] = pd.Categorical(predict_rate['Response_detail'],
                                                   categories=response_detail_order, ordered=True)
dunn_result = sp.posthoc_dunn(predict_rate, val_col='predict_rate', group_col='Response_detail')
print("check predict rate dunn_result:")
print(dunn_result)

groups_pr = predict_rate['Response_detail'].unique()
group_data_pr = [predict_rate[predict_rate['Response_detail'] == g]['predict_rate'] for g in groups_pr]
stat, p_kw = kruskal(*group_data_pr)
x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}

fig, ax = plt.subplots(figsize=(6, 6))
ax.tick_params(axis='both', labelsize=14)
sns.boxplot(data=predict_rate, x='Response_detail', y='predict_rate', linewidth=1.2,
            hue='Response_detail', order=response_detail_order, palette=custom_palette_detail, ax=ax)
sns.stripplot(data=predict_rate, x='Response_detail', y='predict_rate', color='red',
              alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)

# PD→PR bracket (outer, higher)
x1, x2 = 0, 2
y_pos = max(predict_rate['predict_rate']) * 1.1
p_val = dunn_result.loc['PD', 'PR']
ax.plot([x1, x1, x2, x2], [y_pos + 0.01, y_pos + 0.025, y_pos + 0.025, y_pos + 0.01], lw=0.8, color='black')
ax.text((x1 + x2) / 2, y_pos + 0.03, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')

# SD→PR bracket (inner, lower)
x1, x2 = 1, 2
y_pos = max(predict_rate['predict_rate']) * 1.0
p_val = dunn_result.loc['SD', 'PR']
ax.plot([x1, x1, x2, x2], [y_pos + 0.01, y_pos + 0.03, y_pos + 0.03, y_pos + 0.01], lw=0.8, color='black')
ax.text((x1 + x2) / 2, y_pos + 0.04, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')

predict_rate['Response_detailCode'] = predict_rate['Response_detail'].map(category_map_detail)
rho, rp = stats.spearmanr(predict_rate['Response_detailCode'], predict_rate['predict_rate'])
ax.text(0.7, 0.02, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
        ha='center', fontsize=18, color='blue', transform=ax.transAxes)
ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
ax.set_ylabel("Predict Rate", fontsize=14)
ax.set_xlabel("Response_detail", fontsize=14)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_predictRate_box_plot_include0.svg", dpi=300, bbox_inches='tight')
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
        study_log_ad.obs['express_cxcl13'] = (expr > 0).astype(int)
    
        df_toexplore = study_log_ad.obs
        proportion_per_sample = df_toexplore.groupby('sample')['express_cxcl13'].mean()
        proportion_per_sample = proportion_per_sample.reset_index().rename(
            columns={'express_cxcl13': f'proportion_{gene_name}_positive'})
        proportion_per_sample = proportion_per_sample.dropna(subset=[f'proportion_{gene_name}_positive'])
    
        # load saved meta for Response/Response_detail
        meta_csv = out_dir + f'{test_string}_{dense_function_flag}_meta_forReUse.csv'
        if os.path.exists(meta_csv):
            df_meta_ref = pd.read_csv(meta_csv, index_col=0)[['sample', 'Response', 'Response_detail']].drop_duplicates()
        else:
            df_meta_ref = df_filtered[['sample', 'Response', 'Response_detail']].drop_duplicates()
    
        proportion_per_sample = proportion_per_sample.merge(df_meta_ref, on='sample', how='left')
        proportion_per_sample = proportion_per_sample.drop_duplicates()
        proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'] != 'Unknown_Response'].copy()
        proportion_per_sample['study_sample'] = 'chiffelle_15samples_' + proportion_per_sample['sample'].astype(str)
        proportion_per_sample.to_csv(cxcl13_csv)
    
    proportion_per_sample = proportion_per_sample[proportion_per_sample['Response'].notna()].copy()
    proportion_per_sample['predict_rate'] = proportion_per_sample[f'proportion_{gene_name}_positive']
    proportion_per_sample['Response_detail'] = proportion_per_sample['sample'].apply(get_patient_id).map(patient_response_dict)
    proportion_per_sample['Response_detail'] = pd.Categorical(proportion_per_sample['Response_detail'],
                                                         categories=response_detail_order, ordered=True)
    proportion_per_sample['Response_detailCode'] = proportion_per_sample['Response_detail'].map(category_map_detail)

    dunn_result = sp.posthoc_dunn(proportion_per_sample, val_col=f'proportion_{gene_name}_positive',
                                   group_col='Response_detail')
    print(f"check {gene_name} dunn_result:")
    print(dunn_result)
    x_positions = {cat: idx for idx, cat in enumerate(response_detail_order)}
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='both', labelsize=14)
    sns.boxplot(data=proportion_per_sample, x='Response_detail', y=f'proportion_{gene_name}_positive',
                linewidth=1.2, hue='Response_detail', order=response_detail_order,
                palette=custom_palette_detail, ax=ax)
    sns.stripplot(data=proportion_per_sample, x='Response_detail', y=f'proportion_{gene_name}_positive',
                  color='red', alpha=0.6, jitter=True, dodge=True, size=6, ax=ax)
    
    # PD→PR bracket
    x1, x2 = 0, 2
    y_pos = max(proportion_per_sample['predict_rate']) * 1.0
    p_val = dunn_result.loc['PD', 'PR']
    ax.plot([x1, x1, x2, x2], [y_pos + 0.01, y_pos + 0.03, y_pos + 0.03, y_pos + 0.01], lw=0.8, color='black')
    ax.text((x1 + x2) / 2, y_pos + 0.04, f"p={p_val:.3f}", ha='center', fontsize=14, color='red')
    

    rho, rp = stats.spearmanr(proportion_per_sample['Response_detailCode'],
                               proportion_per_sample[f'proportion_{gene_name}_positive'])
    ax.text(0.71, 0.065, f"Spearman's rho: {rho:.3f}\n p-value: {rp:.3f}",
            ha='center', fontsize=18, color='blue', transform=ax.transAxes)
    ax.set_title(f'{str_title} | {cancer_type}', fontsize=14)
    ax.set_ylabel(f"{gene_name} Expression Rate", fontsize=14)
    ax.set_xlabel("Response_detail", fontsize=14)
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
plt.suptitle(f"{str_title} | {cancer_type} — ICB Response Prediction Summary",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/{str_title}_{cancer_type}_assembly_panel.svg", dpi=300, bbox_inches='tight')
plt.close()
"""


print(f"Done. Outputs saved to: {out_dir}")
