"""
plot_ICB_all_panels.py

Loads pre-computed per-study CSVs and generates uniform 1×5 publication panels,
one output file per metric.

Panels produced:
  panel_5studies_predictRate.{svg,pdf}
  panel_5studies_medianPredictSum.{svg,pdf}
  panel_5studies_rankSumNorm.{svg,pdf}
  panel_5studies_CXCL13ExprRate.{svg,pdf}
  panel_5studies_GZMBExprRate.{svg,pdf}

P-value annotation rule:
  - Binary (2 groups): Mann-Whitney, annotate only if p < 0.05.
  - Multi-category (≥3 groups): Dunn post-hoc, annotate only pairs with p < 0.05;
    if no pair is significant, fall back to Kruskal-Wallis, annotated only if p < 0.05.
  Brackets are stacked shortest-span-first so they nest without overlap.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import ttest_ind
import scikit_posthocs as sp
import jenkspy

# ── Global typography / style ─────────────────────────────────────────────────
FS_TITLE  = 16
FS_LABEL  = 14
FS_TICK   = 13
FS_PVAL   = 15
FS_SPEAR  = 15
LW_BOX    = 1.2
DOT_SIZE  = 6
DOT_COLOR = '#C0392B'
DOT_ALPHA = 0.75
SPEAR_COLOR = '#1A5276'

plt.rcParams.update({
    'font.size':            FS_TICK,
    'axes.labelsize':       FS_LABEL,
    'axes.titlesize':       FS_TITLE,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            False,
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
    'figure.dpi':           300,
    'savefig.dpi':          300,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein'
IN_DIR  = f'{BASE}/results/classifier/case_study/case_study89_ICB_0913_combine_6_models/'
OUT_DIR = (f'{BASE}/results/classifier/case_study/'
           'case_study89_ICB_20260630_combine_6_models_Uniform/')
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_FLAG = 'deltarho_30best'

# test_string used in each study's raw CSV filename (for gene expression CSVs)
TEST_STRINGS = dict(
    bassez1   = 'Bassez_cohort1_cd8_QCed_log1p',
    bassez2   = 'Bassez_cohort2_CD8_log1p',
    cappuyns  = 'Cappuyns_tumor_QC_CD8_log1p',
    chen      = 'chen_CRC_cd8_log1p',
    chiffelle = 'chiffelle_15samples_CD8_log1p',
)

# ── Colour palettes ───────────────────────────────────────────────────────────
_s2 = sns.color_palette('Set2', n_colors=8)
PAL_BINARY = {'Non_Response': _s2[2], 'Response': _s2[1]}
PAL_3CAT   = {'SD': _s2[2], 'PR': _s2[3], 'CR': _s2[1]}
PAL_4CAT   = {'PD': _s2[2], 'SD': _s2[0], 'PR': _s2[3], 'CR': _s2[1]}

# ── Study-specific post-load transforms ──────────────────────────────────────

def _chiffelle_add_response_detail(data):
    """Re-derive Response_detail for chiffelle from sample name.

    The gene-expression CSVs are saved in in_dir before Response_detail is
    added (the original script adds it at plot time from sample name).
    """
    if 'Response_detail' in data.columns:
        return data
    _labels = ['SD', 'CR', 'CR', 'SD', 'PD', 'PD', 'PR', 'PR', 'PR', 'SD', 'PD', 'PD', 'PR']
    _pids   = [f'chiffelle_Patient{i+1}' for i in range(len(_labels))]
    _pmap   = dict(zip(_pids, _labels))
    data = data.copy()
    data['Response_detail'] = (
        data['sample']
        .apply(lambda s: '_'.join(str(s).split('_')[:2]))
        .map(_pmap)
    )
    return data


# ── Per-study configuration ───────────────────────────────────────────────────
# binary=True  → single Mann-Whitney, always annotate.
# binary=False → all pairwise Dunn, annotate only p < 0.05.
STUDIES = [
    dict(
        key='bassez1', title='Bassez cohort1', cancer='BC',
        group_col='Response',
        order=['Non_Response', 'Response'],
        cat_map={'Non_Response': 1, 'Response': 2},
        palette=PAL_BINARY,
        binary=True,
        post_load=None,
        extra_filter=None,       # df_filtered already pre-filtered to Pre
    ),
    dict(
        key='bassez2', title='Bassez cohort2', cancer='BC',
        group_col='Response',
        order=['Non_Response', 'Response'],
        cat_map={'Non_Response': 1, 'Response': 2},
        palette=PAL_BINARY,
        binary=True,
        post_load=None,
        extra_filter=None,
    ),
    dict(
        key='cappuyns', title='Cappuyns', cancer='aHCC',
        group_col='Response',
        order=['Non_Response', 'Response'],
        cat_map={'Non_Response': 1, 'Response': 2},
        palette=PAL_BINARY,
        binary=True,
        post_load=None,
        extra_filter=lambda df: df[df['Week'] == 'W0'].copy(),  # multiple timepoints saved
    ),
    dict(
        key='chen', title='Chen', cancer='CRC',
        group_col='Response_detail',
        order=['SD', 'PR', 'CR'],
        cat_map={'SD': 1, 'PR': 2, 'CR': 3},
        palette=PAL_3CAT,
        binary=False,
        post_load=None,
        extra_filter=None,       # df_filtered already filtered to Tumor + Treatment I
    ),
    dict(
        key='chiffelle', title='Chiffelle', cancer='melanoma',
        group_col='Response_detail',
        order=['PD', 'SD', 'PR', 'CR'],
        cat_map={'PD': 1, 'SD': 2, 'PR': 3, 'CR': 4},
        palette=PAL_4CAT,
        binary=False,
        post_load=_chiffelle_add_response_detail,
        extra_filter=None,
    ),
]

# ── Metric → CSV map ──────────────────────────────────────────────────────────
# Gene expression CSVs live in IN_DIR; use gene_csv() to resolve.
def gene_csv(study_key, gene_name):
    fname = f'{TEST_STRINGS[study_key]}_{DENSE_FLAG}_proportion_{gene_name}_positive.csv'
    return os.path.join(IN_DIR, fname)

METRICS = {
    'predictRate': dict(
        y_col='predict_rate',
        y_label='Predicted Reactive Fraction',
        files=dict(
            bassez1   = os.path.join(OUT_DIR, 'Bassez_cohort1_BC_predictRate_box_plot_include0.csv'),
            bassez2   = os.path.join(OUT_DIR, 'Bassez_cohort2_BC_predictRate_box_plot_include0.csv'),
            cappuyns  = os.path.join(OUT_DIR, 'Cappuyns_aHCC_predictRate_box_plot_include0.csv'),
            chen      = os.path.join(OUT_DIR, 'Chen_CRC_predictRate_box_plot_include0.csv'),
            chiffelle = os.path.join(OUT_DIR, 'chiffelle_melanoma_predictRate_box_plot_include0.csv'),
        ),
    ),
    'medianPredictSum': dict(
        y_col='predicts_sum',
        y_label='Median Predicts Sum',
        files=dict(
            bassez1   = os.path.join(OUT_DIR, 'Bassez_cohort1_BC_Median_Predicts_Sum_box_plot.csv'),
            bassez2   = os.path.join(OUT_DIR, 'Bassez_cohort2_BC_Median_Predicts_Sum_box_plot.csv'),
            cappuyns  = os.path.join(OUT_DIR, 'Cappuyns_aHCC_Median_Predicts_Sum_box_plot.csv'),
            chen      = os.path.join(OUT_DIR, 'Chen_CRC_Median_Predicts_Sum_box_plot.csv'),
            chiffelle = os.path.join(OUT_DIR,
                'chiffelle_melanoma_Median_Predicts_Sum_box_plot_ResponseDetails.csv'),
        ),
    ),
    'rankSumNorm': dict(
        y_col='predicts_rank_sum_normalized',
        y_label='Median Predicts Rank Sum Normalized',
        files=dict(
            bassez1   = os.path.join(OUT_DIR,
                'Bassez_cohort1_BC_Median_predicts_rank_sum_normalized.csv'),
            bassez2   = os.path.join(OUT_DIR,
                'Bassez_cohort2_BC_Median_predicts_rank_sum_normalized.csv'),
            cappuyns  = os.path.join(OUT_DIR,
                'Cappuyns_aHCC_Median_predicts_rank_sum_normalized.csv'),
            chen      = os.path.join(OUT_DIR,
                'Chen_CRC_Median_predicts_rank_sum_normalized.csv'),
            chiffelle = os.path.join(OUT_DIR,
                'chiffelle_melanoma_Median_predicts_rank_sum_normalized_box_plot_byresponsedetail.csv'),
        ),
    ),
    'CXCL13ExprRate': dict(
        y_col='proportion_CXCL13_positive',
        y_label='CXCL13 Expression Rate',
        gene='CXCL13',   # → resolved via gene_csv()
    ),
    'GZMBExprRate': dict(
        y_col='proportion_GZMB_positive',
        y_label='GZMB Expression Rate',
        gene='GZMB',
    ),
}


# ── Annotation helpers ────────────────────────────────────────────────────────

def _pval_str(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _compute_sig_pairs(data, y_col, cfg):
    """Return list of (x1, x2, p_val) sorted by x-span, filtered p < 0.05.

    Binary: Mann-Whitney; returns the pair only if p < 0.05.
    Multi-cat: Dunn post-hoc, only p < 0.05.
    """
    group_col = cfg['group_col']
    order     = cfg['order']
    x_pos     = {cat: i for i, cat in enumerate(order)}
    present   = [o for o in order if o in data[group_col].values]

    if cfg['binary']:
        if len(present) < 2:
            return []
        g0 = data[data[group_col] == present[0]][y_col].dropna()
        g1 = data[data[group_col] == present[1]][y_col].dropna()
        if g0.empty or g1.empty:
            return []
        _, p = mannwhitneyu(g0, g1, alternative='two-sided')

        #_, p = ttest_ind(g0, g1, alternative='two-sided')
        if p < 0.05:
            return [(x_pos[present[0]], x_pos[present[1]], p)]
        return []

    # Multi-category: all pairwise Dunn, filter p < 0.05
    df_clean = data.dropna(subset=[y_col, group_col])
    if df_clean[group_col].nunique() < 2:
        return []
    try:
        dunn = sp.posthoc_dunn(df_clean, val_col=y_col, group_col=group_col)
    except Exception as e:
        print(f"  Dunn failed: {e}")
        return []

    pairs = []
    cats = [c for c in present if c in dunn.index]
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            if b not in dunn.columns:
                continue
            p = dunn.loc[a, b]
            if p < 0.05:
                pairs.append((abs(x_pos[b] - x_pos[a]), x_pos[a], x_pos[b], p))

    pairs.sort()  # sort by x-span so shorter brackets sit at lower levels
    return [(xa, xb, p) for (_, xa, xb, p) in pairs]


def _draw_brackets(ax, sig_pairs, y_data_max, y_range,
                   gap_frac=0.06, step_frac=0.13, bar_frac=0.025):
    """Draw stacked significance brackets in data coordinates."""
    for level, (xa, xb, p_val) in enumerate(sig_pairs):
        y_bot = y_data_max + y_range * (gap_frac + level * step_frac)
        y_top = y_bot + y_range * bar_frac
        ax.plot([xa, xa, xb, xb], [y_bot, y_top, y_top, y_bot],
                lw=1.0, color='black', clip_on=False)
        ax.text((xa + xb) / 2, y_top + y_range * 0.01, _pval_str(p_val),
                ha='center', va='bottom', fontsize=FS_PVAL, color='red',
                fontweight='bold', clip_on=False)


# ── RankMin-Study computation ─────────────────────────────────────────────────

def _jenks_thresh(scores):
    breaks = jenkspy.jenks_breaks(scores.tolist(), n_classes=2)
    return breaks[1]  # natural break between class 1 and class 2


def _compute_rankmin_predict_rate(cfg):
    """Load cell-level df_filtered, compute rank_min across 6 model scores,
    apply per-sample Jenks threshold, return per-sample predict_rate_rankmin.

    Result is cached in OUT_DIR so subsequent runs are instant.
    """
    key       = cfg['key']
    cache_path = os.path.join(OUT_DIR, f'{key}_rankMinStudy_predict_rate.csv')

    if os.path.exists(cache_path):
        print(f"  [{key}] loading cached rankMinStudy")
        return pd.read_csv(cache_path, index_col=0)

    # ── Load cell-level predictions ───────────────────────────────────────
    df_filt_path = os.path.join(
        OUT_DIR, f'{TEST_STRINGS[key]}_{DENSE_FLAG}_df_filtered.csv')
    if not os.path.exists(df_filt_path):
        print(f"  [{key}] df_filtered CSV not found: {df_filt_path}")
        return None

    df = pd.read_csv(df_filt_path, index_col=0)

    if cfg['extra_filter'] is not None:
        df = cfg['extra_filter'](df)

    if df.empty:
        print(f"  [{key}] empty after extra_filter")
        return None

    # ── Rank each model's scores, then take per-cell minimum rank ─────────
    score_cols = [f'model{i+1}_score' for i in range(6)]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        print(f"  [{key}] missing score columns: {missing}")
        return None

    score_mat = df[score_cols].astype(float)
    rank_mat  = score_mat.rank(method='average', ascending=True)
    df = df.copy()
    df['rank_min_study'] = rank_mat.min(axis=1)

    # ── Per-sample Jenks prediction ───────────────────────────────────────
    group_col = cfg['group_col']
    pred_rows = []
    for samp, grp in df.groupby('sample'):
        scores = grp['rank_min_study'].dropna().values.astype(float)
        if len(scores) < 2 or np.unique(scores).size < 2:
            # All identical → no natural break → predict all negative
            pred = np.zeros(len(grp), dtype=int)
        else:
            try:
                thresh = _jenks_thresh(scores)
                pred   = (scores > thresh).astype(int)
            except Exception as e:
                print(f"    [{key}] skip {samp}: jenks failed ({e})")
                pred = np.zeros(len(grp), dtype=int)
        tmp = grp.copy()
        tmp['_pred_rankmin'] = pred
        pred_rows.append(tmp)

    if not pred_rows:
        return None

    df_pred = pd.concat(pred_rows)

    # ── Aggregate to per-sample predict rate ──────────────────────────────
    agg_cols = ['sample', group_col]
    rate_df  = (df_pred.groupby(agg_cols)['_pred_rankmin']
                .mean().reset_index()
                .rename(columns={'_pred_rankmin': 'predict_rate_rankmin'}))
    rate_df.dropna(subset=['predict_rate_rankmin'], inplace=True)

    order = cfg['order']
    rate_df[group_col] = pd.Categorical(
        rate_df[group_col], categories=order, ordered=True)

    rate_df.to_csv(cache_path)
    print(f"  [{key}] computed and cached rankMinStudy → {os.path.basename(cache_path)}")
    return rate_df


# Register rankMinStudy after its compute_fn is defined
METRICS['rankMinStudy'] = dict(
    y_col='predict_rate_rankmin',
    y_label='Predicted Reactive Fraction (Rank-Min)',
    compute_fn=_compute_rankmin_predict_rate,
)

# ── Core panel plotting function ──────────────────────────────────────────────

def plot_panel(metric_key, save_stem):
    meta    = METRICS[metric_key]
    y_col   = meta['y_col']
    y_label = meta['y_label']

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    fig.subplots_adjust(wspace=0.25)

    for ax, cfg in zip(axes, STUDIES):
        key = cfg['key']

        title_str = f"{cfg['title']} | {cfg['cancer']}"

        # ── Resolve / load data ───────────────────────────────────────────
        if 'compute_fn' in meta:
            # Computed on-the-fly (e.g. rankMinStudy); result cached to CSV
            data = meta['compute_fn'](cfg)
            if data is None:
                ax.text(0.5, 0.5, 'Computation failed\n(check console)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.set_title(title_str, fontsize=FS_TITLE, fontweight='bold', pad=8)
                continue
        else:
            # Load from pre-computed CSV
            if 'gene' in meta:
                csv_path = gene_csv(key, meta['gene'])
            else:
                csv_path = meta['files'][key]

            if not os.path.exists(csv_path):
                ax.text(0.5, 0.5, f'CSV not found:\n{os.path.basename(csv_path)}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.set_title(title_str, fontsize=FS_TITLE, fontweight='bold', pad=8)
                continue

            data = pd.read_csv(csv_path, index_col=0)
            if cfg['post_load'] is not None:
                data = cfg['post_load'](data)

        group_col = cfg['group_col']

        # Guard: group_col must exist
        if group_col not in data.columns:
            ax.text(0.5, 0.5, f'Column "{group_col}" not found',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.set_title(title_str, fontsize=FS_TITLE, fontweight='bold', pad=8)
            continue

        order   = [o for o in cfg['order'] if o in data[group_col].values]
        palette = {k: v for k, v in cfg['palette'].items() if k in order}
        data[group_col] = pd.Categorical(data[group_col],
                                          categories=order, ordered=True)
        data = data.dropna(subset=[y_col, group_col])

        if data.empty:
            ax.text(0.5, 0.5, 'No data after filtering',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.set_title(title_str, fontsize=FS_TITLE, fontweight='bold', pad=8)
            continue

        # ── Compute significance before plotting (for ylim) ───────────────
        sig_pairs = _compute_sig_pairs(data, y_col, cfg)
        n_br = len(sig_pairs)

        # ── Y-axis limits ─────────────────────────────────────────────────
        y_data_min = data[y_col].min()
        y_data_max = data[y_col].max()
        y_range    = max(y_data_max - y_data_min, 1e-9)

        # Bottom: 35% below data min → room for Spearman annotation
        # Top: headroom proportional to number of brackets
        y_bot = y_data_min - y_range * 0.35
        y_top = y_data_max + y_range * (0.06 + 0.13 * max(n_br, 1) + 0.14)
        ax.set_ylim(y_bot, y_top)

        # ── Boxplot + stripplot ───────────────────────────────────────────
        sns.boxplot(data=data, x=group_col, y=y_col, hue=group_col,
                    order=order, palette=palette,
                    linewidth=LW_BOX, fliersize=0, legend=False, ax=ax)
        sns.stripplot(data=data, x=group_col, y=y_col, order=order,
                      color=DOT_COLOR, alpha=DOT_ALPHA, jitter=True,
                      size=DOT_SIZE, ax=ax)

        # ── Significance annotations ──────────────────────────────────────
        _draw_brackets(ax, sig_pairs, y_data_max, y_range)

        # Multi-category: when no pairwise Dunn pairs pass p < 0.05,
        # fall back to showing the Kruskal-Wallis p at top-centre.
        if not cfg['binary'] and len(sig_pairs) == 0:
            groups_data = [
                data[data[group_col] == g][y_col].dropna()
                for g in order if g in data[group_col].values
            ]
            groups_data = [g for g in groups_data if not g.empty]
            if len(groups_data) >= 2:
                _, p_kw = kruskal(*groups_data)
                if p_kw < 0.05:
                    ax.text(0.5, 0.90,
                            f"Kruskal-Wallis {_pval_str(p_kw)}",
                            ha='center', va='top', fontsize=FS_PVAL, fontweight='bold',
                            color='red', transform=ax.transAxes)

        # ── Spearman inset — bottom-left (avoids strip dots on right) ─────
        data = data.copy()
        data['_code'] = data[group_col].map(cfg['cat_map'])
        valid = data[['_code', y_col]].dropna()
        if len(valid) >= 3:
            rho, rp = stats.spearmanr(valid['_code'], valid[y_col])
            ax.text(0.97, 0.04,
                    f"Spearman's ρ: {rho:.3f}\np-value: {_pval_str(rp)}",
                    ha='right', va='bottom', fontsize=FS_SPEAR,
                    color=SPEAR_COLOR, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec='none', alpha=0.75))

        # ── Labels & ticks ────────────────────────────────────────────────
        ax.set_title(title_str, fontsize=FS_TITLE, fontweight='bold', pad=5)
        ax.set_xlabel('Response', fontsize=FS_LABEL)
        ax.set_ylabel(y_label if ax is axes[0] else '', fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.set_xticklabels([t.replace('_', '\n') for t in order],
                            fontsize=FS_TICK)
        ax.set_axisbelow(True)

    plt.savefig(os.path.join(OUT_DIR, f'{save_stem}.svg'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, f'{save_stem}.pdf'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved → {save_stem}.svg / .pdf")


# ── Run all panels ────────────────────────────────────────────────────────────

plot_panel('predictRate',      'panel_5studies_predictRate')
plot_panel('medianPredictSum', 'panel_5studies_medianPredictSum')
plot_panel('rankSumNorm',      'panel_5studies_rankSumNorm')
plot_panel('CXCL13ExprRate',   'panel_5studies_CXCL13ExprRate')
plot_panel('GZMBExprRate',     'panel_5studies_GZMBExprRate')
plot_panel('rankMinStudy',     'panel_5studies_rankMinStudy')

print(f"\nAll panels saved to:\n  {OUT_DIR}")
