import tarfile
import warnings
from glob import glob

import anndata
import matplotlib.pyplot as plt
import muon as mu
import pandas as pd
import scanpy as sc
import scirpy as ir
import numpy as np

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = 2  # verbosity: errors (0), warnings (1), info (2), hints (3)

import os 
os.getcwd()

import glob
#input_dir = "/mnt/pixstor/dbllab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/no_anno_10x_data/TenX_10k_PBMC_without_Intron/"
#TCR_FILE_PATH = glob.glob(input_dir+"*vdj_t_*_contig_annotations.csv")[0]
#RNA_FILE_PATH = glob.glob(input_dir+"*feature_bc_matrix.h5")[0]



def prep_TCR_RNA (input_dir, n_genes_cut, pct_mt_cut, use_input_RNA=False, RNA_FILE_Input=None, use_input_TCR_FILE=False, TCR_FILE_Input=None, use_gene_subsets=False, geneList=None):
    if len(glob.glob(input_dir+"*feature_bc_matrix.h5")) !=0:
        RNA_FILE_PATH = glob.glob(input_dir+"*feature_bc_matrix.h5")[0]
    if use_input_RNA:
        RNA_FILE_PATH = input_dir+RNA_FILE_Input
        print(f"User specified: use this h5ad file {RNA_FILE_PATH}")
    
    if len(glob.glob(input_dir+"*_t_*_contig_annotations.csv")) != 0:
        TCR_FILE_PATH = glob.glob(input_dir+"*_t_*_contig_annotations.csv")[0]
        print(f"Default: use this contig file {TCR_FILE_PATH}")
    if use_input_TCR_FILE:
        TCR_FILE_PATH = input_dir+TCR_FILE_Input
        print(f"User specified: use this contig file {TCR_FILE_PATH}")
        
    # from https://scirpy.scverse.org/en/latest/tutorials/tutorial_io.html
    # Read 10x data
    
    # Load the associated transcriptomics data
    if use_input_RNA:
        adata = sc.read_h5ad(RNA_FILE_PATH)
    else:
        adata = sc.read_10x_h5(RNA_FILE_PATH)
    adata.var_names_make_unique()
    
    n_cell_orin_adata = adata.shape[0]
    print(adata.shape)
    
    adata.var['mt'] =adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    adata
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    #sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    #sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    
    adata = adata[adata.obs.n_genes_by_counts < n_genes_cut, :]
    adata = adata[adata.obs.pct_counts_mt < pct_mt_cut, :]
    
    print(f"{adata.shape[0]} out of {n_cell_orin_adata} passed the QC step.")
    
    #sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    #sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    
    # Load the TCR data
    adata_tcr = ir.io.read_10x_vdj(TCR_FILE_PATH)
    
    # Creating chain indices
    ir.pp.index_chains(adata_tcr)
    ir.tl.chain_qc(adata_tcr)
    
    #_ = ir.pl.group_abundance(
    #    adata_tcr, groupby="receptor_subtype"
    #)
    
    #
    #_ = ir.pl.group_abundance(
    #    adata_tcr, groupby="chain_pairing"
    #)
    
    print(
        "Fraction of cells with more than one pair of TCRs: {:.2f}".format(
            np.sum(
                adata_tcr.obs["chain_pairing"].isin(
                    ["extra VJ", "extra VDJ", "two full chains", "multichain"]
                )
            )
            / adata_tcr.n_obs
        )
    )
    
    print(
        "Fraction of cells with more than one pair of TCRs: {:.2f}".format(
            np.sum(
                adata_tcr.obs["chain_pairing"].isin(
                    ["orphan VDJ", "orphan VJ", "two full chains"]
                )
            )
            / adata_tcr.n_obs
        )
    )
    
    #mu.pp.filter_obs(adata_tcr, "chain_pairing", lambda x: ~np.isin(x, ["orphan VDJ", "orphan VJ", "two full chains"]))
    
    mu.pp.filter_obs(
        adata_tcr, "chain_pairing", lambda x: ~np.isin(x, ["orphan VDJ", "orphan VJ", "multichain"])
    )
    
    mu.pp.filter_obs(adata_tcr, "receptor_subtype", lambda x: x == "TRA+TRB")
    
    print(adata_tcr.obs['receptor_type'][:5])
    print(adata_tcr.obs['receptor_subtype'][:5])
    
    #_ = ir.pl.group_abundance(
    #    adata_tcr, groupby="chain_pairing"
    #)
    #adata_tcr.obsm["airr"]
    
    cdr3_seq_df = ir.get.airr(adata_tcr, ["cdr3_aa"], ('VDJ_1'))
    
    print(cdr3_seq_df.dropna().shape)
    print(cdr3_seq_df.shape)
    
    cdr3_seq_df = cdr3_seq_df.dropna()
    
    adata_tcr = adata_tcr[adata_tcr.obs_names.isin(cdr3_seq_df.index)]
    
    list_tcr_cell = list(adata_tcr.obs_names)
    list_all_cell = list(adata.obs_names)
    inner_cell = list(set(list_tcr_cell) & set(list_all_cell))
    print(len(inner_cell))
    
    # save the QC-ed adata 
    #adata_full_qc = adata
    
    adata_t = adata[adata.obs_names.isin(inner_cell)]
    adata_tcr = adata_tcr[adata_tcr.obs_names.isin(inner_cell)]
    
    #print(adata_tcr.obsm["airr"][0][1])
    #print(adata_tcr.obsm["airr"][0][2])
    #print(adata_tcr.obsm["airr"][0][0])
    
    # sequence 
    # https://scirpy.scverse.org/en/latest/tutorials/tutorial_3k_tcr.html#define-clonotypes-and-clonotype-clusters
    ir.pp.ir_dist(adata_tcr,sequence="aa")
    ir.tl.define_clonotypes(adata_tcr, receptor_arms="all", dual_ir="primary_only")
    
    # https://scirpy.scverse.org/en/latest/generated/scirpy.get.airr_context.html
    #ir.get.airr(adata_tcr, ["cdr1", "cdr3_aa", "cdr3"], ('VJ_1', 'VDJ_1'))
    
    # This is same with Seurat
    sc.pp.normalize_total(adata_t, target_sum=1e4)
    sc.pp.log1p(adata_t)
    
    mdata = mu.MuData({"gex": adata_t, "airr": adata_tcr})
    mdata
    
    if use_gene_subsets: #  , geneList=None
        geneList = [x for x in geneList if x in adata.var_names]
        adata_t = adata_t[:, geneList]
        mdata = mu.MuData({"gex": adata_t, "airr": adata_tcr})
    else:
        sc.pp.highly_variable_genes(mdata["gex"], flavor="seurat_v3", n_top_genes=2000, subset=True)
    mdata.update()
    sc.tl.pca(mdata["gex"])
    sc.pp.neighbors(mdata["gex"])
    
    mdata.update()
    
    sc.tl.umap(mdata['gex'])
    mdata.update()
    
    #mu.pl.embedding(
    #    mdata,
    #    basis="gex:umap",
    #    #color=["gex:sample", "gex:patient", "gex:cluster"],
    #    ncols=1,
    #    wspace=0.7,
    #)
    
    #adata = mdata["gex"]
    #adata_tcr = mdata["airr"]
    #cdr3_seq_df = ir.get.airr(adata_tcr, ["cdr3_aa"], ('VDJ_1'))
    return(mdata)



def prep_RNA_h5ad (input_dir):
    #TCR_FILE_PATH = glob.glob(input_dir+"*vdj_t_*_contig_annotations.csv")[0]
    RNA_FILE_PATH = glob.glob(input_dir+"*.h5ad")[0]
    
    # from https://scirpy.scverse.org/en/latest/tutorials/tutorial_io.html
    # Read 10x data
    
    # Load the associated transcriptomics data
    adata = sc.read_h5ad(RNA_FILE_PATH)
    adata.var_names_make_unique()
    
    #print(adata_tcr.shape)
    print(adata.shape)
    
    adata.var['mt'] =adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    
    adata = adata[adata.obs.n_genes_by_counts < 4000, :]
    adata = adata[adata.obs.pct_counts_mt < 15, :]
    
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    
    # This is same with Seurat
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, subset=True)

    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    
    sc.tl.umap(adata)

    return(adata)




