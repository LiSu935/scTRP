# test the following in scverse env
import scanpy as sc
import importlib.util
import os 
import scirpy as ir
import muon as mu

import pandas as pd
import getopt
import argparse

# Specify the path to the .py file (without the file extension)
py_file_path = '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/scripts/scRNA_TCRBCR_surfaceProtein/data_prep/load_TCR_RNA_prep'

# Create a module name (can be any valid Python module name)
module_name = 'load_TCR_RNA_prep'

# Load the .py file as a module
spec = importlib.util.spec_from_file_location(module_name, py_file_path + '.py')
load_TCR_RNA_prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_TCR_RNA_prep)
# Now you can access functions or classes from the loaded module
# my_function = load_TCR_RNA_prep.prep_RNA_h5ad

parser = argparse.ArgumentParser(description='This is all in one script for tokenize gex data and also save all the cells to npz files.')

parser.add_argument('--inputDir', type=str, 
                default="/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/no_anno_10x_data/TenX_10k_PBMC_without_Intron/",
                    help='Directory of input:default(/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/no_anno_10x_data/TenX_10k_PBMC_without_Intron/)')

parser.add_argument('--n_genes_cut', type=int, 
                    default=5000, 
                    help='The number of genes with at least 1 count in a cell. Default to be 5000.')

parser.add_argument('--pct_mt_cut', type=int, 
                    default=20, 
                    help='the proportion of total counts for a cell which are mitochondrial. Default to be 20.')

parser.add_argument('--use_input_rna', action='store_true', default=False, help='whether to use user speficied gex h5 file. Default to be False. If True, also need to add `--GEX_FILE_Input`')

parser.add_argument('--RNA_FILE_Input',  
                    default=None, 
                    help='user specified gex h5 file name. Default to be None.')


parser.add_argument('--use_input_TCR', action='store_true', default=False, help='whether to use user speficied TCR file. Default to be False. If True, also need to add `--TCR_FILE_Input`')

parser.add_argument('--TCR_FILE_Input',  
                    default=None, 
                    help='user specified TCR file name. Default to be None.')

parser.add_argument('--use_gene_subsets', action='store_true', default=False, help='whether to use user speficied genelist. Default to be False. If True, also need to add `--Input_geneList_Path`')

parser.add_argument('--Input_geneList_Path',  
                    default=None, 
                    help='user specified TCR file name. Default to be None.')

parser.add_argument('--Output_major_Path',  
                    default="/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/processed_data/blood_model/", 
                    help='The default path for output processed tokenized data and mdata.')

parser.add_argument('--pretrained_model_Path',  
                    default="/cluster/pixstor/xudong-lab/suli/tools_related/scgpt_data_model/scgpt_human", 
                    help='The path to the pretrained model.')


# --Input_geneList_Path '/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/T_cell_gene_list/genelist_final.csv'

args = parser.parse_args()

# Traindata:
#ORI_INPUT_DIR = "/cluster/pixstor/xudong-lab/suli/Alg_development/use_geneformer/data/Benchmark_data_top2000/COVID/"
input_dir = args.inputDir
Input_n_genes_cut= args.n_genes_cut
Input_pct_mt_cut= args.pct_mt_cut

Input_use_input_rna=args.use_input_rna
Input_RNA_FILE_Input=args.RNA_FILE_Input

Input_use_input_TCR_FILE=args.use_input_TCR
Input_TCR_FILE_Input=args.TCR_FILE_Input
Input_use_gene_subsets=args.use_gene_subsets
Input_geneList_Path=args.Input_geneList_Path
Input_Output_major_Path=args.Output_major_Path
Input_pretrained_model_Path=args.pretrained_model_Path

Input_geneList=pd.read_csv(Input_geneList_Path, header=None).iloc[:, 0].tolist()


#input_dir = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/no_anno_10x_data/TenX_10k_PBMC_without_Intron/"
#input_dir = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/data_collection/no_anno_10x_data/TenX_10k_PBMC_without_Intron/"
prefix = os.path.basename(input_dir.rstrip("/"))
print("==================================================================")
print("prefix is: {prefix}")

if Input_use_gene_subsets:
    output_dir = f"{Input_Output_major_Path}{prefix}/tokenized_data_geneFiltered/"
else:
    output_dir = f"{Input_Output_major_Path}{prefix}/tokenized_data/"
    
os.makedirs(output_dir, exist_ok=True)

#out_dir = "/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/results/RNA_untune_benchmark/NSCLC/"
if Input_use_gene_subsets:
    h5ad_out_dir = f"{Input_Output_major_Path}{prefix}/h5ad_out_geneFiltered/"
else:
    h5ad_out_dir = f"{Input_Output_major_Path}{prefix}/h5ad_out/"
os.makedirs(h5ad_out_dir, exist_ok=True)

mdata = load_TCR_RNA_prep.prep_TCR_RNA(input_dir, n_genes_cut=Input_n_genes_cut,pct_mt_cut=Input_pct_mt_cut, use_input_RNA=Input_use_input_rna, RNA_FILE_Input=Input_RNA_FILE_Input,use_input_TCR_FILE=Input_use_input_TCR_FILE, TCR_FILE_Input=Input_TCR_FILE_Input, use_gene_subsets=Input_use_gene_subsets, geneList=Input_geneList)


mdata.write(h5ad_out_dir+prefix+".h5mu")
#mdata_r = mu.read(temp_file.name, backed=True)
# If need to read one of the mod
#adata = mu.read("mudata.h5mu/rna")

adata = mdata["gex"]
adata_tcr = mdata["airr"]
cdr3_seq_df = ir.get.airr(adata_tcr, ["cdr3_aa"], ('VDJ_1'))

adata.var["Gene Symbol"] = adata.var_names

print(cdr3_seq_df.head())

#cdr3_seq_df['VDJ_1_cdr3_aa']['TTTGTCATCGTCGTTC-1']



import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

sys.path.insert(0, "/cluster/pixstor/xudong-lab/suli/tools_related/scGPT/")
import scgpt as scg


from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab

PathLike = Union[str, os.PathLike]


model_dir = Path(Input_pretrained_model_Path)
#cell_type_key = "Celltype"
gene_col = "Gene Symbol"

max_length=1200
batch_size=1
obs_to_save = None
#device = "cuda"
return_new_adata = False


# LOAD MODEL
model_dir = Path(model_dir)
vocab_file = model_dir / "vocab.json"
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]

with open(model_config_file, "r") as f:
    model_configs = json.load(f)

# vocabulary
vocab = scg.tokenizer.GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
adata.var["id_in_vocab"] = [
    vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])

adata = adata[:, adata.var["id_in_vocab"] >= 0] # make sure all the final genes are in the vocabulary

# Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

vocab.set_default_index(vocab["<pad>"])
genes = adata.var[gene_col].tolist()
gene_ids = np.array(vocab(genes), dtype=int)


count_matrix = adata.X
count_matrix = (
    count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
)

if gene_ids is None:
    gene_ids = np.array(adata.var["id_in_vocab"])
    assert np.all(gene_ids >= 0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, count_matrix, gene_ids, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, vocab["<cls>"])
        values = np.insert(values, 0, model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values)
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output

use_batch_labels=False
dataset = Dataset(
    count_matrix, gene_ids, batch_ids if use_batch_labels else None
)

collator = DataCollator(
    do_padding=True,
    pad_token_id=vocab[model_configs["pad_token"]],
    pad_value=model_configs["pad_value"],
    do_mlm=False,
    do_binning=True,
    max_length=max_length,
    sampling=True,
    keep_first_n_tokens=1,
)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset),
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), batch_size),
    pin_memory=True,
)



# Iterate through the data_loader and flatten the batches
count_ = 0
max_len_expdata = 0
for batch in data_loader:
    for sample in range(batch[list(batch.keys())[0]].shape[0]):
        # Extract data from dictionary keys "gene", "expr", and "expr2"
        genes_data = batch["gene"][sample]
        expressions_data = batch["expr"][sample]
        if len(expressions_data) >= max_len_expdata:
            max_len_expdata = len(expressions_data)
        masked_expr_data = batch["masked_expr"][sample]
        cell_id = str(adata.obs_names[count_])
        seq = str(cdr3_seq_df['VDJ_1_cdr3_aa'][cell_id])
        cell_id = adata.obs_names[count_]+"-"+prefix # here to make sure the cell_id in the whole train-test-val pool will be unique.
        outputfile=output_dir+cell_id+".npz"
        np.savez(outputfile,index=cell_id,genes_data=genes_data,expressions_data=expressions_data,masked_expr_data=masked_expr_data, seq=seq)
        count_ +=1

print(f"this count_ should be same with adata's cell # {adata.shape[0]}")
print(count_)
print(max_len_expdata)
print("==================================================================")


del data_loader, dataset, adata
