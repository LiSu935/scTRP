# 20250521
# modified from duolin_contrastiveLearning/Check_embedding_alongTraining.py
# this is for encoding the seq with pretrained ESM2 with npz input.

import tarfile

from scipy.stats import mode
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import numpy as np
from torchvision.transforms import transforms
import logging
from torchvision import transforms, datasets
import os
import sys
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
import yaml
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import glob
import time
import webdataset as wds

from esm import ESM2
import esm

#import esm_adapterH
import cosine_annealing_warmup #you need to install this from github:https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup and study this.

from transformers import Swinv2Config,Swinv2Model
from scipy.ndimage import zoom
#from util_CATH import *

from pathlib import Path
import pandas as pd

import scanpy as sc
import sklearn
import warnings

sys.path.insert(0, "/cluster/pixstor/xudong-lab/suli/tools_related/scGPT/")
import scgpt as scg
import json
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel

from scgpt.tokenizer import GeneVocab

import sys
# Add the project directory to sys.path
sys.path.append('/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/scripts/scRNA_TCRBCR_surfaceProtein/')
from duolin_contrastiveLearning.scgpt_representation_fixpretrain_LIGHT import Transformer_ESM2_representation_fixpretrain, MoBYMLP

from types import SimpleNamespace
"""
# tem debugging:
args_ = dict(
    checkpoints=None,
    data_url='/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/processed_data/classifier/six_studies/test_caushi/hvg2000_by_study_6_89_genes/val_5_S_6Inter89_coarseLabel_withCDR3_tokenized_data.tar',
    run_mode='do_seq_only',
    
)

args = SimpleNamespace(**args_)

--data_url XXX.tar
"""
parser = argparse.ArgumentParser(description='PyTorch SimCLR check embedding from checkpoints')
parser.add_argument('--checkpoints', type=str, default=None, help='path to checkpoints') 

# args.data_url 
parser.add_argument('--data_url', metavar='DIR', default="/cluster/pixstor/xudong-lab/suli/Alg_development/scRNA_TCRBCR_surfaceProtein/processed_data/blood_model/TenX_5k_melanoma.tar", help='path to datasets') 

parser.add_argument('--run_mode', type=str, default='do_seq_only', help='choose from ["run_both_modalities", "do_gex_only", "do_seq_only"]')

args = parser.parse_args()

#"""

model_seq = Transformer_ESM2_representation_fixpretrain(esm2_pretrain='esm2_t33_650M_UR50D', esm2_pretrain_local=None,
                                                        inner_dim=2048,out_dim=128,
                                                        num_projector=2,
                                                        unfix_last_layer=0,
                                                        tune_ESM_table=0).to(torch.device("cuda"))


alphabet = model_seq.alphabet
batch_converter = alphabet.get_batch_converter(truncation_seq_length=25)


def do_seq_only(val_loader, output_dir):
    cell_seq_embeddings = []
    cell_id_list = []
    model_seq.eval()
    for batch_idx, batch_data in enumerate(val_loader):
        batch = batch_data[0]
        bsz = len(batch)
        for d in batch:
            batch_seq = [(d['index'], str(d['seq']))]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_labels = [ str(x) for x in batch_labels]
            batch_tokens = batch_tokens.to(torch.device("cuda"))
            with torch.no_grad(), autocast(enabled=True):
                features_seq, features_residue, no_proj_seq_feature = model_seq(batch_tokens)
            no_proj_seq_feature = no_proj_seq_feature.cpu().numpy()
            outputfile=os.path.join(output_dir,f"{d['index']}"+".npz")
            np.savez(outputfile,index=d['index'],genes_data=d["genes_data"],expressions_data=d["expressions_data"],
                         masked_expr_data=d["masked_expr_data"], seq=d["seq"], esm2_emb=no_proj_seq_feature)
    tar_file_path = output_dir + ".tar"
    print(tar_file_path)      
    # Create a tar file and add the directory to it
    with tarfile.open(tar_file_path, "w") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    print(f"Created tar file at: {tar_file_path}")


if args.checkpoints is not None:
  checkpoint = torch.load(args.checkpoints, map_location=lambda storage, loc: storage)
  print(f"load checkpoints from {args.checkpoints}")
  model_seq.load_state_dict(checkpoint['state_dict1'])


val_url=args.data_url
prefix = os.path.splitext(os.path.basename(val_url.rstrip("/")))[0]





# Use the with statement to ensure that the tar file is closed properly
with wds.WebDataset(val_url).decode().to_tuple("npz") as val_dataset:
    # Proceed with batching while the dataset is still open
    val_dataset = val_dataset.batched(32)
    print("done with loading tar files")

    # Create the WebLoader inside the with block
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, pin_memory=False)
    
    # You can continue using val_loader here
    print("create train loader")
    val_loader = val_loader.unbatched().batched(32, partial=True)

    input_file = args.data_url
    output_dir = input_file.replace(".tar", "_esm2encoded")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if args.run_mode == 'do_seq_only':
        do_seq_only(val_loader, output_dir)
    
    print("loading checkpoint {checkpoint_name} and encoding {prefix} finished!")



          




  

