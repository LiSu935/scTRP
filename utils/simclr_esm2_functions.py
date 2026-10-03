"""
ESM2 TCR-sequence encoder and projector used by the scTRP encoding and training scripts.

Copied from duolin_contrastiveLearning/scgpt_representation_fixpretrain_LIGHT.py so the
scTRP repo does not depend on the external scRNA_TCRBCR_surfaceProtein checkout.
"""
import esm
import torch
from torch import nn


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x


class Transformer_ESM2_representation_fixpretrain(nn.Module):  #embedding table is fixed
    def __init__(self,esm2_pretrain,esm2_pretrain_local,inner_dim=4096,out_dim=256,num_projector=2,unfix_last_layer=4,tune_ESM_table=0):  
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(Transformer_ESM2_representation_fixpretrain, self).__init__()
        esm2_dict = {"esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(), #33 layers embedding=1280
                     "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(), #30 layers embedding=640
                     "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(), #12 layers embedding=480
                     "esm2_t6_8M_UR50D":esm.pretrained.esm2_t6_8M_UR50D(),#6 layers embedding = 320
                    }
        
        if esm2_pretrain_local is None:
            self.esm2, self.alphabet = esm2_dict[esm2_pretrain] #alphabet.all_toks
        else:
          print("load esm2 model from local dir")
          self.esm2, self.alphabet=  esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)
        
        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters(): #frozen all parameters first
                p.requires_grad = False
        
        fix_layer_num = self.num_layers-unfix_last_layer
        fix_layer_index = 0
        for layer in self.esm2.layers: #only fine-tune transformer layers,no contact_head and other parameters
            if fix_layer_index<fix_layer_num:
               fix_layer_index+=1 #keep these layers frozen
               continue
            
            for p in layer.parameters():
                  print("unfix")
                  p.requires_grad = True
        
        if unfix_last_layer!=0: #if need fine-tune last layer, the emb_layer_norm_after for last representation should updated
            for p in self.esm2.emb_layer_norm_after.parameters():
                p.requires_grad = True
            
            if tune_ESM_table:
               for p in self.esm2.embed_tokens.parameters():
                   p.requires_grad = True
        
        self.projectors = MoBYMLP(in_dim=self.esm2.embed_dim,inner_dim=inner_dim,num_layers=num_projector,out_dim=out_dim)
    
    def forward(self, x):
        residue_feature = self.esm2(x,repr_layers=[self.num_layers],return_contacts=False)['representations'][self.num_layers]
        graph_feature = residue_feature[:,0,:]
        no_proj_graph_feature = graph_feature
        graph_feature = self.projectors(graph_feature)
        return graph_feature,residue_feature,no_proj_graph_feature
