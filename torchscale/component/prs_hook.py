import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path
import torch.nn.functional as F

class PRSLogger(object):
    def __init__(self, model,device , spatial:bool = True):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.spatial = spatial
        self.ffn = []
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
    

    @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        #bias_term = self.model.beit3.encoder.layers[self.current_layer].self_attn.out_proj.B.bias

        self.current_layer += 1
        return_value = ret.detach().cpu() # [b , l ,h d]
       
        self.attentions.append(
            return_value
           
        ) 
        return ret
    
    @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        #bias_term = self.model.beit3.encoder.layers[self.current_layer].self_attn.out_proj.B.bias

        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu() # cls token # [b 1 h d  ]
        #return_value = ret.detach().cpu() # cls token
        #print(return_value.shape,"cls token shape of attn before stacking")
        self.attentions.append(
            return_value
            # + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            # / (return_value.shape[1] * return_value.shape[2])
        )  # [b h d]
        return ret


    @torch.no_grad()
    def compute_ffn(self, ret):
        #print("ffn rep shape \n",ret.shape)
        self.ffn.append(ret[:, 0,:].detach().cpu()) # b,(h d)
        return ret

 
    @torch.no_grad()
    def log_post_ln_mean(self, ret): # b , l (L or as in N)
        self.post_ln_mean = ret[:,0].detach().cpu() #b, 1 (one)
        print()
        return ret

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret[:,0].detach().cpu()  # b, 1 
        return ret


    def _normalize_ffn(self):
        len_intermediates = self.attentions.shape[1] + self.ffn.shape[1]

        # This is just the normalization layer:
        mean_centered = (
            self.ffn - self.post_ln_mean[
            :, :, np.newaxis
        ].to(self.device) / len_intermediates
        )
        
        weighted_mean_centered = (
            self.model.beit3.encoder.layer_norm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis
        ].to(self.device)
        bias_term = self.model.beit3.encoder.layer_norm.B.bias.detach().to(self.device) / (
            len_intermediates 
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln 
    
    def _normalize_attentions_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.ffn.shape[1]  # 2*l + 1
        
        # print("self.attentions shape:\n", self.attentions.shape)
        # print("self.post_ln_mean shape:\n", self.post_ln_mean.shape)
        # print("self.post_ln_std shape:\n", self.post_ln_std.shape)
        # print("layer_norm.B.weight shape:\n", self.model.beit3.encoder.layer_norm.B.weight.shape)
        # print("layer_norm.B.bias shape:\n", self.model.beit3.encoder.layer_norm.B.bias.shape)
        #len_intermediates = self.attentions.shape[1] 
        normalization_term = (
            self.attentions.shape[1] * self.attentions.shape[2]
        )  # n * h
        # This is just the normalization layer:
        
        print("shape of attentions \n ", self.attentions.shape)
        print("post ln shape",self.post_ln_mean[
             :, :, np.newaxis, np.newaxis, np.newaxis
        ].shape)
        mean_centered = self.attentions - self.post_ln_mean[
             :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device) / (len_intermediates * normalization_term)
        
        weighted_mean_centered = (
            self.model.beit3.encoder.layer_norm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
             :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device)
        
        
        
        bias_term = self.model.beit3.encoder.layer_norm.B.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        
        post_ln = weighted_mean_by_std + bias_term
        #print(post_ln.shape,"post ln shape\n")
        return post_ln
      

    def _normalize_attentions_non_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.ffn.shape[1]  # 2*l + 1
        
        # print("self.attentions shape:\n", self.attentions.shape)
        # print("self.post_ln_mean shape:\n", self.post_ln_mean.shape)
        # print("self.post_ln_std shape:\n", self.post_ln_std.shape)
        # print("layer_norm.B.weight shape:\n", self.model.beit3.encoder.layer_norm.B.weight.shape)
        # print("layer_norm.B.bias shape:\n", self.model.beit3.encoder.layer_norm.B.bias.shape)
     
        normalization_term = (
          self.attentions.shape[2]
        )  # h
        # This is just the normalization layer:
        # print("post ln shape before newaxis \n" ,self.post_ln_mean.shape)
        # print("shape of attentions \n ", self.attentions.shape)
        # print("post ln shape",self.post_ln_mean[
        #     :, :, np.newaxis,np.newaxis
        # ].shape)
        mean_centered = self.attentions - self.post_ln_mean[
            :, :, np.newaxis,np.newaxis
        ].to(self.device) / (len_intermediates * normalization_term)
        
        weighted_mean_centered = (
            self.model.beit3.encoder.layer_norm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis,np.newaxis
        ].to(self.device)
        
        
        
        bias_term = self.model.beit3.encoder.layer_norm.B.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        
        post_ln = weighted_mean_by_std + bias_term
        #print(post_ln.shape,"post ln shape\n")
        return post_ln
        #return post_ln @ self.model.beit3.encoder.output_projection.to(self.device)  # result should be B , N , C

        
    @torch.no_grad()
    def finalize(self,rep):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, h, d]
        # print(self.attentions.shape,"post stack attentions shape \n")
        self.ffn = torch.stack(self.ffn, axis=1).to(self.device)  # [b, l + 1, d]
        
        if self.spatial:
            norm_attentions = self._normalize_attentions_spatial()
        else:
            norm_attentions = self._normalize_attentions_non_spatial()
            
        #attentions = self._normalize_attentions()
        norm_ffn = self._normalize_ffn()
        #print("norm ffn \n ", norm_ffn.shape)
        projected_attentions = self.model.vision_head(norm_attentions)
        projected_ffn = self.model.vision_head(norm_ffn)
        #print("projected ffn \n ", projected_ffn.shape)
        norm = rep.norm(dim=-1).detach()
        # print(norm.shape, "norm before new axis \n")
        
        if self.spatial:
            return (
                projected_attentions
                / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], 
                projected_ffn / norm[:, np.newaxis, np.newaxis], 
            ) 
        return (
            projected_attentions
            / norm[:, np.newaxis,np.newaxis, np.newaxis], 
            projected_ffn / norm[:, np.newaxis, np.newaxis], 
        ) 
    
        

    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.ffn = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()

def hook_prs_logger(model, device , spatial: bool = True):
    """Hooks a projected residual stream logger to the model."""
    
    prs = PRSLogger( model, device , spatial = spatial)
    if spatial:
        print("SPATIAL \n")
        model.hook_manager.register(
            "beit3.encoder.layer.*.self_attn.out_proj_post",
        prs.compute_attentions_spatial
        )
    else:
        print("using non spatial \n")
        model.hook_manager.register(
            "beit3.encoder.layer.*.self_attn.out_proj_post",
        prs.compute_attentions_non_spatial
        ) 
    model.hook_manager.register(
        "beit3.encoder.layer.*.moe" , prs.compute_ffn
    )
    model.hook_manager.register(
        "beit3.encoder.layer.*.not_moe" , prs.compute_ffn
    )
    # what about layernorm in the forward embedding? ah nvm, its before self attn
    # model.hook_manager.register(
    #     "beit3.encoder.layer.0.self_attn_layer_norm.*.ln_post", prs.compute_ffn
    # )

    #after final layer's layer norm. 
    model.hook_manager.register(
        "beit3.encoder.layer_norm_post.mean",
        prs.log_post_ln_mean
    )
    
    model.hook_manager.register(
        "beit3.encoder.layer_norm_post.sqrt_var",
        prs.log_post_ln_std
    )
    
  
    return prs


# idea: just log the output after the self_attn_layer norm
# as well as the output of ffn or moe layer. for clip, the just log the output of the mlp layer anyway.
# after the ffn are taken care of,
# edit the compute prs to match what was done in the demo file

# then edit whatever in the ablations file

# finally edit the text span file.  
