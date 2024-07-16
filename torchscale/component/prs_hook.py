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
    def __init__(self, model, embed_dim,device):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        self.vision_head = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    @torch.no_grad()
    def compute_attentions(self, ret):
        #bias_term = self.model.beit3.encoder.layers[self.current_layer].self_attn.out_proj.B.bias

        self.current_layer += 1
        #return_value = ret[:, 0].detach().cpu() # cls token
        return_value = ret[:, 0, :].detach().cpu() # cls token
        self.attentions.append(
            return_value
            # + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            # / (return_value.shape[1] * return_value.shape[2])
        )  # [b, n, h, d]
        return ret

    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(ret[:, 0].detach().cpu())  # [b, d]
        return ret

 
    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu()  # [b, 1]
        return ret

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu()  # [b, 1]
        return ret


    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (
            self.mlps
            - self.post_ln_mean[:, :, np.newaxis].to(self.device) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.beit3.encoder.layer_norm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device)
        bias_term = self.model.beit3.encoder.layer_norm.B.bias.detach().to(self.device) / (
            len_intermediates 
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.beit3.encoder.output_projection.to(self.device)

    def _normalize_attentions(self):
        #len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        len_intermediates = self.attentions.shape[1] 
        normalization_term = (
            self.attentions.shape[1] * self.attentions.shape[2]
        )  # n * h
        # This is just the normalization layer:
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
        print(post_ln.shape,"post ln shape\n")
        print(self.model.beit3.encoder.output_projection)
        return self.model.beit3.encoder.output_projection(post_ln)
        #return post_ln @ self.model.beit3.encoder.output_projection.to(self.device)  # result should be B , N , C
        #TypeError: unsupported operand type(s) for @: 'Tensor' and 'Linear'
        
    @torch.no_grad()
    def finalize(self,rep):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, n, h, d]
        #self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        projected_attentions = self._normalize_attentions()
        #attentions = self._normalize_attentions()
        #projected_mlps = self._normalize_mlps()
        
        norm = rep[0].norm(dim=-1).detach()
        
        #vision_cls_proj_attn = self.vision_head(projected_attentions)
        #vision_cls_proj_mlps = self.vision_head(projected_mlps)
        
        #attentions = F.normalize(vision_cls_proj_attn)
        #mlps = F.normalize(vision_cls_proj_mlps)
        
        # return(
        #     attentions,mlps
        # )
        norm = norm[:, np.newaxis, np.newaxis, np.newaxis]
        print(projected_attentions.shape, "proj attentions\n")
        print(norm.shape,"norm")
        return projected_attentions/ norm
        

    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()

def hook_prs_logger(model, embed_dim,device):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, embed_dim,device)
    
    model.hook_manager.register(
            "beit3.encoder.layer.*.self_attn.out_proj_post",
        prs.compute_attentions
    )
    # its not seeing the mlps...
    
    # model.hook_manager.register(
    #     "beit3.encoder.layer.not_moe.ffn.fc2_post",
    #     prs.compute_mlps
    # )
    
    # #MOE FFNs
    # model.hook_manager.register(
    #     "beit3.encoder.layer.moe.expert.*.ffn.fc2_post",
    #     prs.compute_mlps
    # )
    
    # what about layernorm in the forward embedding? ah nvm, its before self attn
    model.hook_manager.register(
        "beit3.encoder.layer.0.self_attn_layer_norm.*.ln_post", prs.compute_mlps
    )

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