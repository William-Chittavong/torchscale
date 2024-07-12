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
        bias_term = self.model.encoder.layers[self.current_layer].self_attn.out_proj.bias

        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()
        self.attentions.append(
            return_value
            + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            / (return_value.shape[1] * return_value.shape[2])
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


    def register_hooks(self):
        self.model.hook_manager.register(
            "encoder.layer.*.self_attn.out_proj_post*",
            self.compute_attentions()
        )
       
        self.model.hook_manager.register(
            "encoder.layer.not_moe.ffn.fc2_post",
            self.compute_mlps
        )
        
        #MOE FFNs
        self.model.hook_manager.register(
            "encoder.layer.moe.expert.*.ffn.fc2_post",
            self.compute_mlps
        )
        
        # IS THE THING BELOW needed? why is the layer norm before
        # the transformer resblocks included in the mlps?
        
        # LN before the other encoder layers but self attn already happened
        # what about layernorm in the forward embedding? ah nvm, its before self attn
        self.model.hook_manager.register(
            "encoder.layer.0.self_attn_layer_norm.*.ln_post",self.compute_mlps()
        )

        #after final layer's layer norm. 
        self.model.hook_manager.register(
            f"encoder.layer_norm_post.mean",
            self.log_post_ln_mean()
        )
        
        self.model.hook_manager.register(
            f"encoder.layer_norm_post.sqrt_var",
            self.log_post_ln_std()
        )


    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (
            self.mlps
            - self.post_ln_mean[:, :, np.newaxis].to(self.device) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.beit3.encoder.layernorm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device)
        bias_term = self.model.beit3.encoder.layernorm.B.bias.detach().to(self.device) / (
            len_intermediates 
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.beit3.encoder.output_projection.detach().to(self.device)

    def _normalize_attentions(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2] * self.attentions.shape[3]
        )  # n * h
        # This is just the normalization layer:
        mean_centered = self.attentions - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device) / (len_intermediates * normalization_term)
        
        weighted_mean_centered = (
            self.model.beit3.encoder.layernorm.B.weight.detach().to(self.device) * mean_centered

        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device)
        
        
        
        bias_term = self.model.beit3.encoder.layernorm.B.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.beit3.encoder.output_projection.detach().to(self.device)

    @torch.no_grad()
    def finalize(self):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, n, h, d]
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        projected_attentions = self._normalize_attentions()
        projected_mlps = self._normalize_mlps()
        
        vision_cls_proj_attn = self.vision_head(projected_attentions)
        vision_cls_proj_mlps = self.vision_head(projected_mlps)
        
        attentions = F.normalize(vision_cls_proj_attn)
        mlps = F.normalize(vision_cls_proj_mlps)
        
        return(
            attentions,mlps
        )

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
    prs.register_hooks()
    return prs