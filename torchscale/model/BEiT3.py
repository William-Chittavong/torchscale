# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
from torchscale.component.multiway_network import MutliwayEmbedding

from torchscale.component.hook import HookManager

from typing import Optional
from torchscale.component import beit3_utils as utils


from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3(nn.Module):
    def __init__(self, args, hook: Optional[HookManager] = None, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.hook = hook or HookManager()
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
            hook= self.hook.fork("encoder"),
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,

            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions
        )
        encoder_out["multiway_split_position"] = multiway_split_position
        
        #self.hook.finalize()

        return encoder_out
    


class BEiT3Wrapper(nn.Module):
    def __init__(self, args,hook: Optional[HookManager] = None, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args,hook = hook)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    
class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self, 
            args,
            hook: Optional[HookManager] = None,
            **kwargs
    ):
        self.hook_manager = hook or HookManager()       
        super(BEiT3ForRetrieval, self).__init__(args=args , hook = self.hook_manager.fork("beit3"))
       
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(), 
            world_size=utils.get_world_size(), 
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, 
                visual_tokens=image, 
                text_padding_position=None
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description, 
                visual_tokens=None, 
                text_padding_position=padding_mask, 
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None
        
        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls




def create_beit3_retrieval_model(model_size='base', img_size=224, **kwargs):
    """
    Create a BEiT3 model for retrieval tasks.
    
    Args:
    model_size (str): 'base' or 'large'
    img_size (int): Image size (assuming square images)
    **kwargs: Additional arguments to pass to the model
    
    Returns:
    BEiT3ForRetrieval: The created model
    """
    if model_size not in ['base', 'large']:
        raise ValueError("model_size must be either 'base' or 'large'")
    
    if model_size == 'base':
        args = get_base_config(img_size=img_size, **kwargs)
    else:  # large
        args = get_large_config(img_size=img_size, **kwargs)
    
    hook_manager = HookManager()
    
    model = BEiT3ForRetrieval(args, hook=hook_manager, **kwargs)
    
    return model

# Example usage:
# base_model_224 = create_beit3_retrieval_model('base', 224)
# large_model_384 = create_beit3_retrieval_model('large', 384)
 