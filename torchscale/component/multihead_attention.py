# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchscale.component.hook import HookManager
import einops
from einops import rearrange
# try:
#     from .fused_norm import FusedLayerNorm as LayerNorm
# except ModuleNotFoundError:
#     from .layer_norm import LayerNorm
from .layer_norm import LayerNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS
from .flash_attention import flash_attn_func
from typing import Optional

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
        hook: Optional[HookManager] = None,
        
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps,hook = hook.fork("inner_attn_ln")))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        if not self.args.flash_attention:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = rearrange(attn_weights, '(b h) t s -> b h t s', h=self.num_heads)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = rearrange(attn_weights, 'b h t s -> (b h) t s')

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )
            attn_probs = self.dropout_module(attn_weights)

            attn = torch.bmm(attn_probs, v)
            attn = rearrange(attn, '(b h) l d -> b l (h d)', h=self.num_heads)
        else:
            assert flash_attn_func is not None
            assert rel_pos is None
            q = rearrange(q, '(b h) l d -> b l h d', h=self.num_heads)
            k = rearrange(k, '(b h) l d -> b l h d', h=self.num_heads)
            v = rearrange(v, '(b h) l d -> b l h d', h=self.num_heads)
            attn, lse = flash_attn_func(q, k, v, self.dropout, attn_mask, None, is_causal)
            attn = rearrange(attn, 'b l h d -> b l (h d)')
            attn_weights = lse[:, :, :attn.size(1)]

        return attn, attn_weights

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)
        
        ####
        # q2 = q
        # k2 = k
        # v2 = v
        # q2 *= self.scaling
        # q2 = rearrange(q2, '(b h) l d -> b h l d', h=self.num_heads)
        # k2 = rearrange(k2, '(b h) l d -> b h l d', h=self.num_heads)
        # v2 = rearrange(v2, '(b h) l d -> b h l d', h=self.num_heads)
        # clip_attn = q2 @ k2.transpose(-2, -1)
        # if attn_mask is not None:
        #     clip_attn += attn_mask
        # clip_attn = clip_attn.softmax(dim=-1)
        
        # x = torch.einsum(
        #     "bhnm,bhmc->bnhc", clip_attn, v2
        # )  
        
        # x =torch.einsum(
        #             "bnhc,dhc->bnhd",
        #             x,
        #             self.out_proj.A.weight.reshape(
        #                 self.embed_dim, self.num_heads, self.head_dim
        #             ))
        # x = x.sum(axis = 2)
        # x = x + self.out_proj.A.bias
        
        ###
        
        attn, attn_weights = self.attention_ops(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)
        
        expose = einops.rearrange(attn, "b n (h c) -> b n h c", h = self.num_heads)
        
        self.hook(
                "out_proj_post",
                ret=torch.einsum(
                    "bnhc,dhc->bnhd",
                    expose,
                    self.out_proj.A.weight.reshape(
                        self.embed_dim, self.num_heads, self.head_dim
                    ),
                ),
            )
        
        # out_proj_post = torch.einsum(
        #             "bnhc,dhc->bnhd",
        #             expose,
        #             self.out_proj.A.weight.reshape(
        #                 self.embed_dim, self.num_heads, self.head_dim
        #             ))
        
        #collapse = out_proj_post.sum(axis=2) + self.out_proj.A.bias
        #to prove it, sum axis=2 and compare this collapse output with out_proj. 
        
        # so then its just b n d which is b l d where d is the embed dim. 
        
        
        attn = self.out_proj(attn) #here would be b l (h d) or batch, 1 or n tokens, embded dim with head and d head dim together)
        # print("distance of clip attn method and beit3 attn \n", torch.norm((attn - x).flatten()))
        # print("\n")
        # print("norm of attn and post collapse:\n ",torch.norm((attn - collapse).flatten()))
        # # hook the reshaped attn to obtain the head without changing the operations
        # #self.hook("out_proj_post", ret = rearrange(attn,"b l (h d) -> b l h d",h = self.num_heads))
        
        self.hook.finalize()
        return attn, attn_weights
