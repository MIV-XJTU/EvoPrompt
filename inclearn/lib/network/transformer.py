# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Q2L Transformer class.

Most borrow from DETR except:
    * remove self-attention by default.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    * using modified multihead attention from nn_multiheadattention.py
"""
import copy
from typing import Optional, List
import math
from loguru import logger

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from inclearn.lib.network.attention import MultiheadAttention


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: int = 0.1,
        activation: int = "relu",
        normalize_before: int = False,
        return_intermediate_dec: int = False,
        rm_self_attn_dec: int = True,
        rm_first_self_attn: int = True,
        reduce_attn_weights: bool = True,
        mh_activation: str = "softmax",
    ):
        """Transformer module composed of transformer encoder-decoder layers of transformer module.

        Args:
            embed_dim (int, optional): dimension of transformer embedding, the query, value,
                                       and positional embedding should have similar dimension as embed_dim. Defaults to 512.
            num_heads (int, optional): number of multi-head attention. Defaults to 8.
            num_encoder_layers (int, optional): number of encoder layer. Defaults to 6.
            num_decoder_layers (int, optional): number of decoder layer. Defaults to 6.
            dim_feedforward (int, optional): hidden dimension encoder-decoder layer. Defaults to 2048.
            dropout (int, optional): dropout probability. Defaults to 0.1.
            activation (int, optional): activation function used on encoder-decore linear. Defaults to "relu".
            normalize_before (int, optional): if true, will normalize the input before feed to multihead attention. Defaults to False.
            return_intermediate_dec (int, optional): if true, will return all intermediate output from decoder layer. Defaults to False.
            rm_self_attn_dec (int, optional): if true, will remove attention module at all decoder layer. Defaults to True.
            rm_first_self_attn (int, optional): if true, will remove attention module at first decoder layer. Defaults to True.
        """
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                embed_dim,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                mh_activation=mh_activation,
            )
            encoder_norm = nn.LayerNorm(embed_dim) if normalize_before else None
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )
        else:
            logger.info("no encoder layer")

        decoder_layer = TransformerDecoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            mh_activation=mh_activation,
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rm_self_attn_dec = rm_self_attn_dec
        self.rm_first_self_attn = rm_first_self_attn
        self.reduce_attn_weights = reduce_attn_weights

        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue

            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        # print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def set_debug_mode(self, status):
        logger.info("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, "encoder"):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, "decoder"):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor = None,
        mask=None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """transformer forward function.

        Args:
            src (torch.Tensor): input/value tensor of shape 3-dim (num tokens, batch_size, embed_dim).
            query_embed (torch.Tensor): query tensor of shape 3-dim (num token, batch size, embed_dim).
            pos_embed (torch.Tensor): positional embedding tensor of shape 3-dim (num token, batch_size, embed_dim).
            mask ([type], optional): key_padding_mask: If specified, a mask of shape (batch_size, height, width) indicating
                    which elements within ``key``to ignore for the purpose of attention (i.e. treat as "padding").
                    For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                    the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                    value will be ignored.

        Returns:
            torch.Tensor: returning features tensor of shape (1, batch_size, any size, e.g. num_class, embed_dim)
                    and encoder output of shape (batch_size, embed_dim, height, width).
        """
        n_token = src.size(0)
        bs = src.size(1)
        c = src.size(2)

        if mask is not None:
            mask = mask.flatten(1)

        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = torch.zeros_like(query_embed)
        decoder_out = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
            reduce_attn_weights=self.reduce_attn_weights,
            need_weights=need_weights,
        )

        hs, memory = decoder_out["output"].transpose(1, 2), memory[:n_token].permute(
            1, 0, 2
        ).view(bs, n_token, c)

        if need_weights:
            return hs, memory, decoder_out["sim_mat"]
        else:
            return hs, memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                need_weights=need_weights,
                **kwargs,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output["tgt"]))
            if len(self.layers) > 1 and i < len(self.layers) - 1:
                output = output["tgt"]
        if need_weights:
            sim_mat = output["sim_mat"]
        if self.norm is not None:
            output = self.norm(output["tgt"])
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        output = dict(output=output.unsqueeze(0))
        if need_weights:
            output["sim_mat"] = sim_mat
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        mh_activation="softmax",
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, activation=mh_activation
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        **kwargs,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            **kwargs,
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        **kwargs,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            **kwargs,
        )[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        mh_activation="softmax",
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, activation=mh_activation
        )
        self.multihead_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, activation=mh_activation
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                **kwargs,
            )

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            **kwargs,
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        out = dict(tgt=tgt)
        if need_weights:
            out["sim_mat"] = sim_mat_2["weights"]
        return out

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            **kwargs,
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, sim_mat_2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            **kwargs,
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        out = dict(tgt=tgt)
        if need_weights:
            out["sim_mat"] = sim_mat_2["weights"]
        return out

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                need_weights=need_weights,
                **kwargs,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            need_weights=need_weights,
            **kwargs,
        )


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    This will generate positional embedding in size (hidden_dim, maxH, maxW).
    """

    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        maxH=30,
        maxW=30,
        keep_spatial: bool = True,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.keep_spatial = keep_spatial

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer("pe", pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor) -> torch.Tensor:
        """
        Get positional embedding given input, e.g. image features
        Args:
            input (Tensor): input of shape (N, C, H, W)

        Returns:
            torch.Tensor: output of shape (N, C, H, W)
        """
        x = input
        if self.keep_spatial:
            return self.pe.repeat((x.size(0), 1, 1, 1))
        else:
            # bs, n_chans, hw
            return self.pe.repeat((x.size(0), 1, 1, 1)).flatten(start_dim=2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
