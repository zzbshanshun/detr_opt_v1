# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math, pdb

from .attention import MultiheadAttention

from util.misc import inverse_sigmoid

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256) 

    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def gen_sineembed_for_position2(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class UpdateGate(nn.Module):
    def __init__(self, d_model, num_layers, gate_type="adaptive", use_highway=False):
        """
        门控更新机制
        
        参数:
            d_model: 特征维度
            num_layers: 解码器层数
            gate_type: 门控类型 ["adaptive", "static", "learned"]
            use_highway: 是否使用高速连接
        """
        super().__init__()
        self.gate_type = gate_type
        self.use_highway = use_highway
        self.num_layers = num_layers
        
        # 自适应门控
        if gate_type == "adaptive":
            self.gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model + d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1),
                    nn.Sigmoid()
                ) for _ in range(num_layers - 1)
            ])
        
        # 静态门控参数
        elif gate_type == "static":
            self.static_gates = nn.Parameter(torch.linspace(0.8, 0.3, num_layers-1))
        
        # 可学习门控参数
        elif gate_type == "learned":
            self.learned_gates = nn.Parameter(torch.ones(num_layers-1) * 0.5)
        
        # 高速连接参数
        if use_highway:
            self.highway_weights = nn.Parameter(torch.zeros(num_layers-1))
        
        # 稳定性控制
        self.gate_clamp = 0.2  # 门控最小变化量
        self.momentum = 0.1    # 指数移动平均动量
    
    def forward(self, layer_id, output, box_off, prev_gate=None):
        """
        计算门控值
        
        参数:
            layer_id: 当前层ID (从1开始)
            output: 解码器输出特征 [num_queries, batch_size, d_model]
            box_off: 预测的框偏移量 [num_queries, batch_size, 4]
            prev_gate: 前一层门控值 (用于稳定性控制)
            
        返回:
            gate_value: 门控值 [num_queries, batch_size, 1]
            highway_weight: 高速连接权重 (如果使用)
        """
        # 第一层不使用门控
        if layer_id == 0:
            return 1.0, None
        
        # 实际索引 (0到num_layers-2)
        gate_idx = layer_id - 1
        
        # 根据门控类型计算门控值
        if self.gate_type == "adaptive":
            # 拼接特征和偏移量
            # box_off_proj = box_off.mean(dim=-1, keepdim=True)  # 简化处理
            gate_input = torch.cat([output, box_off], dim=-1)
            
            # 计算门控值
            gate_value = self.gate_nets[gate_idx](gate_input)
        
        elif self.gate_type == "static":
            gate_value = torch.ones_like(output[:, :, :1]) * self.static_gates[gate_idx]
        
        elif self.gate_type == "learned":
            gate_value = torch.ones_like(output[:, :, :1]) * torch.sigmoid(self.learned_gates[gate_idx])
        
        # 稳定性控制
        if prev_gate is not None:
            # 指数移动平均
            gate_value = (1 - self.momentum) * prev_gate + self.momentum * gate_value
            
            # 限制最小变化量
            min_gate = torch.clamp(prev_gate - self.gate_clamp, 0, 1)
            max_gate = torch.clamp(prev_gate + self.gate_clamp, 0, 1)
            gate_value = torch.clamp(gate_value, min_gate, max_gate)
        
        # 高速连接权重
        highway_weight = None
        if self.use_highway:
            highway_weight = torch.sigmoid(self.highway_weights[gate_idx])
        
        return gate_value, highway_weight

#''' 
# detr################################################
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 num_queries = 300,
                 use_query_embed = False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        if use_query_embed:
            self.decoder = TransformerDecoderEmbed(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec)
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
        # self.src_qpos_ref = nn.Conv2d(d_model, d_model, kernel_size=1)
        # shape = 18
        # self.ref_ac_pool = nn.AdaptiveAvgPool2d(shape)
        # self.ref_ac_pool = nn.AdaptiveMaxPool2d(shape)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
    
        num_queries = refpoint_embed.shape[0]
        tgt = torch.zeros(num_queries, bs, self.d_model, device=pos_embed.device)
        memory = self.encoder(src_flat, src_key_padding_mask=mask, pos=pos_embed)
        
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=refpoint_embed, box_embed=None)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return hs.transpose(1, 2), references

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.wh_sine_proj = nn.Linear(d_model, d_model)
        self.bbox_head = None
        self.qsine_embed_map = MLP(2 * d_model, d_model, d_model, 2)
        
        # for decode layer
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        # self.layers[0].ca_v_proj = None
        # self.layers[0].ca_kcontent_proj = None
        # self.layers[0].ca_kpos_proj = None
        # self.layers[0].ca_qcontent_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                box_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        reference_points_lvl=[]
        # reference points
        reference_points_unsigmoid = query_pos
        reference_points = reference_points_unsigmoid.sigmoid()

        prev_gate = None  # 保存前一层门控值用于稳定性

        for layer_id, layer in enumerate(self.layers):

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                output_ckp = output.detach()
                box, _ = self.bbox_head(output_ckp)                
                box += inverse_sigmoid(reference_points)
                reference_points = box.sigmoid()

            query_sine = gen_sineembed_for_position2(reference_points, self.d_model)
            query_pos_layer = self.qsine_embed_map(query_sine)
            query_sine_xy = query_sine[..., :self.d_model]
            query_sine_wh = query_sine[..., self.d_model:]
            
            if layer_id != 0:
                pos_transformation = self.query_scale(output_ckp + self.wh_sine_proj(query_sine_wh))
            query_sine_embed = query_sine_xy*pos_transformation
            
            # do layer compute
            output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos_layer, query_sine_embed=query_sine_embed,
                            is_first=(layer_id == 0))
             
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                reference_points_lvl.append(reference_points)
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # if self.return_intermediate:
        #     return torch.stack(intermediate)

        if self.return_intermediate:
            return [torch.stack(intermediate), torch.stack(reference_points_lvl).transpose(1, 2)]

        return output.unsqueeze(0), reference_points.unsqueeze(0)

class TransformerDecoderEmbed(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.wh_sine_proj = nn.Linear(d_model, d_model)
        self.bbox_head = None
        
        self.qpos_box_map = nn.Linear(d_model, 4)
        
        # for decode layer
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        # self.layers[0].ca_v_proj = None
        # self.layers[0].ca_kcontent_proj = None
        # self.layers[0].ca_kpos_proj = None
        # self.layers[0].ca_qcontent_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                box_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        reference_points_lvl=[]
        # reference points
        query_pos_proj = query_pos
        reference_points_unsigmoid = self.qpos_box_map(query_pos_proj)
        reference_points = reference_points_unsigmoid.sigmoid()

        for layer_id, layer in enumerate(self.layers):
            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
                query_pos_layer = query_pos_proj
            else:
                output_ckp = output.detach()
                output_pos_layer = self.bbox_head(output_ckp, 1)
                query_pos_layer = output_pos_layer + query_pos_layer
                reference_points = self.qpos_box_map(query_pos_layer).sigmoid()

            query_sine = gen_sineembed_for_position2(reference_points, self.d_model)
            query_sine_xy = query_sine[..., :self.d_model]
            query_sine_wh = query_sine[..., self.d_model:]
            
            if layer_id != 0:
                pos_transformation = self.query_scale(output_ckp + self.wh_sine_proj(query_sine_wh))
            query_sine_embed = query_sine_xy*pos_transformation
            
            # do layer compute
            output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos_layer, query_sine_embed=query_sine_embed,
                            is_first=(layer_id == 0))
             
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                reference_points_lvl.append(reference_points)
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # if self.return_intermediate:
        #     return torch.stack(intermediate)

        if self.return_intermediate:
            return [torch.stack(intermediate), torch.stack(reference_points_lvl).transpose(1, 2)]

        return output.unsqueeze(0), reference_points.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries = args.num_queries,
        use_query_embed = args.use_query_embed
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
