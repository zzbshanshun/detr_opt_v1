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

def gen_sineembed_for_xywh(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    
    # 4输入的编码，编码长度由128下降到64，导致低频信息丢失较多，缺少全局感知;
    # 尝试将1～64变为1～128（间隔为2），效果一般
    fdim = 64
    scale = 2 * math.pi
    dim_t = torch.arange(0, fdim, 1, dtype=torch.float32, device=pos_tensor.device) 
    dim_t = 10000 ** (2 * (dim_t // 2) / fdim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    w_embed = pos_tensor[:, :, 2] * scale
    h_embed = pos_tensor[:, :, 3] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_w = w_embed[:, :, None] / dim_t
    pos_h = h_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
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
                 num_queries = 300):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_dec_layers = num_decoder_layers
        self.num_queries = num_queries

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, src_box_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed, box_embed=src_box_embed)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return hs.transpose(1, 2), references


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        
        # self.query_scale = MLP(d_model, d_model, d_model, 2)
        
        self.query_scale_x = MLP(d_model, d_model, d_model//2, 2)
        self.query_scale_y = MLP(d_model, d_model, d_model//2, 2)
        # self.wh_sine_proj_x = nn.Linear(d_model//2, d_model//2)
        # self.wh_sine_proj_y = nn.Linear(d_model//2, d_model//2)

        self.out_proj = nn.Linear(d_model, d_model//2)

        # self.norm_scale = nn.LayerNorm(d_model)
        
        # self.qref_point_head = MLP(d_model, d_model, 4, 2) # 不使用效果好
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        
        # self.layers[0].ca_qpos_sine_proj = None
        self.layers[0].ca_v_proj = None
        self.layers[0].ca_kcontent_proj = None
        self.layers[0].ca_kpos_proj = None
        self.layers[0].ca_qcontent_proj = None
        
        # ref points
        self.ref_point_maps = nn.ModuleList(nn.Conv2d(d_model, d_model, kernel_size=1) for i in range(num_layers-1))
        self.ref_point_head = nn.Linear(d_model, 4) # box 使用同一个
                
        self.update_gate = UpdateGate(
            d_model=d_model,
            num_layers=num_layers,
            gate_type="adaptive",
            use_highway=False
        )

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
        reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]

        prev_gate = None  # 保存前一层门控值用于稳定性

        for layer_id, layer in enumerate(self.layers):
            # obj_center = reference_points[..., :].transpose(0, 1)

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
                box_off = 0
                reference_points_before_sigmoid_new = reference_points_before_sigmoid
            else:
                # pos_transformation = self.query_scale(output)
                box_embed = box_embed + self.ref_point_maps[layer_id-1](box_embed)
                box_off_embed = box_embed.flatten(2).permute(2, 0, 1)
                box_off = self.ref_point_head(box_off_embed)
                
                # 门控 更新参考点
                # reference_points_before_sigmoid_new = (reference_points_before_sigmoid + box_off)
                gate_value, highway_weight = self.update_gate(layer_id, output, box_off_embed, prev_gate)
                prev_gate = gate_value.detach()  # 保存当前门控值
                # 应用门控
                updated_ref = gate_value * box_off + reference_points_before_sigmoid
                # 高速连接
                if highway_weight is not None:
                    reference_points_before_sigmoid_new = highway_weight * updated_ref + (1 - highway_weight) * reference_points_before_sigmoid
                else:
                    reference_points_before_sigmoid_new = updated_ref
                                
            reference_points = reference_points_before_sigmoid_new.sigmoid().transpose(0, 1)
            obj_box = reference_points[..., :].transpose(0, 1)

            # obj_box_wh = torch.log(obj_box[..., 2:] + 1e-8)
            # obj_box_cen = obj_box[..., :2]
            # query_sine_embed = gen_sineembed_for_position(obj_box) * pos_transformation
            
            if layer_id != 0:
                sin_wh = gen_sineembed_for_position(torch.log(obj_box[..., 2:] + 1e-8))
                # pos_transformation_x = self.query_scale_x(torch.cat([output, self.wh_sine_proj_x(sin_wh[..., 128:])], dim=-1))
                # pos_transformation_y = self.query_scale_y(torch.cat([output, self.wh_sine_proj_y(sin_wh[..., :128])], dim=-1))

                output_half = self.out_proj(output)
                pos_transformation_x = self.query_scale_x(torch.cat([output_half, sin_wh[..., 128:]], dim=-1))
                pos_transformation_y = self.query_scale_y(torch.cat([output_half, sin_wh[..., :128]], dim=-1))
                pos_transformation = torch.cat([pos_transformation_x, pos_transformation_y], dim=-1)
                
            query_sine_embed = gen_sineembed_for_position(obj_box[..., :2])*pos_transformation
            
            # do layer compute
            output, _ = layer(output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
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
            return [torch.stack(intermediate), torch.stack(reference_points_lvl)]

        return output.unsqueeze(0), reference_points.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        # self.multihead_attn = nn.MultiheadAttention(d_model*2, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        head_dim = dim_feedforward//2
        self.linear1 = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(head_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.nhead = nhead
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        # 
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        # self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        
        # split position, test two multihead_attn
        self.ca_v_proj_pos = nn.Linear(d_model, d_model)
        
        # for tgt_pos  mode=0/2
        self.norm_pos2 = nn.LayerNorm(d_model)
        self.norm_pos3 = nn.LayerNorm(d_model)
        self.dropout_pos2 = nn.Dropout(dropout)
        self.dropout_pos3 = nn.Dropout(dropout)
        
        self.linear_pos1 = nn.Linear(d_model, head_dim)
        self.dropout_pos = nn.Dropout(dropout)
        self.linear_pos2 = nn.Linear(head_dim, d_model)
        
        self.norm_add = nn.LayerNorm(d_model)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ========== Begin of Cross-Attention =============
        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        
        q_content = tgt
        k_content = memory
        if is_first:
            k_pos = pos
            q_pos = self.ca_qpos_proj(query_pos)
            
            q = q_content + q_pos
            k = k_content + k_pos
    
            v = memory
        else:
            k_pos = self.ca_kpos_proj(pos)
            
            q = q_content
            # k = k_content
        
            v = self.ca_v_proj(memory)
            k = self.ca_kcontent_proj(k_content) # help for train, ap 0.031->0.036 when testing two multihead_attn
            # q = self.ca_qcontent_proj(q_content)

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        
        atten_mode = 0
        if atten_mode == 0:
            # use multihead_attn twice for context and position
            tgt2 = self.multihead_attn(query=q, key=k, value=v, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
            v_pos = self.ca_v_proj_pos(memory)
            tgt2_pos = self.multihead_attn(query=query_sine_embed, key=k_pos, value=v_pos, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)[0]  
            tgt2 += tgt2_pos
            
            tgt_pos = tgt + self.dropout_pos2(tgt2_pos)
            tgt_pos = self.norm_pos2(tgt_pos)
            tgt2_pos = self.linear_pos2(self.dropout_pos(self.activation(self.linear_pos1(tgt_pos))))
            tgt_pos = tgt_pos + self.dropout_pos3(tgt2_pos)
            tgt_pos = self.norm_pos3(tgt_pos)
            
        elif atten_mode == 1:
            # 0.025
            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape
            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.multihead_attn(query=q, key=k, value=v, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        elif atten_mode == 2:
            v_pos = self.ca_v_proj_pos(memory)
            q_cat = torch.cat([q, query_sine_embed], dim=2)
            k_cat = torch.cat([k, k_pos], dim=2)
            v_cat = torch.cat([v, v_pos], dim=2)
            
            tgt2_cat = self.multihead_attn(query=q_cat, key=k_cat, value=v_cat, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)[0]
            n_q,_, n_model = tgt2_cat.shape
            
            tgt2 = tgt2_cat[..., :n_model//2] + tgt2_cat[..., n_model//2:]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(q, query_sine_embed),
                                        key=self.with_pos_embed(k, k_pos),
                                        value=v, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        if atten_mode == 0:
            tgt = tgt + tgt_pos
            tgt = self.norm_add(tgt)
        else:
            tgt_pos = None

        return tgt, tgt_pos

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)
#'''

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

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

    def forward_post(self,
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

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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
        num_queries = args.num_queries
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
