import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
from PIL import Image


def signed_polygon_area(points):
    area = 0
    ax, ay = points[-1]
    for bx, by in points:
        area += ax * by - bx * ay
        ax = bx
        ay = by
    return area / 2


def polygon_area(points):
    return abs(signed_polygon_area(points))


def find_outlines(mask):
    h, w = mask.shape
    p = np.pad(mask, 1, constant_values=False)
    h_edges = p[:-1, 1:-1] != p[1:, 1:-1]
    v_edges = p[1:-1, :-1] != p[1:-1, 1:]

    outlines = []
    hy, hx = np.nonzero(h_edges)
    for y0, x0 in zip(hy, hx):
        if not h_edges[y0, x0]:
            continue

        r = p[y0, x0 + 1]
        start = (x0 + r, y0)
        x = x0 + 1 - r
        y = y0
        dx = 1 if r else -1
        dy = 0

        h_edges[y0, x0] = False
        outline = [start]

        while True:
            outline.append((x, y))
            if (x, y) == start:
                break
            for ndx, ndy in [(-dy, dx), (dx, dy), (dy, -dx), (-dx, -dy)]:
                if ndx:
                    ex = x if ndx == 1 else x - 1
                    if 0 <= ex < w and h_edges[y, ex]:
                        h_edges[y, ex] = False
                        break
                elif ndy:
                    ey = y if ndy == 1 else y - 1
                    if 0 <= ey < h and v_edges[ey, x]:
                        v_edges[ey, x] = False
                        break
            else:
                break

            x += ndx
            y += ndy
            dx = -ndx
            dy = -ndy

        outlines.append(outline)

    return outlines


def load_safetensors(path):
    with open(path, "rb") as file:
        header_len = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_len))
        data = file.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__": continue
        dtype = {"F32": torch.float32}[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        buffer = bytearray(data[start:end])
        tensor = torch.frombuffer(buffer, dtype=dtype).clone().reshape(shape)
        tensors[name] = tensor
    return tensors


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class HGNetV2ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=nn.ReLU):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.normalization = PPDocLayoutV3FrozenBatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class HGNetV2ConvLayerLight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = HGNetV2ConvLayer(in_channels, out_channels, 1, 1, 1, nn.Identity)
        self.conv2 = HGNetV2ConvLayer(out_channels, out_channels, kernel_size, 1, out_channels)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class HGNetV2Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = HGNetV2ConvLayer(3, 32, 3, 2)
        self.stem2a = HGNetV2ConvLayer(32, 16, 2, 1)
        self.stem2b = HGNetV2ConvLayer(16, 32, 2, 1)
        self.stem3 = HGNetV2ConvLayer(64, 32, 3, 2)
        self.stem4 = HGNetV2ConvLayer(32, 48, 1, 1)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, pixel_values):
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, (0, 1, 0, 1))
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled_emb = self.pool(embedding)
        embedding = torch.cat([pooled_emb, emb_stem_2a], dim=1)
        embedding = self.stem3(embedding)
        embedding = self.stem4(embedding)
        return embedding


class HGNetV2BasicLayer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, residual, ConvLayer):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList([
            ConvLayer(in_channels, middle_channels, kernel_size),
            ConvLayer(middle_channels, middle_channels, kernel_size),
            ConvLayer(middle_channels, middle_channels, kernel_size),
            ConvLayer(middle_channels, middle_channels, kernel_size),
            ConvLayer(middle_channels, middle_channels, kernel_size),
            ConvLayer(middle_channels, middle_channels, kernel_size),
        ])
        self.aggregation = nn.Sequential(
            HGNetV2ConvLayer(in_channels + 6 * middle_channels, out_channels // 2, 1, 1),
            HGNetV2ConvLayer(out_channels // 2, out_channels, 1, 1),
        )

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        return x + identity if self.residual else x


class HGNetV2Stage(nn.Module):
    def __init__(self, downsample, blocks):
        super().__init__()
        self.downsample = downsample
        self.blocks = blocks

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGNetV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            HGNetV2Stage(
                nn.Identity(),
                nn.ModuleList([HGNetV2BasicLayer(48, 48, 128, 3, False, HGNetV2ConvLayer)]),
            ),
            HGNetV2Stage(
                HGNetV2ConvLayer(128, 128, 3, 2, 128, nn.Identity),
                nn.ModuleList([HGNetV2BasicLayer(128, 96, 512, 3, False, HGNetV2ConvLayer)]),
            ),
            HGNetV2Stage(
                HGNetV2ConvLayer(512, 512, 3, 2, 512, nn.Identity),
                nn.ModuleList([
                    HGNetV2BasicLayer(512, 192, 1024, 5, False, HGNetV2ConvLayerLight),
                    HGNetV2BasicLayer(1024, 192, 1024, 5, True, HGNetV2ConvLayerLight),
                    HGNetV2BasicLayer(1024, 192, 1024, 5, True, HGNetV2ConvLayerLight),
                ]),
            ),
            HGNetV2Stage(
                HGNetV2ConvLayer(1024, 1024, 3, 2, 1024, nn.Identity),
                nn.ModuleList([HGNetV2BasicLayer(1024, 384, 2048, 5, False, HGNetV2ConvLayerLight)]),
            ),
        ])

    def forward(self, x):
        h0 = self.stages[0](x)
        h1 = self.stages[1](h0)
        h2 = self.stages[2](h1)
        h3 = self.stages[3](h2)
        return [h0, h1, h2, h3]


class HGNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = HGNetV2Embeddings()
        self.encoder = HGNetV2Encoder()

    def forward(self, x):
        return self.encoder(self.embedder(x))


class PPDocLayoutV3BackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HGNetV2Backbone()

    def forward(self, x):
        return self.model(x)


class PPDocLayoutV3GlobalPointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        query_key_projection = self.dense(inputs).reshape(batch_size, sequence_length, 2, 64)
        query_key_projection = self.dropout(query_key_projection)
        queries, keys = torch.unbind(query_key_projection, dim=2)
        logits = (queries @ keys.transpose(-2, -1)) / 8.0
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=logits.device)).bool()
        logits = logits.masked_fill(mask.unsqueeze(0), -1e4)
        return logits


class PPDocLayoutV3MultiscaleDeformableAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_offsets = nn.Linear(256, 8 * 3 * 4 * 2)
        self.attention_weights = nn.Linear(256, 8 * 3 * 4)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)

    def forward(self, hidden_states, encoder_hidden_states, reference_points, spatial_shapes_list):
        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        value = self.value_proj(encoder_hidden_states)
        value = value.view(batch_size, sequence_length, 8, 32)
        sampling_offsets = self.sampling_offsets(hidden_states).view(batch_size, num_queries, 8, 3, 4, 2)
        attention_weights = self.attention_weights(hidden_states).view(batch_size, num_queries, 8, 3 * 4)
        attention_weights = F.softmax(attention_weights, -1).view(batch_size, num_queries, 8, 3, 4)
        sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / 4 * reference_points[:, :, None, :, None, 2:] * 0.5
        value_list = value.split([height * width for height, width in spatial_shapes_list], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        outputs = []
        for level_id, shape in enumerate(spatial_shapes_list):
            value_l_ = value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * 8, 32, *shape)
            sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, align_corners=False)
            outputs.append(sampling_value_l_)
        attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * 8, 1, num_queries, 3 * 4)
        output = torch.stack(outputs, dim=-2).flatten(-2) * attention_weights
        output = output.sum(-1).view(batch_size, 8 * 32, num_queries)
        output = output.transpose(1, 2).contiguous()
        output = self.output_proj(output)
        return output


class PPDocLayoutV3MLPPredictionHead(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class PPDocLayoutV3ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.silu(self.normalization(self.convolution(x)))


class PPDocLayoutV3ScaleHead(nn.Module):
    def __init__(self, in_channels, feature_channels, fpn_stride, base_stride, align_corners=False):
        super().__init__()
        head_length = max(1, int(np.log2(fpn_stride) - np.log2(base_stride)))
        self.layers = nn.ModuleList()
        for k in range(head_length):
            in_c = in_channels if k == 0 else feature_channels
            self.layers.append(PPDocLayoutV3ConvLayer(in_c, feature_channels))
            if fpn_stride != base_stride:
                self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PPDocLayoutV3MaskFeatFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_heads = nn.ModuleList([
            PPDocLayoutV3ScaleHead(256, 64, 8, 8),
            PPDocLayoutV3ScaleHead(256, 64, 16, 8),
            PPDocLayoutV3ScaleHead(256, 64, 32, 8),
        ])
        self.output_conv = PPDocLayoutV3ConvLayer(64, 64)

    def forward(self, inputs):
        output = self.scale_heads[0](inputs[0])
        output += F.interpolate(self.scale_heads[1](inputs[1]), size=output.shape[2:], mode="bilinear")
        output += F.interpolate(self.scale_heads[2](inputs[2]), size=output.shape[2:], mode="bilinear")
        output = self.output_conv(output)
        return output


class PPDocLayoutV3EncoderMaskOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_conv = PPDocLayoutV3ConvLayer(64, 64)
        self.conv = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        return self.conv(self.base_conv(x))


class PPDocLayoutV3SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.q_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)

    def forward(self, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 32)
        query_key_input = hidden_states + position_embeddings
        query_states = self.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, scale=32**-0.5)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class PPDocLayoutV3ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, activation=nn.Identity):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class PPDocLayoutV3EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = PPDocLayoutV3SelfAttention()
        self.self_attn_layer_norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.final_layer_norm = nn.LayerNorm(256)

    def forward(self, x, spatial_position_embeddings):
        x += self.self_attn(x, spatial_position_embeddings)
        x = self.self_attn_layer_norm(x)
        x += self.fc2(F.gelu(self.fc1(x)))
        x = self.final_layer_norm(x)
        return x


class PPDocLayoutV3DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = PPDocLayoutV3SelfAttention()
        self.self_attn_layer_norm = nn.LayerNorm(256)
        self.encoder_attn = PPDocLayoutV3MultiscaleDeformableAttention()
        self.encoder_attn_layer_norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.final_layer_norm = nn.LayerNorm(256)

    def forward(self, x, object_queries_position_embeddings, encoder_hidden_states, reference_points, spatial_shapes_list):
        x += self.self_attn(x, object_queries_position_embeddings)
        x = self.self_attn_layer_norm(x)
        x += self.encoder_attn(x + object_queries_position_embeddings, encoder_hidden_states, reference_points, spatial_shapes_list)
        x = self.encoder_attn_layer_norm(x)
        x += self.fc2(F.relu(self.fc1(x)))
        x = self.final_layer_norm(x)
        return x


class PPDocLayoutV3RepVggBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PPDocLayoutV3ConvNormLayer(256, 256, 3, 1, padding=1)
        self.conv2 = PPDocLayoutV3ConvNormLayer(256, 256, 1, 1, padding=0)

    def forward(self, x):
        return F.silu(self.conv1(x) + self.conv2(x))


class PPDocLayoutV3CSPRepLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PPDocLayoutV3ConvNormLayer(512, 256, 1, 1, activation=nn.SiLU)
        self.conv2 = PPDocLayoutV3ConvNormLayer(512, 256, 1, 1, activation=nn.SiLU)
        self.bottlenecks = nn.Sequential(*[PPDocLayoutV3RepVggBlock() for _ in range(3)])

    def forward(self, x):
        return self.conv2(x) + self.bottlenecks(self.conv1(x))


class PPDocLayoutV3SinePositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, width, height, device, dtype):
        omega = torch.arange(64, dtype=torch.float64, device=device) / 64
        omega = 10000**-omega
        grid_h = torch.arange(height, dtype=torch.float64, device=device)
        grid_w = torch.arange(width, dtype=torch.float64, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        emb_h = grid_h.flatten().outer(omega)
        emb_w = grid_w.flatten().outer(omega)
        pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
        return pos_embed.to(dtype).unsqueeze(0)


class PPDocLayoutV3AIFILayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = PPDocLayoutV3SinePositionEmbedding()
        self.layers = nn.ModuleList([PPDocLayoutV3EncoderLayer()])

    def forward(self, x):
        batch_size = x.shape[0]
        height, width = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        pos_embed = self.position_embedding(width=width, height=height, device=x.device, dtype=x.dtype)
        x = self.layers[0](x, pos_embed)
        x = x.transpose(1, 2).reshape(batch_size, 256, height, width).contiguous()
        return x


def _get_bounds(values, mask):
    masked_values = values * mask
    max_value = masked_values.flatten(start_dim=-2).max(dim=-1).values + 1
    dtype_max = torch.tensor(torch.finfo(values.dtype).max, device=values.device)
    min_value = torch.where(mask, masked_values, dtype_max).flatten(start_dim=-2).min(dim=-1).values
    return min_value, max_value


def mask_to_box_coordinate(mask, dtype):
    mask = mask.bool()

    height, width = mask.shape[-2:]

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=mask.device),
        torch.arange(width, device=mask.device),
        indexing="ij"
    )
    x_coords = x_coords.to(dtype)
    y_coords = y_coords.to(dtype)

    x_min, x_max = _get_bounds(x_coords, mask)
    y_min, y_max = _get_bounds(y_coords, mask)

    unnormalized_bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    is_mask_non_empty = torch.any(mask, dim=(-2, -1)).unsqueeze(-1)
    unnormalized_bbox = unnormalized_bbox * is_mask_non_empty

    norm_tensor = torch.tensor([width, height, width, height], device=mask.device, dtype=dtype)
    normalized_bbox_xyxy = unnormalized_bbox / norm_tensor

    x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalized_bbox_xyxy.unbind(dim=-1)

    center_x = (x_min_norm + x_max_norm) / 2
    center_y = (y_min_norm + y_max_norm) / 2
    box_width = x_max_norm - x_min_norm
    box_height = y_max_norm - y_min_norm

    return torch.stack([center_x, center_y, box_width, box_height], dim=-1)


class PPDocLayoutV3HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([PPDocLayoutV3AIFILayer()])
        self.lateral_convs = nn.ModuleList([
            PPDocLayoutV3ConvNormLayer(256, 256, 1, 1, activation=nn.SiLU),
            PPDocLayoutV3ConvNormLayer(256, 256, 1, 1, activation=nn.SiLU),
        ])
        self.fpn_blocks = nn.ModuleList([PPDocLayoutV3CSPRepLayer(), PPDocLayoutV3CSPRepLayer()])
        self.downsample_convs = nn.ModuleList([
            PPDocLayoutV3ConvNormLayer(256, 256, 3, 2, activation=nn.SiLU),
            PPDocLayoutV3ConvNormLayer(256, 256, 3, 2, activation=nn.SiLU),
        ])
        self.pan_blocks = nn.ModuleList([PPDocLayoutV3CSPRepLayer(), PPDocLayoutV3CSPRepLayer()])
        self.mask_feature_head = PPDocLayoutV3MaskFeatFPN()
        self.encoder_mask_lateral = PPDocLayoutV3ConvLayer(128, 64)
        self.encoder_mask_output = PPDocLayoutV3EncoderMaskOutput()

    def forward(self, inputs_embeds=None, x4_feat=None):
        feature_maps = inputs_embeds
        feature_maps[2] = self.encoder[0](feature_maps[2])
        fpn_feature_maps = [feature_maps[-1]]
        for idx in range(2):
            backbone_feature_map = feature_maps[1 - idx]
            top_fpn_feature_map = self.lateral_convs[idx](fpn_feature_maps[-1])
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = self.fpn_blocks[idx](fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)
        fpn_feature_maps.reverse()
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx in range(2):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = self.downsample_convs[idx](top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = self.pan_blocks[idx](fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)
        mask_feat = self.mask_feature_head(pan_feature_maps)
        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode="bilinear", align_corners=False)
        mask_feat += self.encoder_mask_lateral(x4_feat)
        mask_feat = self.encoder_mask_output(mask_feat)
        return pan_feature_maps, mask_feat


class PPDocLayoutV3Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PPDocLayoutV3DecoderLayer() for _ in range(6)])
        self.query_pos_head = PPDocLayoutV3MLPPredictionHead(nn.ModuleList([nn.Linear(4, 512), nn.Linear(512, 256)]))

    def forward(self, inputs_embeds, encoder_hidden_states, reference_points, spatial_shapes_list, order_head, global_pointer, mask_query_head, norm, mask_feat, class_embed, bbox_embed):
        hidden_states = inputs_embeds
        intermediate = []
        intermediate_reference_points = []
        intermediate_logits = []
        decoder_out_order_logits = []
        decoder_out_masks = []
        reference_points = F.sigmoid(reference_points)
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            object_queries_position_embeddings = self.query_pos_head(reference_points)
            hidden_states = decoder_layer(hidden_states, object_queries_position_embeddings, encoder_hidden_states, reference_points_input, spatial_shapes_list)
            predicted_corners = bbox_embed(hidden_states)
            new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
            reference_points = new_reference_points.detach()
            intermediate.append(hidden_states)
            intermediate_reference_points.append(new_reference_points)
            out_query = norm(hidden_states)
            mask_query_embed = mask_query_head(out_query)
            batch_size, mask_dim, _ = mask_query_embed.shape
            _, _, mask_h, mask_w = mask_feat.shape
            out_mask = torch.bmm(mask_query_embed, mask_feat.flatten(start_dim=2)).reshape(batch_size, mask_dim, mask_h, mask_w)
            decoder_out_masks.append(out_mask)
            logits = class_embed(out_query)
            intermediate_logits.append(logits)
            valid_query = out_query[:, -300:]
            order_logits = global_pointer(order_head[idx](valid_query))
            decoder_out_order_logits.append(order_logits)
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        intermediate_logits = torch.stack(intermediate_logits, dim=1)
        decoder_out_order_logits = torch.stack(decoder_out_order_logits, dim=1)
        decoder_out_masks = torch.stack(decoder_out_masks, dim=1)
        return intermediate_logits, intermediate_reference_points, decoder_out_order_logits, decoder_out_masks


class PPDocLayoutV3FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.num_features = num_features

    def forward(self, x):
        c = self.num_features
        weight = self.weight.reshape(1, c, 1, 1)
        bias = self.bias.reshape(1, c, 1, 1)
        running_var = self.running_var.reshape(1, c, 1, 1)
        running_mean = self.running_mean.reshape(1, c, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


class PPDocLayoutV3InnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PPDocLayoutV3BackboneWrapper()
        self.encoder_input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
        ])
        self.encoder = PPDocLayoutV3HybridEncoder()
        self.denoising_class_embed = nn.Embedding(25, 256)
        self.enc_output = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
        self.enc_score_head = nn.Linear(256, 25)
        self.enc_bbox_head = PPDocLayoutV3MLPPredictionHead(nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 4)]))
        self.decoder_input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)),
        ])
        self.decoder = PPDocLayoutV3Decoder()
        self.decoder_order_head = nn.ModuleList([nn.Linear(256, 256) for _ in range(6)])
        self.decoder_global_pointer = PPDocLayoutV3GlobalPointer()
        self.decoder_norm = nn.LayerNorm(256)
        self.mask_query_head = PPDocLayoutV3MLPPredictionHead(nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 32)]))

    def generate_anchors(self, spatial_shapes, grid_size=0.05, device="cpu", dtype=torch.float32):
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        eps = 1e-2
        anchors1 = torch.concat(anchors, 1)
        valid_mask = ((anchors1 > eps) * (anchors1 < 1 - eps)).all(-1, keepdim=True)
        anchors2 = torch.log(anchors1 / (1 - anchors1))
        anchors3 = torch.where(valid_mask, anchors2, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))

        return anchors3, valid_mask

    def forward(self, pixel_values):
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        features = self.backbone(pixel_values)
        x4_feat = features[0]
        proj_feats = [self.encoder_input_proj[level](features[level + 1]) for level in range(3)]
        encoder_outputs_last_hidden_state, encoder_outputs_mask_feat = self.encoder(proj_feats, x4_feat)
        sources = [self.decoder_input_proj[level](source) for level, source in enumerate(encoder_outputs_last_hidden_state)]
        source_flatten = []
        spatial_shapes_list = []
        for source in sources:
            height, width = source.shape[-2:]
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)
        dtype = source_flatten.dtype
        anchors, valid_mask = self.generate_anchors(tuple(spatial_shapes_list), device=device, dtype=dtype)
        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, 300, dim=1)
        reference_points_unact = enc_outputs_coord_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1]))
        batch_ind = torch.arange(memory.shape[0], device=output_memory.device).unsqueeze(1)
        target = output_memory[batch_ind, topk_ind]
        out_query = self.decoder_norm(target)
        mask_query_embed = self.mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach()
        _, _, mask_h, mask_w = encoder_outputs_mask_feat.shape
        enc_out_masks = torch.bmm(mask_query_embed, encoder_outputs_mask_feat.flatten(start_dim=2)).reshape(batch_size, mask_dim, mask_h, mask_w)
        reference_points = mask_to_box_coordinate(enc_out_masks > 0, dtype=reference_points_unact.dtype)
        reference_points_unact = inverse_sigmoid(reference_points)
        init_reference_points = reference_points_unact.detach()
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            reference_points=init_reference_points,
            spatial_shapes_list=spatial_shapes_list,
            order_head=self.decoder_order_head,
            global_pointer=self.decoder_global_pointer,
            mask_query_head=self.mask_query_head,
            norm=self.decoder_norm,
            mask_feat=encoder_outputs_mask_feat,
            class_embed=self.enc_score_head,
            bbox_embed=self.enc_bbox_head,
        )
        intermediate_logits, intermediate_reference_points, order_logits, out_masks = decoder_outputs
        pred_boxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]
        order_logits = order_logits[:, -1]
        out_masks = out_masks[:, -1]
        return logits, pred_boxes, order_logits, out_masks


def download(url, path):
    if os.path.exists(path): return

    directory, filename = os.path.split(path)

    if directory:
        os.makedirs(directory, exist_ok=True, parents=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, 100 * downloaded // total_size)
        bar = "█" * (percent // 2) + "-" * (50 - percent // 2)
        print(f"\r{filename}  |{bar}| {percent:3}%", flush=True, end="")

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path, reporthook=_progress)
    print()


class SimplePPDocLayoutV3(nn.Module):
    def __init__(
        self,
        device=None,
        filename=os.path.expanduser("~/.cache/huggingface/hub/models--PaddlePaddle--PP-DocLayoutV3_safetensors/blobs/5ea422c6cc5fe759a47e1357c35639b58173508e025a3131cbe4b6ac59e2b85e"),
        url="https://huggingface.co/zai-org/GLM-OCR/resolve/main/model.safetensors",
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        download(url, filename)

        state_dict = load_safetensors(filename)

        self.device = device

        with torch.device("meta"):
            self.model = PPDocLayoutV3InnerModel()

        self.load_state_dict(state_dict, assign=True)
        self.to(device)
        self.eval()

    def run(self, images, threshold=0.5):
        with torch.no_grad():
            images = [Image.open(image) if isinstance(image, str) else image for image in images]
            images = [image.convert("RGB") for image in images]

            inputs = []
            for image in images:
                image = torch.from_numpy(np.array(image))
                image = image.permute(2, 0, 1).contiguous()
                image = F.interpolate(image.unsqueeze(0), size=[800, 800], mode="bicubic", align_corners=False, antialias=False).squeeze(0) / 255.0
                inputs.append(image)
            inputs = torch.stack(inputs).to(self.device)

            target_sizes = [image.size[::-1] for image in images]
            logits, boxes, order_logits, masks = self.model(inputs)
            order_scores = torch.sigmoid(order_logits)
            batch_size, sequence_length, _ = order_scores.shape
            order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (1.0 - order_scores.transpose(1, 2)).tril(diagonal=-1).sum(dim=1)
            order_pointers = torch.argsort(order_votes, dim=1)
            order_seq = torch.empty_like(order_pointers)
            ranks = torch.arange(sequence_length, device=order_pointers.device, dtype=order_pointers.dtype).expand(batch_size, -1)
            order_seqs = order_seq.scatter_(1, order_pointers, ranks)
            box_centers, box_dims = torch.split(boxes, 2, dim=-1)
            top_left_coords = box_centers - 0.5 * box_dims
            bottom_right_coords = box_centers + 0.5 * box_dims
            boxes = torch.cat([top_left_coords, bottom_right_coords], dim=-1)
            img_height, img_width = torch.as_tensor(target_sizes).unbind(1)
            scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1).to(boxes.device)
            boxes = boxes * scale_factor[:, None, :]
            num_top_queries = logits.shape[1]
            num_classes = logits.shape[2]
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index % num_classes
            index = index // num_classes
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
            masks = masks.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, masks.shape[-2], masks.shape[-1]))
            masks = (masks.sigmoid() > threshold).int()
            order_seqs = order_seqs.gather(dim=1, index=index)

            id2label = [
                "abstract", "algorithm", "aside_text", "chart", "content",
                "formula", "doc_title", "figure_title", "footer", "footer",
                "footnote", "formula_number", "header", "header", "image",
                "formula", "number", "paragraph_title", "reference",
                "reference_content", "seal", "table", "text", "text",
                "vision_footnote"
            ]

            results = []
            for score, label, box, order_seq, mask, image in zip(
                scores, labels, boxes, order_seqs, masks, images
            ):
                valid = score >= threshold
                order_seq, indices = torch.sort(order_seq[valid])
                valid_masks = mask[valid][indices].cpu().numpy().astype(bool)

                outlines = []
                for mask in valid_masks:
                    outline = max(find_outlines(mask), key=polygon_area)
                    mh, mw = mask.shape
                    w, h = image.size
                    outline = np.array(outline) * [w / mw, h / mh]
                    outlines.append(outline)

                labels = [id2label[label_id] for label_id in label[valid][indices]]

                results.append({
                    "image": image,
                    "scores": score[valid][indices].cpu().numpy(),
                    "labels": labels,
                    "boxes": box[valid][indices].cpu().numpy(),
                    "masks": valid_masks,
                    "outlines": outlines,
                    "order_seq": order_seq.cpu().numpy(),
                })
            return results
