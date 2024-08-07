import torch
import torch.nn as nn

import math
from timm.models.layers import trunc_normal_


""" Relative Position Index """
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)])) # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1 # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1

    return relative_coords.sum(-1) # Wh*Ww, Wh*Ww


""" Patch Partition + Linear Embedding """
# 224x224 image -> 4x4 patch (total # of sequence: 56x56(=3136))
# convolution channel -> 96
class PatchPartition(nn.Module):
    def __init__(self, patch_size: int = 4):
        super().__init__()

        self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        x = self.proj(x) # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = self.norm(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


""" Swin Transformer Block (1) """
# Efficient batch computation - [B, C, H, W] -> [B, C, H//w, w, W//w, w] -> [B', C, w, w] (B' = B x H//w x W//w)
# (exclude relative positional encoding)
class W_MSA(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads))
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(7, 7))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(49, 49, -1) # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        # setting
        B, L, C = x.shape
        ws = self.window_size
        w = h = int(math.sqrt(L))
        h_ = int(h // ws)
        w_ = int(w // ws)

        # [B, 3136, C]
        # --- efficient batch computation for shifted configuration ---
        x = x.view(B, h, w, C) # [B, H, W, C]
        x = x.view(B, h_, ws, w_, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, 8, 7, 8, 7, 96] -> [B, 8, 8, 7, 7, 96]
        x = x.view(B * h_ * w_, ws * ws, C) # [B', 49, 96] (B' = B x 8 x 8)

        # --- attention ---
        B_, N, C = x.shape # [B_, 49, 96]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # [B_, N, 3, num_heads, ...] -> # [3, B_, num_heads, N, ...]
        q, k, v = qkv.unbind(0) # each value's shape: [B_, num_heads, N, ...]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_rel_pos_bias()

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x) # [B_, 49, 96]

        # --- make multi-batch tensor to original batch tensor ---
        x = x.view(B, h_, w_, ws, ws, C) # [B, 8, 8, 7, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, 8, 7, 8, 7, 96]
        x = x.view(B, h, w, -1) # [B, 56, 56, 96]
        x = x.view(B, h * w, C) # [B, 3136, 96]

        return x
    

""" Swin Transformer Block (2) """
def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows # [num_windows*B, window_size, window_size, C]


""" Swin Transformer Block (3) """
class SW_MSA(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 input_resolution: tuple = (56, 56)):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads))
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(7, 7))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # calculate attention mask for SW-MSA
        self.input_resolution = input_resolution
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        for h in (slice(0, -self.window_size),
                  slice(-self.window_size, -3),
                  slice(-3, None)):
            for w in (slice(0, -self.window_size),
                      slice(-self.window_size, -3),
                      slice(-3, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size) # [num_win, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        self.attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(49, 49, -1) # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        # setting
        B, L, C = x.shape
        ws = self.window_size
        w = h = int(math.sqrt(L))
        h_ = int(h // ws)
        w_ = int(w // ws)

        # [B, 3136, C]
        # --- efficient batch computation for shifted configuration ---
        x = x.view(B, h, w, C) # [B, H, W, C]
        x = torch.roll(x, shifts=(-3, -3), dims=(1, 2)) # [B, H, W, C]
        x = x.view(B, h_, ws, w_, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * h_ * w_, ws * ws, C) # [B', 49, 96] (B' = B x 8 x 8)

        # --- attention ---
        B_, N, C = x.shape # [B_, 49, 96]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # [B_, N, 3, num_heads, ...] -> # [3, B_, num_heads, N, ...]
        q, k, v = qkv.unbind(0) # each value's shape: [B_, num_heads, N, ...]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_rel_pos_bias()

        num_win = self.attn_mask.shape[0]
        if torch.get_device(q) < 0:
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + self.attn_mask.unsqueeze(1).unsqueeze(0)
        else:
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + self.attn_mask.to(torch.get_device(q)).unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x  = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x) # [B_, 49, 96]

        # --- make multi-batch tensor to original batch tensor ---
        x = x.view(B, h_, w_, ws, ws, C) # [B, 8, 8, 7, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, 8, 7, 8, 7, 96]
        x = x.view(B, h, w, -1) # [B, 56, 56, 96]
        x = torch.roll(x, shifts=(3, 3), dims=(1, 2)) # [B, 56, 56, 96]
        x = x.view(B, h * w, C) # [B, 3136, 96]

        return x

        
""" Patch Merging """
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, downscaling_factor=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.norm = nn.LayerNorm(in_channels * downscaling_factor ** 2)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels, bias=False)

    def forward(self, x):
        b, l, c = x.shape
        h, w = self.input_resolution
        x = x.view(b, h, w, c)
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = x.view(-1, new_h * new_w, c * self.downscaling_factor ** 2)
        x = self.norm(x)
        x = self.linear(x)

        return x
    

class SwinBlock(nn.Module):
    def __init__(self,
                 dim: int = 96,
                 num_heads: int = 3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 input_resolution: tuple = (56, 56)):
        super().__init__()

        # for W-MSA
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.w_msa = W_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # for SW-MSA
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        self.sw_msa = SW_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, input_resolution=input_resolution)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.w_msa(self.norm1_1(x)) # [B, 3136, 96]
        x = x + self.mlp1(self.norm1_2(x)) # [B, 3136, 96]

        x = x + self.sw_msa(self.norm2_1(x)) # [B, 3136, 96]
        x = x + self.mlp2(self.norm2_2(x)) # [B, 3136, 96]

        return x


class SwinTransformer(nn.Module):
    def __init__(self,
                 dim=(96, 192, 384, 768),
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 resolutions=(56, 28, 14, 7),
                 num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # patch partition
            PatchPartition(),

            # swin block 1
            nn.Sequential(*[SwinBlock(96, num_heads[0], input_resolution=(resolutions[0], resolutions[0])) for _ in range(depths[0] // 2)]),

            # patch merging 1
            PatchMerging(dim[0], dim[1], (resolutions[0], resolutions[0])),

            # swin block 2
            nn.Sequential(*[SwinBlock(dim[1], num_heads[1], input_resolution=(resolutions[1], resolutions[1])) for _ in range(depths[1] // 2)]),

            # patch merging 2
            PatchMerging(dim[1], dim[2], (resolutions[1], resolutions[1])),

            # swin block 3
            nn.Sequential(*[SwinBlock(dim[2], num_heads[2], input_resolution=(resolutions[2], resolutions[2])) for _ in range(depths[2] // 2)]),

            # patch merging 3
            PatchMerging(dim[2], dim[3], (resolutions[2], resolutions[2])),

            # swin block 4
            nn.Sequential(*[SwinBlock(dim[3], num_heads[3], input_resolution=(resolutions[3], resolutions[3])) for _ in range(depths[3] // 2)]),
        )
        self.norm = nn.LayerNorm(dim[3])
        self.head = nn.Linear(dim[3], num_classes)

    def forward(self, x):
        # param x: [B, 3, 224, 224]
        x = self.features(x) # [B, 49, 768]
        x = self.norm(x) # [B, 49, 768]
        x = x.mean(dim=1) # [B, 768]
        x = self.head(x) # [B, 1000]

        return x


def build_model(opts):
    model = SwinTransformer()
    return model


if __name__ == '__main__':
    img = torch.randn([2, 3, 224, 224])
    model = SwinTransformer()
    print("SwinTransformer: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model(img).size())