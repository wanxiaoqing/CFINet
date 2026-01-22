import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math


class MSFIM(nn.Module):
    def __init__(self, input_channels_list, fused_channels, norm, act, pool_type="avg"):
        super().__init__()
        self.num_features = len(input_channels_list)

        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fused_channels, 1, bias=False),
                norm(fused_channels),
                act(),
            ) for in_channels in input_channels_list
        ])

        self.attn_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fused_channels, fused_channels, 1, bias=False),
                norm(fused_channels),
                act(),
            ) for _ in range(self.num_features)
        ])

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"不支持的池化方式: {pool_type}")

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        assert len(feature_list) == self.num_features, "输入特征列表的长度必须与 input_channels_list 匹配。"

        aligned_features = []
        for i in range(self.num_features):
            # 将输入特征通道对齐
            aligned_feature = self.align_convs[i](feature_list[i])
            aligned_features.append(aligned_feature)

        updated_features = []
        for i in range(self.num_features):
            x = aligned_features[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)
            x = attn * x + x
            updated_features.append(x)

        fused_output = sum(updated_features)

        return fused_output


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SDAM(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(SDAM, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.relu(self.gn(self.conv(x_h))).view(b, c, h, 1)

        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.relu(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w


class CECAM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(CECAM, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio)

        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(nn.LayerNorm(self.cr), nn.GELU())  # 修正：`normalized_shape` 为 `self.cr`

        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        _scale = 4 ** _time

        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()
        _x = self.norm_act(_x)

        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = v + self.cpe(
            v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)
        ).view(B, C, -1).view(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HAFM(nn.Module):
    def __init__(self, channel, dim):
        super(HAFM, self).__init__()
        self.sda = SDAM(channel)
        self.norm1 = LayerNorm(dim)
        self.ceca = CECAM(dim)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=dim * 4)

    def forward(self, x):
        x = self.sda(x)

        B, C, H, W = x.shape
        x_reshape = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        residual = x_reshape
        x_reshape = self.norm1(x_reshape)
        x_reshape = self.ceca(x_reshape, H, W)
        x_reshape = x_reshape + residual

        residual = x_reshape
        x_reshape = self.norm2(x_reshape)
        x_reshape = self.mlp(x_reshape, H, W)
        x_reshape = x_reshape + residual

        x = x_reshape.permute(0, 2, 1).view(B, C, H, W)

        return x


class FeatureReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hafm = HAFM(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.hafm(x)
        x = self.conv(x)
        return x


class CFINet(nn.Module):
    def __init__(self, channels=1, num_classes=16, drop=0.1):
        super(CFINet, self).__init__()
        self.stem_3D = nn.Sequential(
            nn.Conv3d(channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.stem_2D = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=128, kernel_size=(3, 3)),  # 32 * 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        channels_sequence = [128, 64, 32, 16]

        self.feature_blocks = nn.ModuleList()
        for i in range(len(channels_sequence) - 1):
            in_ch = channels_sequence[i]
            out_ch = channels_sequence[i + 1]
            self.feature_blocks.append(FeatureReductionBlock(in_ch, out_ch))

        self.feature_preserve_convs = nn.ModuleList([
            nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1)
        ])

        input_channels_list = [128, 64, 32, 16, 128]
        fused_channels = 128
        norm = nn.BatchNorm2d
        act = nn.ReLU
        self.msfi = MSFIM(input_channels_list, fused_channels, norm, act)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem_3D(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.stem_2D(x)

        shortcut = x
        intermediate_features = []

        # 遍历特征处理块
        current_feature = x
        for i, block in enumerate(self.feature_blocks):
            current_feature = block(current_feature)

            preserved_feature = self.feature_preserve_convs[i + 1](current_feature) if i > 0 else current_feature
            intermediate_features.append(preserved_feature)

        first_block_output = self.feature_preserve_convs[0](x)
        intermediate_features.insert(0, first_block_output)

        intermediate_features.append(shortcut)

        # 多尺度特征融合
        x = self.msfi(intermediate_features)

        x = x + shortcut
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CFINet(channels=1, num_classes=9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 30, 13, 13).to(device)
    y = model(input)
    print(y.size())
