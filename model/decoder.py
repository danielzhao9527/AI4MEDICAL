from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange
from model.vmamba.vmamba import VSSBlock, LayerNorm2d, Linear2d
from typing import Sequence, Type, Optional
from model.LCSIM import LCSIM
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from functools import partial

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class COCS(nn.Module):
    def __init__(self, channels, activation='relu'):
        super(COCS, self).__init__()

        self.dwConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            act_layer(activation, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.cab = CAB(channels)
        self.sab = SAB()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        u = self.dwConv(x)
        x = u + x
        u = self.cab(x)
        u = self.sab(u)
        x = x + u
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()

        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp_x = x
        for dw_conv in self.dw_convs:
            dw_out = dw_conv(temp_x)
            x = x + dw_out
        return x


class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.cocs = COCS(in_features)
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.hiddens = hidden_features
        self.out_fea = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cocs(x)
        x = self.fc1(x)    # Linear
        x = self.act(x)    # GELU
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = channel_shuffle(x, gcd(self.hiddens, self.out_fea))
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MCAVSS(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP
            ))
        super(MCAVSS, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x

class FLKPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path: Sequence[float] | float,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = LKPE(in_channels)
        #self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.lcsim = LCSIM(out_channels, out_channels)
        self.vss_layer = MCAVSS(dim=in_channels, depth=depth, drop_path=drop_path)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.vss_layer(input)
        out = self.up(out)
        out_temp = self.lcsim(X=out, Y=skip)
        out = out + out_temp
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(depths) - 1) * 2)]

        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

        self.mcavss3 = MCAVSS(dim=dims[0], depth=depths[0], drop_path=dpr[sum(depths[: 0]): sum(depths[: 1])])
        self.lkpe3 = LKPE(dims[0])
        self.lcsim3 = LCSIM(dims[1], dims[1])

        self.mcavss2 = MCAVSS(dim=dims[1], depth=depths[1], drop_path=dpr[sum(depths[: 1]): sum(depths[: 2])])
        self.lkpe2 = LKPE(dims[1])
        self.lcsim2 = LCSIM(dims[2], dims[2])

        self.mcavss1 = MCAVSS(dim=dims[2], depth=depths[2], drop_path=dpr[sum(depths[: 2]): sum(depths[: 3])])
        self.lkpe1 = LKPE(dims[2])
        self.lcsim1 = LCSIM(dims[3], dims[3])

        self.mcavss0 = MCAVSS(dim=dims[3], depth=depths[3], drop_path=dpr[sum(depths[: 3]): sum(depths[: 4])])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        d_3 = self.mcavss3(features[0])
        d_3 = self.lkpe3(d_3)
        d_2 = self.lcsim3(X=d_3, Y=features[1])
        d_2 = d_3 + d_2

        d_2 = self.mcavss2(d_2)
        d_2 = self.lkpe2(d_2)
        d_1 = self.lcsim2(X=d_2, Y=features[2])
        d_1 = d_2 + d_1

        d_1 = self.mcavss1(d_1)
        d_1 = self.lkpe1(d_1)
        d_0 = self.lcsim1(X=d_1, Y=features[3])
        d_0 = d_1 + d_0
        
        out = self.mcavss0(d_0)

        return self.out_layers[0](out)
