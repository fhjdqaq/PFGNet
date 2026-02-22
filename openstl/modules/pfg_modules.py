import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from .layers import (GRN, PFGA, _RepDWLite)
import torch

class MSInit(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k_list=(3,5,7), stride: int = 1, use_gn: bool = True):
        super().__init__()
        # equal split for branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                _RepDWLite(in_ch, K=k, stride=stride),
                nn.Conv2d(in_ch, out_ch // len(k_list), 1, bias=False)
            ) for k in k_list
        ])

        # handle remainder channels
        gap = out_ch - (out_ch // len(k_list)) * len(k_list)
        self.tail = nn.Identity() if gap == 0 else nn.Sequential(
            _RepDWLite(in_ch, K=k_list[0], stride=stride),
            nn.Conv2d(in_ch, gap, 1, bias=False)
        )

        self.fuse = nn.Identity()
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches]
        if not isinstance(self.tail, nn.Identity):
            parts.append(self.tail(x))
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))

class PFG(nn.Module):
    """
    Main PFG block:
    - Token mixing by PFGA (peripheral + frequency gating)
    - Channel mixing by GLU-like depthwise-MLP (1x1 -> DW -> 1x1)
    - LayerScale + DropPath keep identical to original
    """
    def __init__(self,
                 dim: int,
                 groups_pw: int = 1,
                 layerscale_init: float = 1e-6,
                 act_layer=nn.GELU,
                 drop: float = 0.0,
                 drop_path: float = 0.0,
                 pfga_K=(9, 15, 31),
                 mlp_ratio: float = 4.0,
                 dw_kernel: int = 3):
        super().__init__()
        self.dim = dim

        # lightweight norms before token/channel mixers
        self.norm_dw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.norm_pw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)

        # token mixing by PFGA
        self.tm = PFGA(dim, K_list=pfga_K, use_grn=False)

        # GRN after each stage
        self.grn_dw = GRN(dim)
        self.grn_pw = GRN(dim)

        self.mlp_ratio   = mlp_ratio
        self.dw_kernel   = dw_kernel

        # GLU-like channel mixing
        E  = max(dim, int(dim * self.mlp_ratio))

        self.pw_in  = nn.Conv2d(dim, 2*E, kernel_size=1, bias=True, groups=groups_pw)
        self.dw_v   = nn.Conv2d(E, E, kernel_size=self.dw_kernel, padding=1, groups=E, bias=False)
        self.pw_out = nn.Conv2d(E, dim, kernel_size=1, bias=True, groups=groups_pw)

        self.act = act_layer()

        # LayerScale (kept exactly as original)
        self.gamma_dw = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.gamma_pw = nn.Parameter(torch.ones(dim) * layerscale_init)

        from timm.layers import DropPath
        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.drop_path  = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self._init_params()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_dw', 'gamma_pw'}

    def _init_params(self):
        # standard conv/norm init, identical to original
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) token mixing
        y = self.norm_dw(x)
        y = self.tm(y)
        y = self.act(y)
        y = self.grn_dw(y)
        y = self.dropout_dw(y)
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.dim, 1, 1))

        # (2) channel mixing (GLU style)
        z  = self.norm_pw(x)
        uv = self.pw_in(z)
        u, v = torch.chunk(uv, 2, dim=1)
        v  = self.dw_v(v)
        z  = F.silu(u) * v
        z  = self.pw_out(z)

        z = self.grn_pw(z)
        z = self.dropout_pw(z)
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.dim, 1, 1))
        return x