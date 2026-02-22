import torch
from torch import nn
from openstl.modules import ConvSC, MSInit, PFG


def sampling_generator(N: int, reverse: bool = False):
    samplings = [False, True] * (N // 2)
    samplings = samplings[:N]
    return list(reversed(samplings)) if reverse else samplings


class Encoder(nn.Module):
    """Spatial encoder with interleaved downsampling stages."""
    def __init__(self, c_in: int, c_hid: int, n_s: int, k: int, act_inplace: bool = False):
        super().__init__()
        samplings = sampling_generator(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, k, downsampling=samplings[0], act_inplace=act_inplace),
            *[ConvSC(c_hid, c_hid, k, downsampling=s, act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x: torch.Tensor):
        x0 = self.enc[0](x)
        z = x0
        for i in range(1, len(self.enc)):
            z = self.enc[i](z)
        return z, x0


class Decoder(nn.Module):
    """Spatial decoder with symmetric upsampling stages and a residual skip."""
    def __init__(self, c_hid: int, c_out: int, n_s: int, k: int, act_inplace: bool = False):
        super().__init__()
        samplings = sampling_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(c_hid, c_hid, k, upsampling=s, act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(c_hid, c_hid, k, upsampling=samplings[-1], act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1)

    def forward(self, z: torch.Tensor, skip: torch.Tensor):
        for i in range(len(self.dec) - 1):
            z = self.dec[i](z)
        y = self.dec[-1](z + skip)
        return self.readout(y)


class MidPFG(nn.Module):
    """PFG translator operating on the flattened (T*C) channel space."""
    def __init__(self, in_ch: int, depth: int,
                 groups_pw: int = 1, layerscale_init: float = 1e-6,
                 cel_k=(3, 5, 7),
                 drop: float = 0.0, drop_path: float = 0.0,
                 pfga_K=(9, 15, 31)):
        super().__init__()
        self.cel = MSInit(in_ch, in_ch, k_list=cel_k, use_gn=True)

        if drop_path > 0 and depth > 0:
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, depth)]
        else:
            dpr = [0.0] * depth

        self.blocks = nn.Sequential(*[
            PFG(
                in_ch,
                groups_pw=groups_pw,
                layerscale_init=layerscale_init,
                act_layer=nn.GELU,
                drop=drop,
                drop_path=(drop_path if i == depth - 1 else dpr[i]),
                pfga_K=pfga_K,
            )
            for i in range(depth)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.view(b, t * c, h, w)
        x = self.cel(x)
        x = self.blocks(x)
        return x.view(b, t, c, h, w)


class PFG_Model(nn.Module):
    """SimVP backbone with a PFG temporal translator."""
    def __init__(self, in_shape,
                 hid_S: int = 16, N_S: int = 4, N_T: int = 4,
                 spatio_kernel_enc: int = 3, spatio_kernel_dec: int = 3,
                 drop: float = 0.0, drop_path: float = 0.0,
                 groups_pw: int = 1, layerscale_init: float = 1e-6,
                 cel_k=(3, 5, 7), 
                 pfga_K=(9, 15, 31), **kwargs):
        super().__init__()
        t, c, _, _ = in_shape

        act_inplace = False
        self.enc = Encoder(c, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, c, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidPFG(
            in_ch=t * hid_S,
            depth=N_T,
            groups_pw=groups_pw,
            layerscale_init=layerscale_init,
            cel_k=cel_k,
            drop=drop,
            drop_path=drop_path,
            pfga_K=pfga_K
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_raw.shape
        x = x_raw.view(b * t, c, h, w)

        embed, skip = self.enc(x)
        _, c2, h2, w2 = embed.shape
        z = embed.view(b, t, c2, h2, w2)

        z = self.hid(z)
        hid = z.reshape(b * t, c2, h2, w2)

        y = self.dec(hid, skip)
        return y.view(b, t, c, h, w)