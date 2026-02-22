
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2 style)
    x: (B, C, H, W) -> y = x + gamma * (x / ||x||_2) + beta
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))   # learnable scale
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))  # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)     # L2 over spatial dims
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


class _RepDWLite(nn.Module):
    """
    Lightweight re-parameterized depthwise composition:
    DW(1xK) -> DW(Kx1) + DW(3x3) + DW(1x1 ~ identity)
    """
    def __init__(self, dim: int, K: int, stride: int = 1):
        super().__init__()

        # separable large-kernel approximation
        self.dw_h = nn.Conv2d(
            dim, dim, kernel_size=(1, K),
            stride=(1, stride), padding=(0, K // 2),
            groups=dim, bias=False
        )
        self.dw_v = nn.Conv2d(
            dim, dim, kernel_size=(K, 1),
            stride=(stride, 1), padding=(K // 2, 0),
            groups=dim, bias=False
        )

        # complementary 3x3 + identity-like DW(1x1)
        self.dw_s = nn.Conv2d(
            dim, dim, kernel_size=3,
            stride=stride, padding=1, dilation=1,
            groups=dim, bias=False
        )
        self.dw_i = nn.Conv2d(
            dim, dim, kernel_size=1,
            stride=stride, groups=dim, bias=False
        )
        nn.init.dirac_(self.dw_i.weight)  # start as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)


class PFGA(nn.Module):
    """
    Peripheral-Frequency Guided Aggregation (token mixer):
    - Multiple large-kernel depthwise branches (peripheral)
    - Pixel-wise frequency gating (Sobel/Laplacian/variance cues)
    - Optional center suppression
    """
    class Branch(nn.Module):
        def __init__(self, dim: int, K: int, center_suppress: bool = True):
            super().__init__()
            self.center_suppress = center_suppress

            # approximate KxK with DW(1xK) + DW(Kx1)
            self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, K),
                                  padding=(0, K // 2), groups=dim, bias=False)
            self.dw_v = nn.Conv2d(dim, dim, kernel_size=(K, 1),
                                  padding=(K // 2, 0), groups=dim, bias=False)

            # optional 3x3 center path for suppression
            if self.center_suppress:
                self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                      groups=dim, bias=False)
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.register_parameter('beta', None)
                self.dw_c = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.dw_v(self.dw_h(x))
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center   # explicit center suppression
            return y

    def __init__(self, dim: int, K_list=(9, 15, 31), use_grn: bool = False, center_suppress: bool = True):
        super().__init__()
        self.dim = dim
        self.K_list = K_list

        # multi-scale peripheral branches
        self.branches = nn.ModuleList([PFGA.Branch(dim, K, center_suppress=center_suppress) for K in K_list])

        # fixed frequency filters (buffers): Sobel x/y + Laplacian
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0],
                                [1,-4, 1],
                                [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace",  laplace,  persistent=False)

        # 1x1 conv to produce per-scale gating logits
        self.gate_head = nn.Conv2d(3, len(K_list), kernel_size=1, bias=True)

        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)

    # depthwise apply fixed 3x3 kernels to all channels
    def _depthwise_filter(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = k.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, padding=1, groups=C)

    # build frequency maps: gradient magnitude, Laplacian magnitude, local variance
    def _freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        mean  = F.avg_pool2d(x, 3, 1, 1)
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)
        var   = torch.clamp(mean2 - mean * mean, min=0.)

        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)
        return torch.cat([f1, f2, f3], dim=1)  # (B,3,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # peripheral responses at multiple scales
        peris = [b(x) for b in self.branches]

        # per-pixel softmax over scales from frequency cues
        Freq   = self._freq_maps(x)
        logits = self.gate_head(Freq)
        alpha  = torch.softmax(logits, dim=1)  # (B,K,H,W)

        # pixel-wise fusion
        Y = 0.
        for i, y in enumerate(peris):
            Y = Y + y * alpha[:, i:i+1, :, :]

        if self.use_grn:
            Y = self.grn(Y)
        return Y
