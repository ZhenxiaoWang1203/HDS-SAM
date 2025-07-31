import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torch import Tensor



class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ImageEncoderViT3D(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_chans: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            # nn.LayerNorm(out_chans),
        )

        filters = [32, 64, 128, 256]
        self.conv1 = ConvBlock(1, filters[0])
        self.maxpool1 = MPM(filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.maxpool2 = MPM(filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.maxpool3 = MPM(filters[2])
        self.center = ConvBlock(filters[2], filters[3])

        self.up_concat3 = UnetUp(filters[3], filters[2])
        self.up_concat2 = UnetUp(filters[2], filters[1])
        self.up_concat1 = UnetUp(filters[1], filters[0])

        self.final = nn.Conv3d(filters[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        center = self.center(maxpool3)
        up3 = self.up_concat3(conv3, center)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        f = self.final(up1)
        f = self.patch_embed(f)

        x = self.patch_embed(x)
        # x = [2,8,8,8,768]
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # x = [2,8,8,8,768]
        x = x + f

        x = self.neck(x.permute(0, 4, 1, 2, 3))
        # output_size = [2,384,8,8,8]

        return x


class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     use_rel_pos=use_rel_pos,
        #     rel_pos_zero_init=rel_pos_zero_init,
        #     input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        # )
        self.attn = PAM()

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition3D(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)

        return x


def window_partition3D(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(
        windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_d: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int, int],
        k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size

    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
            attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) + rel_d[:, :, :, :, None, None] + rel_h[:, :, :, None, :,
                                                                                         None] + rel_w[:, :, :, None,
                                                                                                 None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16, 16),
            stride: Tuple[int, int] = (16, 16, 16),
            padding: Tuple[int, int] = (0, 0, 0),
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=False),
                                   nn.InstanceNorm3d(out_size),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size=3, padding=1, bias=False),
                                   nn.InstanceNorm3d(out_size),
                                   nn.LeakyReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.p2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)
        x = x1 + x2
        return x


class MPM(nn.Module):
    def __init__(self, in_channels):
        super(MPM, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.x_avgpool = nn.AdaptiveAvgPool3d((1, None, None))
        self.y_avgpool = nn.AdaptiveAvgPool3d((None, 1, None))
        self.z_avgpool = nn.AdaptiveAvgPool3d((None, None, 1))
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = self.conv1(x3)
        y1 = self.x_avgpool(x4)
        y2 = self.y_avgpool(x4)
        y3 = self.z_avgpool(x4)
        x5 = self.conv2(y1).expand(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        x6 = self.conv2(y2).expand(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        x7 = self.conv2(y3).expand(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        x8 = x5 + x6 + x7
        x9 = x8 + x4
        return x9


class PAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        res = x
        x_pooling = [torch.mean(x, dim=i, keepdim=True).expand_as(x) for i in range(2, len(x.shape))]
        x = torch.stack(x_pooling, dim=-1).mean(dim=-1)
        x = torch.sigmoid(x) * res
        return x