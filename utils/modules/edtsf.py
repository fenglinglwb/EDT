import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ResBlockDown(nn.Module):
    def __init__(self, in_chl, out_chl, down=False):
        super(ResBlockDown, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv_1 = nn.Conv2d(in_chl, in_chl, 3, 1, 1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.shortcut = nn.Conv2d(in_chl, out_chl, 1, 1, 0, bias=True)

        self.down = down
        if down:
            self.conv_down = nn.Conv2d(out_chl, out_chl, 4, 2, 1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x += identity

        if self.down:
            x_down = self.conv_down(x)
            return x_down, x
        else:
            return x

    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += H * W * self.in_chl * self.in_chl * 9
        flops += H * W * self.in_chl * self.out_chl * 9
        flops += H * W * self.in_chl * self.out_chl

        if self.down:
            flops += (H // 2) * (W // 2) * self.out_chl * self.out_chl * 16

        return flops


class ResBlockUp(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(ResBlockUp, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv_1 = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(out_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.shortcut = nn.Conv2d(in_chl, out_chl, 1, 1, 0, bias=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x += identity

        return x

    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += H * W * self.in_chl * self.out_chl * 9
        flops += H * W * self.out_chl * self.out_chl * 9
        flops += H * W * self.in_chl * self.out_chl

        return flops


class UpResBlock(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(UpResBlock, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.up = nn.ConvTranspose2d(in_chl, out_chl, kernel_size=2, stride=2, bias=True)
        self.block = ResBlockUp(out_chl * 2, out_chl)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x

    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += (H * 2) * (W * 2) * self.in_chl * self.out_chl * 4
        flops += self.block.flops((H * 2, W * 2))

        return flops


class ResBlockSkip(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(ResBlockSkip, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.block = ResBlockUp(out_chl * 2, out_chl)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x

    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += H * W * self.in_chl * self.out_chl * 9
        flops += self.block.flops((H, W))

        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 5, 1, 5//2, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)


    def forward(self, x):
        B, H, W, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = self.act(x) # B C H W
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc2(x)
        return x


def window_partition(x, window_size, index):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
        index: H or W

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if len(x.shape) == 5:
        _, B, H, W, C = x.shape
    else:
        B, H, W, C = x.shape
    if index == 0:
        h_size, w_size = window_size[0], window_size[1]
    else:
        h_size, w_size = window_size[1], window_size[0]

    if len(x.shape) == 5:
        x = x.view(2, B, H // h_size, h_size, W // w_size, w_size, C)
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(2, -1, h_size, w_size, C)
    else:
        x = x.view(B, H // h_size, h_size, W // w_size, w_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h_size, w_size, C)
    return windows


def window_reverse(windows, window_size, H, W, index):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if index == 0:
        H_window, W_window = window_size[0], window_size[1]
    else:
        H_window, W_window = window_size[1], window_size[0]

    B = int(windows.shape[0] / (H * W / H_window / W_window))
    x = windows.view(B, H // H_window, W // W_window, H_window, W_window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., index=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.index = index
        self.lepe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        qk, v = x[:2], x[2]
        B, H, W, C = v.shape
        qk = window_partition(qk, self.window_size, self.index) #2 B_ H_window W_window C
        _, B_, H_window, W_window, C = qk.shape
        qk = qk.view(2, B_, H_window, W_window, self.num_heads, C//self.num_heads).permute(0,1,4,2,3,5).contiguous()
        qk = qk.view(2, B_, self.num_heads, -1, C//self.num_heads)
        q, k = qk[0], qk[1]
        v, lepe = self.get_v(v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #B_ nh N N (N=H_w*W_w)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, H_window*W_window, H_window*W_window) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H_window*W_window, H_window*W_window)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v + lepe).transpose(1, 2).reshape(B_, H_window*W_window, C)
        x = x.view(B_, H_window, W_window, C)
        x = window_reverse(x, self.window_size, H, W, index=self.index) #B H W C
        return x

    def get_v(self, x):
        x = window_partition(x, self.window_size, self.index)
        x = x.permute(0,3,1,2).contiguous() #B_ C H_winodow W_window
        B_, C, H_window, W_window = x.shape
        lepe = self.lepe(x).view(B_, self.num_heads, C//self.num_heads, H_window, W_window).permute(0,1,3,4,2).contiguous()
        lepe = lepe.view(B_, self.num_heads, -1, C//self.num_heads)
        x = x.view(B_, self.num_heads, C//self.num_heads, H_window, W_window).permute(0,1,3,4,2).contiguous()
        x = x.view(B_, self.num_heads, -1, C//self.num_heads)
        return x, lepe

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, x_size):
        H, W = x_size
        # calculate flops for 1 window with token length of N
        flops = 0
        # lepe
        flops += H * W * self.dim * 9
        N = H // self.window_size[0] * W // self.window_size[1]
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        return flops


class CSwinTransformerBlock(nn.Module):
    r""" CSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.shift_size:
            assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size (H)"
            assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size (W)"

        self.norm1 = norm_layer(dim)
        self.attns = nn.ModuleList(
            [
                WindowAttention(dim//2, window_size=self.window_size, num_heads=num_heads,
             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, index=i) for i in range(2)
            ]
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)


        attn_mask_h = self.calculate_mask(self.input_resolution, index=0)
        attn_mask_v = self.calculate_mask(self.input_resolution, index=1)

        self.register_buffer('attn_mask_h', attn_mask_h)
        self.register_buffer('attn_mask_v', attn_mask_v)

    def calculate_mask(self, x_size, index):
        # calculate attention mask for SW-MSA
        if self.shift_size is None:
            return None

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_window_size, w_window_size = self.window_size[0], self.window_size[1]
        if index == 1:
            h_window_size, w_window_size = self.window_size[1], self.window_size[0]
        h_shift_size, w_shift_size = self.shift_size[0], self.shift_size[1]
        if index == 1:
            h_shift_size, w_shift_size = self.shift_size[1], self.shift_size[0]

        h_slices = (slice(0, -h_window_size),
                    slice(-h_window_size, -h_shift_size),
                    slice(-h_shift_size, None))
        w_slices = (slice(0, -w_window_size),
                    slice(-w_window_size, -w_shift_size),
                    slice(-w_shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size, index)  # nW, h_window_size, w_window_size, 1
        mask_windows = mask_windows.view(-1, h_window_size * w_window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        shortcut = x
        x = self.norm1(x)
        x = self.qkv(x) #B H W 3C
        x = x.view(B, H, W, 3, C).permute(3,0,1,2,4).contiguous()#3 B H W C
        x_h = x[...,:C//2]
        x_v = x[...,C//2:]

        if self.shift_size:
            x_h = torch.roll(x_h, shifts=(-self.shift_size[0],-self.shift_size[1]), dims=(2,3))
            x_v = torch.roll(x_v, shifts=(-self.shift_size[1],-self.shift_size[0]), dims=(2,3))

        if self.input_resolution == x_size:
            attn_windows_h = self.attns[0](x_h, mask=self.attn_mask_h)
            attn_windows_v = self.attns[1](x_v, mask=self.attn_mask_v)
        else:
            mask_h = self.calculate_mask(x_size, index=0).to(x_h.device) if self.shift_size else None
            mask_v = self.calculate_mask(x_size, index=1).to(x_v.device) if self.shift_size else None
            attn_windows_h = self.attns[0](x_h, mask=mask_h)
            attn_windows_v = self.attns[1](x_v, mask=mask_v)

        if self.shift_size:
            attn_windows_h = torch.roll(attn_windows_h, shifts=(self.shift_size[0],self.shift_size[1]), dims=(1,2))
            attn_windows_v = torch.roll(attn_windows_v, shifts=(self.shift_size[1],self.shift_size[0]), dims=(1,2))

        attn_windows = torch.cat([attn_windows_h, attn_windows_v], dim=-1)
        attn_windows = self.proj(attn_windows) #B H W C

        # FFN
        x = shortcut + self.drop_path(attn_windows)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H * W, C)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        # nW = H * W / self.window_size / self.window_size
        # flops += nW * self.attn.flops(self.window_size * self.window_size) * 2
        flops += self.attns[0].flops((H, W))
        flops += self.attns[1].flops((H, W))
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio + H * W * self.dim * 25
        # qkv = self.qkv(x)
        flops += H * W * self.dim * 3 * self.dim
        # proj
        flops += H * W * self.dim * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic CSwin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            CSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=None if (i % 2 == 0) else (window_size[0]//2,window_size[1]//2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        # if self.downsample is not None:
        #     flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual CSwin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

    def forward(self, x, x_size):
        return self.residual_group(x, x_size) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, input_resolution=None):
        self.input_resolution = input_resolution
        self.scale = scale
        self.num_feat = num_feat
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        if (self.scale & (self.scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(self.scale, 2))):
                flops += H * W * self.num_feat * self.num_feat * 4 * 9
        elif self.scale == 3:
            flops += H * W * self.num_feat * self.num_feat * 9 * 9
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.input_resolution = input_resolution
        self.scale = scale
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * self.num_out_ch * self.scale ** 2 * 9
        return flops


class SwinBody(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                 resi_connection='1conv', **kwargs):
        super(SwinBody, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        return self.conv_after_body(self.forward_features(x)) + x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution

        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 9 * self.embed_dim * self.embed_dim

        return flops


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        num_feat = config.MODEL.NUM_FEAT
        img_chl = config.MODEL.IN_CHANNEL
        embed_dim = config.MODEL.EMBED_DIM
        depth = config.MODEL.DEPTH

        self.num_feat = num_feat
        self.embed_dim = embed_dim
        self.depth = depth
        dn_input_size = config.MODEL.IMAGE_SIZE * 2 ** depth
        self.dn_resolution = (dn_input_size, dn_input_size)
        self.sr_resolution = (config.MODEL.IMAGE_SIZE, config.MODEL.IMAGE_SIZE)
        self.scales = config.MODEL.SCALES
        self.noise_levels = config.MODEL.NOISE_LEVELS
        self.rain_levels = config.MODEL.RAIN_LEVELS

        # preprocessing / postprocessing
        self.img_range = config.MODEL.IMAGE_RANGE
        if img_chl == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # heads
        ### sr
        for s in self.scales:
            head = nn.ModuleList()
            head.append(nn.Conv2d(img_chl, num_feat, 3, 1, 1))
            for i in range(depth):
                head.append(ResBlockDown(num_feat*2**i, num_feat*2**(i+1), down=False))
            head.append(nn.Conv2d(num_feat*2**depth, embed_dim, 3, 1, 1))
            setattr(self, 'head_sr_x%d' % s, head)
        ### denoise
        for nl in self.noise_levels:
            head = nn.ModuleList()
            head.append(nn.Conv2d(img_chl, num_feat, 3, 1, 1))
            for i in range(depth):
                head.append(ResBlockDown(num_feat*2**i, num_feat*2**(i+1), down=False))
            head.append(nn.Conv2d(num_feat*2**depth, embed_dim, 3, 1, 1))
            setattr(self, 'head_dn_g%d' % nl, head)
        ### derain
        for rl in self.rain_levels:
            head = nn.ModuleList()
            head.append(nn.Conv2d(img_chl, num_feat, 3, 1, 1))
            for i in range(depth):
                head.append(ResBlockDown(num_feat*2**i, num_feat*2**(i+1), down=False))
            head.append(nn.Conv2d(num_feat*2**depth, embed_dim, 3, 1, 1))
            setattr(self, 'head_dr_%s' % rl, head)

        # body
        self.body = SwinBody(img_size=config.MODEL.IMAGE_SIZE,
                             embed_dim=config.MODEL.EMBED_DIM,
                             depths=config.MODEL.DEPTHS,
                             num_heads=config.MODEL.NUM_HEADS,
                             window_size=config.MODEL.WINDOW_SIZE,
                             mlp_ratio=config.MODEL.MLP_RATIO,
                             resi_connection=config.MODEL.RESI_CONNECTION)

        # tails
        for s in self.scales:
            tail = nn.ModuleList()
            for i in reversed(range(depth)):
                in_chl = embed_dim if i == depth - 1 else num_feat * 2 ** (i + 2)
                out_chl = num_feat * 2 ** (i + 1)
                tail.append(ResBlockSkip(in_chl, out_chl))
            if config.MODEL.UPSAMPLER == 'pixelshuffle':
                tail.append(Upsample(s, out_chl))
                tail.append(nn.Conv2d(out_chl, img_chl, 3, 1, 1))
            else:
                tail.append(UpsampleOneStep(s, out_chl, img_chl, (config.MODEL.IMAGE_SIZE, config.MODEL.IMAGE_SIZE)))
                tail.append(nn.Identity())
            setattr(self, 'tail_sr_x%d' % s, tail)
        for nl in self.noise_levels:
            tail = nn.ModuleList()
            for i in reversed(range(depth)):
                in_chl = embed_dim if i == depth - 1 else num_feat * 2 ** (i + 2)
                out_chl = num_feat * 2 ** (i + 1)
                tail.append(ResBlockSkip(in_chl, out_chl))
            tail.append(nn.Conv2d(out_chl, img_chl, 3, 1, 1))
            setattr(self, 'tail_dn_g%d' % nl, tail)
        for rl in self.rain_levels:
            tail = nn.ModuleList()
            for i in reversed(range(depth)):
                in_chl = embed_dim if i == depth - 1 else num_feat * 2 ** (i + 2)
                out_chl = num_feat * 2 ** (i + 1)
                tail.append(ResBlockSkip(in_chl, out_chl))
            tail.append(nn.Conv2d(out_chl, img_chl, 3, 1, 1))
            setattr(self, 'tail_dr_%s' % rl, tail)

        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, lqs, gt=None):
        # preprocessing
        self.mean = self.mean.type_as(lqs[0])
        lqs = [(lq - self.mean) * self.img_range for lq in lqs]
        n_sr = len(self.scales)
        n_dn = len(self.noise_levels)
        n_all = n_sr + n_dn + len(self.rain_levels)

        # head
        skips_all = []
        outs = []
        ### sr
        for i, s in enumerate(self.scales):
            skips = []
            x = lqs[i].clone()
            head = getattr(self, 'head_sr_x%d' % s)
            for j, block in enumerate(head):
                x = block(x)
                if 0 < j < len(head) - 1:
                    skips.append(x)
            skips_all.append(skips)
            outs.append(x)
        ### denoise
        for i, nl in enumerate(self.noise_levels):
            skips = []
            x = lqs[n_sr+i].clone()
            head = getattr(self, 'head_dn_g%d' % nl)
            for j, block in enumerate(head):
                x = block(x)
                if 0 < j < len(head) - 1:
                    skips.append(x)
            skips_all.append(skips)
            outs.append(x)
        ### derain
        for i, rl in enumerate(self.rain_levels):
            skips = []
            x = lqs[n_sr+n_dn+i].clone()
            head = getattr(self, 'head_dr_%s' % rl)
            for j, block in enumerate(head):
                x = block(x)
                if 0 < j < len(head) - 1:
                    skips.append(x)
            skips_all.append(skips)
            outs.append(x)

        # body
        if gt is not None:
            x = torch.cat(outs, dim=0)
            x = self.body(x)
            x_b = torch.chunk(x, n_all, dim=0)
        else:
            x_b = [self.body(x) for x in outs]

        # tail
        outs = []
        ### sr
        for i, s in enumerate(self.scales):
            x = x_b[i]
            tail = getattr(self, 'tail_sr_x%d' % s)
            for j, block in enumerate(tail):
                if j == len(tail) - 1:
                    lq_up = F.interpolate(lqs[i], scale_factor=s, mode='bilinear', align_corners=False)
                    x = lq_up + block(x)
                elif j == len(tail) - 2:
                    x = block(x)
                else:
                    x = block(x, skips_all[i][-j-1])
            outs.append(x)
        ### denoise
        for i, nl in enumerate(self.noise_levels):
            x = x_b[n_sr+i]
            tail = getattr(self, 'tail_dn_g%d' % nl)
            for j, block in enumerate(tail):
                if j == len(tail) - 1:
                    x = lqs[n_sr+i] + block(x)
                else:
                    x = block(x, skips_all[n_sr+i][-j-1])
            outs.append(x)
        ### derain
        for i, rl in enumerate(self.rain_levels):
            x = x_b[n_sr+n_dn+i]
            tail = getattr(self, 'tail_dr_%s' % rl)
            for j, block in enumerate(tail):
                if j == len(tail) - 1:
                    x = lqs[n_sr+n_dn+i] + block(x)
                else:
                    x = block(x, skips_all[n_sr+n_dn+i][-j-1])
            outs.append(x)

        # preprocessing
        preds = [x / self.img_range + self.mean for x in outs]

        if gt is not None:
            loss_dict = dict()
            for i, name in enumerate(self.scales):
                loss_dict['x%d' % name] = self.criterion(preds[i], gt[i])
            for i, name in enumerate(self.noise_levels):
                loss_dict['g%d' % name] = self.criterion(preds[n_sr+i], gt[n_sr+i])
            for i, name in enumerate(self.rain_levels):
                loss_dict['x%s' % name] = self.criterion(preds[n_sr+n_dn+i], gt[n_sr+n_dn+i])
            return loss_dict
        else:
            return preds

    def flops(self):
        ### for denoising, g15
        flops = 0

        # head
        H, W = self.dn_resolution
        flops += H * W * 3 * self.num_feat * 9
        for i in range(1, len(self.head_dn_g15) - 1):
            flops += self.head_dn_g15[i].flops((H, W))
            H, W = H // 2, W // 2
            # print('down:', i, H, W)
        flops += H * W * self.num_feat*2**self.depth * self.embed_dim * 9

        # body
        flops += self.body.flops()

        # tail
        for i in range(0, len(self.tail_dn_g15) - 1):
            flops += self.tail_dn_g15[i].flops((H, W))
            H, W = H * 2, W * 2
            # print('up:', i, H, W)
        flops += H * W * self.num_feat*2**self.depth * 3 * 9

        # ### for sr, x2
        # flops = 0
        # scale = 2

        # # head
        # H, W = self.sr_resolution
        # flops += H * W * 3 * self.num_feat * 9
        # for i in range(1, len(self.head_sr_x2) - 1):
        #     flops += self.head_sr_x2[i].flops((H, W))
        #     print('down:', i, H, W)
        # flops += H * W * self.num_feat*2**self.depth * self.embed_dim * 9

        # # body
        # flops += self.body.flops()

        # # tail
        # for i in range(0, len(self.tail_sr_x2) - 2):
        #     flops += self.tail_sr_x2[i].flops((H, W))
        #     print('up:', i, H, W)
        # flops += self.tail_sr_x2[-2].flops()
        # if config.MODEL.UPSAMPLER == 'pixelshuffle':
        #     flops += H * scale * W * scale * self.num_feat*2**self.depth * 3 * 9

        return flops


if __name__ == '__main__':
    from easydict import EasyDict as edict

    res = 192

    class Config:
        MODEL = edict()
        MODEL.IN_CHANNEL = 3
        MODEL.DEPTH = 2
        MODEL.IMAGE_SIZE = res // (2 ** MODEL.DEPTH)
        MODEL.SCALES = []
        MODEL.NOISE_LEVELS = [15]
        MODEL.RAIN_LEVELS = []
        MODEL.WINDOW_SIZE = (6, 24)
        MODEL.IMAGE_RANGE = 1.0
        MODEL.NUM_FEAT = 8
        MODEL.DEPTHS = [6, 6, 6, 6]
        MODEL.EMBED_DIM = 60
        MODEL.NUM_HEADS = [6, 6, 6, 6]
        MODEL.MLP_RATIO = 2
        # MODEL.UPSAMPLER = 'pixelshuffle'
        MODEL.UPSAMPLER = 'pixelshuffledirect'
        MODEL.RESI_CONNECTION = '1conv'

    config = Config()

    device = torch.device('cpu')
    model = Network(config).to(device)
    cnt = 0
    for p in model.parameters():
        cnt += p.numel()
    print('param', cnt / 1e6)
    # print('flops', model.flops() / 1e9)

    inp = [torch.randn(1, 3, res, res)]
    out = model(inp)
    print(out[0].shape)

