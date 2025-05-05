import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.CSWin_Transformer import mit
from pytorch_wavelets import DWTForward, DWTInverse
from kymatio.torch import Scattering2D

args = {'hidden_size': 128*3,
        'mlp_dim': 256*3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}
        
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.scale_ratio = scale_ratio
        
        # Upsample high-level features
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        
        # Transit block reduces channels of low_feat
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules on low_feat after transit
        reduced_channels = in_channels_low // scale_ratio
        self.ca = ChannelAttention(reduced_channels)
        self.sa = SpatialAttention()

        # Decoder combines upsampled high-level and attended low-level features
        in_channels = max(in_channels_high, reduced_channels)

        self.low2high = conv1x1(reduced_channels, in_channels_high)
        initialize_weights(self.low2high)

            # right after self.sa = SpatialAttention()
        # ---------------------------------------
        # 1‐level 2D Haar wavelet
        self.dwt = DWTForward(J=1, wave='haar')   # computes LL & [LH,HL,HH]
        self.idwt = DWTInverse(wave='haar')

        self.decode = nn.Sequential(
            conv3x3(in_channels_high, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        # 1) Upsample the high-level feature
        x_up    = self.up(x)  # [B, C_high, H, W]

        # 2) Reduce & attend the low-level feature
        low_tr  = self.transit(low_feat)            # [B, C_low//scale, H, W]
        low_att = low_tr * self.ca(low_tr)          # channel attention
        low_att = low_att * self.sa(low_att)        # spatial attention

        # 3) Project low_att → C_high so DWT bands match
        low_proj = self.low2high(low_att)           # [B, C_high, H, W]

        # 4) Decompose both streams with 1-level Haar DWT
        Yl_x, Yh_x = self.dwt(x_up)                 # Yl_x: [B,C_high,H/2,W/2], Yh_x[0]: [B,C_high,3,H/2,W/2]
        Yl_l, Yh_l = self.dwt(low_proj)

        # 5) Fuse low-frequency (LL) by elementwise max
        fused_LL = torch.max(Yl_x, Yl_l)            # [B,C_high,H/2,W/2]

        # 6) Fuse all high-frequency bands by max
        fused_HH = torch.max(Yh_x[0], Yh_l[0])      # [B,C_high,3,H/2,W/2]

        # 7) Reconstruct the fused feature map
        fused = self.idwt((fused_LL, [fused_HH]))   # [B,C_high,H,W]

        # 8) Decode to the desired out_channels
        out = self.decode(fused)                    # [B, out_channels, H, W]
        return out



class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, input_size=512):
        super(FCN, self).__init__()
        self.input_size = input_size
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)

        # ─── Scattering branch ────────────────────────────────────────────────────────

        # 1) we’ll apply scattering at the same 1/4 resolution as layer1:
        scatt_J     = 2                                        # #scales
        res4        = self.input_size // 4                          # e.g. 512→128
        self.scatt  = Scattering2D(J=scatt_J, shape=(res4,res4))

        # 2) figure out how many scattering channels we get:
        with torch.no_grad():
            dummy  = torch.zeros(1, in_channels, self.input_size, self.input_size)
            x0     = self.layer0(dummy)
            x1     = self.maxpool(x0)
            sc_out = self.scatt(x1)               # [1, C_in, M, H′, W′]
            _, C_in, M, Hs, Ws = sc_out.shape
            C_scatt = C_in * M  

        # 3) project scattering ⟶ 128 dims to match your head
        self.scatt_proj = nn.Sequential(
            nn.Conv2d(C_scatt, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # project fused scattering+head (256 channels) back to 128
        self.fuse_proj = nn.Sequential(
            conv1x1(256, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.fuse_proj)
        initialize_weights(self.scatt_proj)

                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        # --- original ResNet front-end ---
        x      = self.layer0(x)        # → 1/2
        x      = self.maxpool(x)       # → 1/4
        x_low  = self.layer1(x)        # skip (1/4)

        # --- scattering branch ---
        with torch.no_grad():
            S = self.scatt(x)               # [B,C_in,M,Hs,Ws]
        B, C_in, M, Hs, Ws = S.shape
        S = S.view(B, C_in * M, Hs, Ws)      # [B,C_scatt,Hs,Ws]
        S = self.scatt_proj(S)               # [B,128,Hs,Ws]
        S = F.interpolate(S, size=x_low.shape[2:], mode='bilinear', align_corners=False)

        # --- continue ResNet deeper ---
        x      = self.layer2(x_low)    # → 1/8
        x      = self.layer3(x)        # → 1/8
        x      = self.layer4(x)        # → 1/8

        # --- head + fuse with scattering ---
        x_head = self.head(x)          # [B,128,1/8 H,1/8 W]
        x_head = F.interpolate(x_head, size=x_low.shape[2:], mode='bilinear', align_corners=False)

        # concatenate scattering + head, then use the pre-defined fuse_proj
        x_fuse = torch.cat([x_head, S], dim=1)     # [B,256,H_low,W_low]
        x      = self.fuse_proj(x_fuse)            # now [B,128,H_low,W_low]

        return x, x_low


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global avg pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.gate_conv = conv1x1(2*in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.gate_conv)

    def forward(self, x_low, x_high):
        # Apply channel & spatial attention to low-level features
        x_low_att = x_low * self.ca(x_low)
        x_low_att = x_low_att * self.sa(x_low_att)

        # Project attended low-level features up to match x_high’s channels
        low_proj = self.low2high(x_low_att)  # now same C as x_high

        # Compute gate from the concatenation of both streams
        joint = torch.cat([x_high, low_proj], dim=1)  # shape [B, 2*C, H, W]
        g     = self.sigmoid(self.gate_conv(joint))   # shape [B, 1, H, W]

        # Fuse via learned weighted sum
        fused = g * x_high + (1.0 - g) * low_proj     # shape [B, C, H, W]

        return fused


class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(SCanNet, self).__init__()
        feat_size = input_size//4
        self.FCN = FCN(in_channels, pretrained=True, input_size=input_size)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128*3, embed_dim=128*3)
        
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)
        
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
            
        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD, self.classifierCD)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        # --- original ResNet front-end ---
        x = self.layer0(x)        # conv1+bn+relu  →  1/2
        x = self.maxpool(x)       #                 →  1/4
        x_low = self.layer1(x)    # layer1 output  →  1/4

        # --- scattering branch (same 1/4 input) ---
        with torch.no_grad():
            S = self.scatt(x)     # [B, C_scatt, 1/4 H, 1/4 W]
        S = self.scatt_proj(S)    # [B,   128  , 1/4 H, 1/4 W]

        # --- continue ResNet deeper ---
        x = self.layer2(x_low)    # → 1/8
        x = self.layer3(x)        # → 1/8 (stride overrides)
        x = self.layer4(x)        # → 1/8

        # --- your existing head + fusion with scattering ---
        x_head = self.head(x)     # [B,128,1/8 H,1/8 W]
        
        # upsample head back to 1/4 so we can fuse with S
        x_head = F.interpolate(x_head, size=x_low.shape[2:], mode='bilinear', align_corners=False)

        # fuse by concatenation (or you can try max, sum, gated fuse…):
        x_fuse = torch.cat([x_head, S], dim=1)   # [B,256,1/4 H,1/4 W]
        
        # project back to 128 before passing to decoders
        x = conv1x1(256, 128)(x_fuse)
        x = nn.BatchNorm2d(128)(x)
        x = nn.ReLU(inplace=True)(x)

        return x, x_low

    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1, x1_low = self.FCN.base_forward(x1)
        x2, x2_low = self.FCN.base_forward(x2)
        x1, x2, xc = self.CD_forward(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
                
        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]
        
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)
        
        return F.interpolate(change, x_size[2:], mode='bilinear'), F.interpolate(out1, x_size[2:], mode='bilinear'), F.interpolate(out2, x_size[2:], mode='bilinear')