import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.CSWin_Transformer import mit
from models.GuidedFusion import PyramidFusion, ASPP

###############################################################################
# 1) Basic Modules
###############################################################################
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    """Basic residual block (like ResNet)."""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

###############################################################################
# 2) Cross Attention Block (Example)
###############################################################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels,   kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        proj_query_x1 = self.query_conv(x1).view(B, C//2, -1).permute(0,2,1)  # [B,N,C//2]
        proj_key_x2   = self.key_conv(x2).view(B, C//2, -1)                   # [B,C//2,N]
        energy = torch.bmm(proj_query_x1, proj_key_x2)                        # [B,N,N]
        attention = self.softmax(energy)                                      # [B,N,N]

        proj_value_x2 = self.value_conv(x2).view(B, C, -1)   # [B,C,N]
        out_x1 = torch.bmm(proj_value_x2, attention.permute(0,2,1))  # [B,C,N]
        out_x1 = out_x1.view(B, C, H, W)
        out_x1 = self.gamma * out_x1 + x1

        proj_query_x2 = self.query_conv(x2).view(B, C//2, -1).permute(0,2,1)
        proj_key_x1   = self.key_conv(x1).view(B, C//2, -1)
        energy2 = torch.bmm(proj_query_x2, proj_key_x1)
        attention2 = self.softmax(energy2)
        proj_value_x1 = self.value_conv(x1).view(B, C, -1)
        out_x2 = torch.bmm(proj_value_x1, attention2.permute(0,2,1))
        out_x2 = out_x2.view(B, C, H, W)
        out_x2 = self.gamma * out_x2 + x2

        return out_x1, out_x2

###############################################################################
# 3) DecoderBlock with channel + spatial attention
###############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True)
        )
        reduced_ch = in_channels_low // scale_ratio
        self.ca = ChannelAttention(reduced_ch)
        self.sa = SpatialAttention()

        self.merge = nn.Sequential(
            conv3x3(in_channels_high + reduced_ch, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_high, x_low):
        x_high = self.up(x_high)       # e.g., 1/8 -> 1/4
        x_low  = self.transit(x_low)
        # attention
        x_low = x_low * self.ca(x_low)
        x_low = x_low * self.sa(x_low)
        # merge
        x_cat = torch.cat([x_high, x_low], dim=1)
        x_dec = self.merge(x_cat)
        return x_dec

###############################################################################
# 4) Advanced Siamese Encoder
###############################################################################
class AdvancedEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        # Use ResNet50 as base (or replace with ResNet101, etc.).
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv for arbitrary channels
        old_weights = resnet.conv1.weight.data
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :3, :, :].copy_(old_weights[:, :3, :, :])
        if in_channels > 3:
            # replicate or partial copy to handle extra channels
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(
                old_weights[:, 0:(in_channels - 3), :, :]
            )
        resnet.conv1 = newconv1
        
        # We'll extract features from multiple stages:
        self.layer0 = nn.Sequential(
            resnet.conv1,  # [B,64,H/2,W/2]
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool        # -> [B,64,H/4,W/4]
        self.layer1 = resnet.layer1          # -> [B,256,H/4,W/4]
        self.layer2 = resnet.layer2          # -> [B,512,H/8,W/8]
        self.layer3 = resnet.layer3          # -> [B,1024,H/16,W/16]
        self.layer4 = resnet.layer4          # -> [B,2048,H/32,W/32]
 
        self.out_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.out_conv)

    def forward(self, x):
        # x: [B,in_channels,H,W]
        x = self.layer0(x)     # -> [B,64,H/2,W/2]
        x = self.maxpool(x)    # -> [B,64,H/4,W/4]
        x1 = self.layer1(x)    # -> [B,256,H/4,W/4]  (low-level)
        x2 = self.layer2(x1)   # -> [B,512,H/8,W/8]
        x3 = self.layer3(x2)   # -> [B,1024,H/16,W/16]
        x4 = self.layer4(x3)   # -> [B,2048,H/32,W/32]
        x4 = self.out_conv(x4) # -> [B,512,H/32,W/32]

        # Return multi-scale features. E.g., 1/4, 1/8, 1/16, 1/32:
        return {
            'l1': x1,  # 1/4 scale, 256C
            'l2': x2,  # 1/8 scale, 512C
            'l3': x3,  # 1/16, 1024C
            'l4': x4   # 1/32, 512C
        }

###############################################################################
# 5) The Final Model: SCanNetX
###############################################################################
class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super().__init__()
        #######################################################################
        # (A) Siamese Encoder
        #######################################################################
        self.encoder = AdvancedEncoder(in_channels=in_channels, pretrained=True)
        #######################################################################
        # (B) Multi-Scale Difference
        #     We'll compute difference at 2 scales: 1/8 and 1/32
        #######################################################################
        self.diff_8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # for scale=1/8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.diff_32 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # for scale=1/32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        #######################################################################
        # (C) Cross-Attention on some scale (e.g., 1/8)
        #######################################################################
        self.cross_attn_8 = CrossAttentionBlock(in_channels=512)
        #######################################################################
        # (D) Residual Change Block on the highest scale features (1/32)
        #######################################################################
        self.resCD = self._make_layer(ResBlock, 512*2, 512, blocks=3)
        #######################################################################
        # (E) Global Transformer + ASPP
        #     We will combine [f1_32, f2_32, diff_32] -> pass to transformer -> ASPP
        #######################################################################
        # 512 channels each => total 512*3 = 1536 for the transformer
        feat_size = input_size // 32  # if final encoder output is 1/32
        embed_dim = 512 * 3
        self.transformer = mit(
            img_size=feat_size,
            in_chans=embed_dim,
            embed_dim=embed_dim
        )
        self.aspp = ASPP(in_channels=embed_dim, out_channels=embed_dim)
        self.split_conv = nn.Conv2d(embed_dim, 512*3, kernel_size=1, bias=False)  

        self.decA_16 = DecoderBlock(512, 1024, 512, scale_ratio=2)  # from 1/32 up to 1/16, fuse with layer3(1024C)
        self.decA_8  = DecoderBlock(512, 512, 256, scale_ratio=1)   # from 1/16 up to 1/8, fuse with layer2(512C)

        self.decB_16 = DecoderBlock(512, 1024, 512, scale_ratio=2)
        self.decB_8  = DecoderBlock(512, 512, 256, scale_ratio=1)

        self.decC_16 = DecoderBlock(512, 1024, 512, scale_ratio=2)
        self.decC_8  = DecoderBlock(512, 512, 256, scale_ratio=1)

        self.clsA = nn.Conv2d(256, num_classes, kernel_size=1)
        self.clsB = nn.Conv2d(256, num_classes, kernel_size=1)
        self.clsC = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        initialize_weights(
            self.diff_8, self.diff_32, self.resCD, 
            self.decA_16, self.decA_8, self.decB_16, self.decB_8, self.decC_16, self.decC_8,
            self.clsA, self.clsB, self.clsC, self.split_conv
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # x1, x2 shape: [B,in_channels,H,W]
        B, C, H, W = x1.shape

        #############################
        # 1) Siamese Encoding
        #############################
        feats1 = self.encoder(x1)  # dict: l1=1/4(256C), l2=1/8(512C), l3=1/16(1024C), l4=1/32(512C)
        feats2 = self.encoder(x2)

        # For convenience:
        f1_4, f1_8, f1_16, f1_32 = feats1['l1'], feats1['l2'], feats1['l3'], feats1['l4']
        f2_4, f2_8, f2_16, f2_32 = feats2['l1'], feats2['l2'], feats2['l3'], feats2['l4']

        #############################
        # 2) Multi-Scale Difference
        #############################
        # a) 1/8 scale difference
        diff_8 = torch.abs(f1_8 - f2_8)   # [B,512,H/8,W/8]
        diff_8 = self.diff_8(diff_8)      # refine

        # b) 1/32 scale difference
        diff_32 = torch.abs(f1_32 - f2_32) # [B,512,H/32,W/32]
        diff_32 = self.diff_32(diff_32)    

        #############################
        # 3) Cross Attention at 1/8
        #############################
        # Let f1_8, f2_8 cross-attend
        f1_8_ca, f2_8_ca = self.cross_attn_8(f1_8, f2_8)

        #############################
        # 4) Residual Change Block on 1/32
        #############################
        cat_32 = torch.cat([f1_32, f2_32], dim=1)  # [B,1024,H/32,W/32]
        xc_32 = self.resCD(cat_32)                # [B,512,H/32,W/32]

        #############################
        # 5) Transformer + ASPP
        #    Combine f1_32, f2_32, diff_32 
        #    Or combine [xc_32, diff_32, something else].
        #    Let's combine [xc_32, diff_32, maybe average of (f1_32,f2_32)?].
        #############################
        f12_32 = 0.5*(f1_32 + f2_32)  # simple average
        combined_32 = torch.cat([xc_32, diff_32, f12_32], dim=1)  # => [B,512*3,H/32,W/32]

        x_trans = self.transformer(combined_32) # => [B,512*3,H/32,W/32]
        x_aspp  = self.aspp(x_trans)            # => [B,512*3,H/32,W/32]

        # Let's reduce them back to 512*3
        x_aspp = self.split_conv(x_aspp)        # => [B,512*3,H/32,W/32]

        #############################
        # 6) Split into 3 streams: x1, x2, xC
        #############################
        x1_32 = x_aspp[:, 0:512, :, :]
        x2_32 = x_aspp[:, 512:1024, :, :]
        xC_32 = x_aspp[:, 1024:1536, :, :]

        #############################
        # 7) Decoding
        #    e.g. from 1/32 -> 1/16 -> 1/8 using DecoderBlocks
        #############################
        # (A) image-1
        x1_16 = self.decA_16(x1_32, f1_16)  # => [B,512, H/16, W/16]
        x1_8  = self.decA_8(x1_16, f1_8_ca) # => [B,256, H/8, W/8]
        # Then up 2x to 1/4 scale for final classification
        x1_4 = F.interpolate(x1_8, scale_factor=2, mode='bilinear', align_corners=True)  # => [B,256,H/4,W/4]

        # (B) image-2
        x2_16 = self.decB_16(x2_32, f2_16) 
        x2_8  = self.decB_8(x2_16, f2_8_ca)
        x2_4 = F.interpolate(x2_8, scale_factor=2, mode='bilinear', align_corners=True)

        # (C) change
        xC_16 = self.decC_16(xC_32, f1_16)  # or fuse f1_16 + f2_16 if you want
        xC_8  = self.decC_8(xC_16, diff_8) # fuse with difference at 1/8
        xC_4  = F.interpolate(xC_8, scale_factor=2, mode='bilinear', align_corners=True)

        #############################
        # 8) Final Classifiers
        #############################
        out1    = self.clsA(x1_4)   # => [B,num_classes,H/4,W/4]
        out2    = self.clsB(x2_4)   # => [B,num_classes,H/4,W/4]
        change  = self.clsC(xC_4)   # => [B,1,H/4,W/4]

        # upsample to original input size
        out1   = F.interpolate(out1, (H, W), mode='bilinear', align_corners=True)
        out2   = F.interpolate(out2, (H, W), mode='bilinear', align_corners=True)
        change = F.interpolate(change, (H, W), mode='bilinear', align_corners=True)

        return change, out1, out2