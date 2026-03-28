import torch
import torch.nn as nn
import torch.nn.functional as F
from .mk_blocks import mk_irb_bottleneck, MultiKernelInvertedResidualBlock
from .frequency_gating import MultiFrequencyChannelAttention
from .prototype_attention import PrototypeAttention, get_ocr_vector, Graph_Attention_Network, get_correlation_map

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.GELU()

    def forward(self, g, x):
        # g: gating signal (coarser)
        # x: skip connection (finer)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class HDDIBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=2, dw_parallel=True, kernel_sizes=[3,5,7], frequency_selection='top', dct_size=(256, 256)):
        super(HDDIBlock, self).__init__()
        
        self.spatial_block = mk_irb_bottleneck(in_channels, out_channels, 1, stride, expansion_factor, dw_parallel, True, kernel_sizes)
        
        # Out = F_spatial + alpha * F_clean
        
        self.freq_gating = MultiFrequencyChannelAttention(
            out_channels, 
            dct_h=dct_size[0], dct_w=dct_size[1], 
            frequency_branches=16, 
            frequency_selection=frequency_selection
        )
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.1) # Learnable weight

    def forward(self, x):
        f_spatial = self.spatial_block(x)
        
        f_clean = self.freq_gating(f_spatial)
        
        f_fused = f_spatial + self.alpha * f_clean
        
        return f_fused

class FGPAEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=[16, 32, 64, 128, 256], img_size=256):
        super(FGPAEncoder, self).__init__()
        self.channels = channels

        s1 = img_size
        s2 = s1 // 2
        s3 = s2 // 2
        s4 = s3 // 2
        s5 = s4 // 2
        
        # Stage 1
        self.stage1 = HDDIBlock(in_channels, channels[0], stride=1, dct_size=(s1, s1))
        self.down1 = nn.MaxPool2d(2)
        
        # Stage 2
        self.stage2 = HDDIBlock(channels[0], channels[1], stride=1, dct_size=(s2, s2))
        self.down2 = nn.MaxPool2d(2)
        
        # Stage 3
        self.stage3 = HDDIBlock(channels[1], channels[2], stride=1, dct_size=(s3, s3))
        self.down3 = nn.MaxPool2d(2)
        
        # Stage 4
        self.stage4 = HDDIBlock(channels[2], channels[3], stride=1, dct_size=(s4, s4))
        self.down4 = nn.MaxPool2d(2)
        
        # Bottleneck (Stage 5)
        self.bottleneck = HDDIBlock(channels[3], channels[4], stride=1, dct_size=(s5, s5))

    def forward(self, x):
        x1 = self.stage1(x)
        p1 = self.down1(x1)
        
        x2 = self.stage2(p1)
        p2 = self.down2(x2)
        
        x3 = self.stage3(p2)
        p3 = self.down3(x3)
        
        x4 = self.stage4(p3)
        p4 = self.down4(x4)
        
        x5 = self.bottleneck(p4)
        
        return [x1, x2, x3, x4, x5]

class FGPADecoder(nn.Module):
    def __init__(self, channels=[16, 32, 64, 128, 256], num_classes=1, deep_supervision=False):
        super(FGPADecoder, self).__init__()
        self.deep_supervision = deep_supervision

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[4] + channels[3], channels[3], 3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            nn.Conv2d(channels[3], channels[3], 3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU()
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[2] + channels[1], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU()
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[0], channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU()
        )
        
        self.final = nn.Conv2d(channels[0], num_classes, 1)

        self.ag4 = Attention_block(F_g=channels[4], F_l=channels[3], F_int=channels[3]//2)
        self.ag3 = Attention_block(F_g=channels[3], F_l=channels[2], F_int=channels[2]//2)
        self.ag2 = Attention_block(F_g=channels[2], F_l=channels[1], F_int=channels[1]//2)
        self.ag1 = Attention_block(F_g=channels[1], F_l=channels[0], F_int=channels[0]//2)

        # Deep Supervision Heads
        if self.deep_supervision:
            self.ds2 = nn.Conv2d(channels[1], num_classes, 1)
            self.ds3 = nn.Conv2d(channels[2], num_classes, 1)
            self.ds4 = nn.Conv2d(channels[3], num_classes, 1)

    def forward(self, features):
        
        d4 = self.up4(x5)
        x4 = self.ag4(g=d4, x=x4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.conv4(d4)
        
        d1 = self.up1(d2)
        x1 = self.ag1(g=d1, x=x1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv1(d1)
        
        d2 = self.up2(d3)
        x2 = self.ag2(g=d2, x=x2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv2(d2)
        

        
        out = self.final(d1)
        
        if self.deep_supervision and self.training:
            # Upsample intermediate outputs to target size (if needed by loss) or return as is.
            out4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode='bilinear', align_corners=True)
            return [out, out2, out3, out4]
            
        return out

class FGPAModel(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, img_size=256, deep_supervision=False):
        super(FGPAModel, self).__init__()
        
        self.decoder = FGPADecoder(self.channels, num_classes, deep_supervision=deep_supervision)

    def forward(self, x):
        # Encoder
        features = self.encoder(x) # [x1, x2, x3, x4, x5]
        features[-1] = x5_refined
        
        # Decoder
        out = self.decoder(features)
        
        return out

if __name__ == "__main__":
    output = model(input_tensor)
    print(f"Input: {input_tensor.shape}")
    print(f"Output: {output.shape}")
    
    print(f"Total Parameters: {total_params} ({total_params/1e6:.2f} M)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params} ({trainable_params/1e6:.2f} M)")
