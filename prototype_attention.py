import torch
import torch.nn as nn
import torch.nn.functional as F

def get_ocr_vector(x):
    b, c, h, w = x.size()
    probs = x.view(b, c, -1)
    ss_map = F.softmax(probs, dim=2) 
    ss_map = ss_map.view(b, c, h, w)
    pb = get_prototype(x, ss_map.clone().detach())
    return pb

class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2
        
        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)
        B, N, C = feature.size()

        f_div_C = F.softmax(attention, dim=-1)
        g_x = self.g(feature)
        y = torch.matmul(f_div_C, g_x)
        W_y = self.W(y).contiguous().view(B, N, C)
        att_fea = ori_feature.permute(0, 2, 1) + W_y
        return att_fea

class Graph_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Graph_Attention_Network, self).__init__()
        self.transformer = Transformer(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.GELU())


class PrototypeAttention(nn.Module):
    def __init__(self, in_channels):
        super(PrototypeAttention, self).__init__()
        self.GAN = Graph_Attention_Network(in_channels)
        self.out = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1),
                                 nn.BatchNorm2d(in_channels),
                                 nn.GELU())

    def forward(self, x, x_clean=None):
        # x: Spatial features (B, C, H, W)
        # x_clean: Frequency-cleaned features (B, C, H, W). If None, use x.
        
        source = x_clean if x_clean is not None else x
        
        # 1. Generate Prototypes (PB) from Source
        # Note: get_ocr_vector computes prototypes from the *current* image features (x_clean/x).
        # This is Intra-Image logic, so it is safe for Inference (Batch Size = 1) and
        # does not rely on cross-image dependencies (unlike the original Semi-Supervised CSC-PA).
        pb = get_ocr_vector(source) 
        
        # 2. Refine Prototypes via Graph Attention
        graph_pb = self.GAN(pb) # B, C, N
        
        # 3. Correlation between Original X and Refined Prototypes
        # This creates an attention map based on how similar x is to the prototypes
        map = get_correlation_map(x, graph_pb) # B, N, H, W
        
        
        return out
