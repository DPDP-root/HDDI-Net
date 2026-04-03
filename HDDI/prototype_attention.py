import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def get_prototype(x, ss_map):
    B, _, H, W = x.size()
    ss_map = ss_map.view(B, -1, H * W)
    x = x.view(B, -1, H * W)
    prototype_block = torch.bmm(ss_map, x.transpose(1, 2))
    return prototype_block

def get_correlation_map(x, prototype_block):
    B, C, H, W = x.size()
    n_p = prototype_block / (prototype_block.norm(dim=2, keepdim=True) + 1e-8)
    n_x = x.view(B, C, -1) / (x.view(B, C, -1).norm(dim=1, keepdim=True) + 1e-8)
    corr = torch.bmm(n_p, n_x).view(B, -1, H, W)
    return corr

def get_ocr_vector(x):
    b, c, h, w = x.size()
    probs = x.view(b, c, -1)
    ss_map = F.softmax(probs, dim=2) # Interpretation: Self-attention map where each channel attends to spatial locations? 
    # Actually in CSC-PA get_ocr_vector(x) uses x as both features and "probs". 
    # Usually you'd want a separate mask, but if x represents class logits or similar, this makes sense.
    # Here we assume x are features. To get a "map" we might need to project x or just use it.
    # CSC-PA's implementation effectively treats the feature channels as "classes" or "clusters" for the softmax? 
    # If c=channels, dim=2 is spatial. So for each channel, we get a spatial map. 
    # Then we compute prototypes.
    ss_map = ss_map.view(b, c, h, w)
    pb = get_prototype(x, ss_map.clone().detach())
    return pb

class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.GELU(),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)
        B, N, C = feature.size()

        x_theta = self.theta(feature)
        x_phi = self.phi(feature)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

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

    def forward(self, prototype_block):
        # prototype_block: B, C, N (N prototypes)
        att_prototype_block = self.transformer(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1) # B, N, C
        
        # Graph convolutions (GCN style on partial KNN)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        
        graph_prototype_block = graph_prototype.permute(0, 2, 1) # B, C, N
        return graph_prototype_block

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
        
        # 4. Use the map to weight the prototypes and add back to x
        # The correlation map (B, N, H, W) acts as weights for Prototypes (B, C, N).
        # We need to broadcast and multiply.
        
        # Reconstruct "Foreground" features based on prototypes
        # B, C, N x B, N, HW -> B, C, HW
        refined_feat = torch.bmm(graph_pb, map.view(x.size(0), -1, x.size(2)*x.size(3)))
        refined_feat = refined_feat.view(x.size(0), x.size(1), x.size(2), x.size(3))
        
        # 5. Fusion
        out = self.out(torch.cat([x, refined_feat], dim=1))
        
        return out
