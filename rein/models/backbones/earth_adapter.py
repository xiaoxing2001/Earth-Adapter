import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import reduce
from operator import mul


class earth_adapter(nn.Module):
    def __init__(
        self,
        dim = 64,
        adapter_layer = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
        with_token = False,
        token_dim = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.token_dim = token_dim
        self.adapter_layer = adapter_layer
        self.with_token = with_token
        # self.cutoff_ratio = 0.3
        self.scale = nn.Parameter(torch.tensor([0.1]*24))
        if self.with_token:
            self.refine_token = nn.Parameter(torch.empty([24, 1024, self.token_dim]))  # layer, token_length, embed_dims
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (16, 16), 1) + 1024
            )
        )
        # nn.init.uniform_(self.refine_token.data, -val, val)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(1024) for _ in range(24)])
        # Router 控制 Adapter 选择
        # self.router = nn.ModuleList([nn.Linear(1024, 4) for _ in range(24)])  # 输出 4 个分支的权重
        
        # 多个 MLP 分支
        if self.with_token:
            self.mlp_list1 = nn.ModuleList([nn.Sequential(nn.Linear(1024+self.token_dim, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        else:
            self.mlp_list1 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list2 = nn.ModuleList([nn.Sequential(nn.Linear(1024+64, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list3 = nn.ModuleList([nn.Sequential(nn.Linear(1024+64, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list4 = nn.ModuleList([nn.Sequential(nn.Linear(1024+64, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list3 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list4 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list4 = nn.ModuleList([nn.Sequential(nn.Linear(1024+64, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # self.mlp_list5 = nn.ModuleList([nn.Sequential(nn.Linear(1024+64, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        # 初始化 MLP 权重
        for mlp in self.mlp_list1:       
            nn.init.kaiming_uniform_(mlp[0].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(mlp[2].weight, a=math.sqrt(5))

        # self.relu = nn.ReLU()
        

    def decompose_fft(self, feats: Tensor, layer: int) -> Tensor:

        # 将特征分解为低频和高频
        feats = feats.permute(1, 2, 0).reshape(feats.shape[1], feats.shape[2], 32,32)

        assert feats.dim() == 4, "输入特征必须是4D张量 (batch_size, channels, height, width)"
        
        # 傅里叶变换
        fft = torch.fft.fft2(feats, norm='ortho')  # 形状: (batch_size, channels, height, width)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))  # 将低频分量移到中心
        # 创建低频和高频掩码
        batch_size, channels, H, W = fft_shifted.shape
        cutoff = int(min(H, W) * self.cutoff_ratio // 2)  # 截止频率
        mask_low = torch.zeros_like(fft_shifted)
        cx, cy = H // 2, W // 2
        mask_low[:, :, cx - cutoff:cx + cutoff, cy - cutoff:cy + cutoff] = 1  # 中心区域为低频
        mask_high = 1 - mask_low  # 边缘区域为高频

        # 提取低频和高频分量
        fft_low = fft_shifted * mask_low
        fft_high = fft_shifted * mask_high

        # 逆傅里叶变换恢复空间域特征
        feats_low = torch.fft.ifft2(torch.fft.ifftshift(fft_low, dim=(-2, -1)), norm='ortho').real
        feats_high = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)), norm='ortho').real

        return feats_low, feats_high

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        # 调整输入特征形状
        if layer not in self.adapter_layer:
            return feats
        feats = feats.permute(1, 0, 2)#CLIP注释掉
        cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feats: [1024, batch_size, 1024]


        # low_freq,high_freq = self.decompose_fft(feats, layer)
        # batch_size, channels, height, width = low_freq.shape

        # low_freq = low_freq.reshape(batch_size, channels,-1).permute(2,0,1)
        # high_freq = high_freq.reshape(batch_size, channels,-1).permute(2,0,1)
        
        # 获取当前层的 refine_token
        # combined_feats = torch.cat([feats,], dim=-1)
        if self.with_token:
            tokens = self.refine_token[layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
            combined_feats = torch.cat([feats, tokens], dim=-1)
        else:
            combined_feats = feats
        # tokens1 = self.refine_token[0,layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
        # tokens2 = self.refine_token[1,layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
        # tokens3 = self.refine_token[2,layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
        # tokens4 = self.refine_token[3,layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
        # tokens5 = self.refine_token[4,layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  # [1024, batch_size, 64]
        # 将特征和 token 拼接
        # combined_feats1 = torch.cat([feats, tokens1], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats2 = torch.cat([feats, tokens2], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats3 = torch.cat([feats, tokens3], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats4 = torch.cat([feats, tokens4], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats2 = torch.cat([low_freq, tokens2], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats3 = torch.cat([high_freq, tokens3], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats4 = torch.cat([high_freq, tokens4], dim=-1)  # [1024, batch_size, 1024+64]
        # combined_feats5 = torch.cat([low_freq, tokens5], dim=-1)  # [1024, batch_size, 1024+64]
        # 通过 Router 获取权重
        # router_weights = self.router[layer](feats)  # [batch_size, 4]
        # router_weights = torch.softmax(router_weights, dim=-1)  # 归一化为概率分布
        
        # 计算每个 MLP 分支的输出
        # delta_feat1 = self.mlp_list1[layer](combined_feats)  # [1024, batch_size, 1024]
        # delta_feat2 = self.mlp_list2[layer](low_freq)  # [1024, batch_size, 1024]
        # delta_feat3 = self.mlp_list3[layer](high_freq)  # [1024, batch_size, 1024]
        # delta_feat4 = self.mlp_list4[layer](low_freq)  # [1024, batch_size, 1024]
        # delta_feat5 = self.mlp_list5[layer](high_freq)  # [1024, batch_size, 1024]
        # delta_feat1 = self.mlp_list1[layer](combined_feats1)
        # delta_feat2 = self.mlp_list2[layer](combined_feats2)
        # delta_feat3 = self.mlp_list3[layer](combined_feats3)
        # delta_feat4 = self.mlp_list4[layer](combined_feats4)  # [1024, batch_size, 1024]
        # delta_feat5 = self.mlp_list5[layer](combined_feats5)  # [1024, batch_size, 1024]
        # delta_feat = (
        #         router_weights[:,:, 0].unsqueeze(-1) * delta_feat1 +
        #         router_weights[:,:, 1].unsqueeze(-1) * delta_feat2 +        
        #         router_weights[:,:, 2].unsqueeze(-1) * delta_feat3 +
        #         router_weights[:,:, 3].unsqueeze(-1) * delta_feat4 
        #     # router_weights[:,:, 4].unsqueeze(-1) * delta_feat5
        # )  # [1024, batch_size, 1024]
        
        delta_feat = self.mlp_list1[layer](combined_feats)
        feats = feats + self.scale[layer] * delta_feat#TODO:低频
        
        # 恢复 CLS token 和特征顺序
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        
        return feats
