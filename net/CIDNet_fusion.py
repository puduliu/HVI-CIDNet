import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin


class CIDNet_DualHDR(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet_DualHDR, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways - 修改输入通道数从3到6，处理两张RGB图像
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(6, ch1, 3, stride=1, padding=0, bias=False)  # 修改：3→6通道
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # I_ways - 修改输入通道数从1到2，处理两张图像的强度信息
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0, bias=False),  # 修改：1→2通道
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),  # 输出仍为1通道
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
        # 新增：用于融合两个亮度分量的1x1卷积层
        self.intensity_fusion = nn.Conv2d(2, 1, 1, stride=1, padding=0, bias=False)
        
    def forward(self, x1, x2):  # 修改：接受两个输入图像
        """
        Args:
            x1: 第一张低光照图像 (B, 3, H, W)
            x2: 第二张不同HDR亮度的低光照图像 (B, 3, H, W)
        """
        dtypes = x1.dtype
        
        # 对两张图像分别进行HVI变换
        hvi1 = self.trans.HVIT(x1)  # (B, 3, H, W)
        hvi2 = self.trans.HVIT(x2)  # (B, 3, H, W)
        
        # 提取强度分量并拼接
        i1 = hvi1[:,2,:,:].unsqueeze(1).to(dtypes)  # (B, 1, H, W)
        i2 = hvi2[:,2,:,:].unsqueeze(1).to(dtypes)  # (B, 1, H, W)
        i_concat = torch.cat([i1, i2], dim=1)  # (B, 2, H, W) - 拼接两个强度分量
        
        # 提取HV分量并拼接
        hv1 = hvi1[:, :2, :, :]  # (B, 2, H, W)
        hv2 = hvi2[:, :2, :, :]  # (B, 2, H, W) 
        hv_concat = torch.cat([hv1, hv2], dim=1)  # (B, 4, H, W) - 拼接两个HV分量
        
        # 将拼接的HV分量与原始HVI拼接作为HV分支输入
        hvi_concat = torch.cat([hv_concat, i_concat], dim=1)  # (B, 6, H, W)
        
        # low level processing
        i_enc0 = self.IE_block0(i_concat)  # 修改：使用拼接的强度信息
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi_concat)  # 修改：使用拼接的HVI信息
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # 交叉注意力和下采样 - 其余部分保持不变
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # 瓶颈层
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        # 上采样和交叉注意力
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        # 构建输出HVI - 这里需要选择参考的HVI进行残差连接
        # 方案1：使用第一张图像的HVI作为参考
        # output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi1 # 倾向于以第一张图像为"主图像"，第二张图像主要提供补充信息；可能会偏向第一张图像的特征
        
        # 方案2：也可以使用融合后的HVI作为参考
        fused_i = self.intensity_fusion(i_concat)
        fused_hvi = torch.cat([0.5*(hv1+hv2), fused_i], dim=1)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + fused_hvi
        
        
        # TODO 需要进一步进行图像融合，在亮度ok的情况下是不是可以使用torch max, 还是使用torch.mean 都试试
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self, x1, x2):  # 修改：接受双输入
        """返回两张图像的HVI表示"""
        hvi1 = self.trans.HVIT(x1)
        hvi2 = self.trans.HVIT(x2)
        return hvi1, hvi2


# 使用示例：
# model = CIDNet_DualHDR()
# low_light_img1 = torch.randn(1, 3, 512, 512)  # 第一张低光照图像
# low_light_img2 = torch.randn(1, 3, 512, 512)  # 第二张不同HDR亮度的低光照图像
# enhanced_img = model(low_light_img1, low_light_img2)