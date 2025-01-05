import numpy as np
import torch
from torch import nn

#拼接图像和lidar点云
class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    



    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality（BN,C,H,W)
                spatial_features (tensor): Bev features from lidar modality (BN,C*D,H,W)

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']

        # # 打印 img_bev 和 lidar_bev 的维度
        # print("img_bev shape:", img_bev.shape)
        # print("lidar_bev shape:", lidar_bev.shape)


        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        # print("after concat shape:", cat_bev.shape)

        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict
    


    