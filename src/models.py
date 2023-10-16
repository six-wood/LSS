"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):  # 上采样
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # H,W->H*2,W*2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)  # 通道拼接
        return self.conv(x1)  # 卷一手


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 网格深度（41）
        self.C = C  # 特征维度（64）

        self.trunk = EfficientNet.from_pretrained(
            "efficientnet-b0")  # 预训练backboone
        # 上采样in=320+112 out=512
        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(
            512, self.D + self.C, kernel_size=1, padding=0)  # 1*1变维度

    def get_depth_dist(self, x, eps=1e-20):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # 提特征 x: 24x512x8x22
        x = self.get_eff_depth(x)
        # Depth x: 24x105x8x22 =24x(C+D)xfHxfW
        x = self.depthnet(x)

        # depth->dist
        depth = self.get_depth_dist(x[:, :self.D])

        '''
        将特征通道维和通道维利用广播机制相乘 
        depth.unsqueeze(1) -> torch.Size([24, 1, 41, 8, 22])
        x[:, self.D:(self.D + self.C)] -> torch.Size([24, 64, 8, 22])
        x.unsqueeze(2)-> torch.Size([24, 64, 1, 8, 22])
        depth*x-> new_x: torch.Size([24, 64, 41, 8, 22])
        '''

        new_x = depth.unsqueeze(
            1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):  # 提特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # 预测深度和特征

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()
        # resnet18的前3个stage作为backbone
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        # channel对齐？
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)  # 4倍上采样
        self.up2 = nn.Sequential(  # 2倍上采样->3*3卷积->1*1卷积
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: 4 x 64 x 200 x 200
        x = self.conv1(x)  # x: 4 x 64 x 100 x 100
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 100 x 100
        x = self.layer2(x1)  # x: 4 x 128 x 50 x 50
        x = self.layer3(x)  # x: 4 x 256 x 25 x 25

        x = self.up1(x, x1)
        # 给x进行4倍上采样然后和x1 concat 在一起  x: 4 x 256 x 100 x 100
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: 4 x 1 x 200 x 200

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf  # 网格参数
        self.data_aug_conf = data_aug_conf  # 数据增强参数

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )  # 网格划分
        self.dx = nn.Parameter(dx, requires_grad=False)
        # dx: x,y,z方向上的网格间距 [0.5,0.5,20]
        self.bx = nn.Parameter(bx, requires_grad=False)
        # bx: 第一个网格的中心坐标 [-49.5,-49.5,0]
        self.nx = nn.Parameter(nx, requires_grad=False)
        # nx: 分别为x, y, z三个方向上格子的数量 [200,200,1]

        self.downsample = 16  # 图像下采样倍数
        self.camC = 64  # 图像特征维度
        self.frustum = self.create_frustum()  # 单个相机的伪点云DxfHxfWx3(41x8x22x3)
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        # D: 41 C:64 downsample:16
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):  # 为每一张图片生成一个棱台状（frustum）的点云
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        # 16倍下采样后的图像尺寸

        '''
        ds:在深度方向上划分网格 
        dbound: [4.0, 45.0, 1.0]  
        arange后-> [4.0,5.0,6.0,...,44.0]
        view后(相当于reshape操作)-> (41x1x1)    
        expand后(扩展张量中某维数据的尺寸)->  ds: DxfHxfW(41x8x22)
        '''

        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape

        '''
        xs: 在宽度方向上划分网格
        linspace 后(在[0,ogfW)区间内,均匀划分fW份)-> [0,16,32..336]  大小=fW(22)   
        view后-> 1x1xfW(1x1x22)
        expand后-> xs: DxfHxfW(41x8x22)
        '''
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)

        '''
        ys: 在高度方向上划分网格
        linspace 后(在[0,ogfH)区间内,均匀划分fH份)-> [0,16,32..112]  大小=fH(8)
        view 后-> 1xfHx1 (1x8x1)
        expand 后-> ys: DxfHxfW (41x8x22)
        '''

        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        '''
        frustum: 把xs,ys,ds堆叠到一起
        stack后-> frustum: DxfHxfWx3
        堆积起来形成网格坐标, frustum[d,h,w,0]就是(h,w)位置,深度为d的像素的宽度方向上的栅格坐标
        '''
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        把在相机坐标系(ego frame)下的坐标 (x,y,z) 转换成自车坐标系下的点云坐标
        返回 B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化，先变化像素
        # afterAG = rots * beforeAG + trans
        points = self.frustum - \
            post_trans.view(B, N, 1, 1, 1, 3)  # 广播机制，伪点云复制B*N份，平移一维就够[3]->[3]
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # 旋转要增加一维，[3*3]*[3*1]->[3*1]([3]->[3*1])

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 化为三维坐标，[u,v,d]->[u*d,v*d,d]
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans) 相机外参矩阵，先平移再旋转
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 变换到ego

        return points

    def get_cam_feats(self, x):  # 提取单图特征
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        # B和N两个维度合起来  x: 24 x 3 x 128 x 352
        x = self.camencode(x)
        # 进行图像编码  x: B*N x C x D x fH x fW (24 x 64 x 41 x 8 x 22)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW//self.downsample)
        # 将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 8 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W
        # geom_feats: B x N x D x fH x fW x 3 (4 x 6 x 41 x 8 x 22 x 3)
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        # flatten x
        x = x.reshape(Nprime, C)  # 图像展平，一共B*N*D*H*W 个点

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        # 将[-50,50] [-10 10]的范围平移到[0,100] [0,20]，计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # geom_feats: B*N*D*H*W x 4(173184 x 4), geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]  # 过滤掉超出范围的点

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
            # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x 1 x 200 x 200
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 坐标变换矩阵计算，获取自车伪点云
        geom = self.get_geometry(rots, trans, intrins,
                                 post_rots, post_trans)  # 生成全视角伪点云，这个东西和x无关，可以提前计算
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # 4，6=batch_size
        # x:[4,6,3,128,352]
        # rots: [4,6,3,3]
        # trans: [4,6,3]
        # intrins: [4,6,3,3]
        # post_rots: [4,6,3,3]
        # post_trans: [4,6,3]
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        # 将图像变换到BEV下，B x C x 200 x 200 (4 x 64 x 200 x 200)
        x = self.bevencode(x)
        # 用resnet18提取特征  x: 4 x 1 x 200 x 200
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
