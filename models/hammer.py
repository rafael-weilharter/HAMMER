import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import *
from .module import *

from models.features.UNet import unet as FeatureNet

Align_Corners_Range = False

#HAMMER
class HAMMER(nn.Module):
    def __init__(self, blocks=5, heads=1, channels=32, stages=10, psi=0.6, output_scaling=1):
        super(HAMMER, self).__init__()

        self.grad_method = "detach"

        c = channels
        self.num_stage = stages 
        self.depth_scale_psi = psi
        self.depth_intervals_ratio = [8, 4, 2, 1]
        self.output_scaling = output_scaling

        num_blocks = blocks
        num_heads = heads

        if (self.num_stage == 8):
            self.decoders = torch.nn.ModuleList([decoderBlock(num_blocks,c,c, att=False),
                                decoderBlock(num_blocks,c,c, att=False),
                                decoderBlock(num_blocks,c,c, att=False),
                                decoderBlock(num_blocks,c,c, att=False)])
        else:
            self.decoders = torch.nn.ModuleList([decoderBlock(num_blocks,c,c, att=False),
                               decoderBlock(num_blocks,c,c, att=False),
                               decoderBlock(num_blocks,c,c, att=False),
                               decoderBlock(num_blocks,c,c, att=False),
                               decoderBlock(num_blocks,c,c, att=False)])
            self.depth_intervals_ratio = [16, 8, 4, 2, 1]

        self.entropy_layer = decoderBlockPooling(2,c,c, pool=True)

        self.stage_infos = {
            "stage0":{
                "scale": 2,
            },
            "stage1":{
                "scale": 4,
            },
            "stage2": {
                "scale": 8,
            },
            "stage3": {
                "scale": 16,
            },
            "stage4": {
                "scale": 32,
            },
            "stage5": {
                "scale": 64,
            }
        }

        self.feature_net = FeatureNet(stages=self.num_stage//2, num_chan=c, heads=num_heads, first_stride=1)

        #freeze layers
        # for param in self.feature_net.parameters():
        #     param.requires_grad = False

        print(f"created hrtmvsnet_v18 network with: {self.num_stage} stages, {c} channels, {num_heads} heads")

    def get_volume_variance(self, features, proj_matrices, depth_values, num_depth):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_ms(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        return volume_variance

    def get_volume_variance_and_mask(self, features, proj_matrices, depth_values, num_depth):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        b, c, d, w, h = volume_sum.shape
        cv_mask = torch.cuda.FloatTensor(b,w,h).fill_(0)

        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_ms(src_fea, src_proj_new, ref_proj_new, depth_values)

            cv_mask += torch.count_nonzero(torch.count_nonzero(warped_volume, dim=1), dim=1)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2) 

            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        return volume_variance, cv_mask

    def forward(self, imgs, proj_mats, depth_values_init):
        depth_min = float(depth_values_init[0, 0].cpu().numpy())
        depth_max = float(depth_values_init[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / 4 #depth_values_init.size(1)

        #order projection matrices
        proj_matrices = {
            "stage0": proj_mats["stage6"], # 1/2
            "stage1": proj_mats["stage0"], # 1/4
            "stage2": proj_mats["stage1"], # 1/8
            "stage3": proj_mats["stage2"], # 1/16
            "stage4": proj_mats["stage3"], # 1/32
            "stage5": proj_mats["stage4"]  # 1/64
        }

        b, n, c, output_h, output_w = imgs.shape
        output_h = output_h // self.output_scaling
        output_w = output_w // self.output_scaling

        print("output h w: ", output_h, output_w)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature_net(img))

        print("CUDA mem usage for feature extraction: ", torch.cuda.memory_allocated())

        # step 2. cost volume from homography warping
        depth, cur_depth = None, None
        feat_2x = None
        entropy = None
        stacked = []
        entropy_stacked = []

        cv_mask = torch.zeros(b, output_h, output_w)

        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            scale_idx = stage_idx//2

            features_stage = [feat["stage{}".format(self.num_stage//2 - scale_idx)] for feat in features] #reverse order
            proj_matrices_stage = proj_matrices["stage{}".format(self.num_stage//2 - (scale_idx+1))]
            stage_scale = self.stage_infos["stage{}".format(self.num_stage//2 - (scale_idx+1))]["scale"]

            cur_h = img.shape[2]//int(stage_scale)
            cur_w = img.shape[3]//int(stage_scale)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
            else:
                cur_depth = depth_values_init
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=4,
                                                        depth_interval_pixel=depth_interval,
                                                        dtype=img.dtype,
                                                        device=img.device,
                                                        shape=[b, cur_h, cur_w], #img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            print("depth interval: ", depth_interval)
            depth_interval = depth_interval*self.depth_scale_psi #0.55

            depth_tmp = depth_range_samples

            if(stage_idx == 0):
                volume_variance, cv_mask = self.get_volume_variance_and_mask(features_stage, proj_matrices_stage, 
                                                    depth_values=depth_tmp,
                                                    num_depth=4)
                cv_mask = cv_mask.unsqueeze(1)
                cv_mask = F.interpolate(cv_mask, [output_h, output_w], mode='nearest') #nearest
                cv_mask = cv_mask.squeeze(1)
            else:
                volume_variance = self.get_volume_variance(features_stage, proj_matrices_stage, 
                                                    depth_values=depth_tmp,
                                                    num_depth=4)

            cost = self.decoders[scale_idx](volume_variance)
            print("cost shape: ", cost.shape)

            if(stage_idx == self.num_stage - 1):
                depth_tmp_up = F.interpolate(depth_range_samples.unsqueeze(1),
                                [4, output_h, output_w], mode='trilinear', #trilinear
                                align_corners=Align_Corners_Range).squeeze(1)
                cost = F.interpolate(cost, [output_h, output_w], mode='bilinear') #bilinear

            else:
                if(stage_idx%2 == 1):
                    depth_tmp_up = F.interpolate(depth_range_samples.unsqueeze(1),
                                    [4, cur_h*2, cur_w*2], mode='trilinear', #trilinear
                                    align_corners=Align_Corners_Range).squeeze(1)
                    cost = F.interpolate(cost, [cur_h*2, cur_w*2], mode='bilinear') #bilinear
                else:
                    depth_tmp_up = depth_tmp

            depth, entropy = depth_regression(F.softmax(cost,1), depth_values=depth_tmp_up, confidence=True)
            
            if not self.training:
                entropy_stacked.append(torch.squeeze(entropy))
            else:
                entropy_stacked.append(entropy)

            if(stage_idx == self.num_stage - 1):
                cost_entropy = self.entropy_layer(volume_variance)
                cost_entropy = F.interpolate(cost_entropy, [output_h, output_w], mode='bilinear') #bilinear
                depth_entropy, entropy_loss = depth_regression(F.softmax(cost_entropy,1), depth_values=depth_tmp_up, confidence=True)
                if not self.training:
                    entropy_stacked.append(torch.squeeze(entropy_loss))
                else:
                    entropy_stacked.append(entropy_loss)
            
            stacked.append(depth)
            
            print("CUDA mem usage after stage{}: ".format(stage_idx + 1), torch.cuda.memory_allocated())

        stacked.reverse()
        entropy_stacked.reverse()

        if self.training:
            return stacked, entropy_stacked, cv_mask
        else:
            return depth, entropy_stacked, cv_mask


def hammer_loss(stacked, depth_gt_ms, cv_mask):
    cv_mask = cv_mask.type(torch.cuda.FloatTensor)

    depth_gt = depth_gt_ms["fullRes"]
    
    depth_gt_0 = depth_gt_ms["stage0"]
    depth_gt_1 = depth_gt_ms["stage1"]
    depth_gt_2 = depth_gt_ms["stage2"]
    depth_gt_3 = depth_gt_ms["stage3"]
    depth_gt_4 = depth_gt_ms["stage4"]

    print("cv_mask: ", cv_mask.shape)

    cv_mask_0 = F.interpolate(cv_mask.unsqueeze(1), [depth_gt_0.shape[1], depth_gt_0.shape[2]], mode='nearest').squeeze(1) > 0.5
    cv_mask_1 = F.interpolate(cv_mask.unsqueeze(1), [depth_gt_1.shape[1], depth_gt_1.shape[2]], mode='nearest').squeeze(1) > 0.5
    cv_mask_2 = F.interpolate(cv_mask.unsqueeze(1), [depth_gt_2.shape[1], depth_gt_2.shape[2]], mode='nearest').squeeze(1) > 0.5
    cv_mask_3 = F.interpolate(cv_mask.unsqueeze(1), [depth_gt_3.shape[1], depth_gt_3.shape[2]], mode='nearest').squeeze(1) > 0.5
    cv_mask_4 = F.interpolate(cv_mask.unsqueeze(1), [depth_gt_4.shape[1], depth_gt_4.shape[2]], mode='nearest').squeeze(1) > 0.5

    mask_0 = depth_gt_0 > 0 & cv_mask_0
    mask_0.detach_()
    mask_1 = depth_gt_1 > 0 & cv_mask_1
    mask_1.detach_()
    mask_2 = depth_gt_2 > 0 & cv_mask_2
    mask_2.detach_()
    mask_3 = depth_gt_3 > 0 & cv_mask_3
    mask_3.detach_()
    mask_4 = depth_gt_4 > 0 & cv_mask_4
    mask_4.detach_()

    if(stacked[0].shape == depth_gt.shape):
        mask_output = (depth_gt > 0) & (cv_mask > 0.5)
        mask_output.detach_()
        depth_gt_output = depth_gt
    else:
        mask_output = mask_0
        depth_gt_output = depth_gt_0

    if len(stacked) == 8:
        loss = (8./30)*F.smooth_l1_loss(stacked[0][mask_output], depth_gt_output[mask_output], reduction='mean') + \
                (8./30)*F.smooth_l1_loss(stacked[1][mask_0], depth_gt_0[mask_0], reduction='mean') + \
                (4./30)*F.smooth_l1_loss(stacked[2][mask_0], depth_gt_0[mask_0], reduction='mean') + \
                (4./30)*F.smooth_l1_loss(stacked[3][mask_1], depth_gt_1[mask_1], reduction='mean') + \
                (2./30)*F.smooth_l1_loss(stacked[4][mask_1], depth_gt_1[mask_1], reduction='mean') + \
                (2./30)*F.smooth_l1_loss(stacked[5][mask_2], depth_gt_2[mask_2], reduction='mean') + \
                (1./30)*F.smooth_l1_loss(stacked[6][mask_2], depth_gt_2[mask_2], reduction='mean') + \
                (1./30)*F.smooth_l1_loss(stacked[7][mask_3], depth_gt_3[mask_3], reduction='mean')

    if len(stacked) == 10:
        loss = (32./93)*F.smooth_l1_loss(stacked[0][mask_output], depth_gt_output[mask_output], reduction='mean') + \
                (16./93)*F.smooth_l1_loss(stacked[1][mask_0], depth_gt_0[mask_0], reduction='mean') + \
                (16./93)*F.smooth_l1_loss(stacked[2][mask_0], depth_gt_0[mask_0], reduction='mean') + \
                (8./93)*F.smooth_l1_loss(stacked[3][mask_1], depth_gt_1[mask_1], reduction='mean') + \
                (8./93)*F.smooth_l1_loss(stacked[4][mask_1], depth_gt_1[mask_1], reduction='mean') + \
                (4./93)*F.smooth_l1_loss(stacked[5][mask_2], depth_gt_2[mask_2], reduction='mean') + \
                (4./93)*F.smooth_l1_loss(stacked[6][mask_2], depth_gt_2[mask_2], reduction='mean') + \
                (2./93)*F.smooth_l1_loss(stacked[7][mask_3], depth_gt_3[mask_3], reduction='mean') + \
                (2./93)*F.smooth_l1_loss(stacked[8][mask_3], depth_gt_3[mask_3], reduction='mean') + \
                (1./93)*F.smooth_l1_loss(stacked[9][mask_4], depth_gt_4[mask_4], reduction='mean')

    return loss