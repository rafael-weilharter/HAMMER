from torch.utils.data import Dataset
import torchvision
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

import cv2
import math

# max_h = 1056
# max_w = 1920

# max_h = 576
# max_w = 768

#scaled image will be centrally cropped to this size
# max_h = 1024
# max_w = 1536

# max_h = 768
# max_w = 1024

base_image_size = 64 #cropped/scaled image size will be divisible by this number
# base_image_size = 128

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        self.scaling = 1.0
        self.augment_data = False
        self.depth_scaling = 1.0

        self.max_h = 1200
        self.max_w = 1600

        for key, value in kwargs.items(): 
            print ("%s == %s" %(key, value))
            if(key == 'scaling'):
                self.scaling = value
            if(key == 'augment_data'):
                self.augment_data = value
            if(key == 'depth_scaling'):
                self.depth_scaling = 1.0/value

        self.crop_h = 512 + 64*2 #288*2
        self.crop_w = 512 + 64*2 #400*2

        print("Dataset mode: ", self.mode)
        # if (self.mode == "train"):
        #     self.max_h = 768
        #     self.max_w = 1024

        # assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if(len(src_views) < self.nviews - 1): #not enough src views
                        continue
                    # light conditions 0-6
                    if(self.mode == "test"):
                        metas.append((scan, 3, ref_view, src_views))
                    else:    
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))

        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def generate_stage_depth(self, depth):
        h, w = depth.shape
        depth_ms = {
            "stage0": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage1": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w // 16, h // 16), interpolation=cv2.INTER_NEAREST),
            "stage4": cv2.resize(depth, (w // 32, h // 32), interpolation=cv2.INTER_NEAREST),
            "fullRes": depth
        }
        return depth_ms

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        #only change extrinsics in 50% of cases
        # if(self.augment_data and np.random.binomial(1, 0.5)):
        #     R = extrinsics[:3,:3]
        #     # print("det(R) = {}".format(np.linalg.det(R)))
        #     random_rotation = np.random.uniform(-1, 1, 1) #1° max
        #     print("data augmentation rotation: ", random_rotation[0])
        #     theta = random_rotation[0]*math.pi/180 #to radians
        #     Rx = np.array([
        #         [1, 0, 0],
        #         [0, math.cos(theta), -math.sin(theta)],
        #         [0, math.sin(theta), math.cos(theta)]
        #     ])
        #     Ry = np.array([
        #         [math.cos(theta), 0, math.sin(theta)],
        #         [0, 1, 0],
        #         [-math.sin(theta), 0, math.cos(theta)]
        #     ])
        #     Rz = np.array([
        #         [math.cos(theta), -math.sin(theta), 0],
        #         [math.sin(theta), math.cos(theta), 0],
        #         [0, 0, 1]
        #     ])
        #     R = np.matmul(np.matmul(np.matmul(Rz, Ry), Rx), R)
        #     assert R.transpose().all() == np.linalg.inv(R).all(), "R.transpose() != R.inv()"
        #     # assert np.linalg.det(R) == 1, "det(R) = {}".format(np.linalg.det(R))
        #     extrinsics[:3,:3] = R

        #     t = extrinsics[:3,3]
        #     random_translation = np.random.uniform(0.99, 1.01, 3) # up to 1% translation error
        #     t = np.multiply(t, random_translation)
        #     extrinsics[:3,3] = t

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        if(self.augment_data and np.random.binomial(1, 0.5)):
            # photometric unsymmetric-augmentation
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            img = torchvision.transforms.functional.adjust_brightness(img, random_brightness[0])
            img = torchvision.transforms.functional.adjust_gamma(img, random_gamma[0])
            img = torchvision.transforms.functional.adjust_contrast(img, random_contrast[0])
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_and_crop(self, scaling, img_in, intrinsics_in, depth_image_in=None, start_h=0, start_w=0):
        #do not manipulate input data directly
        img = np.copy(img_in)
        intrinsics = np.copy(intrinsics_in)
        depth_image = None
        if not (depth_image_in is None):
            depth_image = np.copy(depth_image_in)
        
        if(scaling != 1.0):
            img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
            intrinsics[:2, :] *= scaling
            if not (depth_image is None):
                depth_image = cv2.resize(depth_image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

        # crop images and cameras
        h, w = img.shape[0:2]
        new_h = self.crop_h
        new_w = self.crop_w
        # print("scale and crop h, w: ", h, w)
        # if new_h > self.max_h:
        #     new_h = self.max_h
        # else:
        #     new_h = int(math.floor(h / base_image_size) * base_image_size)
        # if new_w > self.max_w:
        #     new_w = self.max_w
        # else:
        #     new_w = int(math.floor(w / base_image_size) * base_image_size)
        # # print("new h, w: ", new_h, new_w)
        # start_h = int(math.ceil((h - new_h) / 2))
        # start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        img = img[start_h:finish_h, start_w:finish_w]

        #intrinsics are already scaled by factor 4
        intrinsics[0,2] = intrinsics[0,2] - start_w
        intrinsics[1,2] = intrinsics[1,2] - start_h

        # crop depth image
        if not depth_image is None:
            depth_image = depth_image[start_h:finish_h, start_w:finish_w]
            # depth_image = cv2.resize(depth_image, None, fx=self.depth_scaling, fy=self.depth_scaling, interpolation=cv2.INTER_AREA)
            return img, intrinsics, depth_image
        else:
            return img, intrinsics

    def __getitem__(self, idx):
        # print("getitem")
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        border_px = 10
        start_h = 0
        start_w = 0
        # start_h = np.random.random_integers(border_px, self.max_h - self.crop_h - border_px)
        # start_w = np.random.random_integers(border_px, self.max_w - self.crop_w - border_px)

        for i, vid in enumerate(view_ids):
            # img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            # proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            # depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            # mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid)) #Depths_raw
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(vid)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            img = self.read_img(img_filename)

            # depth_interval = 2 #should still cover full range (?) -> results get worse
            depth_num = 256 #according to MVSNet paper
            depth_interval_scaling = depth_num/self.ndepths
            depth_interval *= depth_interval_scaling

            # depth_interval = (depth_max - depth_min) / self.ndepths

            if i == 0:  # reference view
                depth_max = depth_interval * (self.ndepths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval,
                                         dtype=np.float32)
                print("depth values shape: ", depth_values.shape)
                print("depth range: ", depth_min, depth_max)
                # depth_values = np.arange(depth_min, depth_max, depth_interval,
                #                          dtype=np.float32)
                depth = self.read_depth(depth_filename)

                try_count = 0
                while(True):
                    try_count += 1
                    start_h = np.random.random_integers(border_px, self.max_h - self.crop_h - border_px)
                    start_w = np.random.random_integers(border_px, self.max_w - self.crop_w - border_px)

                    img_crop, intrinsics_tmp, depth_crop = self.scale_and_crop(self.scaling, img, intrinsics, depth, start_h, start_w)

                    if(np.count_nonzero(depth_crop) > depth_crop.size//8):
                        break

                    if(try_count > 20):
                        print("FAILED TO FIND VALID IMAGE PATCH!")
                        break

                intrinsics = intrinsics_tmp
                depth = depth_crop
                img = img_crop

                print(f"image after {try_count}: ", img_filename, img.shape)
                # print("depth: ", depth_filename, depth.shape)
                # cv2.imshow("img", img)
                # cv2.imshow("depth", depth)
                # cv2.waitKey(0)
            else:
                img, intrinsics = self.scale_and_crop(self.scaling, img, intrinsics, depth_image_in=None, start_h=start_h, start_w=start_w)

            # print("img_shape: ", img.shape)

            imgs.append(img)

            # print("img shape: ", img.shape)
            # print("intrinsics: ", intrinsics)

            # intrinsics and extrinsics separately
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        print("imgs shape: ", imgs.shape)

        print("len images: ", len(imgs))

        # mvsnet_scale = 0.25
        # if(mvsnet_scale != 1.0):
        #     depth = cv2.resize(depth, None, fx=mvsnet_scale, fy=mvsnet_scale, interpolation=cv2.INTER_NEAREST) #cv2.INTER_CUBIC

        mask = np.ones(depth.shape)
        super_threshold_indices = (depth <= 0)
        mask[super_threshold_indices] = 0
        print("gt invalid values: ", np.count_nonzero(mask==0))

        # print("mask shape: ", mask.shape)
        # print("depth shape: ", depth.shape)
        # cv2.imshow("mask", mask)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(0)

        # super_threshold_indices = (depth == 0)
        # mask[super_threshold_indices] = 0
        # print("gt invalid values: ", np.count_nonzero(mask==0))
        # super_threshold_indices = np.isnan(depth)
        # mask[super_threshold_indices] = 0
        # print("gt invalid values: ", np.count_nonzero(mask==0))
        # super_threshold_indices = np.isinf(depth)
        # mask[super_threshold_indices] = 0
        # print("gt invalid values: ", np.count_nonzero(mask==0))

        # print("depth values: ", depth_values.shape)

        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 16
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 32
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 64

        stage5_pjmats = proj_matrices.copy()
        stage6_pjmats = proj_matrices.copy()
        stage6_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2

        proj_matrices_ms = {
            "stage0": stage0_pjmats,
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats,
            "stage5": stage5_pjmats,
            "stage6": stage6_pjmats
        }

        # print("mask shape: ", mask.shape)

        depth_ms = self.generate_stage_depth(depth)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))