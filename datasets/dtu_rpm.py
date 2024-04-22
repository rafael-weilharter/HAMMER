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

        self.crop_h = 64*10 #256 #288*2
        self.crop_w = 64*14 #384 #400*2

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

    def reproject_with_depth(self, depth_ref, intr_ref, extrinsics_ref, depth_src, intr_src, extrinsics_src):
    
        intrinsics_ref = np.copy(intr_ref)
        intrinsics_src = np.copy(intr_src)
        intrinsics_ref[:2, :] *= self.depth_scaling
        intrinsics_src[:2, :] *= self.depth_scaling

        # print("intrinsics ref: ", intrinsics_ref)
        # print("intrinsics src: ", intrinsics_src)

        depth_src = cv2.resize(depth_src, None, fx=self.depth_scaling, fy=self.depth_scaling, interpolation=cv2.INTER_AREA)

        width, height = depth_ref.shape[1], depth_ref.shape[0]
        ## step1. project reference pixels to the source view
        # reference view x, y
        x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
        # reference 3D space
        xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                            np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
        # source 3D space
        xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                            np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
        # source view x, y
        K_xyz_src = np.matmul(intrinsics_src, xyz_src)
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        ## step2. reproject the source view points with source view depth estimation
        # find the depth estimation of the source view
        x_src = xy_src[0].reshape([height, width]).astype(np.float32)
        y_src = xy_src[1].reshape([height, width]).astype(np.float32)

        # print("x max min: ", np.max(x_src), np.min(x_src))
        # print("y max min: ", np.max(y_src), np.min(y_src))

        # sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

        # save_pfm("./sampled_depth.pfm", sampled_depth_src)
        # save_pfm("./depth_ref.pfm", depth_ref)
        # save_pfm("./depth_src.pfm", depth_src)

        return x_src, y_src
        # # mask = sampled_depth_src > 0

        # # source 3D space
        # # NOTE that we should use sampled source-view depth_here to project back
        # xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
        #                     np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
        # # reference 3D space
        # xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        #                             np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
        # # source view x, y, depth
        # depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
        # K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
        # xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
        # x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
        # y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

        # return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

    def search_patch_match(self, scaling, img, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
        img_adj = None
        intrinsics_adj = None
        start_h = 0
        start_w = 0

        success = False

        # h, w = depth_ref.shape
        center_h = self.crop_h//2
        center_w = self.crop_w//2

        # print("centers: ", center_h, center_w)

        x_avg = 0
        y_avg = 0

        # print("depth shapes: ", depth_ref.shape, depth_src.shape)

        x_src, y_src = self.reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)

        # print("x_src.shape: ", x_src.shape)

        search_px = 10 #20
        dil = 10
        count_valid = 0
        for i in range(search_px):
            for j in range(search_px):
                x = center_w - dil*search_px//2 + i*dil
                y = center_h - dil*search_px//2 + j*dil
                if x_src[y,x] > 0 and x_src[y,x] < self.max_w*self.depth_scaling:
                    if y_src[y,x] > 0 and y_src[y,x] < self.max_h*self.depth_scaling:
                        count_valid +=1
                        x_avg += x_src[y,x]
                        y_avg += y_src[y,x]


        if(count_valid > 0):
            start_w = int((x_avg/self.depth_scaling)/count_valid - self.crop_w//2)
            start_h = int((y_avg/self.depth_scaling)/count_valid - self.crop_h//2)

            #wiggle patch
            start_w = start_w - self.crop_w//4 + np.random.random_integers(0, self.crop_w//2)
            start_h = start_h - self.crop_h//4 + np.random.random_integers(0, self.crop_h//2)

        if start_h < 0:
            start_h = 0
        if start_h > self.max_h - self.crop_h - 1:
            start_h = self.max_h - self.crop_h - 1

        if start_w < 0:
            start_w = 0
        if start_w > self.max_w - self.crop_w - 1:
            start_w = self.max_w - self.crop_w - 1

        # print("start_h: ", start_h)
        # print("start_w: ", start_w)

        img_adj, intrinsics_adj, depth_crop = self.scale_and_crop(self.scaling, img, intrinsics_src, depth_image=depth_src, start_h=start_h, start_w=start_w)

        if count_valid > search_px*search_px//4: #25% valid
            success = True

            if(np.count_nonzero(depth_crop) < depth_crop.size//4): #4
                success = False

            # save_pfm("./depth_crop.pfm", depth_crop)
            # save_pfm("./img_src.pfm", img_adj)


        return img_adj, intrinsics_adj, success

    def search_patch(self, scaling, img, intrinsics, depth):
        border_px = 10
        depth_adj = np.zeros(depth.shape)
        img_adj = None
        intrinsics_adj = None
        start_h = 0
        start_w = 0

        counter = 0
        while (True):
            counter += 1
            start_h = np.random.random_integers(border_px, self.max_h - self.crop_h - border_px)
            start_w = np.random.random_integers(border_px, self.max_w - self.crop_w - border_px)
            img_adj, intrinsics_adj, depth_adj = self.scale_and_crop(self.scaling, img, intrinsics, depth, start_h, start_w)
            if(np.count_nonzero(depth_adj) > depth_adj.size//4): #at least 25% valid gt values
                print(f"found valid patch after {counter} tries.")
                break
            if(counter > 20): # max tries
                print(f"did not find a good patch after {counter} tries!")
                break

        return img_adj, intrinsics_adj, depth_adj, start_h, start_w


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
            if not depth_image is None:
                depth_image = cv2.resize(depth_image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

        # crop images and cameras
        h, w = img.shape[0:2]
        new_h = self.crop_h
        new_w = self.crop_w
        finish_h = start_h + new_h
        finish_w = start_w + new_w

        img = img[start_h:finish_h, start_w:finish_w]

        #intrinsics are already scaled by factor 4
        intrinsics[0,2] = intrinsics[0,2] - start_w
        intrinsics[1,2] = intrinsics[1,2] - start_h

        # crop depth image
        if not depth_image is None:
            depth_image = depth_image[start_h:finish_h, start_w:finish_w]
            depth_image = cv2.resize(depth_image, None, fx=self.depth_scaling, fy=self.depth_scaling, interpolation=cv2.INTER_AREA)
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

        vid = 0
        total_tries = 0

        intrinsics_ref = None
        extrinsics_ref = None

        batch_invalid = False

        while vid < self.nviews:
            total_tries += 1
            if (total_tries > 30): 
                batch_invalid = True
            # if (total_tries > 30): #30
            #     print("FAILED to find a valid stack!")
            #     return {"imgs": [],
            #     "proj_matrices": [],
            #     "depth": None,
            #     "depth_values": None,
            #     "mask": None,
            #     "filename": "FAILED"}

            # img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            # proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            # depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            # mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(vid)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            img = self.read_img(img_filename)

            # depth_interval = 2 #should still cover full range (?) -> results get worse
            depth_num = 256 #according to MVSNet paper
            depth_interval_scaling = depth_num/self.ndepths
            depth_interval *= depth_interval_scaling

            # depth_interval = (depth_max - depth_min) / self.ndepths

            if vid == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)
                # depth_values = np.arange(depth_min, depth_max, depth_interval,
                #                          dtype=np.float32)
                depth = self.read_depth(depth_filename)
                # img, intrinsics, depth = self.scale_and_crop(self.scaling, img, intrinsics, depth, start_h, start_w)
                img, intrinsics, depth, start_h, start_w = self.search_patch(self.scaling, img, intrinsics, depth)
                print("image: ", img_filename, img.shape)
                print("depth: ", depth_filename, depth.shape)
                intrinsics_ref = intrinsics
                extrinsics_ref = extrinsics
            else:
                depth_src = self.read_depth(depth_filename)
                success = True
                # img, intrinsics, _ = self.scale_and_crop(self.scaling, img, intrinsics, depth_src, start_h=start_h, start_w=start_w)
                img, intrinsics, success = self.search_patch_match(self.scaling, img, depth, intrinsics_ref, extrinsics_ref, depth_src, intrinsics, extrinsics)

                # if(total_tries < 20):
                #     success = False

                if(success == False) and not batch_invalid: #reset
                    print("no match")
                    vid = 0
                    imgs = []
                    mask = None
                    depth = None
                    depth_values = None
                    intrinsics_ref = None
                    extrinsics_ref = None
                    proj_matrices = []
                    continue

            # print("img shape: ", img.shape)

            imgs.append(img)

            # print("while imgs length: ", len(imgs))

            # print("img shape: ", img.shape)
            # print("intrinsics: ", intrinsics)

            # intrinsics and extrinsics separately
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            vid += 1

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        # print("imgs shape: ", imgs.shape)
        # print("proj_matrices length: ", len(proj_matrices))

        #print("len images: ", len(imgs))
        print(f"found valid stack after {total_tries} tries")

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

        batch_filename = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        if(batch_invalid):
            batch_filename = "FAILED"

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth,
                "depth_values": depth_values,
                "mask": mask,
                "filename": batch_filename}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
