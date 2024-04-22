import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
import math
from plyfile import PlyData, PlyElement
from PIL import Image

cudnn.benchmark = False #True

parser = argparse.ArgumentParser(description='A PyTorch Implementation for testing ATLAS-MVSNet')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=384, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.0, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')

parser.add_argument('--consistent', type=int, default=3, help='number of consistent views for geometric verification')
parser.add_argument('--confidence', type=float, default=-1.0, help='confidence for photometric verification')

parser.add_argument('--depth_only', action='store_true', help='only run network for depth estimation without pointcloud fusion')
parser.add_argument('--evaluate', action='store_true', help='compare output to groundtruth depth map')
parser.add_argument('--output_scale', type=int, default=1, help='scaling for the output depth image, integer')
parser.add_argument('--neighbors', type=int, default=5, help='number of image neighbors stacked in the network')
parser.add_argument('--input_scale', type=float, default=1.0, help='scaling for the input rgb image')

parser.add_argument('--ent_high', type=float, default=-1.0, help='entropy upper limit, disabled if set to negative value')
parser.add_argument('--ent_low', type=float, default=-1.0, help='entropy lower limit, disabled if set to negative value')
parser.add_argument('--cv_mask', type=int, default=0, help='min number of overlaps in the cost volume')
parser.add_argument('--dist', type=float, default=0.25, help='max distance for geometric verification')
parser.add_argument('--rel_dist', type=int, default=100, help='max relative distance factor for geometric verification')
parser.add_argument('--dyn', type=int, default=0, help='use dynamic filtering instead of distance filtering')

parser.add_argument('--num_blocks', type=int, default=5, help='number of 3D regularization blocks')
parser.add_argument('--num_heads', type=int, default=1, help='number of attention heads for 2D and 3D')
parser.add_argument('--num_channels', type=int, default=32, help='number of maximum channels, has to be divisible by 4 (and 3 if 3D HAB is used)')
parser.add_argument('--num_stages', type=int, default=10, help='number of regularization stages, should be 8 or 10')
parser.add_argument('--psi', type=float, default=0.6, help='depth interval scale parameter psi')

parser.add_argument('--filter_isolated', action='store_true', help='filter isolated single pixels')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

MVSDataset = find_dataset_def(args.dataset)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.neighbors, args.numdepth, args.interval_scale, scaling=args.input_scale, depth_scaling=args.output_scale)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False) #num_workers=4

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: fix this scaling
    # intrinsics[:2, :] /= 2
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

            # add neighbors
            count = 0
            while len(src_views) < 48: #48
                count += 1
                neighbor_idx_pos = int(ref_view + count)
                if neighbor_idx_pos < num_viewpoint and neighbor_idx_pos not in src_views:
                    src_views.append(neighbor_idx_pos)
                
                neighbor_idx_neg = int(ref_view - count)
                if neighbor_idx_neg >= 0 and neighbor_idx_neg not in src_views:
                    src_views.append(neighbor_idx_neg)


            data.append((ref_view, src_views))
    return data

def scale_and_crop_eval(scaling, img_in, intrinsics_in, depth_image_in):
    #scaled image will be centrally cropped to this size
    max_h = 10000
    max_w = 10000

    #avoid direct manipulation
    intrinsics = np.copy(intrinsics_in)
    img = np.copy(img_in)
    depth_image = np.copy(depth_image_in)

    #cropped/scaled image size will be divisible by this number
    base_image_size = 64 #64 #128

    if(scaling != 1.0):
        img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
        #depth_image = cv2.resize(depth_image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
        intrinsics[:2, :] *= scaling

    # crop images and cameras
    h, w = img.shape[0:2]
    new_h = h
    new_w = w
    # print("scale and crop h, w: ", h, w)
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.floor(h / base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.floor(w / base_image_size) * base_image_size)

    # print("new h, w: ", new_h, new_w)
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    img = img[start_h:finish_h, start_w:finish_w]

    #intrinsics are already scaled by factor 4
    intrinsics[0,2] = intrinsics[0,2] - start_w
    intrinsics[1,2] = intrinsics[1,2] - start_h

    depth_h, depth_w = depth_image.shape[0:2]

    color_to_depth_ratio = depth_h/new_h
    # print("color_to_depth_ratio: ", color_to_depth_ratio)
    # print("sizes: ", depth_h, depth_w, new_h, new_w)
    # print("start: ", start_h, start_w)

    assert color_to_depth_ratio == depth_w/new_w, "depth map does not match color image ratio: {} != {}".format(color_to_depth_ratio,depth_w/new_w)

    intrinsics[:2, :] *= color_to_depth_ratio
    img = cv2.resize(img, None, fx=color_to_depth_ratio, fy=color_to_depth_ratio, interpolation=cv2.INTER_NEAREST)

    # depth_image = depth_image[start_h:finish_h, start_w:finish_w]
    return img, intrinsics, depth_image


# run MVS model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    # MVSDataset = find_dataset_def(args.dataset)
    # test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale, scaling=args.scaling)
    # TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    model = HAMMER(blocks=args.num_blocks, heads=args.num_heads, channels=args.num_channels, stages=args.num_stages, psi=args.psi, output_scaling=args.output_scale)

    model = nn.DataParallel(model)
    model.cuda()
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            
            start_time = time.time()
            outputs, entropy, cv_mask = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            print("time: {:.3f}".format(time.time() - start_time))

            outputs = tensor2numpy(outputs)
            entropy = tensor2numpy(entropy)
            cv_mask = tensor2numpy(cv_mask)

            del sample_cuda
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence, warp_mask in zip(filenames, outputs,
                                                                   entropy, cv_mask):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cv_mask_filename = os.path.join(args.outdir, filename.format('cv_mask', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cv_mask_filename.rsplit('/', 1)[0], exist_ok=True)
                
                # save depth maps
                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, entropy[0])
                save_pfm(cv_mask_filename, warp_mask)

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
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
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < args.dist, relative_depth_diff < 1/args.rel_dist) #0.01
    #mask = np.logical_and(dist < 2, relative_depth_diff < 0.02)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

def check_geometric_consistency_dyn(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, filter_num):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(0,filter_num):
        mask = np.logical_and(dist < (i+2)/args.dyn, relative_depth_diff < (i+2)/args.rel_dist)
        masks.append(mask)
    # mask = masks[1] #only average good close estimates
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src

def load_photo_mask(ref_view, ref_depth_est):
    try:
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
    except:
        confidence = np.ones_like(ref_depth_est)

    if(ref_depth_est.shape != confidence.shape):
        import cv2
        ratio = ref_depth_est.shape[0]/confidence.shape[0]
        confidence = cv2.resize(confidence, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    if (args.confidence > 0):
        photo_mask = confidence > args.confidence
    elif (args.ent_high > 0):
        photo_mask = np.logical_and(confidence < args.ent_high, confidence > args.ent_low)
    else:
        photo_mask = confidence > 0

    return photo_mask

def load_photo_mask_ent(ref_view, ref_depth_est, ent_high, ent_low):
    try:
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
    except:
        confidence = np.ones_like(ref_depth_est)

    if(ref_depth_est.shape != confidence.shape):
        import cv2
        ratio = ref_depth_est.shape[0]/confidence.shape[0]
        confidence = cv2.resize(confidence, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    if (args.confidence > 0):
        photo_mask = confidence > args.confidence
    elif (args.ent_high > 0):
        photo_mask = np.logical_and(confidence < ent_high, confidence > ent_low)
    else:
        photo_mask = confidence > 0

    return photo_mask


def filter_depth(scan_folder, out_folder, plyfilename, scan_name, use_dyn=False):
    # the pair file
    print("args.dataset: ", args.dataset)
    if(args.dataset=="blended_mvs"):
        pair_file = os.path.join(scan_folder, "cams/pair.txt")
    elif (args.dataset=="dtu_full_res"):
        pair_file = os.path.join(args.testpath, "Cameras/pair.txt")
    else:
        pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    use_dyn_consistency = use_dyn

    print("use_dyn: ", use_dyn)
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    print("nviews: ", nviews)
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    #Set to True to generate cross filtered GT depth maps (works only for DTU)
    generate_gt = False

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        print("ref_view: ", ref_view)

        if len(src_views) < args.consistent:
            continue

        readCamSuccess = True

        # load the camera parameters
        if (args.dataset=="dtu_full_res"):
            ref_intrinsics, ref_extrinsics, readCamSuccess = read_camera_parameters(
                os.path.join(args.testpath, 'Cameras/{:0>8}_cam.txt'.format(ref_view)))
        else:
            ref_intrinsics, ref_extrinsics, readCamSuccess = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))

        if not readCamSuccess:
            continue

        # load the reference image
        if(args.dataset=="blended_mvs"):
            ref_img = read_img(os.path.join(scan_folder, 'blended_images/{:0>8}.jpg'.format(ref_view)))
        elif (args.dataset=="dtu_full_res"):
            ref_img = read_img(os.path.join(args.testpath, "Rectified/" + scan_name + '/rect_{:0>3}_max.png'.format(ref_view+1)))
        else:
            ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))

        # load the estimated depth of the reference view
        if(generate_gt):
            ref_depth_est = read_pfm(os.path.join(args.testpath, 'Depths_raw/' + scan_name + '/depth_map_{:0>4}.pfm'.format(ref_view)))[0]
        else:
            ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]

        ref_img, ref_intrinsics, ref_depth_est = scale_and_crop_eval(args.input_scale, ref_img, ref_intrinsics, ref_depth_est)

        if(np.isnan(ref_depth_est.any())):
            continue

        photo_mask = load_photo_mask(ref_view, ref_depth_est)

        if(args.cv_mask > 0):
            cv_mask = read_pfm(os.path.join(out_folder, 'cv_mask/{:0>8}.pfm'.format(ref_view)))[0]
            if(ref_depth_est.shape != cv_mask.shape):
                import cv2
                ratio = ref_depth_est.shape[0]/cv_mask.shape[0]
                cv_mask = cv2.resize(cv_mask, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

            cv_mask = cv_mask > args.cv_mask
            # mask_new = photo_mask + cv_mask
            # print("cv-mask: ", cv_mask.mean())
            photo_mask = np.logical_and(photo_mask, cv_mask)
            
        if(True): #geometric filtering
            all_srcview_depth_ests = []
            all_srcview_x = []
            all_srcview_y = []
            all_srcview_geomask = []

            # compute the geometric mask
            geo_mask_sum = 0
            geo_mask_sums = []
            n = len(src_views) - 2
            # print("n: ", n)
            ct = 0
            for src_view in src_views:

                readCamSuccess = True

                # camera parameters of the source view
                if (args.dataset=="dtu_full_res"):
                    src_intrinsics, src_extrinsics, readCamSuccess = read_camera_parameters(
                    os.path.join(args.testpath, 'Cameras/{:0>8}_cam.txt'.format(src_view)))
                else:
                    src_intrinsics, src_extrinsics, readCamSuccess = read_camera_parameters(
                        os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))

                if not readCamSuccess:
                    continue

                ct += 1        

                if(args.dataset=="blended_mvs"):
                    src_img = read_img(os.path.join(scan_folder, 'blended_images/{:0>8}.jpg'.format(src_view)))
                elif (args.dataset=="dtu_full_res"):
                    src_img = read_img(os.path.join(args.testpath, "Rectified/" + scan_name + '/rect_{:0>3}_max.png'.format(src_view+1)))
                else:
                    src_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(src_view)))

                # src_img, src_intrinsics = MVSDataset.scale_and_crop(test_dataset, args.scaling, src_img, src_intrinsics)

                # the estimated depth of the source view
                if(generate_gt):
                    src_depth_est = read_pfm(os.path.join(args.testpath, 'Depths_raw/' + scan_name + '/depth_map_{:0>4}.pfm'.format(src_view)))[0]
                else:
                    src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

                src_img, src_intrinsics, src_depth_est = scale_and_crop_eval(args.input_scale, src_img, src_intrinsics, src_depth_est)

                if(np.isnan(src_depth_est.any())):
                    ct -= 1
                    continue

                print("Shapes: ", src_img.shape, ref_img.shape, src_depth_est.shape, ref_depth_est.shape)

                # filter entropy also in src views
                # photo_mask_src = load_photo_mask_ent(src_view, src_depth_est, 1.2, 0.1)

                # photo_mask_src = load_photo_mask(src_view, src_depth_est)
                # src_depth_est *= photo_mask_src

                if(use_dyn_consistency):
                    masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency_dyn(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                            src_depth_est,
                                                                            src_intrinsics, src_extrinsics, n)
                    # print("len(masks): ", len(masks))
                    if (ct==1):
                        for i in range(0,n):
                            geo_mask_sums.append(masks[i].astype(np.int32))
                    else :
                        for i in range(0,n):
                            # print("i: ", i)
                            geo_mask_sums[i]+=masks[i].astype(np.int32)
                else:
                    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                            src_depth_est,
                                                                            src_intrinsics, src_extrinsics)

                geo_mask_sum += geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)
                all_srcview_x.append(x2d_src)
                all_srcview_y.append(y2d_src)
                all_srcview_geomask.append(geo_mask)

            depth_est_averaged = ref_depth_est #(sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

            if(use_dyn_consistency):
                geo_mask=geo_mask_sum>=n
                min_views = args.consistent
                for i in range (min_views,n+min_views):
                    geo_mask=np.logical_or(geo_mask, geo_mask_sums[i-min_views]>=i)
                    # print(geo_mask.mean())
            else:
                geo_mask = geo_mask_sum >= args.consistent

            final_mask = np.logical_and(photo_mask, geo_mask)
        else:
            final_mask = photo_mask
            depth_est_averaged = ref_depth_est
            geo_mask = np.zeros_like(final_mask)

        if(args.filter_isolated):
            import cv2
            kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], np.uint8)
            kernel2 = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], np.uint8)

            final_mask_comp = np.invert(final_mask)
            hitormiss1 = cv2.morphologyEx(final_mask.astype(np.float32), cv2.MORPH_ERODE, kernel1)
            hitormiss2 = cv2.morphologyEx(final_mask_comp.astype(np.float32), cv2.MORPH_ERODE, kernel2)
            hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)

            mask_single = hitormiss > 0.5
            mask_single = np.invert(mask_single)

            final_mask = np.logical_and(final_mask, mask_single)

        if(generate_gt):
            os.makedirs(os.path.join(args.testpath, "Depths_cross/" + scan_name), exist_ok=True)
            filename_gt_cross = os.path.join(args.testpath, 'Depths_cross/' + scan_name + '/depth_map_{:0>4}.pfm'.format(ref_view))
            ref_depth_cross = ref_depth_est * geo_mask.astype(np.float32)
            save_pfm(filename_gt_cross, ref_depth_cross)
            continue
        else:
            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        print("valid_points", valid_points.mean())

        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        # if(args.model == 'mvsnet'):
        #     color = ref_img[1::4, 1::4, :][valid_points]
        # else:
        #     color = ref_img[1::2, 1::2, :][valid_points] #[1::4, 1::4, :]

        color = ref_img[valid_points]


        # color = None
        # if(args.scaling != 1):
        #     import cv2
        #     ref_img = cv2.resize(ref_img, None, fx=args.scaling, fy=args.scaling, interpolation=cv2.INTER_CUBIC)

        # crop_off_h = ref_img.shape[0]%32
        # crop_off_w = ref_img.shape[1]%32
        # if(crop_off_h > 0):
        #     color = ref_img[1:-crop_off_h:4, 1::4, :][valid_points]
        # if(crop_off_w > 0):
        #     color = ref_img[1::4, 1:-crop_off_w:4, :][valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]

        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    if(generate_gt):
        return

    print("1")
    vertexs = np.concatenate(vertexs, axis=0)
    print("2")
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    print("3")
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print("4")
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    print("5")

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    print("6")
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    print("7")
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    print("8")

    el = PlyElement.describe(vertex_all, 'vertex')
    print("9")
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def evaluate():
    eval_file_path = os.path.join(args.outdir, "eval_log.txt")
    os.makedirs(eval_file_path.rsplit('/', 1)[0], exist_ok=True)
    total_perc_error = 0
    total_abs_error = 0
    total_entropy_down_error = 0
    total_entropy_up_error = 0
    total_cv_mask_error = 0
    total_all_error = 0
    total_2mm_error = 0
    total_4mm_error = 0
    total_8mm_error = 0
    total_cv_mask = 0
    total_entropy_up = 0
    total_entropy_down = 0
    total_entropy_avg = 0
    eval_file = open(eval_file_path, "a")

    for batch_idx, sample in enumerate(TestImgLoader):
        filename = sample["filename"][0]
        eval_file.write(filename + "\n")

        depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
        # confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm_12.pfm'))
        confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
        cv_mask_filename = os.path.join(args.outdir, filename.format('cv_mask', '.pfm'))

        # eval_all_filename = os.path.join(args.outdir, filename.format('eval_all', '.pfm'))
        # os.makedirs(eval_all_filename.rsplit('/', 1)[0], exist_ok=True)
        # eval_perc_filename = os.path.join(args.outdir, filename.format('eval_perc', '.pfm'))
        # os.makedirs(eval_perc_filename.rsplit('/', 1)[0], exist_ok=True)
        # eval_conf_filename = os.path.join(args.outdir, filename.format('eval_conf', '.pfm'))
        # os.makedirs(eval_conf_filename.rsplit('/', 1)[0], exist_ok=True)

        entropy = read_pfm(confidence_filename)[0]
        depth_est = read_pfm(depth_filename)[0]
        cv_mask = read_pfm(cv_mask_filename)[0]

        num_pixels = depth_est.shape[0] * depth_est.shape[1]

        if "depth" in sample.keys():
            mask = sample["mask"].squeeze(0).data.cpu().numpy()
            depth_gt = sample["depth"].squeeze(0).data.cpu().numpy()
            # print("depth_gt_shape: ", depth_gt.shape)
            # print("depth_est_shape: ", depth_est.shape)
            eval_output = np.zeros(depth_gt.shape)
            eval_perc_output = np.zeros(depth_gt.shape)
        
            mask = mask > 0.5
            mask_num_valid = mask.sum()
            eval_output[mask] = abs(depth_est[mask] - depth_gt[mask])
            # save_pfm(eval_all_filename, np.float32(eval_output))

            eval_perc_output[mask] = eval_output[mask]/depth_gt[mask]
            # save_pfm(eval_perc_filename, np.float32(eval_perc_output))

            avg_error_perc = (eval_output[mask]/depth_gt[mask]).sum()/mask_num_valid
            total_perc_error += avg_error_perc

            avg_error_abs = eval_output.sum()/mask_num_valid
            total_abs_error += avg_error_abs

            total_2mm_error += (eval_output > 2).sum()/mask_num_valid
            total_4mm_error += (eval_output > 4).sum()/mask_num_valid
            total_8mm_error += (eval_output > 8).sum()/mask_num_valid

            # num_pixels = depth_gt.shape[0] * depth_gt.shape[1]

            #cv_mask
            max_cv_mask = 12 #112
            best_cv_mask = 0
            best_cv_mask_score = 0
            num_tries = 12 #112
            for i in range(num_tries):
                cv_mask_output = np.zeros(depth_gt.shape)
                mask_thresh = max_cv_mask - i
                if i == (num_tries-1):
                    mask_thresh = best_cv_mask
                mask_cv = (mask > 0.5) & (cv_mask > mask_thresh)
                mask_cv_num_valid = mask_cv.sum()
                cv_mask_output[mask_cv] = abs(depth_est[mask_cv] - depth_gt[mask_cv])
                avg_error_cv_mask = cv_mask_output.sum()/mask_cv_num_valid
                # eval_file.write("cv mask " + str(mask_thresh) + " avg error: " + str(avg_error_cv_mask) + "\n")
                if i == 0:
                    best_cv_mask_score = avg_error_cv_mask
                    best_cv_mask = mask_thresh
                else:
                    if(avg_error_cv_mask < best_cv_mask_score):
                        best_cv_mask_score = avg_error_cv_mask
                        best_cv_mask = mask_thresh

            total_cv_mask_error += best_cv_mask_score
            total_cv_mask += best_cv_mask

            #entropy up
            min_entropy = 0.001
            best_entropy_up = 0
            best_entropy_up_score = 0
            num_tries = 100
            for i in range(num_tries):
                ent_up_output = np.zeros(depth_gt.shape)
                # entropy_up_thresh = max_entropy - (i/100)
                entropy_up_thresh = min_entropy + (i/100)
                if i == (num_tries-1):
                    entropy_up_thresh = best_entropy_up
                mask_ent_up = (mask > 0.5) & (entropy.squeeze() > entropy_up_thresh)
                mask_ent_up_num_valid = mask_ent_up.sum()
                ent_up_output[mask_ent_up] = abs(depth_est[mask_ent_up] - depth_gt[mask_ent_up])
                avg_error_ent_up = ent_up_output.sum()/mask_ent_up_num_valid
                if i == 0:
                    best_entropy_up_score = avg_error_ent_up
                    best_entropy_up = entropy_up_thresh
                else:
                    if(avg_error_ent_up < best_entropy_up_score):
                        best_entropy_up_score = avg_error_ent_up
                        best_entropy_up = entropy_up_thresh

            # save_pfm(eval_conf_filename, np.float32(ent_up_output))
            total_entropy_up_error += best_entropy_up_score
            total_entropy_up += best_entropy_up

            #entropy down
            max_entropy = 1.38
            best_entropy_down = 0
            best_entropy_down_score = 0
            num_tries = 100
            for i in range(num_tries):
                ent_down_output = np.zeros(depth_gt.shape)
                entropy_down_thresh = max_entropy - (i/100)
                # entropy_down_thresh = min_entropy + (i/100)
                if i == (num_tries-1):
                    entropy_down_thresh = best_entropy_down
                mask_ent_down = (mask > 0.5) & (entropy.squeeze() < entropy_down_thresh)
                mask_ent_down_num_valid = mask_ent_down.sum()
                ent_down_output[mask_ent_down] = abs(depth_est[mask_ent_down] - depth_gt[mask_ent_down])
                avg_error_ent_down = ent_down_output.sum()/mask_ent_down_num_valid
                if i == 0:
                    best_entropy_down_score = avg_error_ent_down
                    best_entropy_down = entropy_down_thresh
                else:
                    if(avg_error_ent_down < best_entropy_down_score):
                        best_entropy_down_score = avg_error_ent_down
                        best_entropy_down = entropy_down_thresh

            total_entropy_down_error += best_entropy_down_score
            total_entropy_down += best_entropy_down

            all_output = np.zeros(depth_gt.shape)
            mask_all = (mask_ent_down > 0.5)
            mask_test = (mask_all > 0.5) & (mask_cv > 0.5)
            if mask_test.sum() > 0: # check if there are any valid values
                mask_all = mask_test
            mask_test = (mask_all > 0.5) & (mask_ent_up > 0.5)
            if mask_test.sum() > 0: # check if there are any valid values
                mask_all = mask_test
            mask_all_num_valid = mask_all.sum()
            print("mask_all_num_valid: ", mask_all_num_valid)
            all_output[mask_all] = abs(depth_est[mask_all] - depth_gt[mask_all])
            avg_error_all = all_output.sum()/mask_all_num_valid

            # save_pfm(eval_conf_filename, np.float32(all_output))

            total_all_error += avg_error_all

            eval_file.write("avg error: " + str(avg_error_abs) + " num valid: " + str(mask_num_valid) + "\n")
            eval_file.write("avg error perc: " + str(avg_error_perc) + "\n")
            eval_file.write("entropy down " + str(best_entropy_down) + " avg error: " + str(best_entropy_down_score) + " num valid: " + str(mask_ent_down_num_valid) + "\n")
            eval_file.write("entropy up " + str(best_entropy_up) + " avg error: " + str(best_entropy_up_score) + " num valid: " + str(mask_ent_up_num_valid) + "\n")
            eval_file.write("cv mask " + str(best_cv_mask) + " avg error: " + str(best_cv_mask_score) + " num valid: " + str(mask_cv_num_valid) + "\n")
            eval_file.write("avg error all " + str(avg_error_all) + " num valid: " + str(mask_all_num_valid) + "\n")

        avg_entropy_value = entropy.squeeze().sum()/num_pixels
        total_entropy_avg += avg_entropy_value

        eval_file.write("avg entropy value: " + str(avg_entropy_value) + "\n")
        eval_file.write("\n\n")

    total_perc_error = total_perc_error / len(TestImgLoader)
    total_abs_error = total_abs_error / len(TestImgLoader)
    total_2mm_error = total_2mm_error / len(TestImgLoader)
    total_4mm_error = total_4mm_error / len(TestImgLoader)
    total_8mm_error = total_8mm_error / len(TestImgLoader)
    total_entropy_down_error = total_entropy_down_error / len(TestImgLoader)
    total_entropy_down = total_entropy_down / len(TestImgLoader)
    total_entropy_up_error = total_entropy_up_error / len(TestImgLoader)
    total_entropy_up = total_entropy_up / len(TestImgLoader)
    total_all_error = total_all_error / len(TestImgLoader)
    total_cv_mask = total_cv_mask / len(TestImgLoader)
    total_entropy_avg = total_entropy_avg / len(TestImgLoader)
    print("Total perc error: ", total_perc_error)
    print("Total abs error: ", total_abs_error)
    print("Total ent down error: ", total_entropy_down_error)
    print("Total ent up error: ", total_entropy_up_error)
    eval_file.write("total perc error: " + str(total_perc_error) + "\n")
    eval_file.write("total abs error: " + str(total_abs_error) + "\n")
    eval_file.write("total ent down error: " + str(total_entropy_down_error) + "\n")
    eval_file.write("total ent up error: " + str(total_entropy_up_error) + "\n")
    eval_file.write("total all error: " + str(total_all_error) + "\n")
    eval_file.write("total 2mm error: " + str(total_2mm_error) + "\n")
    eval_file.write("total 4mm error: " + str(total_4mm_error) + "\n")
    eval_file.write("total 8mm error: " + str(total_8mm_error) + "\n\n")
    eval_file.write("average entropy value: " + str(total_entropy_avg) + "\n")
    eval_file.write("best entropy down value: " + str(total_entropy_down) + "\n")
    eval_file.write("best entropy up value: " + str(total_entropy_up) + "\n")
    eval_file.write("best cv mask value: " + str(total_cv_mask) + "\n")
    eval_file.close()


if __name__ == '__main__':
    #step 1: save all the depth maps and the masks in outputs directory
    save_depth()

    print("Finished depth inference...")

    #step 2: evaluate depth maps
    if(args.evaluate):
        evaluate()

    #step 3: fusion
    if not args.depth_only:
        with open(args.testlist) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        dyn_filtering = False

        filename = ""
        ent_low = str(args.ent_low).replace('.','')
        ent_high = str(args.ent_high).replace('.','')

        if(args.dyn > 0):
            dyn_filtering = True
            filename = f"c{args.consistent}_dyn{args.dyn}_r{args.rel_dist}_cv{args.cv_mask}_ent{ent_low}_{ent_high}"
        else:
            dist = str(args.dist).replace('.','')
            filename = f"c{args.consistent}_dist{dist}_r{args.rel_dist}_cv{args.cv_mask}_ent{ent_low}_{ent_high}"

        filename = "{}_" + filename + "_geo48.ply"
        # filename="{}.ply"

        print("filename: ", filename)

        count = 0
        for scan in scans:
            count += 1
            if (count <= 0): #skip x image sets 
                continue
            scan_folder = os.path.join(args.testpath, scan)
            out_folder = os.path.join(args.outdir, scan)
            # step2. filter saved depth maps with photometric confidence maps and geometric constraints
            # filter_depth(scan_folder, out_folder, os.path.join(args.outdir, 'mvsnet_{}_c3_dist025.ply'.format(scan)), scan)
            filter_depth(scan_folder, out_folder, os.path.join(args.outdir, filename.format(scan)), scan, use_dyn=dyn_filtering)
            
