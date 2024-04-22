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
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation for training ATLAS-MVSNet')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') #0.001
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=384, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.0, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed') #1

parser.add_argument('--augment_data', action='store_true', help='augment data')
parser.add_argument('--output_scale', type=int, default=2, help='dividing scaling factor (int) for the output depth image')
parser.add_argument('--input_scale', type=float, default=1.0, help='multiplicative scaling factor (float) for the input rgb image')

parser.add_argument('--neighbors', type=int, default=3, help='number of image neighbors stacked in the network')

parser.add_argument('--num_blocks', type=int, default=5, help='number of 3D regularization blocks')
parser.add_argument('--num_heads', type=int, default=1, help='number of attention heads for 2D and 3D')
parser.add_argument('--num_channels', type=int, default=32, help='number of maximum channels, has to be divisible by 4 (and 3 if 3D HAB is used)')
parser.add_argument('--num_stages', type=int, default=10, help='number of regularization stages, should be 8 or 10')
parser.add_argument('--psi', type=float, default=0.6, help='depth interval scale parameter psi')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("current time", current_time_str)
print("creating new summary file")
logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.neighbors, args.numdepth, args.interval_scale, augment_data=args.augment_data, depth_scaling=args.output_scale, scaling=args.input_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.neighbors, args.numdepth, args.interval_scale, augment_data=args.augment_data, depth_scaling=args.output_scale, scaling=args.input_scale)
#test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True) #8
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False) #4

# create model
print("training HAMMER model")
model = HAMMER(blocks=args.num_blocks, heads=args.num_heads, channels=args.num_channels, stages=args.num_stages, psi=args.psi, output_scaling=args.output_scale)
model_loss = hammer_loss

model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd) #betas=(0.9, 0.999)

# load parameters
start_epoch = 0
if (args.resume):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

print("CUDA mem usage: ", torch.cuda.memory_allocated())

# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            batch_invalid = False
            for filename in sample["filename"]:
                if(filename == "FAILED"):
                    batch_invalid = True
            if(batch_invalid):
                print("Skipping invalid batch!")
                continue

            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)

            if( (loss != loss) ): #check for nan
                print("NAN LOSS at sample: ", sample["filename"])
                error_file_path = os.path.join(args.logdir, "error_log.txt")
                error_file = open(error_file_path, "a")
                error_file.write("NAN LOSS at sample: ")
                for fn in sample["filename"]:
                    error_file.write(fn + " ")
                error_file.write("\nglobal_step: " + str(global_step) + "\n")
                error_file.close()
                do_summary=False

            if( loss < -1 ): #check loss (< -1 to ignore)
                loss_file_path = os.path.join(args.logdir, "loss_log.txt")
                loss_file = open(loss_file_path, "a")
                loss_file.write(f"{loss} loss at sample: ")
                for fn in sample["filename"]:
                    loss_file.write(fn + " ")
                loss_file.write("\n")
                loss_file.close()
                do_summary=False

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs , image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    depth_gt = depth_gt_ms["fullRes"]
    mask = sample_cuda["mask"]

    outputs, entropy, cv_mask = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    cv_mask = cv_mask > 0 #1 #32
    cv_mask = tocuda(cv_mask)

    cv_mask = cv_mask > 0.5
    depth_gt = depth_gt_ms["fullRes"]

    if(outputs[0].shape == depth_gt.shape):
        mask_output = depth_gt > 0 & cv_mask
        mask_output.detach_()
        depth_gt_output = depth_gt
    else:
        mask_output = depth_gt_ms["stage0"] > 0 & cv_mask
        depth_gt_output = depth_gt_ms["stage0"]

    mask_output.detach_()

    depth_est = outputs[0]

    loss_depth = model_loss(outputs, depth_gt_ms, cv_mask)
    # loss_entropy = calc_loss_entropy(entropy, mask)
    depth_min = float(sample_cuda["depth_values"][0, 0].cpu().numpy())
    depth_max = float(sample_cuda["depth_values"][0, -1].cpu().numpy())
    loss_ent_0, loss_ent_max, thresh = calc_masked_loss_entropy_last_stage(outputs, depth_gt_output, entropy, mask_output, depth_min, depth_max)

    loss = loss_depth + (loss_ent_0 + loss_ent_max) * thresh

    if (loss != loss).any():
        print("NAN LOSS! loss.backward() will be skipped!")
    elif (loss < -1):
        print(f"Loss is {loss}! loss.backward() will be skipped!")
    else:
        loss.backward()
        optimizer.step()

    mask_output = mask_output.type(torch.cuda.FloatTensor)

    scalar_outputs = {"loss": loss,
                      "loss_depth": loss_depth,
                      "loss_ent_0": loss_ent_0,
                      "loss_ent_max": loss_ent_max,
                      "thresh": thresh}
    image_outputs = {"depth_est": depth_est, 
                     "depth_gt": depth_gt_output, #depth_est * mask
                     "ref_img": sample["imgs"][:, 0],
                     "entropy0": entropy[0],
                     "entropy2": entropy[2],
                     "entropy4": entropy[4],
                     "entropy6": entropy[6],
                     "mask": mask_output} #sample["mask"]
    if detailed_summary:
        # image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt_output, mask_output > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt_output, mask_output > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt_output, mask_output > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt_output, mask_output > 0.5, 8)

    print("loss: ", loss)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

def calc_masked_loss_entropy_last_stage(stacked, depth_gt_output, entropy, mask_output, depth_min=0, depth_max=0):
    thresh = torch.tensor(1.) #F.smooth_l1_loss(stacked[0][mask], depth_gt[mask], reduction='mean')
    if(depth_min > 0):
        thresh = 1* (depth_max - depth_min) / pow(2, 9)
        thresh = torch.tensor(thresh)

    print("threshold: ", thresh)
    ent_max = (-0.25*np.log(0.25))*4

    mask0 = F.smooth_l1_loss(stacked[0][mask_output], depth_gt_output[mask_output], reduction='none') <= thresh*1

    loss_ent_0 = F.smooth_l1_loss(entropy[0][mask_output][mask0], torch.zeros_like(entropy[0][mask_output][mask0]), reduction='mean')

    mask0 = F.smooth_l1_loss(stacked[0][mask_output], depth_gt_output[mask_output], reduction='none') > thresh*1

    loss_ent_max = F.smooth_l1_loss(entropy[0][mask_output][mask0], torch.zeros_like(entropy[0][mask_output][mask0]).fill_(ent_max), reduction='mean')
    
    if(loss_ent_0 != loss_ent_0):
        loss_ent_0 = torch.tensor(0.)

    if(loss_ent_max != loss_ent_max):
        loss_ent_max = torch.tensor(0.)

    return loss_ent_0, loss_ent_max, thresh


if __name__ == '__main__':
    train()